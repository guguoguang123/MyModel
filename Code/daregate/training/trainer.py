"""Lifecycle manager for training, validation and checkpoint recovery."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..model import DAREGateModel
from .batch import DAREGateBatch, as_daregate_batch
from .config import ObjectiveConfig, TrainerConfig
from .losses import BehaviorObjective
from .metrics import ResponseMetrics


class DAREGateTrainer:
    """Manage optimization without coupling training state to the model class."""

    def __init__(
        self,
        model: DAREGateModel,
        *,
        trainer_config: TrainerConfig | None = None,
        objective_config: ObjectiveConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.trainer_config = trainer_config or TrainerConfig()
        self.objective = BehaviorObjective(objective_config)
        self.device = self._resolve_device(self.trainer_config.device)
        self.model.to(self.device)
        self.objective.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.trainer_config.learning_rate,
            weight_decay=self.trainer_config.weight_decay,
        )
        self.scheduler = scheduler
        self.use_amp = self.trainer_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.epoch_callback = epoch_callback

        self.start_epoch = 0
        self.best_metric: float | None = None
        self.history: list[dict[str, Any]] = []
        self._epochs_without_improvement = 0
        self._pair_generator = torch.Generator(device=self.device)
        self._pair_generator.manual_seed(self.trainer_config.seed)
        torch.manual_seed(self.trainer_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.trainer_config.seed)

        self.checkpoint_dir = (
            Path(self.trainer_config.checkpoint_dir)
            if self.trainer_config.checkpoint_dir is not None
            else None
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    @property
    def objective_config(self) -> ObjectiveConfig:
        return self.objective.config

    def _expected_signature_dim(self) -> int:
        config = self.model.config
        return config.num_probes * config.behavior_channels

    def _prepare_batch(self, raw_batch: Any) -> DAREGateBatch:
        batch = as_daregate_batch(raw_batch)
        batch.validate(expected_signature_dim=self._expected_signature_dim())
        return batch.to(self.device, non_blocking=self.device.type == "cuda")

    def _run_epoch(
        self,
        loader: Iterable[Any],
        *,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        self.model.train(training)
        metrics = ResponseMetrics()
        loss_sums = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "similarity_loss": 0.0,
            "decoupling_loss": 0.0,
        }
        node_count = 0
        batch_count = 0
        start_time = time.perf_counter()
        grad_context = torch.enable_grad if training else torch.no_grad

        with grad_context():
            for batch_number, raw_batch in enumerate(loader, start=1):
                batch = self._prepare_batch(raw_batch)
                batch_count += 1
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    output = self.model(**batch.model_kwargs())
                    losses = self.objective(
                        output,
                        batch.signature,
                        epoch=epoch,
                        batch_index=batch.batch_index,
                        generator=self._pair_generator,
                    )

                if not torch.isfinite(losses.total):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1}, batch {batch_number}"
                    )
                if training:
                    self.scaler.scale(losses.total).backward()
                    if self.trainer_config.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.trainer_config.gradient_clip_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                detached = losses.detached()
                for name in loss_sums:
                    loss_sums[name] += detached[name] * batch.num_nodes
                node_count += batch.num_nodes
                metrics.update(output.signature.detach(), batch.signature.detach())

                interval = self.trainer_config.log_interval
                if training and interval > 0 and batch_number % interval == 0:
                    print(
                        f"epoch={epoch + 1} batch={batch_number} "
                        f"loss={detached['loss']:.6f}"
                    )

        if batch_count == 0 or node_count == 0:
            raise RuntimeError("training or validation loader produced no batches")
        result = {name: value / node_count for name, value in loss_sums.items()}
        result.update(metrics.compute())
        result["seconds"] = time.perf_counter() - start_time
        result["batches"] = float(batch_count)
        result["nodes"] = float(node_count)
        return result

    def train_epoch(self, loader: Iterable[Any], epoch: int) -> dict[str, float]:
        return self._run_epoch(loader, epoch=epoch, training=True)

    def evaluate(self, loader: Iterable[Any], *, epoch: int | None = None) -> dict[str, float]:
        effective_epoch = self.start_epoch if epoch is None else epoch
        return self._run_epoch(loader, epoch=effective_epoch, training=False)

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.trainer_config.monitor_mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def _step_scheduler(self, monitored_metric: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(monitored_metric)
        else:
            self.scheduler.step()

    def fit(
        self,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any] | None,
        *,
        num_epochs: int,
    ) -> list[dict[str, Any]]:
        """Train until ``num_epochs`` total epochs, respecting resumed state."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.start_epoch >= num_epochs:
            return self.history

        for epoch in range(self.start_epoch, num_epochs):
            train_result = self.train_epoch(train_loader, epoch)
            validation_result = (
                self.evaluate(validation_loader, epoch=epoch)
                if validation_loader is not None
                else None
            )
            monitored_result = validation_result or train_result
            monitor_name = self.trainer_config.monitor
            if monitor_name not in monitored_result:
                raise KeyError(
                    f"monitor '{monitor_name}' is not available; "
                    f"choose from {sorted(monitored_result)}"
                )
            monitored_metric = monitored_result[monitor_name]
            improved = self._is_improvement(monitored_metric)
            if improved:
                self.best_metric = monitored_metric
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

            self._step_scheduler(monitored_metric)
            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_result,
                "validation": validation_result,
                "improved": improved,
            }
            self.history.append(record)
            self.start_epoch = epoch + 1

            if self.checkpoint_dir is not None:
                if improved:
                    self.save_checkpoint(self.checkpoint_dir / "model_best.pt")
                if self.start_epoch % self.trainer_config.save_every == 0:
                    self.save_checkpoint(self.checkpoint_dir / "model_last.pt")

            validation_text = (
                f" val_{monitor_name}={monitored_metric:.6f}"
                if validation_result is not None
                else ""
            )
            print(
                f"epoch={epoch + 1}/{num_epochs} "
                f"train_loss={train_result['loss']:.6f}{validation_text}"
            )
            if self.epoch_callback is not None:
                self.epoch_callback(epoch + 1, record)

            patience = self.trainer_config.early_stopping_patience
            if patience is not None and self._epochs_without_improvement >= patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir / "model_last.pt")
        return self.history

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "epoch": self.start_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "epochs_without_improvement": self._epochs_without_improvement,
            "history": self.history,
            "model_config": asdict(self.model.config),
            "trainer_config": asdict(self.trainer_config),
            "objective_config": asdict(self.objective_config),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save all state required for exact training continuation."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_state(), temporary)
        os.replace(temporary, destination)
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Restore model and, by default, the complete optimization state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None and checkpoint.get("scheduler") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler") is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.start_epoch = int(checkpoint.get("epoch", 0))
            self.best_metric = checkpoint.get("best_metric")
            self._epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
            self.history = list(checkpoint.get("history", []))
        return checkpoint

