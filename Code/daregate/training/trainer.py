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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    tqdm = None

from ..model import DAREGateModel
from .batch import DAREGateBatch, as_daregate_batch
from .config import ObjectiveConfig, TrainerConfig
from .joint_losses import JointObjective
from .metrics import ResponseMetrics
from .output import TrainingOutputStore


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
        self.objective = JointObjective(objective_config)
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
        self._active_stage: str | None = None
        self._pair_generator = torch.Generator(device=self.device)
        self._pair_generator.manual_seed(self.trainer_config.seed)
        torch.manual_seed(self.trainer_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.trainer_config.seed)

        self.output_store = (
            TrainingOutputStore(self.trainer_config.output_dir)
            if self.trainer_config.output_dir is not None
            else None
        )
        if self.trainer_config.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
        elif self.output_store is not None:
            self.checkpoint_dir = self.output_store.directory("checkpoints")
        else:
            self.checkpoint_dir = None
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


    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, message: str) -> None:
        print(f"[{self._timestamp()}] {message}", flush=True)

    def _progress(
        self,
        loader: Iterable[Any],
        *,
        epoch: int,
        training: bool,
    ) -> Any | None:
        if tqdm is None or not self.trainer_config.show_progress:
            return None
        try:
            total = len(loader)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None
        phase = "train" if training else "valid"
        target_epochs = getattr(self, "_target_epochs", None)
        epoch_text = (
            f"{epoch + 1}/{target_epochs}"
            if target_epochs is not None
            else str(epoch + 1)
        )
        stage = self.objective.stage_for_epoch(epoch)
        return tqdm(
            loader,
            total=total,
            desc=f"{self._timestamp()} {phase} E{epoch_text} {stage}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            mininterval=self.trainer_config.progress_min_interval,
            disable=None,
        )

    def _progress_postfix(
        self, loss_sums: dict[str, float], node_count: int, *, epoch: int
    ) -> dict[str, str]:
        values = {
            "loss": f"{loss_sums['loss'] / node_count:.4f}",
            "prob": f"{loss_sums['probability_loss'] / node_count:.4f}",
            "rc": f"{loss_sums['reconvergence_loss'] / node_count:.4f}",
            "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
        }
        if epoch >= self.objective_config.functionality_start_epoch:
            values["func"] = f"{loss_sums['functionality_loss'] / node_count:.4f}"
        if epoch >= self.objective_config.behavior_start_epoch:
            values["recon"] = (
                f"{loss_sums['reconstruction_loss'] / node_count:.4f}"
            )
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            values["gpu"] = f"{allocated:.2f}G"
        return values

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
            "probability_loss": 0.0,
            "reconvergence_loss": 0.0,
            "functionality_loss": 0.0,
            "reconstruction_loss": 0.0,
            "similarity_loss": 0.0,
            "decoupling_loss": 0.0,
        }
        node_count = 0
        batch_count = 0
        start_time = time.perf_counter()
        grad_context = torch.enable_grad if training else torch.no_grad
        progress = self._progress(loader, epoch=epoch, training=training)
        epoch_loader = progress if progress is not None else loader

        with grad_context():
            for batch_number, raw_batch in enumerate(epoch_loader, start=1):
                batch = self._prepare_batch(raw_batch)
                batch_count += 1
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    model_kwargs = batch.model_kwargs()
                    model_kwargs["compute_behavior"] = (
                        epoch >= self.objective_config.behavior_start_epoch
                    )
                    output = self.model(**model_kwargs)
                    losses = self.objective(
                        output, batch,
                        epoch=epoch,
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

                if progress is not None:
                    progress.set_postfix(
                        self._progress_postfix(loss_sums, node_count, epoch=epoch),
                        refresh=False,
                    )
                interval = self.trainer_config.log_interval
                progress_visible = progress is not None and not progress.disable
                if (
                    training and not progress_visible and interval > 0
                    and batch_number % interval == 0
                ):
                    self._log(
                        f"epoch={epoch + 1} batch={batch_number} "
                        f"stage={losses.stage} loss={detached['loss']:.6f}"
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

    @torch.no_grad()
    def export_outputs(
        self, loader: Iterable[Any], *, split: str
    ) -> list[tuple[Path, Path]]:
        """Stream embeddings and predictions into classified output folders."""
        if self.output_store is None:
            raise RuntimeError("TrainerConfig.output_dir is required for export")
        if not split or "/" in split or "\\" in split:
            raise ValueError("split must be a non-empty directory-safe name")
        was_training = self.model.training
        saved: list[tuple[Path, Path]] = []
        self.model.eval()
        try:
            for batch_number, raw_batch in enumerate(loader):
                batch = self._prepare_batch(raw_batch)
                output = self.model(**batch.model_kwargs())
                saved.append(
                    self.output_store.save_batch(
                        split, batch_number, batch, output
                    )
                )
        finally:
            self.model.train(was_training)
        return saved

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
        late_train_loader: Iterable[Any] | None = None,
        train_loader_switch_epoch: int | None = None,
    ) -> list[dict[str, Any]]:
        """Train until ``num_epochs`` total epochs, respecting resumed state."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if (late_train_loader is None) != (train_loader_switch_epoch is None):
            raise ValueError(
                "late_train_loader and train_loader_switch_epoch "
                "must be provided together"
            )
        if train_loader_switch_epoch is not None and train_loader_switch_epoch < 0:
            raise ValueError("train_loader_switch_epoch cannot be negative")
        if self.start_epoch >= num_epochs:
            return self.history

        self._target_epochs = num_epochs
        active_loader_phase: str | None = None

        for epoch in range(self.start_epoch, num_epochs):
            stage = self.objective.stage_for_epoch(epoch)
            if stage != self._active_stage:
                self._active_stage = stage
                self.best_metric = None
                self._epochs_without_improvement = 0
                self._log(f"enter training stage: {stage}")
            use_late_loader = (
                late_train_loader is not None
                and train_loader_switch_epoch is not None
                and epoch >= train_loader_switch_epoch
            )
            epoch_train_loader = late_train_loader if use_late_loader else train_loader
            loader_phase = "joint" if use_late_loader else "prebehavior"
            if loader_phase != active_loader_phase:
                active_loader_phase = loader_phase
                batch_size = getattr(epoch_train_loader, "batch_size", "unknown")
                self._log(
                    f"use {loader_phase} train loader: batch_size={batch_size}"
                )
            train_result = self.train_epoch(epoch_train_loader, epoch)
            validation_result = (
                self.evaluate(validation_loader, epoch=epoch)
                if validation_loader is not None
                else None
            )
            monitored_result = validation_result or train_result
            monitor_name = (
                self.trainer_config.monitor
                if stage == "joint"
                else "loss"
            )
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
                "stage": stage,
                "monitor": monitor_name,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_result,
                "validation": validation_result,
                "improved": improved,
            }
            self.history.append(record)
            self.start_epoch = epoch + 1
            if self.output_store is not None:
                self.output_store.write_epoch(record, self.history)

            if (
                self.checkpoint_dir is not None
                and self.start_epoch % self.trainer_config.save_every == 0
            ):
                self.save_checkpoint(self.checkpoint_dir / "model_last.pt")

            validation_text = (
                f" val_{monitor_name}={monitored_metric:.6f}"
                if validation_result is not None
                else ""
            )
            self._log(
                f"epoch={epoch + 1}/{num_epochs} "
                f"stage={stage} train_loss={train_result['loss']:.6f}{validation_text}"
            )
            if self.epoch_callback is not None:
                self.epoch_callback(epoch + 1, record)

            patience = self.trainer_config.early_stopping_patience
            if (
                stage == "joint"
                and patience is not None
                and self._epochs_without_improvement >= patience
            ):
                self._log(f"early stopping after epoch {epoch + 1}")
                break

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
            "active_stage": self._active_stage,
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
            self._active_stage = checkpoint.get("active_stage")
            self._epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )
            self.history = list(checkpoint.get("history", []))
        return checkpoint

