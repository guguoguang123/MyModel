"""Whole-design splits, staged optimization, validation and exact resume."""

from __future__ import annotations

import json
import os
import random
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .utils.checkpoint import atomic_save, load as load_checkpoint, random_state, restore_random_state
from .utils.losses import loss_terms
from .utils.config import DEFAULT_EXP_DIR
from .utils.logger import AverageMeter, BatchProgress, TrainingLogger, format_seconds, info
from .utils.metrics import batch_metrics
from .utils.random_seed import set_seed
from .utils.model_utils import model_outputs
from .utils.batch import SCHEDULE_KEYS, prepare_topology


def move(item, device):
    # PyG .to() mutates its stores; use a shallow copy to keep cached CPU graphs.
    return prepare_topology(copy(item)).to(device)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run_directory(args):
    if args.resume is not None:
        path = Path(args.resume)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name != "last.pt":
            raise ValueError("resume must point to the run's last.pt")
        if args.output is not None and args.output.resolve() != path.parent.resolve():
            raise ValueError("--output conflicts with --resume run directory")
        return path.parent
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = DEFAULT_EXP_DIR / f"triwave_gnn_{stamp}"
    else:
        path = Path(args.output)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory already contains a run: {path}; use --resume or a new path")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint(model, optimizer, dataset, args, splits, stage, epoch, best_val, completed):
    return {
        "architecture_id": model.architecture_id,
        "graph_backend": model.graph_backend,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage": stage, "epoch": epoch, "stage_complete": completed, "best_val": best_val,
        "split_ids": {key: [dataset.ids[i] for i in value] for key, value in splits.items()},
        "source_sha256": dataset.source_sha256,
        "logic_vocab": dataset.logic_vocab, "role_vocab": dataset.role_vocab, "pin_vocab": dataset.pin_vocab,
        "feature_normalization": {
            relation: {"mean": getattr(dataset, relation + "_mean"), "std": getattr(dataset, relation + "_std")}
            for relation in ("net", "cell")
        },
        "random_state": random_state(),
    }


class TriWaveTrainer:
    """Train the three representation stages over datasets supplied by NpzParser."""

    def __init__(self, model, args, dataset, splits):
        self.model = model.to(args.device)
        self.args = args
        self.dataset = dataset
        self.splits = {name: list(indices) for name, indices in splits.items()}
        if args.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if args.patience < 0 or args.hidden <= 0 or args.lr <= 0 or args.lr_step < 0 or args.batch_size <= 0 or args.num_workers < 0 or args.progress_interval < 0:
            raise ValueError("patience, lr_step, num_workers and progress_interval must be nonnegative; hidden, lr and batch_size must be positive")
        if any(getattr(args, f"stage{stage}_epochs") <= 0 for stage in (1, 2, 3)):
            raise ValueError("every stage needs at least one epoch")
        if not all(self.splits.values()):
            raise ValueError("train, validation and test datasets must be nonempty")
        set_seed(args.seed)
        self.run_dir = _run_directory(args)
        args.output = self.run_dir
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.history = []
        self.next_stage, self.next_epoch = 1, 1
        self.selected_stage = None
        split_ids = {name: [dataset.ids[i] for i in indices]
                     for name, indices in self.splits.items()}
        if args.resume is None:
            dataset.fit_normalization(self.splits["train"])
            _write_json(self.run_dir / "split.json", split_ids)
            _write_json(self.run_dir / "run.json", {
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                "source_sha256": dataset.source_sha256,
                "graph_backend": model.graph_backend,
                "design_count": len(dataset),
                "selected_design_count": sum(map(len, self.splits.values())),
                "schema_version": str(dataset.graph["schema_version"]),
            })
            info(f"Run: {self.run_dir}")
            info(f"Device={args.device} | batch_size={args.batch_size} | "
                 f"designs={sum(map(len, self.splits.values()))}/{len(dataset)} | "
                 f"resume checkpoint every {args.save_interval} epochs")
        else:
            saved = load_checkpoint(args.resume, args.device)
            if saved.get("architecture_id") != model.architecture_id:
                raise ValueError("checkpoint uses an older model architecture; start a new run")
            if saved["source_sha256"] != dataset.source_sha256:
                raise ValueError("resume data hash differs; start a new run for replaced NPZ files")
            for key in ("hidden", "seed", "max_nodes", "max_designs", "lr", "lr_step", "split_mode"):
                recorded = saved["config"].get(key, {"lr_step": 20, "max_designs": 0,
                                                     "split_mode": "random"}.get(key))
                if recorded != getattr(args, key):
                    raise ValueError(f"resume configuration differs: {key}")
            if saved["split_ids"] != split_ids:
                raise ValueError("resume split differs from the checkpoint")
            for key, current in (("logic_vocab", dataset.logic_vocab),
                                 ("role_vocab", dataset.role_vocab),
                                 ("pin_vocab", dataset.pin_vocab)):
                if saved[key] != current:
                    raise ValueError(f"resume {key} differs from the checkpoint")
            for relation, stats in saved["feature_normalization"].items():
                setattr(dataset, relation + "_mean", stats["mean"])
                setattr(dataset, relation + "_std", stats["std"])
            model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            restore_random_state(saved["random_state"])
            # Later unsaved epochs may have replaced the best file. Restore the
            # selection as it stood at this resume point before replaying them.
            if saved.get("stage_best_checkpoint") is not None:
                atomic_save(saved["stage_best_checkpoint"],
                            self.run_dir / f"stage{saved['stage']}_best.pt")
            self.history = json.loads((self.run_dir / "history.json").read_text())
            self.history = [row for row in self.history
                            if row["stage"] < saved["stage"] or
                            (row["stage"] == saved["stage"] and row["epoch"] <= saved["epoch"])]
            _write_json(self.run_dir / "history.json", self.history)
            self.next_stage = saved["stage"] + int(saved["stage_complete"])
            self.next_epoch = 1 if saved["stage_complete"] else saved["epoch"] + 1
            info(f"Resume: stage={self.next_stage} epoch={self.next_epoch} | {args.resume}")
            previous_batch_size = saved["config"].get("batch_size", 1)
            if previous_batch_size != args.batch_size:
                info(f"Resume batch_size={previous_batch_size} -> {args.batch_size}; "
                     "optimizer state retained; subsequent updates use the new batch size")
        total = sum(getattr(args, f"stage{stage}_epochs") for stage in (1, 2, 3))
        self.timeline = TrainingLogger(self.run_dir, total=max(1, total - len(self.history)))

    def run_batch(self, item, stage):
        """Embed one graph, apply task heads and calculate stage losses."""
        output = model_outputs(self.model, item, include_behavior=stage >= 3)
        total, terms = loss_terms(output, item, stage)
        return output, total, terms

    def _save_resume(self, stage, epoch, best, completed=False):
        state = _checkpoint(self.model, self.optimizer, self.dataset, self.args,
                            self.splits, stage, epoch, best, completed)
        # Keep the best checkpoint in the same atomic file as the resume state,
        # so interruptions between periodic saves cannot change model selection.
        state["stage_best_checkpoint"] = load_checkpoint(
            self.run_dir / f"stage{stage}_best.pt", "cpu")
        atomic_save(state, self.run_dir / "last.pt")

    def _run_epoch(self, indices, stage, training=False, epoch=0, phase=None):
        self.model.train(training)
        rows = []
        loader = DataLoader(Subset(self.dataset, indices), batch_size=self.args.batch_size,
                            shuffle=False, num_workers=self.args.num_workers,
                            exclude_keys=SCHEDULE_KEYS)
        batch_loss = AverageMeter()
        term_stats = {}
        metric_sums = {}
        metric_counts = {}
        phase = phase or ("train" if training else "val")
        with BatchProgress(phase, stage, epoch, len(loader), self.args.progress_interval,
                           epoch_total=getattr(self.args, f"stage{stage}_epochs")) as progress:
            for batch_number, item in enumerate(loader, 1):
                item = move(item, self.args.device)
                with torch.set_grad_enabled(training):
                    output, total, terms = self.run_batch(item, stage)
                metrics = batch_metrics(output, item, stage)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                names = item["name"] if isinstance(item["name"], list) else [item["name"]]
                rows.append({"designs": names, "loss": float(total.detach()),
                             "terms": terms, "metrics": {
                                 key: value / count for key, (value, count) in metrics.items()}})
                batch_loss.update(float(total.detach()), len(names))
                for key, value in terms.items():
                    term_stats.setdefault(key, AverageMeter()).update(value, len(names))
                for key, (value, count) in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                    metric_counts[key] = metric_counts.get(key, 0) + count
                progress.update(batch_number, batch_loss.avg,
                                {key: meter.avg for key, meter in term_stats.items()},
                                {key: metric_sums[key] / metric_counts[key] for key in metric_sums})
        count = sum(len(row["designs"]) for row in rows)
        mean = sum(row["loss"] * len(row["designs"]) for row in rows) / count
        return {"mean_loss": float(mean),
                "terms": {key: meter.avg for key, meter in term_stats.items()},
                "metrics": {key: metric_sums[key] / metric_counts[key] for key in metric_sums},
                "per_batch": rows}

    def _indices(self, subset, split_name):
        if getattr(subset, "dataset", None) is not self.dataset:
            raise ValueError(f"{split_name} must come from this trainer's NpzParser")
        indices = list(subset.indices)
        if indices != self.splits[split_name]:
            raise ValueError(f"{split_name} indices differ from parser split")
        return indices

    def set_training_args(self, stage, lr=None):
        """Select one of the three cumulative objectives before train()."""
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2 or 3")
        self.selected_stage = stage
        if lr is not None and not (stage == self.next_stage and self.next_epoch > 1):
            if lr <= 0:
                raise ValueError("learning rate must be positive")
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def train(self, num_epoch, train_dataset, val_dataset):
        """Train one selected stage, then carry its best weights to the next."""
        stage = self.selected_stage
        if stage is None:
            raise RuntimeError("call set_training_args(stage=...) before train()")
        if stage < self.next_stage:
            info(f"Stage {stage}/3 already completed; skipping")
            return
        if stage != self.next_stage:
            raise ValueError(f"expected stage {self.next_stage}, got {stage}")
        train_indices = self._indices(train_dataset, "train")
        val_indices = self._indices(val_dataset, "val")
        history = [row for row in self.history if row["stage"] == stage]
        best = min((row["val"]["mean_loss"] for row in history), default=float("inf"))
        best_epoch = min((row["epoch"] for row in history
                          if row["val"]["mean_loss"] == best), default=0)
        stage_best_path = self.run_dir / f"stage{stage}_best.pt"
        first_epoch = self.next_epoch
        last_epoch = first_epoch - 1
        task_names = {1: "probability + reconvergence",
                      2: "probability + reconvergence + functional distance",
                      3: "probability + reconvergence + functional distance + 4 behavior targets"}
        info(f"Stage {stage}/3 | {task_names[stage]} | up to {num_epoch} epochs")
        for epoch in range(first_epoch, num_epoch + 1):
            order = list(train_indices)
            random.shuffle(order)
            train = self._run_epoch(order, stage, training=True, epoch=epoch)
            val = self._run_epoch(val_indices, stage, epoch=epoch)
            row = {"stage": stage, "epoch": epoch, "train": train, "val": val}
            self.history.append(row)
            _write_json(self.run_dir / "history.json", self.history)
            score = val["mean_loss"]
            timing = self.timeline.epoch(stage, epoch, train, val)
            if self.args.lr_step and epoch % self.args.lr_step == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] *= 0.1
            if score < best:
                best, best_epoch = score, epoch
                atomic_save(_checkpoint(self.model, self.optimizer, self.dataset, self.args,
                                        self.splits, stage, epoch, best, False), stage_best_path)
            if epoch % self.args.save_interval == 0:
                self._save_resume(stage, epoch, best)
            info(f"Stage {stage}/3 | Epoch {epoch}/{num_epoch} | "
                 f"TrainLoss={train['mean_loss']:.5f} | ValLoss={score:.5f} | BestVal={best:.5f} | "
                 f"Elapsed={format_seconds(timing['elapsed_seconds'])} | "
                 f"ETA={format_seconds(timing['eta_seconds'])}")
            for phase, report in (("train", train), ("val", val)):
                info(f"{phase.upper()} losses: " + " | ".join(
                    f"{key}={value:.4f}" for key, value in report["terms"].items()))
                info(f"{phase.upper()} metrics: " + " | ".join(
                    f"{key}={value:.1%}" if "accuracy" in key else f"{key}={value:.4f}"
                    for key, value in report["metrics"].items()))
            if epoch % self.args.save_interval == 0:
                info(f"Resume checkpoint saved: last.pt | stage={stage} epoch={epoch}")
            last_epoch = epoch
            if self.args.patience and epoch - best_epoch >= self.args.patience:
                info(f"Stage {stage}/3 early stop at epoch {epoch}; best epoch={best_epoch}")
                break
        if not stage_best_path.is_file():
            raise RuntimeError(f"missing best checkpoint for stage {stage}")
        best_state = load_checkpoint(stage_best_path, self.args.device)
        self.model.load_state_dict(best_state["model"])
        self.optimizer.load_state_dict(best_state["optimizer"])
        self._save_resume(stage, last_epoch, best, completed=True)
        info(f"Stage {stage}/3 complete | best epoch={best_epoch} | last.pt saved")
        self.next_stage, self.next_epoch = stage + 1, 1

    def evaluate(self, test_dataset):
        """Select the stage-3 best model and score the held-out circuits."""
        indices = self._indices(test_dataset, "test")
        if self.next_stage <= 3:
            raise RuntimeError("all three training stages must complete before evaluation")
        final = load_checkpoint(self.run_dir / "stage3_best.pt", self.args.device)
        self.model.load_state_dict(final["model"])
        atomic_save(final, self.run_dir / "best.pt")
        test = self._run_epoch(indices, 3, phase="test")
        _write_json(self.run_dir / "test.json", test)
        info(f"Test complete | best stage-3 epoch={final['epoch']} | loss={test['mean_loss']:.5f}")
        info("TEST metrics: " + " | ".join(
            f"{key}={value:.1%}" if "accuracy" in key else f"{key}={value:.4f}"
            for key, value in test["metrics"].items()))
        info(f"Results and checkpoints: {self.run_dir}")
        return test

    def fit(self, train_dataset, val_dataset, test_dataset):
        """Convenience path for the explicit three-stage example below."""
        for stage in (1, 2, 3):
            self.set_training_args(stage, lr=self.args.lr)
            self.train(getattr(self.args, f"stage{stage}_epochs"), train_dataset, val_dataset)
        return self.evaluate(test_dataset)


Trainer = TriWaveTrainer
