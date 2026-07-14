"""Resume an interrupted or previously stopped DARE-Gate run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from ..config import DAREGateConfig
from ..data import DAREGateNPZDataset
from ..model import DAREGateModel
from ..training import DAREGateTrainer, ObjectiveConfig, TrainerConfig
from .common import (
    build_loaders,
    create_or_load_split,
    infer_output_dir,
    load_json,
    print_data_summary,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume DARE-Gate from a complete training checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default="Code/data/output/default/checkpoints/model_last.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="total target epoch count, not the number of additional epochs (default: 60)",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override behavior-stage batch size (default: saved value or 32)",
    )
    parser.add_argument(
        "--prebehavior-batch-size", type=int, default=None,
        help="override pre-behavior batch size (default: saved value or 64)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="save model_last.pt every N completed epochs (default: 5)",
    )
    parser.add_argument("--device", default=None, help="override saved device")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true", help="disable interactive progress bars"
    )
    parser.add_argument(
        "--progress-min-interval", type=float, default=None,
        help="override the saved progress refresh interval",
    )
    return parser


def _checkpoint_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model",
        "optimizer",
        "model_config",
        "trainer_config",
        "objective_config",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise KeyError(f"checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


def run(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _checkpoint_dict(checkpoint_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else infer_output_dir(checkpoint_path)
    )
    run_config_path = output_dir / "logs" / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.is_file() else {}
    data_dir = args.data_dir or run_config.get("data_dir", "Code/data/train")
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(run_config.get("batch_size", 32))
    )
    prebehavior_batch_size = (
        args.prebehavior_batch_size
        if args.prebehavior_batch_size is not None
        else int(run_config.get("prebehavior_batch_size", 64))
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prebehavior_batch_size <= 0:
        raise ValueError("prebehavior_batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive")
    if args.progress_min_interval is not None and args.progress_min_interval <= 0.0:
        raise ValueError("progress_min_interval must be positive")

    saved_trainer = dict(checkpoint["trainer_config"])
    saved_trainer["device"] = args.device or saved_trainer.get("device")
    saved_trainer["use_amp"] = (
        False if args.no_amp else bool(saved_trainer.get("use_amp", True))
    )
    saved_trainer["output_dir"] = str(output_dir)
    saved_trainer["save_every"] = args.save_every
    saved_trainer["show_progress"] = (
        False if args.no_progress else bool(saved_trainer.get("show_progress", True))
    )
    if args.progress_min_interval is not None:
        saved_trainer["progress_min_interval"] = args.progress_min_interval
    saved_trainer["checkpoint_dir"] = str(output_dir / "checkpoints")
    trainer_config = TrainerConfig(**saved_trainer)
    objective_config = ObjectiveConfig(**checkpoint["objective_config"])
    model_config = DAREGateConfig(**checkpoint["model_config"])

    dataset = DAREGateNPZDataset(
        data_dir,
        derive_reconvergence_sources=not bool(
            run_config.get("no_rc_source", False)
        ),
    )
    split = create_or_load_split(
        dataset,
        output_dir / "logs" / "data_split.json",
        validation_ratio=float(run_config.get("validation_ratio", 0.1)),
        test_ratio=float(run_config.get("test_ratio", 0.1)),
        seed=trainer_config.seed,
        require_existing=True,
    )
    pin_memory = (
        (trainer_config.device or "cuda").startswith("cuda")
        and torch.cuda.is_available()
    )
    prebehavior_train_loader, _, _ = build_loaders(
        dataset,
        split,
        batch_size=prebehavior_batch_size,
        num_workers=args.num_workers,
        seed=trainer_config.seed,
        pin_memory=pin_memory,
    )
    joint_train_loader, validation_loader, _ = build_loaders(
        dataset,
        split,
        batch_size=batch_size,
        num_workers=args.num_workers,
        seed=trainer_config.seed,
        pin_memory=pin_memory,
    )
    print_data_summary(dataset, split)

    trainer = DAREGateTrainer(
        DAREGateModel(model_config),
        trainer_config=trainer_config,
        objective_config=objective_config,
    )
    trainer.load_checkpoint(checkpoint_path)
    if args.epochs <= trainer.start_epoch:
        raise ValueError(
            f"--epochs must exceed checkpoint epoch {trainer.start_epoch}; "
            "it is the total target epoch count"
        )
    save_json(
        output_dir / "logs" / "resume_config.json",
        {
            "python_entry": "Code/resume_train.py",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": trainer.start_epoch,
            "target_epochs": args.epochs,
            "data_dir": str(data_dir),
            "batch_size": batch_size,
            "prebehavior_batch_size": prebehavior_batch_size,
            "num_workers": args.num_workers,
            "save_every": args.save_every,
            "resolved_device": str(trainer.device),
        },
    )
    print(
        f"resuming {checkpoint_path} at completed epoch {trainer.start_epoch}; "
        f"target total epochs={args.epochs}, device={trainer.device}; "
        f"batch schedule: {prebehavior_batch_size} before epoch "
        f"{objective_config.behavior_start_epoch}, then {batch_size}"
    )
    try:
        trainer.fit(
            prebehavior_train_loader,
            validation_loader,
            num_epochs=args.epochs,
            late_train_loader=joint_train_loader,
            train_loader_switch_epoch=objective_config.behavior_start_epoch,
        )
    except KeyboardInterrupt:
        print(
            "training interrupted; no interruption checkpoint was saved; "
            "resume from the latest periodic model_last.pt"
        )
        return 130
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
