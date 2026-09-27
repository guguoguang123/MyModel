"""First-run DARE-Gate training command."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import torch

from ..config import DAREGateConfig
from ..data import DAREGateNPZDataset
from ..model import DAREGateModel
from ..training import DAREGateTrainer, ObjectiveConfig, TrainerConfig
from .common import (
    build_loaders,
    create_or_load_split,
    print_data_summary,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a new three-stage DARE-Gate training run."
    )
    parser.add_argument("--data-dir", default="Code/data/train")
    parser.add_argument("--output-dir", default="Code/data/output/default")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="batch size from the behavior stage onward (default: 32)",
    )
    parser.add_argument(
        "--prebehavior-batch-size", type=int, default=64,
        help="batch size before the behavior stage (default: 64)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0 or cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--no-progress", action="store_true", help="disable interactive progress bars"
    )
    parser.add_argument(
        "--progress-min-interval", type=float, default=0.5,
        help="minimum seconds between progress bar refreshes (default: 0.5)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--behavior-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--functional-start-epoch", type=int, default=10)
    parser.add_argument("--behavior-start-epoch", type=int, default=20)
    parser.add_argument("--auxiliary-warmup-epochs", type=int, default=5)
    parser.add_argument("--probability-weight", type=float, default=3.0)
    parser.add_argument("--reconvergence-weight", type=float, default=1.0)
    parser.add_argument("--functionality-weight", type=float, default=2.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--similarity-weight", type=float, default=0.1)
    parser.add_argument("--decoupling-weight", type=float, default=0.01)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="save model_last.pt every N completed epochs (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--no-rc-source",
        action="store_true",
        help="disable topology-derived behavior reconvergence correction",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.prebehavior_batch_size <= 0:
        parser.error("--prebehavior-batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.progress_min_interval <= 0.0:
        parser.error("--progress-min-interval must be positive")
    if args.functional_start_epoch < 0:
        parser.error("--functional-start-epoch cannot be negative")
    if args.behavior_start_epoch < args.functional_start_epoch:
        parser.error("--behavior-start-epoch cannot precede functional start")


def _reset_output_dir(output_dir: Path) -> None:
    output_root = Path("Code/data/output").resolve()
    resolved = output_dir.resolve()
    if resolved == output_root or output_root not in resolved.parents:
        raise ValueError(
            "--output-dir must be a concrete experiment directory below "
            f"{output_root}"
        )
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        shutil.rmtree(resolved)
        print(f"removed previous run directory: {resolved}")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    _reset_output_dir(output_dir)

    dataset = DAREGateNPZDataset(
        args.data_dir,
        derive_reconvergence_sources=not args.no_rc_source,
    )
    split_path = output_dir / "logs" / "data_split.json"
    split = create_or_load_split(
        dataset,
        split_path,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        require_existing=False,
    )
    pin_memory = (args.device or "cuda").startswith("cuda") and torch.cuda.is_available()
    prebehavior_train_loader, _, _ = build_loaders(
        dataset,
        split,
        batch_size=args.prebehavior_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    joint_train_loader, validation_loader, _ = build_loaders(
        dataset,
        split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    print_data_summary(dataset, split)

    model_config = DAREGateConfig(
        structure_dim=args.embedding_dim,
        function_dim=args.embedding_dim,
        behavior_dim=args.behavior_dim,
        attention_heads=args.attention_heads,
    )
    objective_config = ObjectiveConfig(
        probability_weight=args.probability_weight,
        reconvergence_weight=args.reconvergence_weight,
        functionality_weight=args.functionality_weight,
        reconstruction_weight=args.reconstruction_weight,
        similarity_weight=args.similarity_weight,
        decoupling_weight=args.decoupling_weight,
        functionality_start_epoch=args.functional_start_epoch,
        behavior_start_epoch=args.behavior_start_epoch,
        auxiliary_warmup_epochs=args.auxiliary_warmup_epochs,
    )
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        use_amp=not args.no_amp,
        device=args.device,
        output_dir=str(output_dir),
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience,
        show_progress=not args.no_progress,
        progress_min_interval=args.progress_min_interval,
        seed=args.seed,
    )
    trainer = DAREGateTrainer(
        DAREGateModel(model_config),
        trainer_config=trainer_config,
        objective_config=objective_config,
    )
    run_config = vars(args).copy()
    run_config.update(
        {
            "python_entry": "Code/train.py",
            "resolved_device": str(trainer.device),
            "torch_version": torch.__version__,
            "split_path": str(split_path),
        }
    )
    save_json(output_dir / "logs" / "run_config.json", run_config)
    print(
        f"starting new run on {trainer.device}; checkpoints every "
        f"{args.save_every} completed epoch(s); "
        f"batch schedule: {args.prebehavior_batch_size} before epoch "
        f"{args.behavior_start_epoch}, then {args.batch_size}"
    )
    try:
        trainer.fit(
            prebehavior_train_loader,
            validation_loader,
            num_epochs=args.epochs,
            late_train_loader=joint_train_loader,
            train_loader_switch_epoch=args.behavior_start_epoch,
        )
    except KeyboardInterrupt:
        print(
            "training interrupted; no interruption checkpoint was saved; "
            "resume from the latest periodic model_last.pt"
        )
        return 130
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
