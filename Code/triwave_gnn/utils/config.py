"""One place for the training command's reproducible experiment settings."""

import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"
DEFAULT_EXP_DIR = PROJECT_ROOT / "examples" / "exp"


def training_parser():
    parser = argparse.ArgumentParser(description="TriWave-GNN staged representation learning")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="run directory; omitted creates a timestamped directory")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume exactly from an existing run's last.pt")
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage3-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="complete circuits per optimizer step; 1 is safest for deep graphs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=10,
                        help="print detailed metrics every N batches; 0 hides details")
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="0 trains every design; positive limits are for workflow checks")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-step", type=int, default=20,
                        help="decay learning rate by 10 every N epochs within each stage; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser
