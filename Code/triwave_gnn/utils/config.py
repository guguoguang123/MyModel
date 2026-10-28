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
    parser.add_argument("--split-mode", choices=("fixed", "random"), default="fixed",
                        help="fixed manifest for full training; random for bounded checks")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="fixed split manifest; defaults to the data directory's split.json")
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage3-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5,
                        help="save last.pt every N epochs and at every stage end")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="complete circuits per optimizer step; reduce if GPU memory is insufficient")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="redirected-output snapshot interval in batches; 0 only shows completion; terminals always redraw in place")
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="per-design node limit; 0 leaves node counts unlimited")
    parser.add_argument("--max-designs", type=int, default=0,
                        help="limit total train/val/test designs after node filtering; 0 uses all")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-step", type=int, default=20,
                        help="decay learning rate by 10 every N epochs within each stage; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser
