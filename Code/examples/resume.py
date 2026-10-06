"""Resume staged training from Code/examples/exp/<run>/last.pt."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.trainer import TriWaveTrainer
from triwave_gnn.utils.config import training_parser
from triwave_gnn.utils.checkpoint import load as load_checkpoint
from triwave_gnn.utils.random_seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="experiment directory under Code/examples/exp")
    args, extra = parser.parse_known_args()
    saved = load_checkpoint(args.run / "last.pt", "cpu")
    recorded = saved["config"]
    restored_options = []
    for key in (
        "stage1_epochs", "stage2_epochs", "stage3_epochs", "patience", "hidden",
        "batch_size", "num_workers", "progress_interval", "max_nodes", "lr",
        "lr_step", "seed",
    ):
        if key in recorded:
            restored_options.extend(("--" + key.replace("_", "-"), str(recorded[key])))
    settings = training_parser().parse_args(
        ["--data", str(DATA_DIR), "--resume", str(args.run / "last.pt"),
         *restored_options, *extra]
    )
    circuit_path = settings.data / "graphs.npz"
    label_path = settings.data / "labels.npz"
    data_parser = NpzParser(settings.data, circuit_path, label_path,
                            seed=settings.seed,
                            max_nodes=settings.max_nodes)
    train_dataset, val_dataset = data_parser.get_dataset()
    test_dataset = data_parser.get_test_dataset()
    dataset = data_parser.dataset
    set_seed(settings.seed)
    model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                       len(dataset.pin_vocab), hidden=settings.hidden)
    trainer = TriWaveTrainer(model, settings, dataset, data_parser.get_split_indices())
    for stage in (1, 2, 3):
        trainer.set_training_args(stage, lr=settings.lr)
        trainer.train(getattr(settings, f"stage{stage}_epochs"), train_dataset, val_dataset)
    trainer.evaluate(test_dataset)
