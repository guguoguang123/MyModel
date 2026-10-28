"""Resume staged training from Code/examples/exp/<run>/last.pt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.trainer import TriWaveTrainer
from triwave_gnn.utils.config import training_parser
from triwave_gnn.utils.checkpoint import load as load_checkpoint
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.utils.split_utils import resolve_split_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"
#Breakpoint Directory
RUN_DIR = PROJECT_ROOT / "examples" / "exp" / "triwave_gnn_20261026_144750_012230"


if __name__ == "__main__":
    parser = training_parser()
    parser.add_argument("run", type=Path, nargs="?", default=RUN_DIR,
                        help="experiment directory; defaults to RUN_DIR in this script")
    args = parser.parse_args()
    saved = load_checkpoint(args.run / "last.pt", "cpu")
    recorded = saved["config"]
    # Runs created before manifests used seeded random selection.
    restored_options = ["--split-mode", recorded.get("split_mode", "random")]
    if recorded.get("split_file"):
        restored_options.extend(("--split-file", recorded["split_file"]))
    for key in (
        "stage1_epochs", "stage2_epochs", "stage3_epochs", "patience", "hidden",
        "batch_size", "num_workers", "progress_interval", "max_nodes", "max_designs", "lr",
        "lr_step", "seed", "save_interval",
    ):
        if key in recorded:
            restored_options.extend(("--" + key.replace("_", "-"), str(recorded[key])))
    settings = parser.parse_args(
        ["--data", str(DATA_DIR), "--resume", str(args.run / "last.pt"),
         *restored_options, *sys.argv[1:]]
    )
    del settings.run
    settings.split_file = resolve_split_file(settings)
    circuit_path = settings.data / "graphs.npz"
    label_path = settings.data / "labels.npz"
    data_parser = NpzParser(settings.data, circuit_path, label_path,
                            seed=settings.seed,
                            max_nodes=settings.max_nodes, max_designs=settings.max_designs,
                            split_file=settings.split_file)
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
