"""Create the parser, representation model and three-stage trainer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.trainer import TriWaveTrainer
from triwave_gnn.utils.config import training_parser
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.utils.split_utils import resolve_split_file
from triwave_gnn.utils.logger import info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"
EXP_DIR = PROJECT_ROOT / "examples" / "exp"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = training_parser().parse_args(["--data", str(DATA_DIR), *argv])
    args.split_file = resolve_split_file(args)
    circuit_path = args.data / "graphs.npz"
    label_path = args.data / "labels.npz"
    info(f"Loading graph and label bundles: {args.data}; validation may take a few minutes")
    parser = NpzParser(args.data, circuit_path, label_path,
                       seed=args.seed, max_nodes=args.max_nodes, max_designs=args.max_designs,
                       split_file=args.split_file)
    train_dataset, val_dataset = parser.get_dataset()
    test_dataset = parser.get_test_dataset()
    dataset = parser.dataset
    info(f"Split mode={args.split_mode} | train={len(train_dataset)} | "
         f"validation={len(val_dataset)} | test={len(test_dataset)}")
    set_seed(args.seed)
    model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                       len(dataset.pin_vocab), hidden=args.hidden)
    trainer = TriWaveTrainer(model, args, dataset, parser.get_split_indices())
    for stage in (1, 2, 3):
        trainer.set_training_args(stage, lr=args.lr)
        trainer.train(getattr(args, f"stage{stage}_epochs"), train_dataset, val_dataset)
    return trainer.evaluate(test_dataset)


if __name__ == "__main__":
    main()
