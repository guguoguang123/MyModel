"""Create the parser, representation model and three-stage trainer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn import NpzParser, TriWaveGNN
from triwave_gnn.trainer import TriWaveTrainer
from triwave_gnn.utils.config import training_parser
from triwave_gnn.utils.random_seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data" / "train"
EXP_DIR = PROJECT_ROOT / "examples" / "exp"


if __name__ == "__main__":
    args = training_parser().parse_args(["--data", str(DATA_DIR), *sys.argv[1:]])
    circuit_path = args.data / "graphs.npz"
    label_path = args.data / "labels.npz"
    parser = NpzParser(args.data, circuit_path, label_path,
                       seed=args.seed, max_nodes=args.max_nodes)
    train_dataset, val_dataset = parser.get_dataset()
    test_dataset = parser.get_test_dataset()
    dataset = parser.dataset
    set_seed(args.seed)
    model = TriWaveGNN(len(dataset.logic_vocab), len(dataset.role_vocab),
                       len(dataset.pin_vocab), hidden=args.hidden)
    trainer = TriWaveTrainer(model, args, dataset, parser.get_split_indices())
    for stage in (1, 2, 3):
        trainer.set_training_args(stage, lr=args.lr)
        trainer.train(getattr(args, f"stage{stage}_epochs"), train_dataset, val_dataset)
    trainer.evaluate(test_dataset)
