"""Read a complete pin-graph bundle and expose circuit-level datasets."""

from pathlib import Path

from .dataset import CircuitDataset
from .utils.data_utils import split_designs
from torch.utils.data import Subset


class NpzParser:
    """Parse aligned physical pin graphs and labels into design subsets."""

    def __init__(self, data_dir, circuit_path=None, label_path=None,
                 seed=42, max_nodes=0, cache_mode="auto"):
        data_dir = Path(data_dir)
        if circuit_path is not None and Path(circuit_path).resolve() != (data_dir / "graphs.npz").resolve():
            raise ValueError("circuit_path must be data_dir/graphs.npz")
        if label_path is not None and Path(label_path).resolve() != (data_dir / "labels.npz").resolve():
            raise ValueError("label_path must be data_dir/labels.npz")
        self.dataset = CircuitDataset(data_dir, cache_mode=cache_mode)
        self.splits = split_designs(self.dataset, seed, max_nodes or None)

    def get_dataset(self):
        """Return train and validation subsets of complete designs."""
        splits = self.get_splits()
        return splits["train"], splits["val"]

    def get_test_dataset(self):
        return self.get_splits()["test"]

    def get_splits(self):
        return {name: Subset(self.dataset, indices)
                for name, indices in self.splits.items()}

    def get_split_indices(self):
        return self.splits
