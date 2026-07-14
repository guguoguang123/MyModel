"""Tests for aligned NPZ loading and disconnected-graph collation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Code.daregate import DAREGateNPZDataset, collate_daregate
from Code.daregate.data import CircuitSplit, design_family, split_by_design_family


class DAREGateDatasetTest(unittest.TestCase):
    def test_two_archives_are_aligned_and_pair_indices_are_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.asarray(
                [[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 0], [4, 1, 0]],
                dtype=np.float32,
            )
            edges = np.asarray(
                [[0, 2], [1, 2], [0, 3], [2, 4], [3, 4]], dtype=np.int64
            )
            graphs = {
                name: {"x": x.copy(), "edge_index": edges.copy()}
                for name in ("a", "b")
            }
            labels = {
                name: {
                    "prob": np.linspace(0.1, 0.5, 5, dtype=np.float32),
                    "tt_pair_index": np.asarray([[0, 2], [3, 4]], dtype=np.int64),
                    "tt_dis": np.asarray([0.2, 0.8], dtype=np.float32),
                    "min_tt_dis": np.asarray([0.1, 0.1], dtype=np.float32),
                    "rc_pair_index": np.asarray([[0, 2], [0, 3]], dtype=np.int64),
                    "is_rc": np.asarray([1, 1], dtype=np.int64),
                }
                for name in graphs
            }
            behavior = {}
            for name in graphs:
                response = np.zeros((8, 5, 2), dtype=np.float32)
                response[:, :, 0] = np.linspace(-0.1, 0.1, 8)[:, None]
                behavior[name] = {
                    "R": response,
                    "S": response.transpose(1, 0, 2).reshape(5, 16),
                }
            np.savez_compressed(root / "graphs.npz", circuits=graphs)
            final_labels = {
                name: {**labels[name], **behavior[name]} for name in graphs
            }
            np.savez_compressed(
                root / "finalLabels.npz", labels=final_labels, meta={"version": 1}
            )

            dataset = DAREGateNPZDataset(root)
            first, second = dataset[0], dataset[1]
            self.assertEqual(len(dataset), 2)
            repeated = dataset[0]
            self.assertEqual(dataset.topology_cache_size, 2)
            self.assertEqual(
                first.forward_level.data_ptr(), repeated.forward_level.data_ptr()
            )
            self.assertEqual(first.signature.shape, (5, 16))
            self.assertEqual(first.response.shape, (5, 8, 2))
            self.assertEqual(first.probability.shape, (5,))
            self.assertEqual(first.tt_pair_index.shape, (2, 2))
            self.assertEqual(first.rc_pair_index.shape, (2, 2))
            self.assertEqual(int(first.rc_source[4]), 0)

            batch = collate_daregate([first, second])
            self.assertEqual(batch.num_nodes, 10)
            self.assertEqual(batch.edge_index.shape, (2, 10))
            self.assertEqual(batch.batch_index.bincount().tolist(), [5, 5])
            torch.testing.assert_close(
                batch.tt_pair_index[:, 2:], second.tt_pair_index + 5
            )
            torch.testing.assert_close(
                batch.rc_pair_index[:, 2:], second.rc_pair_index + 5
            )
            self.assertEqual(int(batch.rc_source[9]), 5)


    def test_design_family_split_is_disjoint_and_persistent(self) -> None:
        names = tuple(
            f"circuit_{family}_syn_{variant:03d}"
            for family in range(5)
            for variant in range(3)
        )
        split = split_by_design_family(
            names, validation_ratio=0.2, test_ratio=0.2, seed=17
        )
        train_families = {design_family(name) for name in split.train}
        validation_families = {design_family(name) for name in split.validation}
        test_families = {design_family(name) for name in split.test}
        self.assertFalse(train_families & validation_families)
        self.assertFalse(train_families & test_families)
        self.assertFalse(validation_families & test_families)
        self.assertEqual(
            set(split.train) | set(split.validation) | set(split.test), set(names)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "split.json")
            split.save(path)
            restored = CircuitSplit.load(path)
            self.assertEqual(restored, split)

if __name__ == "__main__":
    unittest.main()
