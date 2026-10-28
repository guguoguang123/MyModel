"""Bounded, reproducible design selection without loading the large NPZs."""

import unittest
from types import SimpleNamespace

import numpy as np

from triwave_gnn.utils.data_utils import split_designs


class TrainingSelection(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleNamespace(
            ids=[f"synthetic__{i}" for i in range(200)],
            graph={"num_nodes": np.array([100 if i % 3 else 20000 for i in range(200)])},
        )

    def test_limit_applies_after_node_filter_and_keeps_disjoint_splits(self):
        splits = split_designs(self.dataset, seed=42, max_nodes=10000, max_designs=30)
        selected = [index for indices in splits.values() for index in indices]
        self.assertEqual(len(selected), 30)
        self.assertEqual(len(set(selected)), 30)
        self.assertTrue(all(splits.values()))
        self.assertTrue(all(self.dataset.graph["num_nodes"][i] <= 10000 for i in selected))
        self.assertEqual(splits, split_designs(self.dataset, 42, 10000, 30))
        self.assertNotEqual(splits, split_designs(self.dataset, 43, 10000, 30))

    def test_unlimited_default_keeps_all_designs(self):
        splits = split_designs(self.dataset, seed=42)
        self.assertEqual(sum(map(len, splits.values())), len(self.dataset.ids))

    def test_limit_larger_than_available_keeps_all_eligible_designs(self):
        splits = split_designs(self.dataset, seed=42, max_nodes=10000, max_designs=1000)
        expected = int((self.dataset.graph["num_nodes"] <= 10000).sum())
        self.assertEqual(sum(map(len, splits.values())), expected)

    def test_invalid_limits_fail(self):
        for limit in (-1, 1, 2):
            with self.assertRaisesRegex(ValueError, "max_designs"):
                split_designs(self.dataset, seed=42, max_designs=limit)


if __name__ == "__main__":
    unittest.main()
