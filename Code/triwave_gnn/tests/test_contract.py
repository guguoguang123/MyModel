"""Checks that matter when a new graphs.npz/labels.npz pair replaces the pilot."""

import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from triwave_gnn import CircuitDataset, TriWaveGNN
from triwave_gnn.utils.losses import loss_terms
from triwave_gnn.utils.schema import validate_bundle
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.batch import collate_circuits


DATA = Path(__file__).resolve().parents[2] / "examples" / "data" / "train"


@unittest.skipUnless((DATA / "graphs.npz").is_file(), "training bundle unavailable")
class BundleAndTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = CircuitDataset(DATA)

    def test_cross_design_edge_is_rejected(self):
        graph = dict(self.dataset.graph)
        edge = graph["net_edge_index"].copy()
        edge[1, 0] = graph["node_ptr"][1]
        graph["net_edge_index"] = edge
        with self.assertRaisesRegex(ValueError, "crosses graph boundary"):
            validate_bundle(graph, self.dataset.label)

    def test_reordered_physical_columns_are_rejected(self):
        graph = dict(self.dataset.graph)
        graph["net_cause_names"] = graph["net_cause_names"][::-1].copy()
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_bundle(graph, self.dataset.label)

    def test_pi_initialization_and_stage_objectives(self):
        data = self.dataset[self.dataset.ids.index("iscas85__c1908")]
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab), len(self.dataset.pin_vocab), 16)
        _, states = model.initialization(data)
        pi = torch.where(data["node_role"] == data["pi_role_index"])[0]
        self.assertGreater(pi.numel(), 1)
        self.assertFalse(torch.allclose(states["s"][pi[0]], states["s"][pi[1]]))
        self.assertTrue(torch.allclose(states["f"][pi[0]], states["f"][pi[1]]))
        output = model_outputs(model, data, include_behavior=True)
        _, stage1 = loss_terms(output, data, 1)
        _, stage2 = loss_terms(output, data, 2)
        _, stage3 = loss_terms(output, data, 3)
        self.assertNotIn("func", stage1)
        self.assertIn("func", stage2)
        self.assertIn("behavior", stage3)
        mask = data["hb_unstable_window_mask"]
        expected = F.smooth_l1_loss(
            output["hb_raw"][mask, 9],
            torch.log1p(data["hb_unstable_window"][mask]),
        )
        self.assertAlmostEqual(stage3["hb_unstable_window"], expected.item(), places=6)

    def test_physical_features_only_change_behavior_state(self):
        data = self.dataset[self.dataset.ids.index("iscas85__c1908")]
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16).eval()
        changed = dict(data)
        changed["net_features"] = data["net_features"] + 1
        changed["cell_features"] = data["cell_features"] + 1
        with torch.no_grad():
            original = model(data)
            altered = model(changed)
        self.assertTrue(torch.allclose(original[0], altered[0]))
        self.assertTrue(torch.allclose(original[1], altered[1]))
        self.assertFalse(torch.allclose(original[2], altered[2]))

    def test_batched_graph_matches_separate_inference(self):
        names = ("iscas85__c17", "epfl__random_control__ctrl")
        items = [self.dataset[self.dataset.ids.index(name)] for name in names]
        batch = collate_circuits(items)
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16).eval()
        with torch.no_grad():
            separate = [model(item) for item in items]
            combined = model(batch)
        start = 0
        for item, expected in zip(items, separate):
            stop = start + item["node_role"].numel()
            for branch in range(3):
                self.assertTrue(torch.allclose(combined[branch][start:stop],
                                               expected[branch], atol=1e-5, rtol=1e-5))
            start = stop


if __name__ == "__main__":
    unittest.main()
