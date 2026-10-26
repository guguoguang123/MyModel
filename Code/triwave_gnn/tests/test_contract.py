"""Checks that matter when a new graphs.npz/labels.npz pair replaces the pilot."""

import unittest
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from triwave_gnn import CircuitDataset, TriWaveGNN
from triwave_gnn.utils.losses import loss_terms
from triwave_gnn.utils.losses import BEHAVIOR, STAGE_WEIGHTS
from triwave_gnn.utils.schema import validate_bundle
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.batch import SCHEDULE_KEYS, collate_circuits
from triwave_gnn.arch.message import RelationMessage
from triwave_gnn.tests.fixtures import make_bundle
from triwave_gnn.trainer import move


class BundleAndTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="triwave_test_")
        make_bundle(Path(cls.scratch.name))
        cls.dataset = CircuitDataset(cls.scratch.name)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

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
        data = self.dataset[1]
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab), len(self.dataset.pin_vocab), 16)
        states = model.initialization(data)
        pi = torch.where(data["pin"].is_pi)[0]
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
        self.assertEqual(tuple(name for name, _, _ in BEHAVIOR),
                         ("hb_tp", "hb_first_mean", "hb_stable_mean", "hb_glitch_rate"))
        self.assertEqual(STAGE_WEIGHTS[3],
                         {"prob": 3.0, "rc": 1.0, "func": 2.0, "behavior": 2.0})
        self.assertEqual(output["hb_raw"].shape[1], 4)
        self.assertNotIn("hb_tp_01", stage3)
        self.assertNotIn("hb_unstable_window", stage3)
        mask = data["pin"].hb_stable_train_mask
        expected = F.smooth_l1_loss(
            output["hb_raw"][mask, 2],
            torch.log1p(data["pin"].hb_stable_mean[mask]),
        )
        self.assertAlmostEqual(stage3["hb_stable_mean"], expected.item(), places=6)

    def test_messages_use_cross_branch_predecessors_without_edge_type_context(self):
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16)
        for branch, source_dim in (("s", 16), ("f", 32), ("b", 48)):
            for relation, physical_dim in (("net", 12), ("cell", 14)):
                module = model.aggregators[branch + "_" + relation]
                edge_dim = physical_dim if branch == "b" else 0
                self.assertIsInstance(module, MessagePassing)
                self.assertEqual(module.message_mlp.layers[0].in_features, source_dim + edge_dim)
                if relation == "cell":
                    self.assertEqual(module.attention.score.layers[0].in_features,
                                     source_dim + 16 + edge_dim)

    def test_cell_attention_reads_target_state_before_update(self):
        module = RelationMessage(source_dim=2, hidden_dim=2, physical_dim=0,
                                 cell_attention=True)
        with torch.no_grad():
            for layer in (module.message_mlp.layers[0], module.message_mlp.layers[2]):
                layer.weight.copy_(torch.eye(2))
                layer.bias.zero_()
            first = module.attention.score.layers[0]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1.0  # predecessor z_u
            first.weight[0, 2] = 1.0  # target's pre-update h_v
            last = module.attention.score.layers[2]
            last.weight.zero_()
            last.bias.zero_()
            last.weight[0, 0] = 1.0
            predecessor = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
            targets = torch.zeros((3, 2))
            edges = torch.tensor([[0, 1], [2, 2]])
            physical = torch.empty((2, 0))
            first_result = module(predecessor, targets, edges, physical, torch.tensor([2]))
            targets[2, 0] = 3.0
            second_result = module(predecessor, targets, edges, physical, torch.tensor([2]))
        self.assertFalse(torch.allclose(first_result, second_result))

    def test_physical_features_only_change_behavior_state(self):
        data = self.dataset[1]
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16).eval()
        changed = data.clone()
        changed.__dict__.pop("_topology_ready", None)
        for relation in ("net", "cell"):
            changed["pin", relation, "pin"].edge_attr += 1
        with torch.no_grad():
            original = model(data)
            altered = model(changed)
        self.assertTrue(torch.allclose(original[0], altered[0]))
        self.assertTrue(torch.allclose(original[1], altered[1]))
        self.assertFalse(torch.allclose(original[2], altered[2]))

    def test_batched_graph_matches_separate_inference(self):
        items = [self.dataset[0], self.dataset[1]]
        batch = collate_circuits(items)
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16).eval()
        with torch.no_grad():
            separate = [model(item) for item in items]
            combined = model(batch)
        start = 0
        for item, expected in zip(items, separate):
            stop = start + item["pin"].num_nodes
            for branch in range(3):
                self.assertTrue(torch.allclose(combined[branch][start:stop],
                                               expected[branch], atol=1e-5, rtol=1e-5))
            start = stop

    def test_pyg_batch_offsets_pairs_edges_and_round_trips(self):
        items = [self.dataset[0], self.dataset[1], self.dataset[2]]
        self.assertIsInstance(items[0], HeteroData)
        batch = next(iter(DataLoader(items, batch_size=3, exclude_keys=SCHEDULE_KEYS)))
        offset = 0
        for key in ("hs_pairs", "hf_pairs"):
            expected = []
            offset = 0
            for item in items:
                expected.append(item[key] + offset)
                offset += item["pin"].num_nodes
            self.assertTrue(torch.equal(batch[key], torch.cat(expected, dim=1)))
        for relation in ("net", "cell"):
            edge = batch["pin", relation, "pin"].edge_index
            membership = batch["pin"].batch
            self.assertTrue(torch.equal(membership[edge[0]], membership[edge[1]]))
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16)
        with torch.no_grad():
            model(batch)  # Building the level schedule must not invalidate unbatching.
        for actual, expected in zip(batch.to_data_list(), items):
            for relation in ("net", "cell"):
                self.assertTrue(torch.equal(actual["pin", relation, "pin"].edge_index,
                                            expected["pin", relation, "pin"].edge_index))
            self.assertTrue(torch.equal(actual.hs_pairs, expected.hs_pairs))
            self.assertTrue(torch.equal(actual["pin"].topological_order, expected["pin"].topological_order))

    def test_legacy_four_task_weights_load_without_changing_parameters(self):
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16)
        original = model.state_dict()
        legacy = {key.replace(".message_mlp.layers.", ".message.layers.").replace(
            ".attention.score.", ".aggregate.score."): value.clone() for key, value in original.items()}
        model.load_state_dict(legacy)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_pyg_cuda_matches_cpu_and_keeps_cached_graph_on_cpu(self):
        data = self.dataset[1]
        model = TriWaveGNN(len(self.dataset.logic_vocab), len(self.dataset.role_vocab),
                           len(self.dataset.pin_vocab), 16).eval()
        with torch.no_grad():
            cpu = model(data)
            gpu = model.to("cuda")(move(data, "cuda"))
        self.assertEqual(data["pin"].node_role.device.type, "cpu")
        for expected, actual in zip(cpu, gpu):
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
