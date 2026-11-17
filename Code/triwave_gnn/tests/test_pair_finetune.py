"""Sampling, full-pair validation and optimizer checks on in-memory pin graphs."""

from copy import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ..model import TriWaveGNN
from ..parser_func import PinCircuitData
from ..utils.batch import collate_circuits
from ..utils.losses import functional_distance, loss_terms
from ..utils.model_utils import model_outputs
from ..utils.pair_sampling import MODES, STRATA, allocate_strata, sample_functional_pairs


def tiny_item(name, distances, roles):
    """In-memory contract fixture, never written to or substituted for real NPZs."""
    item = PinCircuitData()
    item.name = name
    pin = item["pin"]
    pin.num_nodes = 8
    pin.node_role = torch.tensor([roles.index(role) for role in
                                 ("pi", "pi", "cell_input", "cell_input", "cell_output", "cell_input", "cell_output", "po")])
    pin.node_logic = torch.zeros(8, dtype=torch.long)
    pin.node_pin = torch.zeros(8, dtype=torch.long)
    pin.is_pi = torch.tensor([True, True, False, False, False, False, False, False])
    pin.depth = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5])
    pin.topological_order = torch.arange(8)
    pin.hf_prob = torch.linspace(.1, .9, 8)
    for relation, edges, width in (("net", [[0, 1, 4, 6], [2, 3, 5, 7]], 12),
                                   ("cell", [[2, 3, 5], [4, 4, 6]], 14)):
        item["pin", relation, "pin"].edge_index = torch.tensor(edges)
        item["pin", relation, "pin"].edge_attr = torch.zeros(len(edges[0]), width)
    item["hs_pairs"] = torch.tensor([[2, 2], [3, 5]])
    item["hs_target"] = torch.tensor([0., 1.])
    item["hs_mask"] = torch.ones(2, dtype=torch.bool)
    distance = torch.tensor(distances, dtype=torch.float32)
    size = distance.numel()
    item["hf_pairs"] = torch.tensor([[4, 6], [6, 4]])[:, torch.arange(size) % 2]
    item["hf_tt_target"] = distance
    item["hf_tt_mask"] = torch.ones(size, dtype=torch.bool)
    for name in ("hb_tp", "hb_first_mean", "hb_stable_mean", "hb_glitch_rate"):
        pin[name] = torch.linspace(.1, .8, 8)
    for name in ("hb_tp_mask", "hb_first_train_mask", "hb_stable_train_mask", "hb_glitch_rate_train_mask"):
        pin[name] = torch.ones(8, dtype=torch.bool)
    return item


def tiny_items(roles):
    a = tiny_item("selfcheck_a", [0.] + [.1] * 1999 + [.9] * 100 + [.5] * 300 + [float("nan"), 0.], roles)
    a.hf_tt_mask[-1] = False
    b = tiny_item("selfcheck_b", [.1] * 7, roles)
    c = tiny_item("selfcheck_c", [.5] * 3, roles)
    return [a, b, c]


def tiny_source():
    roles = ["pi", "cell_input", "cell_output", "po"]
    model = TriWaveGNN(1, 4, 1, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    return {"model": model.state_dict(), "logic_vocab": ["logic"], "role_vocab": roles, "pin_vocab": ["pin"],
            "config": {"hidden": 8, "batch_size": 32, "seed": 42, "num_workers": 0},
            "optimizer": optimizer.state_dict()}


class TinyDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return copy(self.items[index])


def check_selection(batch, mode):
    before = {key: batch[key].clone() for key in ("hf_pairs", "hf_tt_target", "hf_tt_mask")}
    random_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(1000045)
    view, selected, rows = sample_functional_pairs(batch, mode, generator)
    assert torch.equal(torch.get_rng_state(), random_state), "Sampler consumed global RNG"
    same_view, same_indices, _ = sample_functional_pairs(batch, mode, torch.Generator().manual_seed(1000045))
    assert torch.equal(selected, same_indices), "Same independent seed is not reproducible"
    for key in before:
        assert torch.allclose(batch[key], before[key], equal_nan=True), "Sampler mutated the full batch"
    assert torch.equal(view.hf_pairs, batch.hf_pairs[:, selected])
    assert torch.equal(view.hf_tt_target, batch.hf_tt_target[selected])
    assert torch.equal(view.hf_tt_mask, batch.hf_tt_mask[selected])
    assert torch.equal(view.hs_pairs, batch.hs_pairs) and torch.equal(view.hs_target, batch.hs_target)
    assert view["pin"].hf_prob.data_ptr() == batch["pin"].hf_prob.data_ptr()
    assert all(row["sampled_pairs"] == (row["valid_pairs"] if mode == "original" else min(row["valid_pairs"], 1024)) for row in rows)
    assert rows[0]["source_strata"] == dict(zero=1, near=1999, far=100, other=300)
    if mode == "circuit_cap":
        assert all(row["duplicate_pairs"] == 0 for row in rows)
        _, subsequent, _ = sample_functional_pairs(batch, mode, generator)
        assert not torch.equal(selected, subsequent), "Training sampling did not change"
    if mode == "circuit_cap_stratified":
        assert all(value > 0 for value in rows[0]["sampled_strata"].values()), "Other distances lost coverage"
        assert rows[0]["duplicate_pairs"] > 0, "Undersupplied positive bin did not use replacement"
        assert rows[1]["sampled_strata"] == dict(zero=0, near=7, far=0, other=0)
        assert rows[2]["sampled_strata"] == dict(zero=0, near=0, far=0, other=3)
    assert same_view.hf_pairs.shape == view.hf_pairs.shape
    return view, rows


def self_check(source, device):
    from examples.finetune_functional_pairs import evaluate_validation, epoch_orders, make_optimizer, train_epoch, large_circuit_metrics
    from triwave_gnn.trainer import move

    roles = source["role_vocab"]
    items = tiny_items(roles)
    dataset = TinyDataset(items)
    batch = collate_circuits(items)
    source_snapshot = {key: value.clone() for key, value in source["model"].items()}
    report = {"fixture": "in-memory tiny graphs; actual checkpoint architecture and weights",
              "device": device, "source_weights_unchanged": True, "modes": {}}
    names = [item.name for item in items]
    orders = [epoch_orders(names, source["config"]["seed"], 5) for _ in MODES]
    assert orders[0] == orders[1] == orders[2]
    args = SimpleNamespace(batch_size=source["config"]["batch_size"], seed=source["config"]["seed"],
                           num_workers=0, device=device, chunk_size=128, progress_interval=0,
                           epochs=5, sampler_seed=1000045, cap=1024)
    indices = {name: i for i, name in enumerate(names)}
    for mode in MODES:
        view, rows = check_selection(batch, mode)
        model = TriWaveGNN(len(source["logic_vocab"]), len(roles), len(source["pin_vocab"]), source["config"]["hidden"])
        model.load_state_dict(source["model"])
        model.to(device).train()
        optimizer = make_optimizer(model, source, 3e-5)
        assert not optimizer.state
        full, sampled = move(batch, device), move(view, device)
        output = model_outputs(model, full)
        original_loss, original_terms = loss_terms(output, full, 3)
        sampled_loss, sampled_terms = loss_terms(output, sampled, 3)
        for key in original_terms.keys() - {"func"}:
            assert original_terms[key] == sampled_terms[key], f"Nonfunctional task changed: {key}"
        if mode == "original":
            assert torch.allclose(original_loss, sampled_loss, atol=1e-7, rtol=1e-6)
        sampled_loss.backward()
        gradients = {name: float(parameter.grad.abs().sum()) for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        assert sum(gradients.values()) > 0
        for branch in ("s", "f", "b"):
            assert sum(value for name, value in gradients.items() if name.startswith("updates." + branch + "_")) > 0
        optimizer.zero_grad(set_to_none=True)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        args.mode = mode
        with TemporaryDirectory() as temporary:
            train = train_epoch(model, optimizer, dataset, orders[0][0], indices, args, 1, Path(temporary))
            stats = json.loads((Path(temporary) / "sampling.jsonl").read_text())
            assert stats["lr"] == [3e-5] and train["optimizer_steps"] == 1
        maximum = max(float((value - before[key]).abs().max()) for key, value in model.state_dict().items())
        assert maximum > 0 and model.training
        assert optimizer.state and all(int(state["step"]) == 1 for state in optimizer.state.values())
        report["modes"][mode] = {"sampled_pairs": sum(row["sampled_pairs"] for row in rows),
                                  "unique_pairs": sum(row["unique_pairs"] for row in rows),
                                  "duplicate_pairs": sum(row["duplicate_pairs"] for row in rows),
                                  "first_circuit_strata": rows[0]["sampled_strata"],
                                  "gradient_absolute_sum": sum(gradients.values()),
                                  "max_parameter_update": maximum, "optimizer_steps": 1,
                                  "train_mode": model.training, "nonfunctional_losses_unchanged": True}
        if mode == "circuit_cap_stratified":
            with patch("examples.finetune_functional_pairs.sample_functional_pairs", side_effect=AssertionError("Validation called training sampler")):
                validation = evaluate_validation(model, dataset, names, indices, args, 0)
            assert validation["functional_equivalence"]["positive_count"] == 1
            assert validation["functional_equivalence"]["negative_count"] == 2409
            assert validation["metric_counts"]["tt_distance_mae"] == 2410
            assert validation["metric_counts"]["prob_mae"] == 24
            assert validation["metric_counts"]["rc_accuracy"] == 6
            for key in ("hb_tp_mae", "hb_first_mean_mae", "hb_stable_mean_mae", "hb_glitch_rate_mae"):
                assert validation["metric_counts"][key] == 24
            report["full_validation_pairs"] = 2410
            report["validation_sampler_calls"] = 0
        del model, output, full, sampled, optimizer, original_loss, sampled_loss
    assert all(torch.equal(value, source_snapshot[key]) for key, value in source["model"].items())
    report["identical_epoch_orders"] = True
    report["independent_sampling_rng"] = True
    group = large_circuit_metrics([{"positive_count": 3, "negative_count": 9997, "tp": 2, "fp": 100},
                                    {"positive_count": 1, "negative_count": 5, "tp": 0, "fp": 5}])
    assert group["circuit_count"] == 1 and group["recall"] == 2 / 3
    assert group["false_positive_rate"] == 100 / 9997
    return report


class PairSamplingTest(unittest.TestCase):
    def test_modes_and_label_alignment(self):
        batch = collate_circuits(tiny_items(["pi", "cell_input", "cell_output", "po"]))
        for mode in MODES:
            with self.subTest(mode=mode):
                check_selection(batch, mode)

    def test_cross_circuit_pair_rejected(self):
        batch = collate_circuits(tiny_items(["pi", "cell_input", "cell_output", "po"]))
        batch.hf_pairs[1, 0] = 12
        with self.assertRaisesRegex(ValueError, "crosses"):
            sample_functional_pairs(batch, "circuit_cap", torch.Generator())

    def test_stratified_allocation(self):
        self.assertEqual(allocate_strata(dict(zero=1000, near=1000, far=2000, other=0), 1024),
                         dict(zero=256, near=256, far=512, other=0))
        self.assertEqual(allocate_strata(dict(zero=0, near=0, far=0, other=3), 3),
                         dict(zero=0, near=0, far=0, other=3))
        for counts in (dict(zero=1, near=1, far=1, other=10000), dict(zero=1, near=1, far=1, other=1)):
            result = allocate_strata(counts, min(1024, sum(counts.values())))
            self.assertEqual(sum(result.values()), min(1024, sum(counts.values())))
            self.assertTrue(all(result[key] > 0 for key in counts if counts[key]))

    def test_optimizer_and_full_validation(self):
        report = self_check(tiny_source(), "cpu")
        self.assertTrue(report["source_weights_unchanged"])

    def test_chunked_functional_distance_and_gradient(self):
        generator = torch.Generator().manual_seed(125)
        hf = torch.randn(15, 12, dtype=torch.float64, generator=generator, requires_grad=True)
        pairs = torch.randint(15, (2, 47), generator=generator)
        full = functional_distance(hf, pairs)
        chunks = functional_distance(hf, pairs, chunk_size=7)
        self.assertTrue(torch.equal(full, chunks))
        weight = torch.linspace(.1, .9, full.numel(), dtype=torch.float64)
        grad_full = torch.autograd.grad((full * weight).sum(), hf)[0]
        grad_chunks = torch.autograd.grad((chunks * weight).sum(), hf)[0]
        torch.testing.assert_close(grad_full, grad_chunks, atol=1e-12, rtol=1e-12)

    def test_complete_finetune_lifecycle_on_temporary_graphs(self):
        from examples import finetune_functional_pairs as entry
        source = tiny_source()
        source.update(architecture_id=TriWaveGNN.architecture_id, epoch=40, stage=3, stage_complete=True,
                      source_sha256={"graphs.npz": "fixture", "labels.npz": "fixture"},
                      feature_normalization={"net": {"mean": [0.], "std": [1.]}})
        source["config"]["lr_step"] = 20
        items = tiny_items(source["role_vocab"]) + tiny_items(source["role_vocab"])
        for index, item in enumerate(items):
            item.name = ("train_" if index < 3 else "val_") + item.name
        dataset = TinyDataset(items)
        dataset.ids = [item.name for item in items]
        dataset.source_sha256 = source["source_sha256"]
        source["split_ids"] = {"train": dataset.ids[:3], "val": dataset.ids[3:], "test": ["must_never_be_loaded"]}

        def temporary_load(path, supplied_dataset, device):
            self.assertIs(supplied_dataset, dataset)
            model = TriWaveGNN(1, 4, 1, hidden=8).to(device)
            model.load_state_dict(source["model"])
            return model.eval(), source

        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            checkpoint_path = root / "stage3_best.pt"
            torch.save(source, checkpoint_path)
            args = SimpleNamespace(checkpoint=checkpoint_path, data=root / "unchanged_data", experiment_root=root / "experiments",
                                   lr=3e-5, epochs=1, cap=1024, device="cpu", chunk_size=128, progress_interval=0)
            with patch.object(entry, "data_distribution", return_value={}), \
                 patch.object(entry, "CircuitDataset", return_value=dataset), \
                 patch.object(entry, "load_model", side_effect=temporary_load), \
                 patch.object(entry, "evaluate_validation", wraps=entry.evaluate_validation) as validation_call:
                entry.prepare(args, source)
                for mode in MODES:
                    args.mode = mode
                    entry.run(args)
                    directory = args.experiment_root / mode
                    self.assertTrue((directory / "best_loss.pt").exists())
                    self.assertTrue((directory / "best_equiv.pt").exists())
                    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
                    self.assertEqual(last["epoch"], 1)
                    self.assertTrue(all(int(state["step"]) == 1 for state in last["optimizer"]["state"].values()))
                    self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "completed")
                    result = json.loads((directory / "val_epoch_1.json").read_text())
                    self.assertEqual(result["metric_counts"]["tt_distance_mae"], 2410)
                    with self.assertRaises(FileExistsError):
                        entry.run(args)
                # Baseline once, then one full validation per mode/epoch.
                self.assertEqual(validation_call.call_count, 4)
            baselines = [json.loads((args.experiment_root / mode / "baseline.json").read_text()) for mode in MODES]
            self.assertEqual(baselines[0], baselines[1])
            self.assertEqual(baselines[1], baselines[2])


if __name__ == "__main__":
    unittest.main()
