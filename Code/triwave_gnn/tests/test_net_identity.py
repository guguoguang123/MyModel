"""Identity wiring, causal gradients, fanout, batching and strict checkpoints."""

from copy import deepcopy
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
from ..utils.batch import collate_circuits
from ..utils.losses import loss_terms
from ..utils.model_utils import model_outputs
from ..utils.net_identity import annotate_net_identity_roles, checkpoint_net_identity
from .test_pair_finetune import tiny_item, tiny_source, TinyDataset

ROLES = ["pi", "cell_input", "cell_output", "po"]


def identity_fixture(name="fanout", fanout=True):
    item = tiny_item(name, [0., .1, .9, .1], ROLES)
    if fanout:
        # Output 4 now drives both input 5 and PO 7, before cell 5->6.
        item["pin", "net", "pin"].edge_index[0, -1] = 4
        item["pin"].depth[-1] = 3
        item["pin"].topological_order = torch.tensor([0, 1, 2, 3, 4, 5, 7, 6])
    return annotate_net_identity_roles(item, ROLES, ["AND"])


def make_model(source=None, identity=False):
    if source is None:
        return TriWaveGNN(1, 4, 1, 8, functional_net_identity=identity)
    model = TriWaveGNN(len(source["logic_vocab"]), len(source["role_vocab"]),
                       len(source["pin_vocab"]), source["config"]["hidden"],
                       functional_net_identity=identity)
    model.load_state_dict(source["model"], strict=True)
    return model


def legacy_model(source, original_propagation_source):
    """Execute the preserved pre-change propagation code as a reference."""
    namespace = {"__name__": "triwave_gnn.arch._pre_identity_reference",
                 "__package__": "triwave_gnn.arch"}
    exec(compile(original_propagation_source, "pre_identity_propagation.py", "exec"), namespace)
    reference = namespace["TopologicalPropagation"]()

    class Adapter(torch.nn.Module):
        def forward(self, *args, functional_net_identity=False, **kwargs):
            assert not functional_net_identity
            return reference(*args, **kwargs)

    model = make_model(source)
    model.propagation = Adapter()
    return model


def check_real_models(source, items, original_propagation_source, device):
    """Temporary actual-checkpoint models only; never save updated weights."""
    before = {key: value.clone() for key, value in source["model"].items()}
    # CUDA scatter reductions may differ in their last bits between identical
    # forwards. Establish exact CPU agreement, then bound GPU roundoff.
    cpu_reference = legacy_model(source, original_propagation_source).eval()
    cpu_disabled = make_model(source).eval()
    cpu_batch = collate_circuits(items)
    with torch.no_grad():
        cpu_old, cpu_off = cpu_reference(cpu_batch), cpu_disabled(cpu_batch)
        cpu_differences = [float((a - b).abs().max()) for a, b in zip(cpu_old, cpu_off)]
        assert cpu_differences == [0., 0., 0.], cpu_differences
    del cpu_reference, cpu_disabled, cpu_batch, cpu_old, cpu_off
    reference = legacy_model(source, original_propagation_source).to(device).eval()
    disabled = make_model(source).to(device).eval()
    enabled = make_model(source, identity=True).to(device).eval()
    batch = collate_circuits(items).to(device)
    with torch.no_grad():
        old = reference(batch)
        off = disabled(batch)
        on = enabled(batch)
        disabled_differences = [float((a - b).abs().max()) for a, b in zip(old, off)]
        for a, b in zip(old, off):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        edges = batch["pin", "net", "pin"].edge_index
        assert torch.equal(on[1][edges[0]], on[1][edges[1]])
        torch.testing.assert_close(off[0], on[0], atol=1e-6, rtol=1e-5)
        hs_difference = float((off[0] - on[0]).abs().max())
        max_batch_difference = 0.
        for index, item in enumerate(items):
            separate = enabled(item.clone().to(device))
            start, stop = batch["pin"].ptr[index:index + 2]
            for expected, combined in zip(separate, on):
                actual = combined[start:stop]
                torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
                max_batch_difference = max(max_batch_difference, float((expected - actual).abs().max()))
    enabled.train()
    optimizer = torch.optim.Adam(enabled.parameters(), lr=3e-5)
    output = model_outputs(enabled, batch)
    total, terms = loss_terms(output, batch, 3, functional_chunk_size=128)
    assert set(("prob", "rc", "func", "behavior")) <= terms.keys()
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(enabled.parameters(), 1., error_if_nonfinite=True)
    for family in ("aggregators", "updates"):
        assert all(p.grad is None for p in getattr(enabled, family)["f_net"].parameters())
    optimizer.step()
    maximum_update = max(float((value.cpu() - before[key].cpu()).abs().max())
                         for key, value in enabled.state_dict().items())
    assert maximum_update > 0
    assert all(torch.equal(source["model"][key], value) for key, value in before.items())
    return {"device": device, "circuits": [item.name for item in items],
            "legacy_vs_disabled_cpu_max_hs_hf_hb": cpu_differences,
            "legacy_vs_disabled_max_hs_hf_hb": disabled_differences,
            "cuda_comparison_tolerance": {"atol": 1e-6, "rtol": 1e-5},
            "enabled_net_endpoint_hf_max_difference": 0.,
            "enabled_single_vs_batch_max_difference": max_batch_difference,
            "hs_path_unchanged": True, "hs_numerical_max_difference": hs_difference,
            "f_net_message_and_gru_bypassed": True,
            "all_four_task_losses_present": True, "gradient_norm": float(grad_norm),
            "gradients_finite": True, "temporary_optimizer_steps": 1,
            "maximum_parameter_update": maximum_update, "source_weights_unchanged": True}


class NetIdentityTest(unittest.TestCase):
    def test_identity_uses_current_generated_source_and_accumulates_fanout_gradients(self):
        model = make_model(identity=True).train()
        generated = []

        def capture(module, inputs, value):
            value.retain_grad()
            generated.append(value)

        hook = model.updates["f_cell"].register_forward_hook(capture)
        _, hf, _ = model(identity_fixture())
        hook.remove()
        self.assertTrue(torch.equal(hf[4], hf[5]))
        self.assertTrue(torch.equal(hf[4], hf[7]))
        # Source output 4 was generated by cell logic at depth 2, not copied
        # from its uncomputed initialization. Both fanout paths accumulate.
        self.assertTrue(torch.equal(hf[4], generated[0][0]))
        (hf[5].square().sum() + hf[7].square().sum()).backward()
        torch.testing.assert_close(generated[0].grad[0], 4 * generated[0][0])
        self.assertGreater(float(model.updates["f_cell"].weight_ih.grad.norm()), 0.)
        self.assertGreater(float(model.initialization.pi_function.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.aggregators["f_net"].parameters()))
        self.assertTrue(all(p.grad is None for p in model.updates["f_net"].parameters()))

    def test_unequal_depth_batch_and_default_off(self):
        off = make_model().eval()
        explicit_off = deepcopy(off)
        explicit_off.functional_net_identity = False
        on = deepcopy(off)
        on.functional_net_identity = True
        items = [identity_fixture(), identity_fixture("longer", fanout=False)]
        batch = collate_circuits(items)
        with torch.no_grad():
            for a, b in zip(off(batch), explicit_off(batch)):
                self.assertTrue(torch.equal(a, b))
            combined = on(batch)
            for i, item in enumerate(items):
                for separate, joined in zip(on(item), combined):
                    torch.testing.assert_close(separate, joined[i*8:(i+1)*8], atol=1e-6, rtol=1e-5)
            self.assertTrue(torch.equal(off(batch)[0], combined[0]))
        self.assertEqual(list(off.state_dict()), list(on.state_dict()))

    def test_invalid_multidriver_depth_roles_and_mixed_relations_raise(self):
        model = make_model(identity=True)
        for kind in ("multi", "depth", "role", "logic", "mixed"):
            item = identity_fixture()
            if kind == "multi":
                store = item["pin", "net", "pin"]
                store.edge_index = torch.cat((store.edge_index, torch.tensor([[0], [5]])), dim=1)
                store.edge_attr = torch.cat((store.edge_attr, store.edge_attr[:1]), dim=0)
            elif kind == "depth":
                item["pin"].depth[5] = item["pin"].depth[4]
            elif kind == "role":
                item["pin"].functional_net_receiver_ok[5] = False
            elif kind == "logic":
                annotate_net_identity_roles(item, ROLES, ["DFF"])
            else:
                store = item["pin", "cell", "pin"]
                store.edge_index = torch.cat((store.edge_index, torch.tensor([[2], [5]])), dim=1)
                store.edge_attr = torch.cat((store.edge_attr, store.edge_attr[:1]), dim=0)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "refuses unverified"):
                model(item)

    def test_missing_weights_fail_and_metadata_is_explicit(self):
        model = make_model(identity=True)
        state = dict(model.state_dict())
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)
        self.assertFalse(checkpoint_net_identity({"config": {}}))
        self.assertTrue(checkpoint_net_identity({"functional_net_identity": True,
                                               "config": {"functional_net_identity": True}}))
        with self.assertRaises(ValueError):
            checkpoint_net_identity({"functional_net_identity": True,
                                     "config": {"functional_net_identity": False}})

    def test_temporary_f_lifecycle_keeps_full_validation_and_checkpoint_epochs(self):
        from examples import finetune_net_identity as entry
        from examples import finetune_functional_pairs as base
        from examples.evaluate import digest_file
        from ..utils.pair_sampling import sample_functional_pairs
        source = tiny_source()
        items = [identity_fixture(name, fanout=i % 2 == 0)
                 for i, name in enumerate(("train_a", "train_b", "val_a", "val_b"))]
        dataset = TinyDataset(items)
        dataset.ids = [item.name for item in items]
        dataset.source_sha256 = {"graphs.npz": "fixture", "labels.npz": "fixture"}
        splits = {"train": dataset.ids[:2], "val": dataset.ids[2:], "test": ["never_inferred"]}
        source.update(architecture_id=TriWaveGNN.architecture_id, source_sha256=dataset.source_sha256,
                      split_ids=splits, feature_normalization={}, epoch=40, stage=3)
        source["config"]["lr_step"] = 20
        orders = base.epoch_orders(splits["train"], 42, 2)
        indices = {name: i for i, name in enumerate(dataset.ids)}
        args = SimpleNamespace(device="cpu", progress_interval=0, batch_size=32, seed=42,
                               num_workers=0, chunk_size=128, epochs=2)

        def fake_load(path, data, device):
            return make_model(source).to(device).eval(), source

        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            b_run, output = root / "B", root / "F"
            b_run.mkdir()
            output.mkdir()
            args.output = output
            checkpoint = root / "stage3_best.pt"
            torch.save(source, checkpoint)
            for folder in (b_run, output):
                base.write_json(folder / "split.json", splits)
                base.write_json(folder / "epoch_orders.json", orders)
            sampling = []
            for epoch, order in enumerate(orders, 1):
                batch = collate_circuits([items[indices[name]] for name in order])
                _, _, rows = sample_functional_pairs(batch, "circuit_cap",
                    torch.Generator().manual_seed(1000045 + epoch - 1), 1024)
                sampling.append({"epoch": epoch, "batch": 1, "circuits": rows})
            (b_run / "sampling.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sampling))
            reference = base.evaluate_validation(make_model(source).eval(), dataset,
                                                 splits["val"], indices, args, 0)
            base.write_json(b_run / "baseline.json", reference)
            for epoch in (1, 2):
                base.write_json(b_run / f"val_epoch_{epoch}.json", dict(reference, epoch=epoch))
            base.write_json(output / "B_comparison_reference.json", {
                "last": {"actual_epoch": 2, "validation": entry.compact_validation(dict(reference, epoch=2))},
                "best_equiv": {"actual_epoch": 1, "validation": entry.compact_validation(dict(reference, epoch=1))}})
            base.write_json(output / "net_edge_audit.json", {"status": "passed"})
            base.write_json(output / "self_check.json", {"passed": True})
            plan = {"group": "F", "status": "prepared", "self_check_passed": True,
                    "data": str(root / "data"), "initial_checkpoint": str(checkpoint),
                    "initial_checkpoint_sha256": digest_file(checkpoint),
                    "comparison_b_run": str(b_run), "lr": 3e-5, "epochs": 2, "cap": 1024,
                    "batch_size": 32, "seed": 42, "num_workers": 0, "sampler_seed": 1000045,
                    "evaluation_chunk_size": 128, "lr_step": 20, "code_sha256": {},
                    "b_sampling_sha256": digest_file(b_run / "sampling.jsonl"),
                    "self_check_sha256": digest_file(output / "self_check.json"),
                    "net_edge_audit_sha256": digest_file(output / "net_edge_audit.json")}
            base.write_json(output / "plan.json", plan)
            with patch.object(entry, "CircuitDataset", return_value=dataset), \
                 patch.object(entry, "load_model", side_effect=fake_load), \
                 patch.object(base, "sample_functional_pairs", wraps=sample_functional_pairs) as sampler:
                entry.run(args)
                self.assertEqual(sampler.call_count, 2, "Validation must never invoke the training sampler")
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 2)
            for row in history:
                self.assertEqual(row["val"]["metric_counts"]["tt_distance_mae"], 8)
                self.assertIn("groups_by_original_valid_pairs", row["val"])
                self.assertEqual(set(("prob", "rc", "func", "behavior")) & row["train"]["terms"].keys(),
                                 set(("prob", "rc", "func", "behavior")))
            result = json.loads((output / "comparison_B_F.json").read_text())
            self.assertEqual(result["B"]["last"]["validation"]["epoch"], 2)
            self.assertEqual(result["B"]["best_equiv"]["validation"]["epoch"], 1)
            for selection in ("last", "best_equiv"):
                card = result["F"][selection]
                state = torch.load(output / (selection + ".pt"), map_location="cpu", weights_only=False)
                self.assertEqual(card["actual_epoch"], state["epoch"])
                self.assertEqual(card["validation"]["functional_equivalence"]["f1"], state["validation_f1"])
                self.assertTrue(checkpoint_net_identity(state))
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "completed")
            self.assertEqual(digest_file(checkpoint), plan["initial_checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
