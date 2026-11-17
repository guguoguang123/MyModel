"""F: B's unchanged four-task fine-tuning with identity functional net wiring.

Preparation and temporary self-checks do not start formal training. --run is a
separate manual action. Neither path evaluates test-set predictions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import finetune_functional_pairs as base
from examples.compare_functional_stages import within_circuit_auc
from examples.evaluate import RUN_DIR, digest_file, validate_splits
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.utils.batch import collate_circuits
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint, random_state
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.net_identity import checkpoint_net_identity
from triwave_gnn.utils.net_identity_audit import audit_graph_bundle
from triwave_gnn.utils.pair_sampling import sample_functional_pairs
from triwave_gnn.utils.random_seed import set_seed

DEFAULT_B = RUN_DIR.parent / "functional_pair_sampling_stage3/circuit_cap"


def code_paths():
    root = Path(__file__).resolve().parents[1]
    return sorted((root / "triwave_gnn").rglob("*.py")) + [
        Path(__file__).resolve(), Path(base.__file__).resolve(),
        root / "examples/compare_functional_stages.py", root / "examples/evaluate.py"]


def grouped_saved_reference(validation):
    """B persisted circuit metrics, not pair scores: do not fabricate pooled AUC."""
    groups = {}
    for name, low, high in (("1-999", 1, 999), ("1000-9999", 1000, 9999),
                            (">=10000", 10000, sys.maxsize)):
        rows = [row for row in validation["per_circuit"]
                if low <= row["positive_count"] + row["negative_count"] <= high]
        positive = sum(row["positive_count"] for row in rows)
        negative = sum(row["negative_count"] for row in rows)
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        groups[name] = {"circuit_count": len(rows), "sample_count": positive + negative,
                        "positive_count": positive, "negative_count": negative,
                        "positive_ratio": positive / (positive + negative) if positive + negative else None,
                        "tp": tp, "fp": fp, "fn": fn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / positive if positive else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                        "roc_auc": None,
                        "pooled_auc_unavailable_reason": "B validation per-pair scores were not saved; no B validation inference rerun",
                        **within_circuit_auc(rows)}
    return groups


def compact_validation(validation):
    result = {key: value for key, value in validation.items() if key != "per_circuit"}
    if "groups_by_original_valid_pairs" not in result:
        result["groups_by_original_valid_pairs"] = grouped_saved_reference(validation)
    return result


def b_checkpoint_reference(run, filename):
    path = run / filename
    state = load_checkpoint(path, "cpu")
    epoch = state["epoch"]
    validation_path = run / (f"val_epoch_{epoch}.json" if epoch else "baseline.json")
    validation = json.loads(validation_path.read_text())
    if (validation["epoch"] != epoch or
            validation["functional_equivalence"]["f1"] != state["validation_f1"] or
            validation["functional_equivalence"]["similarity_threshold"] != state["validation_threshold"]):
        raise ValueError(f"B checkpoint/validation record mismatch: {path}")
    return {"checkpoint": str(path.resolve()), "checkpoint_sha256": digest_file(path),
            "actual_epoch": epoch, "validation_record": str(validation_path.resolve()),
            "validation": compact_validation(validation)}


def set_plan_args(args, plan):
    args.data, args.checkpoint = Path(plan["data"]), Path(plan["initial_checkpoint"])
    args.b_run = Path(plan["comparison_b_run"])
    for key in ("lr", "epochs", "cap", "batch_size", "seed", "num_workers", "sampler_seed"):
        setattr(args, key, plan[key])
    args.mode = "circuit_cap"
    args.chunk_size = plan["evaluation_chunk_size"]


def prepare(args):
    continuing = getattr(args, "finish_prepare", False)
    if args.output.exists() and not continuing:
        raise FileExistsError(f"Preserving existing F directory: {args.output}")
    if continuing:
        previous = json.loads((args.output / "plan.json").read_text())
        if previous["status"] != "preparing" or (args.output / "run.json").exists():
            raise ValueError("Only unfinished preparation can be completed; formal runs are never reset")
    bplan = json.loads((args.b_run / "plan.json").read_text())
    brun = json.loads((args.b_run / "run.json").read_text())
    if brun["status"] != "completed" or bplan["mode"] != "circuit_cap":
        raise ValueError("F requires the completed original B circuit_cap run")
    if (bplan["lr"], bplan["epochs"], bplan["cap"]) != (3e-5, 5, 1024):
        raise ValueError("B is not the requested 3e-5 / 5 epochs / cap1024 control")
    plan = dict(bplan, group="F", status="preparing", functional_net_identity=True,
                comparison_b_run=str(args.b_run.resolve()),
                unique_experiment_variable="functional_net_identity False -> True",
                evaluation_chunk_size=brun.get("evaluation_chunk_size", 100000),
                baseline="new full validation at epoch 0 with identity enabled; B baseline never reused as F baseline",
                f_net_bypass=["aggregators.f_net", "updates.f_net"],
                task_weights={"prob": 3., "rc": 1., "func": 2., "behavior": 2.},
                effective_weights_all_active={"prob": .375, "rc": .125, "func": .25, "behavior": .25},
                gradient_policy="normal autograd through current source Hf; no detach or freezing",
                compare_b_sampling_each_batch=True, generator_source_available=False,
                validation_groups=["1-999", "1000-9999", ">=10000"],
                test_evaluated=False)
    set_plan_args(args, plan)
    if digest_file(args.checkpoint) != plan["initial_checkpoint_sha256"]:
        raise ValueError("Original stage3 checkpoint differs from B")
    source = load_checkpoint(args.checkpoint, "cpu")
    if source["stage"] != 3 or checkpoint_net_identity(source):
        raise ValueError("F must initialize from the same original, identity-disabled stage3 checkpoint as B")
    splits = json.loads((args.b_run / "split.json").read_text())
    validate_splits(splits, source)
    orders = json.loads((args.b_run / "epoch_orders.json").read_text())
    if orders != base.epoch_orders(splits["train"], args.seed, args.epochs):
        raise ValueError("B epoch orders disagree with recorded seed")
    args.output.mkdir(parents=True, exist_ok=continuing)
    with base.direct_log(args.output / "prepare.log"):
        base.write_json(args.output / "plan.json", plan)
        for filename in ("split.json", "epoch_orders.json"):
            shutil.copyfile(args.b_run / filename, args.output / filename)
        info("F preparation only: loading full data for net-edge checks; no formal training")
        dataset = CircuitDataset(args.data, cache_mode="none")
        if dataset.source_sha256 != source["source_sha256"]:
            raise ValueError("Data files differ from B/original checkpoint")
        audit_path = args.output / "net_edge_audit.json"
        if continuing and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["source_sha256"] != dataset.source_sha256:
                raise ValueError("Data changed since the stored net-edge audit")
            info("Reusing completed net-edge audit on identical data; not repeating full graph audit")
        else:
            audit = audit_graph_bundle(dataset.graph, dataset.label, notify=info)
            audit["source_sha256"] = dataset.source_sha256
            base.write_json(audit_path, audit)
        info(f"Net audit: {audit['status']} | totals={audit['totals']} | failures={audit['failure_counts']}")
        if audit["status"] != "passed":
            raise ValueError("Net-edge conditions failed; F is not prepared; see net_edge_audit.json")
        info(audit["semantics_evidence"]["limitation"])
        # This original tracked source exactly matches the pre-F propagation
        # hash recorded by E. Preserve it for the default-off regression check.
        root = Path(__file__).resolve().parents[2]
        original = subprocess.check_output(["git", "show", "HEAD:Code/triwave_gnn/arch/propagation.py"], cwd=root).decode()
        original_hash = hashlib.sha256(original.encode()).hexdigest()
        expected_hash = "172a85fc7a5eb7969e2a358a140df0c234dde88568c9687d28f58d7c16e6b411"
        if original_hash != expected_hash:
            raise ValueError("Pre-F reference propagation source differs from the recorded original")
        (args.output / "original_propagation_reference.py").write_text(original)
        set_seed(args.seed)
        # Apply source vocabulary/normalization with strict state loading.
        initial_model, loaded = load_model(args.checkpoint, dataset, "cpu")
        del initial_model, loaded
        lookup = {name: i for i, name in enumerate(dataset.ids)}
        pair_counts = np.diff(dataset.label["hf_tt_pair_ptr"])
        candidates = sorted([name for name in splits["train"]
                             if 20 <= int(dataset.graph["num_nodes"][lookup[name]]) <= 2000
                             and pair_counts[lookup[name]] >= 2],
                            key=lambda name: (int(dataset.graph["num_nodes"][lookup[name]]), name))
        if len(candidates) < 3:
            raise ValueError("Need at least three small real training graphs for the self-check")
        names = [candidates[round((len(candidates) - 1) * q)] for q in (.05, .5, .95)]
        from triwave_gnn.tests.test_net_identity import check_real_models
        check_items = [dataset[lookup[name]] for name in names]
        atomic_save(check_items, args.output / "self_check_graphs.pt")
        check = check_real_models(source, check_items, original, args.device)
        batch = collate_circuits([dataset[lookup[name]] for name in orders[0][:args.batch_size]])
        _, _, rows = sample_functional_pairs(batch, "circuit_cap", torch.Generator().manual_seed(args.sampler_seed), args.cap)
        with (args.b_run / "sampling.jsonl").open() as stream:
            expected = json.loads(next(stream))
        if (expected["epoch"], expected["batch"]) != (1, 1) or expected["circuits"] != rows:
            raise ValueError("First real F graph batch or sampled pairs differ from B")
        check["first_real_B_batch_pair_columns_match"] = True
        # Round-trip flag + every weight through the regular inference loader.
        with TemporaryDirectory(prefix="F_identity_selfcheck_") as temporary:
            temp_state = dict(source, functional_net_identity=True,
                              config=dict(source["config"], functional_net_identity=True))
            path = Path(temporary) / "identity.pt"
            atomic_save(temp_state, path)
            reloaded, _ = load_model(path, dataset, "cpu")
            assert reloaded.functional_net_identity
            assert all(torch.equal(value, source["model"][key])
                       for key, value in reloaded.state_dict().items())
            check["strict_checkpoint_roundtrip_and_flag_restore"] = True
            del reloaded
        check["original_propagation_sha256"] = original_hash
        check["initial_checkpoint_sha256"] = digest_file(args.checkpoint)
        if check["initial_checkpoint_sha256"] != plan["initial_checkpoint_sha256"]:
            raise RuntimeError("Original checkpoint changed")
        base.write_json(args.output / "self_check.json", check)
        info(f"Actual-checkpoint self-check: {check}")
        b_reference = {"last": b_checkpoint_reference(args.b_run, "last.pt"),
                       "best_equiv": b_checkpoint_reference(args.b_run, "best_equiv.pt")}
        base.write_json(args.output / "B_comparison_reference.json", b_reference)
        plan.update(status="prepared", source_sha256=dataset.source_sha256,
                    b_sampling_sha256=digest_file(args.b_run / "sampling.jsonl"),
                    net_edge_audit_sha256=digest_file(args.output / "net_edge_audit.json"),
                    self_check_sha256=digest_file(args.output / "self_check.json"),
                    self_check_passed=True,
                    code_sha256={str(path.resolve()): digest_file(path) for path in code_paths()})
        base.write_json(args.output / "plan.json", plan)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--output", str(args.output.resolve()), "--run", "--device", "cuda"]
        script = "#!/usr/bin/env bash\nset -euo pipefail\nexport OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec " + shlex.join(command) + "\n"
        (args.output / "run.sh").write_text(script)
        info(f"Prepared F; no formal training or validation/test inference started: {args.output}")


def save_checkpoint(path, model, optimizer, source, args, epoch, validation, sampler_state=None):
    state = {key: deepcopy(source[key]) for key in (
        "architecture_id", "logic_vocab", "role_vocab", "pin_vocab", "source_sha256", "split_ids", "feature_normalization")}
    state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 config=dict(source["config"], lr=args.lr, output=str(args.output), resume=None,
                             functional_net_identity=True),
                 functional_net_identity=True, experiment_group="F", stage=3, epoch=epoch,
                 stage_complete=False, finetune_mode="circuit_cap", finetune_epochs=args.epochs,
                 initial_checkpoint=str(args.checkpoint), sampler_state=sampler_state,
                 validation_loss=validation["mean_loss"],
                 validation_f1=validation["functional_equivalence"]["f1"],
                 validation_threshold=validation["functional_equivalence"]["similarity_threshold"],
                 random_state=random_state())
    atomic_save(state, path)


def print_validation(result, label, b_reference=None):
    base.print_validation(result, label)
    eq = result["functional_equivalence"]
    info(f"{label} confusion: TP={eq['tp']} FP={eq['fp']} FN={eq['fn']}")
    for name, row in result["groups_by_original_valid_pairs"].items():
        info(f"{label} original-pair-count group={name}: " + " | ".join(
            f"{key}={row[key]}" for key in ("circuit_count", "sample_count", "positive_ratio", "roc_auc",
                                           "precision", "recall", "f1", "tp", "fp", "fn")))
    if b_reference:
        delta = {key: result["metrics"][key] - value for key, value in b_reference["metrics"].items()
                 if value is not None and result["metrics"].get(key) is not None}
        info(f"{label} task_metric_delta_vs_B_same_epoch={delta}")
        info(f"{label} task_metric_degraded_vs_B_same_epoch=" + str(
            {key: (value < 0 if key == "rc_accuracy" else value > 0) for key, value in delta.items()}))


def run(args):
    plan = json.loads((args.output / "plan.json").read_text())
    if plan["status"] != "prepared" or plan["group"] != "F" or not plan["self_check_passed"]:
        raise ValueError("Not a prepared, checked F experiment")
    if (args.output / "run.json").exists():
        raise FileExistsError(f"Preserving previously started F experiment: {args.output}")
    for name, expected in plan["code_sha256"].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f"Code changed after F preparation: {name}; prepare a new output directory")
    for filename, key in (("net_edge_audit.json", "net_edge_audit_sha256"), ("self_check.json", "self_check_sha256")):
        if digest_file(args.output / filename) != plan[key]:
            raise ValueError(f"Prepared F evidence changed: {filename}")
    set_plan_args(args, plan)
    if digest_file(args.checkpoint) != plan["initial_checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed")
    if digest_file(args.b_run / "sampling.jsonl") != plan["b_sampling_sha256"]:
        raise ValueError("B sampling reference changed")
    splits = json.loads((args.output / "split.json").read_text())
    orders = json.loads((args.output / "epoch_orders.json").read_text())
    if splits != json.loads((args.b_run / "split.json").read_text()) or orders != json.loads((args.b_run / "epoch_orders.json").read_text()):
        raise ValueError("F split/batch order differs from B")
    with base.direct_log(args.output / "console.log"):
        record = dict(plan, status="running", completed_epochs=0, device=args.device)
        base.write_json(args.output / "run.json", record)
        info(f"F: functional_net_identity=True | source={args.checkpoint} | fresh optimizer | lr={args.lr} "
             f"| batch_size={args.batch_size} seed={args.seed} epochs={args.epochs} cap={args.cap}")
        info("Only functional net message+GRU bypassed; original four-task loss, active denominator and clipping unchanged")
        info("Hs propagation unchanged; Hb formula unchanged and can respond to changed upstream Hf")
        set_seed(args.seed)
        dataset = CircuitDataset(args.data, cache_mode="small")
        model, source = load_model(args.checkpoint, dataset, args.device)
        if source["source_sha256"] != dataset.source_sha256:
            raise ValueError("Original data files changed")
        model.functional_net_identity = True  # Strictly loaded original weights first.
        info("Strict original checkpoint load succeeded: no missing/unexpected parameters; f_net parameters retained but bypassed")
        val_key = validate_splits(splits, source)
        indices = {name: i for i, name in enumerate(dataset.ids)}
        optimizer = base.make_optimizer(model, source, args.lr)
        if optimizer.state:
            raise RuntimeError("F must start without optimizer state")
        baseline = base.evaluate_validation(model, dataset, splits[val_key], indices, args, 0,
                                            include_pair_count_groups=True)
        base.write_json(args.output / "baseline.json", baseline)
        b_baseline = json.loads((args.b_run / "baseline.json").read_text())
        print_validation(baseline, "F BASELINE epoch=0 (identity enabled)", b_baseline)
        best_loss, best_f1 = baseline["mean_loss"], baseline["functional_equivalence"]["f1"]
        best_loss_epoch = best_equiv_epoch = 0
        for filename in ("best_loss.pt", "best_equiv.pt"):
            save_checkpoint(args.output / filename, model, optimizer, source, args, 0, baseline)
        history = []
        for epoch, order in enumerate(orders, 1):
            info(f"F epoch={epoch}/{args.epochs} | actual_lr={[group['lr'] for group in optimizer.param_groups]}")
            training = base.train_epoch(model, optimizer, dataset, order, indices, args, epoch, args.output,
                                        sampling_reference=args.b_run / "sampling.jsonl")
            validation = base.evaluate_validation(model, dataset, splits[val_key], indices, args, epoch,
                                                  include_pair_count_groups=True)
            base.write_json(args.output / f"val_epoch_{epoch}.json", validation)
            sampler_state = training.pop("sampler_state")
            if validation["mean_loss"] < best_loss:
                best_loss, best_loss_epoch = validation["mean_loss"], epoch
                save_checkpoint(args.output / "best_loss.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            f1 = validation["functional_equivalence"]["f1"]
            if f1 is not None and (best_f1 is None or f1 > best_f1):
                best_f1, best_equiv_epoch = f1, epoch
                save_checkpoint(args.output / "best_equiv.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            save_checkpoint(args.output / "last.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            history.append({"epoch": epoch, "train": training, "val": compact_validation(validation)})
            base.write_json(args.output / "history.json", history)
            info(f"F epoch={epoch} sampling: {training['sampling']} | exact B pair columns matched")
            b_validation = json.loads((args.b_run / f"val_epoch_{epoch}.json").read_text())
            print_validation(validation, f"F VAL epoch={epoch}", b_validation)
            record.update(completed_epochs=epoch, best_loss=best_loss, best_loss_epoch=best_loss_epoch,
                          best_f1=best_f1, best_equiv_epoch=best_equiv_epoch, B_pair_columns_match=True)
            base.write_json(args.output / "run.json", record)
            if plan["lr_step"] and epoch % plan["lr_step"] == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
        best_path = args.output / (f"val_epoch_{best_equiv_epoch}.json" if best_equiv_epoch else "baseline.json")
        comparison = {"B": json.loads((args.output / "B_comparison_reference.json").read_text()),
                      "F": {"last": {"actual_epoch": args.epochs, "checkpoint": str(args.output / "last.pt"),
                                      "validation": compact_validation(validation)},
                            "best_equiv": {"actual_epoch": best_equiv_epoch, "checkpoint": str(args.output / "best_equiv.pt"),
                                           "validation": compact_validation(json.loads(best_path.read_text()))}},
                      "scope": "complete original validation; each checkpoint uses its own validation-selected threshold; no test evaluation"}
        base.write_json(args.output / "comparison_B_F.json", comparison)
        for group, selections in comparison.items():
            if group not in ("B", "F"):
                continue
            for selection, row in selections.items():
                result = row["validation"]
                eq = result["functional_equivalence"]
                info(f"SUMMARY {group} / {selection} actual_epoch={row['actual_epoch']} | " + " | ".join(
                    f"{key}={eq[key]}" for key in ("precision", "recall", "f1", "tp", "fp", "fn", "roc_auc"))
                    + f" | macro_auc={result['circuit_macro_roc_auc']} | within_circuit_weighted_auc={result['within_circuit_weighted_auc']}")
        record.update(status="completed", test_evaluated=False)
        base.write_json(args.output / "run.json", record)
        info(f"Completed F: best_loss epoch={best_loss_epoch}, best_equiv epoch={best_equiv_epoch}; test not evaluated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, default=DEFAULT_B)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-interval", type=int, default=100)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="audit data and self-check temporary models only")
    action.add_argument("--finish-prepare", action="store_true", help="complete unfinished preparation, reusing its data audit")
    action.add_argument("--run", action="store_true", help="manually start the prepared five-epoch F experiment")
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("progress-interval must be nonnegative")
    args.output = args.output.resolve()
    (run if args.run else prepare)(args)


if __name__ == "__main__":
    main()
