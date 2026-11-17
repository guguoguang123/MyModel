"""Five-fold, circuit-group score calibration on saved G1 validation logits only."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import expanded_frozen_head_validation as reuse
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import write_json
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.logger import info

G1 = original.EXP / "frozen_head_dense_sampling_G1_20261113"
FINAL = original.EXP / "final_B_head33438_20261113"
FOLDS, SEED, WALL_SECONDS, DEFAULT_MAX_ITER = 5, 42, 600, 60
GROUPS = original.GROUPS
NOTICE = ("仅使用原始seed=20261113 G1 step43438已保存的完整val逐对logits、标签、电路ID和原始有效配对数；"
          "C0/C1五折交叉校准开发结果，不读取test，不重复模型推理，不覆盖G1/H0/H1记录")


class CalibrationTimeout(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def group_for_count(count):
    if count < 1000:
        return "1-999"
    if count < 10000:
        return "1000-9999"
    return ">=10000"


def metric_from_pred(labels, pred):
    labels = labels.bool()
    pred = pred.bool()
    positive, negative = int(labels.sum()), int((~labels).sum())
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    tn = negative - fp
    fn = positive - tp
    return {
        "positive_count": positive, "negative_count": negative,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positive if positive else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def grouped_pred_metrics(labels, pred, groups):
    return {group: metric_from_pred(labels[groups == group], pred[groups == group])
            for group in GROUPS}


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise CalibrationTimeout("10-minute wall-clock limit reached")


def load_saved_scores():
    source = read(G1 / "comparison.json")
    if not source.get("complete_val_evaluation") or source.get("test_evaluated"):
        raise ValueError("G1 saved full-val result is incomplete or includes test evaluation")
    val = torch.load(FINAL / "val_scores.pt", map_location="cpu", weights_only=False)
    circuits, offsets, labels = val["circuits"], val["offsets"], val["labels"].bool()
    if len(circuits) != 1663 or len(offsets) != 1664 or offsets[-1] != labels.numel():
        raise ValueError("Expected the saved complete 1663-circuit val index")
    parts = []
    cache_dir = G1 / "val_score_cache"
    for index, (name, start, stop) in enumerate(zip(circuits, offsets, offsets[1:])):
        check = cache_dir / f"{reuse.cache_key(name)}.pt"
        if not check.is_file():
            raise FileNotFoundError(f"Missing saved G1 val score cache: {check}")
        row = torch.load(check, map_location="cpu", weights_only=False)
        logits = row["logits"].detach().cpu().reshape(-1)
        local_labels = row["labels"].detach().cpu().bool().reshape(-1)
        if (row.get("circuit") != name or logits.numel() != stop - start
                or not torch.equal(local_labels, labels[start:stop])
                or not bool(torch.isfinite(logits).all())):
            raise ValueError(f"Saved G1 score/label/order mismatch: {name}")
        parts.append(logits)
    logits = torch.cat(parts) if parts else torch.empty(0)
    counts = torch.tensor([b - a for a, b in zip(offsets, offsets[1:])], dtype=torch.long)
    group_by_circuit = [group_for_count(int(n)) for n in counts.tolist()]
    # Integer group codes avoid object tensors and keep the saved pair order unchanged.
    codes = {name: i for i, name in enumerate(GROUPS)}
    pair_group_codes = torch.cat([torch.full((int(n),), codes[group], dtype=torch.long)
                                  for n, group in zip(counts.tolist(), group_by_circuit)])
    return {
        "source": source, "circuits": circuits, "offsets": offsets, "labels": labels,
        "logits": logits, "counts": counts, "group_by_circuit": group_by_circuit,
        "pair_group_codes": pair_group_codes,
        "g1_reference_correction": source["methods"]["G1"]["maximum_f1"]["at_fixed_val_threshold"],
    }


def make_folds(circuits, counts):
    order = list(range(len(circuits)))
    random.Random(SEED).shuffle(order)
    folds = []
    start = 0
    for fold in range(FOLDS):
        size = len(order) // FOLDS + (fold < len(order) % FOLDS)
        folds.append(order[start:start + size])
        start += size
    fold_of = torch.empty(len(circuits), dtype=torch.long)
    for fold, indices in enumerate(folds):
        fold_of[torch.tensor(indices)] = fold
    pair_fold = torch.cat([torch.full((int(n),), int(fold_of[i]), dtype=torch.long)
                           for i, n in enumerate(counts.tolist())])
    return folds, fold_of, pair_fold


def fit_c1(logits, labels, groups, mask, device, deadline, max_iter):
    x = logits[mask].to(device=device, dtype=torch.float64)
    y = labels[mask].to(device=device, dtype=torch.float64)
    g = groups[mask].to(device=device)
    raw_a = torch.nn.Parameter(torch.tensor(0.541324854612918, device=device, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([raw_a, bias], lr=0.5, max_iter=max_iter,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")
    calls = [0]

    def closure():
        check_deadline(deadline)
        optimizer.zero_grad(set_to_none=True)
        a = F.softplus(raw_a) + 1e-9
        z = a * x + bias[g]
        loss = F.binary_cross_entropy_with_logits(z, y, reduction="mean")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite calibration BCE")
        loss.backward()
        calls[0] += 1
        return loss

    optimizer.step(closure)
    check_deadline(deadline)
    with torch.no_grad():
        a = float((F.softplus(raw_a) + 1e-9).cpu())
        b = [float(v) for v in bias.cpu()]
        fitted = a * x + bias[g]
        loss = float(F.binary_cross_entropy_with_logits(fitted, y, reduction="mean").cpu())
    del x, y, g, fitted, optimizer, raw_a, bias
    if not math.isfinite(loss) or a <= 0:
        raise FloatingPointError("invalid fitted calibration parameters")
    return {"a": a, "bias": b, "fit_bce": loss, "optimizer_closure_calls": calls[0]}


def apply_c1(logits, group_codes, params):
    return params["a"] * logits.double() + torch.tensor(params["bias"], dtype=torch.float64)[group_codes]


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = started + WALL_SECONDS
    run_record = {"status": "preparing", "notice": NOTICE, "seed": SEED, "folds": FOLDS,
                  "max_optimizer_iterations": args.max_iter, "wall_seconds": WALL_SECONDS,
                  "device": args.device, "source_g1": str(G1), "test_evaluated": False,
                  "encoder_forwards": 0, "head_forwards": 0, "calibration_feature": "original_pair_count_group only"}
    write_json(args.output / "run.json", run_record)
    folds_done, fold_records = [], []
    try:
        data = load_saved_scores()
        folds, fold_of, pair_fold = make_folds(data["circuits"], data["counts"])
        fold_assignment = {name: int(fold_of[i]) for i, name in enumerate(data["circuits"])}
        write_json(args.output / "fold_assignment.json", fold_assignment)
        circuit_group_counts = {group: sum(n > 0 and group_for_count(int(n)) == group
                                            for n in data["counts"].tolist()) for group in GROUPS}
        circuit_group_counts["zero_valid_pairs"] = int((data["counts"] == 0).sum())
        zero_by_fold = {str(fold): sum(int(data["counts"][i]) == 0 for i in indices)
                        for fold, indices in enumerate(folds)}
        write_json(args.output / "plan.json", {
            "notice": NOTICE, "seed": SEED, "folds": FOLDS,
            "source_g1_comparison": str(G1 / "comparison.json"),
            "source_g1_comparison_sha256": file_hash(G1 / "comparison.json"),
            "source_val_index": str(FINAL / "val_scores.pt"),
            "source_val_index_sha256": file_hash(FINAL / "val_scores.pt"),
            "score_cache": str(G1 / "val_score_cache"), "calibration": "z=a*logit+b[group]",
            "groups": list(GROUPS), "fit_objective": "ordinary mean BCE on four training folds; raw class ratios",
            "threshold_rule": "maximum F1 with whole equal-score groups; higher threshold on equal F1",
            "max_optimizer_iterations": args.max_iter, "device": args.device,
            "test_evaluated": False, "encoder_forwards": 0,
        })
        run_record.update(status="running", valid_pairs=int(data["labels"].numel()),
                          positive_count=int(data["labels"].sum()),
                          circuit_group_counts=circuit_group_counts, zero_valid_by_fold=zero_by_fold,
                          g1_reference_correction={"recall": data["g1_reference_correction"]["recall"],
                                                   "tp": data["g1_reference_correction"]["tp"],
                                                   "fp": data["g1_reference_correction"]["fp"]})
        write_json(args.output / "run.json", run_record)
        info("G1 reference correction: Recall=%.15f TP=%d FP=%d" %
             (data["g1_reference_correction"]["recall"], data["g1_reference_correction"]["tp"],
              data["g1_reference_correction"]["fp"]))
        logits, labels, groups = data["logits"], data["labels"], data["pair_group_codes"]
        pred0 = torch.zeros(labels.numel(), dtype=torch.bool)
        pred1 = torch.zeros(labels.numel(), dtype=torch.bool)
        fold_of_pairs = pair_fold
        for fold, circuit_indices in enumerate(folds):
            check_deadline(deadline)
            train_mask = fold_of_pairs != fold
            held_mask = ~train_mask
            train_logits, train_labels = logits[train_mask], labels[train_mask]
            held_logits, held_labels = logits[held_mask], labels[held_mask]
            threshold0 = select_threshold(train_logits, train_labels)
            if threshold0 is None:
                raise ValueError(f"C0 training folds lack both classes at fold {fold}")
            params = fit_c1(logits, labels, groups, train_mask, args.device, deadline, args.max_iter)
            train_c1 = apply_c1(train_logits, groups[train_mask], params)
            threshold1 = select_threshold(train_c1, train_labels)
            if threshold1 is None:
                raise ValueError(f"C1 training folds lack both classes at fold {fold}")
            held_c1 = apply_c1(held_logits, groups[held_mask], params)
            p0 = held_logits >= threshold0
            p1 = held_c1 >= threshold1
            pred0[held_mask], pred1[held_mask] = p0, p1
            m0, m1 = metric_from_pred(held_labels, p0), metric_from_pred(held_labels, p1)
            fold_record = {"fold": fold, "circuit_count": len(circuit_indices),
                           "circuits": [data["circuits"][i] for i in circuit_indices],
                           "zero_valid_circuits": sum(int(data["counts"][i]) == 0 for i in circuit_indices),
                           "train_pair_count": int(train_mask.sum()), "heldout_pair_count": int(held_mask.sum()),
                           "c0_threshold": float(threshold0), "c1_threshold": float(threshold1),
                           "c1_parameters": params, "C0": m0, "C1": m1,
                           "C1_minus_C0": {"tp": m1["tp"] - m0["tp"], "fp": m1["fp"] - m0["fp"]}}
            fold_records.append(fold_record)
            folds_done.append(fold)
            write_json(args.output / "fold_results.json", fold_records)
            info("fold=%d C0 P/R/F1=%.6f/%.6f/%.6f TP/FP=%d/%d | C1 P/R/F1=%.6f/%.6f/%.6f TP/FP=%d/%d" %
                 (fold, m0["precision"], m0["recall"], m0["f1"], m0["tp"], m0["fp"],
                  m1["precision"], m1["recall"], m1["f1"], m1["tp"], m1["fp"]))
        overall = {}
        group_results = {}
        for name, pred in (("C0", pred0), ("C1", pred1)):
            overall[name] = metric_from_pred(labels, pred)
            group_results[name] = {}
            for code, group in enumerate(GROUPS):
                group_results[name][group] = metric_from_pred(labels[groups == code], pred[groups == code])
        report = {"status": "completed", "notice": NOTICE, "development_only": True,
                  "test_evaluated": False, "encoder_forwards": 0, "source": str(G1),
                  "fold_assignment": str(args.output / "fold_assignment.json"),
                  "g1_reference_correction": {"recall": data["g1_reference_correction"]["recall"],
                                              "tp": data["g1_reference_correction"]["tp"],
                                              "fp": data["g1_reference_correction"]["fp"]},
                  "folds": fold_records, "overall": overall, "groups": group_results,
                  "C1_minus_C0": {"tp": overall["C1"]["tp"] - overall["C0"]["tp"],
                                  "fp": overall["C1"]["fp"] - overall["C0"]["fp"]},
                  "completed_folds": folds_done}
        write_json(args.output / "comparison.json", report)
        for name in ("C0", "C1"):
            m = overall[name]
            info("OVERALL %s P/R/F1=%.6f/%.6f/%.6f TP/FP/TN/FN=%d/%d/%d/%d" %
                 (name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["tn"], m["fn"]))
        info("C1 minus C0: TP delta=%d; FP delta=%d" %
             (report["C1_minus_C0"]["tp"], report["C1_minus_C0"]["fp"]))
        run_record.update(status="completed", completed_folds=folds_done, stop_reason="completed")
    except CalibrationTimeout as error:
        run_record.update(status="timeout", completed_folds=folds_done, stop_reason="wall_time_limit",
                          error=str(error))
        write_json(args.output / "comparison.json", {"status": "timeout", "notice": NOTICE,
                                                       "test_evaluated": False, "completed_folds": fold_records,
                                                       "error": str(error)})
        info("Calibration stopped at wall-clock limit; incomplete result.")
    except Exception as error:
        run_record.update(status="failed", completed_folds=folds_done, stop_reason="calibration_failure",
                          error=f"{type(error).__name__}: {error}")
        write_json(args.output / "comparison.json", {"status": "failed", "notice": NOTICE,
                                                       "test_evaluated": False, "completed_folds": fold_records,
                                                       "error": run_record["error"]})
        raise
    finally:
        run_record["elapsed_wall_seconds"] = time.monotonic() - started
        write_json(args.output / "fold_results.json", fold_records)
        write_json(args.output / "run.json", run_record)
        info("status=%s; completed_folds=%s; elapsed=%.1fs" %
             (run_record["status"], folds_done, run_record["elapsed_wall_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    args = parser.parse_args()
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    run(args)


if __name__ == "__main__":
    main()
