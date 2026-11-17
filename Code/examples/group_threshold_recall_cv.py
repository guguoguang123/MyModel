"""C2: per-pair-count-group thresholds under calibration-fold TP constraints."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import group_score_calibration_cv as base
from examples.finetune_functional_pairs import write_json
from triwave_gnn.utils.functional_equivalence import grouped_curve
from triwave_gnn.utils.logger import info

DEFAULT_OUTPUT = base.original.EXP / "group_threshold_recall_cv_20261114"
CV_SOURCE = base.original.EXP / "group_score_calibration_cv_20261113"
WALL_SECONDS = 600


def threshold_candidates(scores, labels, c0_threshold):
    scores = scores.detach().cpu().double()
    labels = labels.detach().cpu().bool()
    if not scores.numel():
        return [(float(c0_threshold), 0, 0)]
    thresholds, tp, fp = grouped_curve(scores, labels)
    candidates = [(float(t), int(p), int(n)) for t, p, n in zip(thresholds, tp, fp)]
    # Include the exact C0 decision point and the predict-none point, including when
    # their thresholds fall outside this group's observed score range.
    for t in (float(c0_threshold), math.nextafter(float(scores.max()), math.inf)):
        pred = scores >= t
        candidates.append((t, int((pred & labels).sum()), int((pred & ~labels).sum())))
    by_counts = {}
    for threshold, tp_count, fp_count in candidates:
        key = (tp_count, fp_count)
        by_counts[key] = max(threshold, by_counts.get(key, -math.inf))
    return [(threshold, tp_count, fp_count)
            for (tp_count, fp_count), threshold in by_counts.items()]


def choose_group_thresholds(logits, labels, group_codes, c0_threshold, min_tp):
    options = []
    for group_index in range(len(base.GROUPS)):
        mask = group_codes == group_index
        candidates = threshold_candidates(logits[mask], labels[mask], c0_threshold)
        feasible = [candidate for candidate in candidates if candidate[1] >= min_tp[group_index]]
        if not feasible:
            raise ValueError(f"C2 infeasible in calibration group {base.GROUPS[group_index]}")
        options.append(feasible)
    positive_total = int(labels.sum())
    selected = []
    # Dinkelbach iteration for max 2TP/(P+TP+FP); each iteration optimizes
    # the three groups independently, avoiding their Cartesian product.
    c0_selected = []
    for group_index in range(len(base.GROUPS)):
        mask = group_codes == group_index
        pred = logits[mask] >= c0_threshold
        local_labels = labels[mask]
        c0_selected.append((float(c0_threshold), int((pred & local_labels).sum()),
                            int((pred & ~local_labels).sum())))
    c0_tp = sum(row[1] for row in c0_selected)
    c0_fp = sum(row[2] for row in c0_selected)
    ratio = 2.0 * c0_tp / (positive_total + c0_tp + c0_fp) if positive_total + c0_tp + c0_fp else 0.0
    if any(c0_selected[i][1] < min_tp[i] for i in range(len(base.GROUPS))):
        raise ValueError("Saved C0 global threshold is not feasible for its own group-TP constraints")
    for iteration in range(100):
        selected = []
        for candidates in options:
            # candidates preserve exact score-group decisions; ties prefer the
            # highest threshold and then fewer false positives.
            best = max(candidates, key=lambda r: (2 * r[1] - ratio * (r[1] + r[2]),
                                                  r[0], -r[2]))
            selected.append(best)
        tp_total = sum(row[1] for row in selected)
        fp_total = sum(row[2] for row in selected)
        updated = 2.0 * tp_total / (positive_total + tp_total + fp_total)
        residual = 2 * tp_total - ratio * (positive_total + tp_total + fp_total)
        if residual <= 1e-9 or abs(updated - ratio) <= 1e-12:
            ratio = updated
            break
        ratio = updated
    else:
        raise RuntimeError("C2 fractional threshold optimization did not converge in 100 iterations")
    selected_tp = sum(row[1] for row in selected)
    selected_fp = sum(row[2] for row in selected)
    c0_f1 = 2 * c0_tp / (positive_total + c0_tp + c0_fp)
    selected_f1 = 2 * selected_tp / (positive_total + selected_tp + selected_fp)
    if selected_f1 + 1e-12 < c0_f1:
        raise AssertionError(f"C2 calibration F1 {selected_f1} below feasible C0 {c0_f1}")
    return {
        "thresholds": [row[0] for row in selected],
        "calibration_tp": [row[1] for row in selected],
        "calibration_fp": [row[2] for row in selected],
        "minimum_tp": list(map(int, min_tp)),
        "constraints_satisfied": all(row[1] >= min_tp[i] for i, row in enumerate(selected)),
        "calibration_f1": selected_f1,
        "c0_calibration_f1": c0_f1,
        "c0_is_feasible_candidate": True,
        "iterations": iteration + 1,
    }


def summarize(labels, pred):
    return base.metric_from_pred(labels, pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    deadline = started + WALL_SECONDS
    result_file = args.output / "comparison.json"
    try:
        data = base.load_saved_scores()
        saved_folds = json.loads((CV_SOURCE / "fold_results.json").read_text())
        saved_assign = json.loads((CV_SOURCE / "fold_assignment.json").read_text())
        if len(saved_folds) != base.FOLDS or len(saved_assign) != len(data["circuits"]):
            raise ValueError("Expected the existing complete five-fold assignment and C0/C1 records")
        folds, fold_of, fold_pair = base.make_folds(data["circuits"], data["counts"])
        if any(int(saved_assign[name]) != int(fold_of[i]) for i, name in enumerate(data["circuits"])):
            raise ValueError("Reconstructed fold assignment differs from saved C0/C1 folds")
        circuits = data["circuits"]
        logits, labels = data["logits"].double(), data["labels"].bool()
        groups = data["pair_group_codes"]
        pair_fold = fold_pair
        c2_pred = torch.zeros_like(labels)
        c0_pred = torch.zeros_like(labels)
        c1_pred = torch.zeros_like(labels)
        fold_rows = []
        for fold, original_record in enumerate(saved_folds):
            if time.monotonic() >= deadline:
                raise TimeoutError("10-minute wall-clock limit reached")
            if original_record["fold"] != fold:
                raise ValueError("Saved fold result order changed")
            held_circuits = set(original_record["circuits"])
            held_mask = torch.tensor([name in held_circuits for name in circuits], dtype=torch.bool)
            held_pair_mask = pair_fold == fold
            train_pair_mask = ~held_pair_mask
            if int(held_mask.sum()) != original_record["circuit_count"]:
                raise ValueError(f"Fold circuit assignment mismatch at fold {fold}")
            c0_threshold = float(original_record["c0_threshold"])
            train_groups, held_groups = groups[train_pair_mask], groups[held_pair_mask]
            train_logits, train_labels = logits[train_pair_mask], labels[train_pair_mask]
            held_logits, held_labels = logits[held_pair_mask], labels[held_pair_mask]
            min_tp = []
            for group_index in range(len(base.GROUPS)):
                m = (train_groups == group_index)
                pred = train_logits[m] >= c0_threshold
                min_tp.append(int((pred & train_labels[m]).sum()))
            selected = choose_group_thresholds(train_logits, train_labels, train_groups,
                                               c0_threshold, min_tp)
            if time.monotonic() >= deadline:
                raise TimeoutError("10-minute wall-clock limit reached")
            thresholds = torch.tensor(selected["thresholds"], dtype=torch.float64)
            local_c2 = held_logits >= thresholds[held_groups]
            local_c0 = held_logits >= c0_threshold
            # C1 scores and predictions are from the already completed C0/C1 run.
            c1_threshold = float(original_record["c1_threshold"])
            c1_params = original_record["c1_parameters"]
            held_c1 = base.apply_c1(held_logits, held_groups, c1_params)
            local_c1 = held_c1 >= c1_threshold
            c2_pred[held_pair_mask] = local_c2
            c0_pred[held_pair_mask] = local_c0
            c1_pred[held_pair_mask] = local_c1
            held_c2_metrics = summarize(held_labels, local_c2)
            held_c0_metrics = summarize(held_labels, local_c0)
            held_c1_metrics = summarize(held_labels, local_c1)
            fold_group_metrics = {}
            fold_group_deltas = {}
            for group_index, group_name in enumerate(base.GROUPS):
                m = held_groups == group_index
                fold_group_metrics[group_name] = {
                    "C0": summarize(held_labels[m], local_c0[m]),
                    "C1": summarize(held_labels[m], local_c1[m]),
                    "C2": summarize(held_labels[m], local_c2[m]),
                }
                fold_group_deltas[group_name] = {
                    key: fold_group_metrics[group_name]["C2"][key] - fold_group_metrics[group_name]["C0"][key]
                    for key in ("tp", "fp", "fn")
                }
            cal_check = []
            for group_index, group_name in enumerate(base.GROUPS):
                m = train_groups == group_index
                p = train_logits[m] >= thresholds[group_index]
                tp = int((p & train_labels[m]).sum())
                cal_check.append({"group": group_name, "required_tp": min_tp[group_index],
                                  "actual_tp": tp, "satisfied": tp >= min_tp[group_index]})
            fold_rows.append({
                "fold": fold, "circuit_count": int(held_mask.sum()),
                "calibration_folds": [i for i in range(base.FOLDS) if i != fold],
                "zero_valid_circuits": original_record["zero_valid_circuits"],
                "calibration_circuit_count": len(circuits) - int(held_mask.sum()),
                "c0_global_threshold": c0_threshold,
                "c2_thresholds_by_group": {g: selected["thresholds"][i] for i, g in enumerate(base.GROUPS)},
                "calibration_group_constraints": cal_check,
                "calibration_constraints_satisfied": selected["constraints_satisfied"],
                "constraint_source": "C0 global threshold from this saved fold record, applied only to the other four folds",
                "calibration_c0_f1": selected["c0_calibration_f1"],
                "calibration_c2_f1": selected["calibration_f1"],
                "calibration_f1_not_below_c0": selected["calibration_f1"] + 1e-12 >= selected["c0_calibration_f1"],
                "optimizer_iterations": selected["iterations"],
                "C0_heldout": held_c0_metrics, "C1_heldout": held_c1_metrics,
                "C2_heldout": held_c2_metrics,
                "C2_minus_C0_heldout": {"tp": held_c2_metrics["tp"] - held_c0_metrics["tp"],
                                          "fp": held_c2_metrics["fp"] - held_c0_metrics["fp"],
                                          "fn": held_c2_metrics["fn"] - held_c0_metrics["fn"]},
                "groups_heldout": fold_group_metrics,
                "groups_C2_minus_C0_heldout": fold_group_deltas,
            })
            write_json(args.output / "fold_results.json", fold_rows)
            info("fold=%d C2 calibration F1=%.6f vs C0=%.6f feasible=%s thresholds=%s" %
                 (fold, selected["calibration_f1"], selected["c0_calibration_f1"],
                  selected["constraints_satisfied"], selected["thresholds"]))
            info("fold=%d calibration_folds=%s C0 group TP requirements/actual=%s" %
                 (fold, [i for i in range(base.FOLDS) if i != fold],
                  [(r["group"], r["required_tp"], r["actual_tp"], r["satisfied"]) for r in cal_check]))
            for method, metrics in (("C0", held_c0_metrics), ("C1", held_c1_metrics),
                                    ("C2", held_c2_metrics)):
                info("fold=%d heldout %s P/R/F1=%s/%s/%s TP/FP/TN/FN=%d/%d/%d/%d" %
                     (fold, method, metrics["precision"], metrics["recall"], metrics["f1"],
                      metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]))
            for group_name in base.GROUPS:
                row = fold_group_metrics[group_name]
                info("fold=%d group=%s C0=%s/%s/%s %d/%d/%d/%d C1=%s/%s/%s %d/%d/%d/%d C2=%s/%s/%s %d/%d/%d/%d" %
                     (fold, group_name,
                      row["C0"]["precision"], row["C0"]["recall"], row["C0"]["f1"],
                      row["C0"]["tp"], row["C0"]["fp"], row["C0"]["tn"], row["C0"]["fn"],
                      row["C1"]["precision"], row["C1"]["recall"], row["C1"]["f1"],
                      row["C1"]["tp"], row["C1"]["fp"], row["C1"]["tn"], row["C1"]["fn"],
                      row["C2"]["precision"], row["C2"]["recall"], row["C2"]["f1"],
                      row["C2"]["tp"], row["C2"]["fp"], row["C2"]["tn"], row["C2"]["fn"]))

        overall, group_results = {}, {}
        for name, pred in (("C0", c0_pred), ("C1", c1_pred), ("C2", c2_pred)):
            overall[name] = summarize(labels, pred)
            group_results[name] = {
                group: summarize(labels[groups == i], pred[groups == i])
                for i, group in enumerate(base.GROUPS)
            }
        macro_average = {}
        group_macro_average = {}
        for name in ("C0", "C1", "C2"):
            macro_average[name] = {
                key: sum(row[f"{name}_heldout"][key] for row in fold_rows) / len(fold_rows)
                for key in ("precision", "recall", "f1")
            }
            group_macro_average[name] = {}
            for group in base.GROUPS:
                group_macro_average[name][group] = {}
                for key in ("precision", "recall", "f1"):
                    values = [row["groups_heldout"][group][name][key] for row in fold_rows
                              if row["groups_heldout"][group][name][key] is not None]
                    group_macro_average[name][group][key] = sum(values) / len(values) if values else None
        recall_declines = []
        for row in fold_rows:
            for group in base.GROUPS:
                c0_recall = row["groups_heldout"][group]["C0"]["recall"]
                c2_recall = row["groups_heldout"][group]["C2"]["recall"]
                if c0_recall is not None and c2_recall is not None and c2_recall < c0_recall:
                    recall_declines.append({"fold": row["fold"], "group": group,
                                            "C0_recall": c0_recall, "C2_recall": c2_recall,
                                            "delta": c2_recall - c0_recall})
        preserves_group_recall = all(group_results["C2"][g]["recall"] is not None
                                     and group_results["C2"][g]["recall"] >= group_results["C0"][g]["recall"]
                                     for g in base.GROUPS)
        report = {
            "status": "completed", "notice": base.NOTICE,
            "protocol": "C0/C1/C2 use the same saved seed42 five-fold circuit split; for each fold the C0 threshold and C2 group TP constraints/thresholds are evaluated on the other four folds only; held-out labels are used only for reporting",
            "test_evaluated": False, "encoder_forwards": 0, "head_forwards": 0,
            "source_g1_corrected_reference": {"recall": data["g1_reference_correction"]["recall"],
                                               "tp": data["g1_reference_correction"]["tp"],
                                               "fp": data["g1_reference_correction"]["fp"]},
            "circuit_count": len(circuits), "zero_valid_circuits": int((data["counts"] == 0).sum()),
            "folds": fold_rows, "overall": overall, "groups": group_results,
            "macro_average_over_folds": macro_average,
            "group_macro_average_over_folds": group_macro_average,
            "C2_minus_C0": {"tp": overall["C2"]["tp"] - overall["C0"]["tp"],
                            "fp": overall["C2"]["fp"] - overall["C0"]["fp"],
                            "fn": overall["C2"]["fn"] - overall["C0"]["fn"]},
            "heldout_group_recall_delta_C2_minus_C0": {
                g: group_results["C2"][g]["recall"] - group_results["C0"][g]["recall"]
                for g in base.GROUPS},
            "C2_preserves_all_three_aggregate_group_recalls": preserves_group_recall,
            "C2_improves_aggregate_overall_f1_vs_C0": overall["C2"]["f1"] > overall["C0"]["f1"],
            "heldout_group_recall_declines_by_fold": recall_declines,
        }
        write_json(result_file, report)
        write_json(args.output / "run.json", {"status": "completed", "folds_completed": 5,
                                               "wall_seconds": WALL_SECONDS,
                                               "elapsed_wall_seconds": time.monotonic() - started,
                                               "test_evaluated": False, "encoder_forwards": 0,
                                               "source_fold_results": str(CV_SOURCE / "fold_results.json")})
        for name in ("C0", "C1", "C2"):
            m = overall[name]
            info("OVERALL %s P/R/F1=%s/%s/%s TP/FP/TN/FN=%d/%d/%d/%d" %
                 (name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["tn"], m["fn"]))
            info("FOLD MACRO %s P/R/F1=%s/%s/%s" %
                 (name, macro_average[name]["precision"], macro_average[name]["recall"], macro_average[name]["f1"]))
            for group in base.GROUPS:
                x = group_results[name][group]
                info("%s group=%s P/R/F1=%s/%s/%s TP/FP/TN/FN=%d/%d/%d/%d" %
                     (name, group, x["precision"], x["recall"], x["f1"], x["tp"], x["fp"], x["tn"], x["fn"]))
                gm = group_macro_average[name][group]
                info("FOLD MACRO %s group=%s P/R/F1=%s/%s/%s" %
                     (name, group, gm["precision"], gm["recall"], gm["f1"]))
        info("C2 minus C0 aggregate TP/FP/FN delta=%d/%d/%d" %
             (report["C2_minus_C0"]["tp"], report["C2_minus_C0"]["fp"], report["C2_minus_C0"]["fn"]))
        for group, delta in report["heldout_group_recall_delta_C2_minus_C0"].items():
            info("C2 minus C0 held-out group=%s Recall delta=%+.8f" % (group, delta))
        info("C2 preserves all aggregate group recalls=%s; overall F1 improves vs C0=%s" %
             (report["C2_preserves_all_three_aggregate_group_recalls"],
              report["C2_improves_aggregate_overall_f1_vs_C0"]))
        info("C2 calibration constraints apply to four-fold calibration data only; they do not guarantee held-out group recall.")
        info("Completed C2 saved: %s" % result_file)
    except Exception as error:
        write_json(args.output / "run.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                               "test_evaluated": False,
                                               "elapsed_wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
