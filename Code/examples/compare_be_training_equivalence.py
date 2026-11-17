"""Evaluate B/E best-equivalence checkpoints on all original training pairs.

No training or validation/test inference. Validation results and thresholds are
read from the exact saved checkpoint epoch. Score archives enable later reuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.compare_functional_stages import cosine_scores64, file_hash, within_circuit_auc
from examples.evaluate import validate_splits
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.functional_equivalence import grouped_curve
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

EXP = Path(__file__).resolve().parent / "exp"
DEFAULT_RUNS = {
    "B": EXP / "functional_pair_sampling_stage3" / "circuit_cap",
    "E": EXP / "functional_pair_sampling_stage3_E_functional_only_20261109",
}
AUC_KEYS = ("circuit_macro_roc_auc", "macro_participating_circuits",
            "within_circuit_weighted_auc", "within_circuit_positive_negative_comparison_count")
SUMMARY_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "roc_auc",
                *AUC_KEYS[:3], "positive_count", "negative_count", "similarity_threshold")


def training_threshold(scores, labels):
    """Same grouped-tie rule as validation; never label this a validation score."""
    positives = int(labels.sum())
    if not positives or positives == labels.numel():
        return None
    thresholds, tp, fp = grouped_curve(scores, labels)
    f1 = 2 * tp.double() / (positives + tp + fp)
    return float(thresholds[int(f1.argmax())])


def summarize_archive(archive, threshold):
    scores, labels, offsets = (archive[key] for key in ("scores", "labels", "offsets"))
    rows = []
    for name, start, stop in zip(archive["circuit_names"], offsets, offsets[1:]):
        rows.append({"circuit": name, "original_valid_pairs": stop - start,
                     **pair_metrics(scores[start:stop], labels[start:stop], threshold)})
    result = pair_metrics(scores, labels, threshold)
    result.update(within_circuit_auc(rows))
    result.update(circuit_count=len(rows),
                  circuits_with_pairs=sum(row["original_valid_pairs"] > 0 for row in rows),
                  per_circuit=rows)
    groups = {}
    for name, low, high in (("1-999", 1, 999), ("1000-9999", 1000, 9999),
                            (">=10000", 10000, sys.maxsize)):
        slices = [(start, stop) for start, stop in zip(offsets, offsets[1:])
                  if low <= stop - start <= high]
        group_scores = torch.cat([scores[a:b] for a, b in slices]) if slices else scores[:0]
        group_labels = torch.cat([labels[a:b] for a, b in slices]) if slices else labels[:0]
        groups[name] = {"circuit_count": len(slices), "sample_count": group_scores.numel(),
                        **pair_metrics(group_scores, group_labels, threshold)}
    result["groups_by_original_valid_pairs"] = groups
    result["zero_pair_circuits_excluded_from_groups"] = sum(
        row["original_valid_pairs"] == 0 for row in rows)
    return result


def read_record(group, run):
    path = run / "best_equiv.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint["epoch"]
    val_path = run / (f"val_epoch_{epoch}.json" if epoch else "baseline.json")
    validation = json.loads(val_path.read_text())
    threshold = checkpoint["validation_threshold"]
    if (validation["epoch"] != epoch or
            validation["functional_equivalence"]["similarity_threshold"] != threshold or
            validation["functional_equivalence"]["f1"] != checkpoint["validation_f1"]):
        raise ValueError(f"{group}: best_equiv epoch/threshold/F1 does not match validation record")
    history = json.loads((run / "history.json").read_text())
    available_f1 = [row["val"]["functional_equivalence"]["f1"]
                    for row in history]
    baseline = json.loads((run / "baseline.json").read_text())
    available_f1.append(baseline["functional_equivalence"]["f1"])
    if checkpoint["validation_f1"] != max(value for value in available_f1 if value is not None):
        raise ValueError(f"{group}: saved best_equiv is not the maximum recorded validation F1")
    splits = json.loads((run / "split.json").read_text())
    validate_splits(splits, checkpoint)
    val_result = {**validation["functional_equivalence"],
                  **{key: validation[key] for key in AUC_KEYS}}
    info(f"{group}: best_equiv.pt epoch={epoch}, saved validation threshold={threshold:.17g}")
    return checkpoint, splits, {"checkpoint": str(path), "checkpoint_sha256": file_hash(path),
                               "epoch": epoch, "validation_threshold": threshold,
                               "validation_record": str(val_path), "val": val_result,
                               "validation_record_sha256": file_hash(val_path)}


def collect_train(model, dataset, names, args, metadata):
    indices = {name: index for index, name in enumerate(dataset.ids)}
    loader = DataLoader(Subset(dataset, [indices[name] for name in names]),
                        batch_size=metadata["batch_size"], shuffle=False, num_workers=0,
                        exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(metadata["seed"]))
    parts, label_parts, offsets, circuit_names = [], [], [0], []
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for number, batch in enumerate(loader, 1):
            item = move(batch, args.device)
            _, hf, _ = model(item, include_behavior=False)
            valid = item["hf_tt_mask"].bool() & torch.isfinite(item["hf_tt_target"])
            scores = cosine_scores64(hf, item["hf_pairs"], valid, args.chunk_size)
            target = item["hf_tt_target"][valid].cpu()
            labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
            groups = item["pin"].batch[item["hf_pairs"][0, valid]].cpu()
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("Nonfinite cosine scores")
            for circuit, name in enumerate(item.name):
                selected = groups == circuit
                local_scores = scores[selected]
                parts.append(local_scores)
                label_parts.append(labels[selected])
                offsets.append(offsets[-1] + local_scores.numel())
                circuit_names.append(name)
            del batch, item, hf
            if number == 1 or number % args.progress_interval == 0 or number == len(loader):
                info(f"{metadata['group']} train: batch={number}/{len(loader)} | "
                     f"circuits={len(circuit_names)}/{len(names)} | pairs={offsets[-1]} | "
                     f"elapsed={time.monotonic() - started:.1f}s")
    if circuit_names != names:
        raise ValueError("Training circuit order changed")
    return {"metadata": metadata, "scores": torch.cat(parts), "labels": torch.cat(label_parts),
            "offsets": offsets, "circuit_names": circuit_names,
            "pair_order": "circuit_names order; within each circuit original valid hf_pairs column order",
            "inference_seconds": time.monotonic() - started}


def summary_line(group, split, result, epoch):
    return f"{group} / {split} epoch={epoch} | " + " | ".join(
        f"{key}={result[key]:.12g}" if isinstance(result[key], float)
        else f"{key}={result[key]}" for key in SUMMARY_KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/train")
    parser.add_argument("--b-run", type=Path, default=DEFAULT_RUNS["B"])
    parser.add_argument("--e-run", type=Path, default=DEFAULT_RUNS["E"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    if min(args.chunk_size, args.progress_interval) < 1:
        parser.error("chunk-size and progress-interval must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    with direct_log(args.output / "console.log"):
        write_json(args.output / "run.json", {"status": "running", "training": False,
                                              "evaluated_splits": ["train"], "test_evaluated": False})
        records, states, splits_by_group = {}, {}, {}
        for group, run in (("B", args.b_run), ("E", args.e_run)):
            states[group], splits_by_group[group], records[group] = read_record(group, run)
        if splits_by_group["B"] != splits_by_group["E"]:
            raise ValueError("B/E data splits differ")
        info("Existing B/E results have no training pair score archives; one full train inference per checkpoint")
        info("Loading original dataset once; no circuit_cap, no balancing, no validation/test inference")
        dataset = CircuitDataset(args.data, cache_mode="none")
        report = {"metadata": {
            "scope": "完整训练集评价；主结果固定使用各检查点对应轮次的验证阈值",
            "training_threshold_scope": "训练集阈值诊断，不是验证或测试成绩，不替代主结果",
            "score_method": "original float32 eval model; endpoint embeddings converted to float64 BEFORE chunked cosine(dim=-1, eps=1e-8)",
            "valid_pairs": "all original hf_tt_mask & isfinite(hf_tt_target), no sampling",
            "positive_definition": "isclose(TT, 0, atol=1e-8, rtol=0)",
            "threshold_rule": "score >= threshold; equal scores grouped; highest threshold on max-F1 ties",
            "within_circuit_weighted_auc": "both-class circuits only; sum(AUC*n_positive*n_negative)/sum(n_positive*n_negative), excludes cross-circuit comparisons",
            "undefined_values": "null with original pair_metrics status and undefined_reasons",
            "source_sha256": dataset.source_sha256, "data": str(args.data),
            "chunk_size": args.chunk_size, "device": args.device,
            "validation_inference": False, "test_evaluated": False, "optimizer_steps": 0},
                  "checkpoints": records}
        write_json(args.output / "split.json", splits_by_group["B"])
        for group in ("B", "E"):
            state, record = states.pop(group), records[group]
            if state["source_sha256"] != dataset.source_sha256:
                raise ValueError(f"{group}: data differs from checkpoint")
            seed = state["config"]["seed"]
            set_seed(seed)
            model, loaded = load_model(Path(record["checkpoint"]), dataset, args.device)
            metadata = {"group": group, "checkpoint_sha256": record["checkpoint_sha256"],
                        "epoch": record["epoch"], "seed": seed,
                        "batch_size": state["config"]["batch_size"],
                        "source_sha256": dataset.source_sha256,
                        "validation_threshold": record["validation_threshold"]}
            info(f"{group}: eval mode, batch_size={metadata['batch_size']}, seed={seed}")
            archive = collect_train(model, dataset, splits_by_group[group]["train"], args, metadata)
            archive_path = args.output / f"{group}_train_scores.pt"
            atomic_save(archive, archive_path)
            info(f"{group}: original training scores/labels saved: {archive_path}")
            record["train_score_archive"] = str(archive_path)
            record["train"] = summarize_archive(archive, record["validation_threshold"])
            selected_threshold = training_threshold(archive["scores"], archive["labels"])
            diagnostic = pair_metrics(archive["scores"], archive["labels"], selected_threshold)
            diagnostic["scope"] = "训练集阈值诊断：阈值在同一完整训练集上选择，不是验证或测试成绩"
            record["training_threshold_diagnostic"] = diagnostic
            write_json(args.output / f"{group}.json", record)
            info(summary_line(group, "train (fixed val threshold)", record["train"], record["epoch"]))
            info(summary_line(group, "val (saved exact epoch)", record["val"], record["epoch"]))
            info(f"{group} 训练集阈值诊断: " + " | ".join(
                f"{key}={diagnostic[key]}" for key in ("precision", "recall", "f1", "similarity_threshold")))
            for name, metrics in record["train"]["groups_by_original_valid_pairs"].items():
                info(f"{group} train group={name}: " + " | ".join(
                    f"{key}={metrics[key]}" for key in ("circuit_count", "sample_count", "positive_ratio",
                                                       "roc_auc", "precision", "recall", "f1")))
            write_json(args.output / "comparison.json", report)
            del model, loaded, state, archive
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        lines = [summary_line(group, split, records[group][split], records[group]["epoch"])
                 for group in ("B", "E") for split in ("train", "val")]
        for group, record in records.items():
            if file_hash(Path(record["checkpoint"])) != record["checkpoint_sha256"]:
                raise RuntimeError(f"{group} checkpoint changed during evaluation")
            if file_hash(Path(record["validation_record"])) != record["validation_record_sha256"]:
                raise RuntimeError(f"{group} validation record changed during evaluation")
        # Report evidence without turning a train/validation comparison into causality.
        evidence = []
        for group, record in records.items():
            train, val, diag = record["train"], record["val"], record["training_threshold_diagnostic"]
            evidence.append({"group": group, "train_minus_val_pooled_auc": train["roc_auc"] - val["roc_auc"],
                             "train_minus_val_macro_auc": train["circuit_macro_roc_auc"] - val["circuit_macro_roc_auc"],
                             "train_minus_val_within_circuit_weighted_auc": train["within_circuit_weighted_auc"] - val["within_circuit_weighted_auc"],
                             "train_f1_gain_from_reselecting_threshold": diag["f1"] - train["f1"]})
        report["diagnostic_evidence"] = evidence
        report["metadata"]["status"] = "completed"
        report["metadata"]["original_checkpoints_and_validation_records_unchanged"] = True
        write_json(args.output / "comparison.json", report)
        write_json(args.output / "run.json", {"status": "completed", "evaluated_splits": ["train"],
                                              "optimizer_steps": 0, "test_evaluated": False})
        (args.output / "summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            info(line)
        info(f"Diagnostic evidence: {evidence}")
        info(f"Completed B/E full training equivalence: {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
