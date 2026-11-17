"""Compare stage 2/3 on the original train/validation splits, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.evaluate import RUN_DIR, validate_splits
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine_scores64(hf, pairs, valid, chunk_size):
    """Only pair endpoint chunks become double; model inference stays float32."""
    selected = torch.nonzero(valid, as_tuple=True)[0]
    scores = []
    for chunk in selected.split(chunk_size):
        left, right = pairs[:, chunk]
        scores.append(F.cosine_similarity(hf[left].double(), hf[right].double(),
                                          dim=-1, eps=1e-8).cpu())
    return torch.cat(scores) if scores else torch.empty(0, dtype=torch.float64)


def within_circuit_auc(rows):
    """Average AUC using only within-circuit positive/negative comparisons."""
    eligible = [row for row in rows if row["positive_count"] and row["negative_count"]]
    weight = sum(row["positive_count"] * row["negative_count"] for row in eligible)
    return {
        "circuit_macro_roc_auc": math.fsum(row["roc_auc"] for row in eligible) / len(eligible)
        if eligible else None,
        "macro_participating_circuits": len(eligible),
        "within_circuit_weighted_auc": math.fsum(
            row["roc_auc"] * (row["positive_count"] * row["negative_count"])
            for row in eligible) / weight if weight else None,
        "within_circuit_positive_negative_comparison_count": weight,
    }


def collect_split(model, dataset, names, indices, device, chunk_size, tag, interval):
    scores, labels, offsets, circuit_rows = [], [], [0], []
    probability_error, probability_count = 0., 0
    distance_error, distance_count = 0., 0
    started = time.monotonic()
    with torch.inference_mode():
        for number, name in enumerate(names, 1):
            item = move(dataset[indices[name]], device)
            # The behavior branch has no path back into Hs/Hf in this architecture.
            _, hf, _ = model(item, include_behavior=False)
            target = item["hf_tt_target"]
            valid = item["hf_tt_mask"].bool() & torch.isfinite(target)
            local_scores = cosine_scores64(hf, item["hf_pairs"], valid, chunk_size)
            local_target = target[valid].cpu()
            if not bool(torch.isfinite(local_scores).all()):
                raise ValueError(f"Nonfinite similarities: {tag}, {name}")
            scores.append(local_scores)
            labels.append(torch.isclose(local_target, torch.zeros_like(local_target), atol=1e-8, rtol=0))
            offsets.append(offsets[-1] + local_scores.numel())
            local_distance_error = float(((1 - local_scores) - local_target.double()).abs().sum())
            probability = model.pred_prob_logits(hf).sigmoid()
            probability_target = item["pin"].hf_prob
            local_probability_error = float((probability.double() - probability_target.double()).abs().sum())
            local_probability_count = probability_target.numel()
            if not math.isfinite(local_probability_error):
                raise ValueError(f"Nonfinite probability prediction: {tag}, {name}")
            probability_error += local_probability_error
            probability_count += local_probability_count
            distance_error += local_distance_error
            distance_count += local_scores.numel()
            circuit_rows.append({"circuit": name, "node_count": item["pin"].num_nodes,
                                 "prob_mae": local_probability_error / local_probability_count,
                                 "tt_distance_mae": local_distance_error / local_scores.numel()
                                 if local_scores.numel() else None})
            del item, hf, probability, probability_target, target, valid
            if number % interval == 0 or number == len(names):
                print(f"{tag}: {number}/{len(names)} circuits, {offsets[-1]} pairs, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
    return {"scores": torch.cat(scores), "labels": torch.cat(labels), "offsets": offsets,
            "per_circuit": circuit_rows, "prob_mae": probability_error / probability_count,
            "prob_count": probability_count,
            "tt_distance_mae": distance_error / distance_count if distance_count else None,
            "tt_distance_count": distance_count}


def summarize(collected, threshold):
    scores, labels = collected.pop("scores"), collected.pop("labels")
    offsets = collected.pop("offsets")
    if not scores.numel():
        raise ValueError("No valid functional pairs in split")
    result = pair_metrics(scores, labels, threshold)
    for row, start, stop in zip(collected["per_circuit"], offsets, offsets[1:]):
        row.update(pair_metrics(scores[start:stop], labels[start:stop], threshold))
    result.update(within_circuit_auc(collected["per_circuit"]))
    result.update(collected)
    result["circuit_count"] = len(result["per_circuit"])
    return result


def format_summary(stage, split, result):
    prefix = f"{stage} / {split}"
    if result is None:
        return prefix + " | MISSING checkpoint (no substitute used)"
    keys = ("precision", "recall", "f1", "roc_auc", "circuit_macro_roc_auc",
            "macro_participating_circuits", "within_circuit_weighted_auc",
            "positive_count", "negative_count", "prob_mae", "tt_distance_mae", "similarity_threshold")
    return prefix + " | " + " | ".join(
        f"{key}={result[key]:.12g}" if isinstance(result[key], float)
        else f"{key}={result[key]}" for key in keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.chunk_size, args.progress_interval) < 1:
        parser.error("chunk-size and progress-interval must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = args.output_dir or args.run / ("stage_functional_comparison_" + timestamp)
    destination.mkdir(parents=True, exist_ok=False)
    protected = {path: file_hash(path) for path in args.run.iterdir()
                 if path.is_file() and path.suffix in (".pt", ".json")}
    splits = json.loads((args.run / "split.json").read_text())
    checkpoints = {stage: args.run / (stage + "_best.pt") for stage in ("stage2", "stage3")}
    available = {stage: path for stage, path in checkpoints.items() if path.is_file()}
    for stage, path in checkpoints.items():
        print(f"{stage}: {path}" + ("" if stage in available else " MISSING; no substitute"), flush=True)
    report = {"metadata": {
        "device": args.device, "inference": "eval + inference_mode, float32 model, behavior branch skipped",
        "score_method": "float64 endpoint embeddings before chunked cosine_similarity(dim=-1, eps=1e-8)",
        "chunk_size": args.chunk_size, "evaluated_splits": ["train", "val"],
        "positive_definition": "isclose(hf_tt_target, 0, atol=1e-8, rtol=0)",
        "valid_pair_definition": "hf_tt_mask & isfinite(hf_tt_target)",
        "threshold_selection": "one pooled validation maximum-F1 threshold per checkpoint; equal scores grouped; highest threshold on equal F1",
        "decision_rule": "score >= checkpoint validation threshold; same threshold for train and val",
        "mae_averaging": "prob_mae pooled over all nodes; tt_distance_mae pooled over valid pairs, predicted distance = 1 - float64 cosine",
        "within_circuit_weighted_auc_definition": "sum(AUC_c * n_positive_c * n_negative_c) / sum(n_positive_c * n_negative_c), both-class circuits only; excludes cross-circuit comparisons",
        "original_files_sha256": {str(path): digest for path, digest in protected.items()},
    }, "checkpoints": {}}
    if available:
        print(f"Loading full dataset once: {args.data}", flush=True)
        dataset = CircuitDataset(args.data, cache_mode="none")
        report["metadata"]["source_sha256"] = dataset.source_sha256
        indices = {name: i for i, name in enumerate(dataset.ids)}
        for stage, path in available.items():
            model, checkpoint = load_model(path, dataset, args.device)
            val_key = validate_splits(splits, checkpoint)
            if checkpoint.get("stage") != int(stage[-1]):
                raise ValueError(f"{path}: checkpoint stage does not match its filename")
            print(f"{stage}: epoch={checkpoint.get('epoch')}; validation inference", flush=True)
            validation = collect_split(model, dataset, splits[val_key], indices, args.device,
                                       args.chunk_size, stage + " / val", args.progress_interval)
            threshold = select_threshold(validation["scores"], validation["labels"])
            val_result = summarize(validation, threshold)
            print(f"{stage}: fixed validation threshold={threshold}; training-set inference", flush=True)
            training = collect_split(model, dataset, splits["train"], indices, args.device,
                                     args.chunk_size, stage + " / train", args.progress_interval)
            train_result = summarize(training, threshold)
            report["checkpoints"][stage] = {
                "status": "ok", "path": str(path.resolve()), "epoch": checkpoint.get("epoch"),
                "threshold": threshold, "train": train_result, "val": val_result,
            }
            (destination / (stage + ".json")).write_text(json.dumps(
                report["checkpoints"][stage], indent=2, ensure_ascii=False, allow_nan=False))
            del model, checkpoint, validation, training, train_result, val_result
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    for stage in checkpoints.keys() - available.keys():
        report["checkpoints"][stage] = {"status": "missing", "path": str(checkpoints[stage])}
    for path, digest in protected.items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Original checkpoint/result changed: {path}")
    report["metadata"]["original_files_unchanged"] = True
    (destination / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    summary = []
    for stage in ("stage2", "stage3"):
        for split in ("train", "val"):
            summary.append(format_summary(stage, split, report["checkpoints"][stage].get(split)))
    (destination / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"Results saved: {destination / 'comparison.json'}", flush=True)
    for line in summary:
        print(line, flush=True)


if __name__ == "__main__":
    main()
