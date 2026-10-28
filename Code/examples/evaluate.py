"""Evaluate prediction accuracy; calibrate equivalence on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.losses import BEHAVIOR, functional_distance
from triwave_gnn.utils.utils import get_function_acc
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.logger import BatchProgress, info
from triwave_gnn.utils.functional_equivalence import equivalence_samples, select_threshold
from triwave_gnn.utils.equivalence_audit import pair_metrics, circuit_macro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 修改此实验目录后，可直接在 VS Code 中运行本文件。
RUN_DIR = PROJECT_ROOT / "examples" / "exp" / "triwave_gnn_20261026_144750_012230"
# None：自动读取 RUN_DIR/best.pt，没有时回退到 last.pt。
# 指定文件示例：CHECKPOINT = RUN_DIR / "last.pt"
CHECKPOINT = None
PROGRESS_INTERVAL = 100
OUTPUT_NAME = "metrics_extended.json"

LEGACY_METRICS = {
    "hf_prob_mae": "mae", "hf_tt_cosine_distance_mae": "mae",
    "hf_tt_rank_accuracy": "accuracy", "hs_rc_error_rate": "mae",
    **{key + "_mae": "mae" for key, _, _ in BEHAVIOR},
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(splits, checkpoint):
    val_keys = [key for key in ("val", "validation") if key in splits]
    if len(val_keys) != 1 or "test" not in splits:
        raise ValueError("Expected exactly one val/validation split and a test split")
    val_key = val_keys[0]
    seen = set()
    for key, names in splits.items():
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError(f"Duplicate or overlapping circuit IDs in {key}; evaluation would leak")
        seen.update(names)
    if checkpoint.get("split_ids") is not None:
        def canonical(value):
            return {"val" if k == "validation" else k: v for k, v in value.items()}
        if canonical(checkpoint["split_ids"]) != canonical(splits):
            raise ValueError("Evaluation split.json differs from checkpoint split_ids")
    if not splits[val_key] or not splits["test"]:
        raise ValueError("Validation and test splits must be nonempty")
    return val_key


def check_pair_alignment(dataset, index, item):
    """Verify dataset conversion against the original NPZ slices before GPU transfer."""
    y, g = dataset.label, dataset.graph
    f0, f1 = map(int, y["hf_tt_pair_ptr"][index:index + 2])
    n0, n1 = map(int, g["node_ptr"][index:index + 2])
    target = torch.as_tensor(y["hf_tt_distance"][f0:f1], dtype=torch.float32)
    mask = torch.as_tensor(y["hf_tt_pair_mask"][f0:f1], dtype=torch.bool)
    pairs = torch.as_tensor(y["hf_tt_pair_node_index"][:, f0:f1] - n0, dtype=torch.long)
    actual = item["hf_tt_target"]
    if actual.shape != target.shape:
        raise ValueError(f"Functional target shape mismatch: {dataset.ids[index]}")
    equal_target = (actual == target) | (torch.isnan(actual) & torch.isnan(target))
    if (not bool(equal_target.all())
            or not torch.equal(mask, item["hf_tt_mask"])
            or not torch.equal(pairs, item["hf_pairs"])):
        raise ValueError(f"Functional pair/target/mask alignment failure: {dataset.ids[index]}")
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= n1 - n0):
        raise ValueError(f"Functional pair crosses circuit boundary: {dataset.ids[index]}")
    if item.name != dataset.ids[index]:
        raise ValueError("Circuit ID alignment failure")


def collect_legacy(output, item):
    sums, counts = {}, {}
    pins = item["pin"]

    def collect(key, prediction, target, mask):
        valid = mask.bool() & torch.isfinite(target)
        counts[key] = int(valid.sum())
        if counts[key]:
            if not bool(torch.isfinite(prediction[valid]).all()):
                raise ValueError(f"Nonfinite predictions for {key}")
            sums[key] = float((prediction[valid] - target[valid]).abs().sum().detach().cpu())

    collect("hf_prob_mae", output["hf_prob_logits"].sigmoid(), pins.hf_prob,
            torch.ones_like(pins.hf_prob, dtype=torch.bool))
    distance = functional_distance(output["hf"], item["hf_pairs"])
    collect("hf_tt_cosine_distance_mae", distance, item["hf_tt_target"], item["hf_tt_mask"])
    rank, count = get_function_acc(output["hf"], item["hf_pairs"], item["hf_tt_target"], item["hf_tt_mask"])
    counts["hf_tt_rank_accuracy"] = count
    if count:
        sums["hf_tt_rank_accuracy"] = rank * count
    collect("hs_rc_error_rate", (output["hs_rc_logits"].sigmoid() >= .5).float(),
            item["hs_target"], item["hs_mask"])
    for j, (key, mask_key, kind) in enumerate(BEHAVIOR):
        pred = output["hb_raw"][:, j].sigmoid() if kind == "probability" else torch.expm1(output["hb_raw"][:, j]).clamp_min(0)
        collect(key + "_mae", pred, pins[key], pins[mask_key])
    scores, labels = equivalence_samples(distance, item["hf_tt_target"], item["hf_tt_mask"])
    return sums, counts, scores, labels


def legacy_results(sums, counts):
    result = {}
    for key, measure in LEGACY_METRICS.items():
        count = counts.get(key, 0)
        entry = {measure: sums.get(key, 0.) / count if count else None, "count": count}
        if not count:
            entry["undefined_reason"] = "no informative comparisons" if key == "hf_tt_rank_accuracy" else "no finite valid labels"
        result[key] = entry
    error = result["hs_rc_error_rate"]
    result["hs_rc_accuracy"] = {"accuracy": 1 - error["mae"] if error["mae"] is not None else None,
                                "count": error["count"]}
    if not error["count"]:
        result["hs_rc_accuracy"]["undefined_reason"] = "no finite valid structural labels"
    return result


def evaluate_split(model, dataset, names, indices, device, split_name, interval):
    sums, counts, rows, score_parts, label_parts, offsets = {}, {}, [], [], [], [0]
    info(f"Evaluate {split_name}: {len(names)} circuits")
    with torch.no_grad(), BatchProgress(split_name, None, 0, len(names), interval, show_loss=False) as progress:
        for number, name in enumerate(names, 1):
            index = indices[name]
            item = dataset[index]
            check_pair_alignment(dataset, index, item)
            item = move(item, device)
            output = model_outputs(model, item)
            local_sums, local_counts, scores, labels = collect_legacy(output, item)
            for key, count in local_counts.items():
                sums[key] = sums.get(key, 0.) + local_sums.get(key, 0.)
                counts[key] = counts.get(key, 0) + count
            rows.append({"circuit": name, "node_count": item["pin"].num_nodes,
                         **legacy_results(local_sums, local_counts)})
            score_parts.append(scores)
            label_parts.append(labels)
            offsets.append(offsets[-1] + scores.numel())
            panel = {"prob_mae": sums["hf_prob_mae"] / max(counts["hf_prob_mae"], 1),
                     "tt_distance_mae": sums["hf_tt_cosine_distance_mae"] / max(counts["hf_tt_cosine_distance_mae"], 1),
                     "rc_accuracy": 1 - sums["hs_rc_error_rate"] / max(counts["hs_rc_error_rate"], 1)}
            progress.update(number, None, {}, panel)
            del output, item
    scores, labels = torch.cat(score_parts), torch.cat(label_parts)
    if not scores.numel():
        raise ValueError(f"{split_name}: no valid functional pairs (mask=true and finite target)")
    return legacy_results(sums, counts), rows, scores, labels, offsets


def attach_equivalence(result, rows, scores, labels, offsets, threshold):
    pooled = pair_metrics(scores, labels, threshold)
    pooled["averaging"] = "pooled/micro: concatenate all valid pairs; no circuit weighting"
    result["functional_equivalence"] = pooled
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        row["functional_equivalence"] = pair_metrics(scores[start:stop], labels[start:stop], threshold)
    result["functional_equivalence_macro"] = circuit_macro([row["functional_equivalence"] for row in rows])
    result["functional_equivalence_macro"]["similarity_threshold"] = threshold
    result["per_circuit"] = rows
    return pooled


def print_summary(test):
    """Print three test summary lines; JSON retains full precision and counts."""
    def display(value, percentage=False, reason="undefined"):
        if value is None:
            return f"N/A ({reason})"
        return f"{value:.4%}" if percentage else f"{value:.8f}"

    fields = (
        ("prob_mae", "hf_prob_mae", "mae", False),
        ("rc_accuracy", "hs_rc_accuracy", "accuracy", True),
        ("tt_distance_mae", "hf_tt_cosine_distance_mae", "mae", False),
        ("tt_rank_accuracy", "hf_tt_rank_accuracy", "accuracy", True),
        *((key + "_mae", key + "_mae", "mae", False) for key, _, _ in BEHAVIOR),
    )
    metrics = []
    for label, key, measure, percentage in fields:
        entry = test[key]
        value = display(entry[measure], percentage, entry.get("undefined_reason", "no valid labels"))
        metrics.append(f"{label}={value}")
    info("TEST metrics: " + " | ".join(metrics))

    functional = test["functional_equivalence"]
    reasons = functional.get("undefined_reasons", {})
    equivalence = []
    for key in ("precision", "recall", "f1", "accuracy", "roc_auc"):
        value = display(functional[key], key != "roc_auc", reasons.get(key, functional.get("status", "undefined")))
        equivalence.append(f"{key}={value}")
    info("TEST equivalence: " + " | ".join(equivalence))

    threshold = functional["similarity_threshold"]
    threshold_text = repr(threshold) if threshold is not None else "N/A (validation requires both positive and negative pairs)"
    details = [f"threshold={threshold_text}", f"positive_count={functional['positive_count']}",
               f"negative_count={functional['negative_count']}"]
    for key in ("tp", "fp", "tn", "fn"):
        value = functional[key]
        details.append(f"{key.upper()}={value if value is not None else 'N/A (validation threshold unavailable)'}")
    info("TEST equivalence details: " + " | ".join(details))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="checkpoint path; defaults to run/best.pt, then run/last.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL)
    args = parser.parse_args()
    if args.progress_interval < 0:
        parser.error("--progress-interval must be nonnegative")
    checkpoint_path = args.checkpoint or args.run / "best.pt"
    if args.checkpoint is None and not checkpoint_path.is_file():
        checkpoint_path = args.run / "last.pt"
        warnings.warn(f"best.pt absent; evaluating fallback {checkpoint_path} (may be unfinished)")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")
    original_path = args.run / "metrics.json"
    original_hash = digest_file(original_path) if original_path.is_file() else None
    original = json.loads(original_path.read_text()) if original_hash else {}
    checkpoint_hash = digest_file(checkpoint_path)
    split_path = args.run / "split.json"
    split_hash = digest_file(split_path)
    splits = json.loads(split_path.read_text())
    info(f"Checkpoint: {checkpoint_path} | device={args.device}")
    info(f"Loading and validating dataset: {args.data}; this may take a few minutes")
    dataset = CircuitDataset(args.data, cache_mode="none")
    model, checkpoint = load_model(checkpoint_path, dataset, args.device)
    model.eval()
    if checkpoint.get("source_sha256") is not None and checkpoint["source_sha256"] != dataset.source_sha256:
        raise ValueError("evaluation NPZ files differ from those used in this run")
    val_key = validate_splits(splits, checkpoint)
    indices = {name: index for index, name in enumerate(dataset.ids)}
    results = {}
    val, val_rows, val_scores, val_labels, val_offsets = evaluate_split(
        model, dataset, splits[val_key], indices, args.device, val_key, args.progress_interval)
    threshold = select_threshold(val_scores, val_labels)
    attach_equivalence(val, val_rows, val_scores, val_labels, val_offsets, threshold)
    info(f"Validation similarity_threshold={threshold if threshold is not None else 'N/A (validation lacks one class)'}; fixed for test")
    results[val_key] = val
    del val_rows, val_scores, val_labels, val_offsets
    test, rows, scores, labels, offsets = evaluate_split(
        model, dataset, splits["test"], indices, args.device, "test", args.progress_interval)
    pooled = attach_equivalence(test, rows, scores, labels, offsets, threshold)
    results["test"] = test
    print_summary(test)
    if not pooled["positive_count"] or not pooled["negative_count"]:
        info("WARNING: test split lacks one class; undefined metrics are displayed as N/A")
    elif pooled["positive_count"] < 100:
        info("WARNING: fewer than 100 test positive pairs; precision/recall/F1 estimates may be unstable")
    results["evaluation_metadata"] = {
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_stage": checkpoint.get("stage"), "checkpoint_epoch": checkpoint.get("epoch"),
        "source_sha256": dataset.source_sha256, "split_sha256": split_hash,
        "threshold_split": val_key, "device": args.device,
        "evaluated_splits": [val_key, "test"], "model_mode": "eval + no_grad",
        "original_metrics_sha256": original_hash,
        "original_checkpoint_record": original.get("evaluation_metadata", {}),
        "threshold_objective": "maximum pooled validation F1, ties grouped, highest threshold for equal F1",
        "decision_rule": "similarity >= validation threshold",
        "positive_definition": "torch.isclose(hf_tt_target, 0, atol=1e-8, rtol=0)",
        "json_units": "classification ratios in [0,1]; terminal percentages explicitly marked with %",
    }
    if original:
        results["previous_metrics"] = original
    if digest_file(checkpoint_path) != checkpoint_hash or digest_file(split_path) != split_hash:
        raise RuntimeError("Checkpoint or split file changed during evaluation; results not saved")
    if original_hash is not None and digest_file(original_path) != original_hash:
        raise RuntimeError("Original metrics.json changed during evaluation; results not saved")
    output_path = args.run / OUTPUT_NAME
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(output_path)
    info(f"Saved full results and all {len(rows)} circuit details: {output_path}")


if __name__ == "__main__":
    main()
