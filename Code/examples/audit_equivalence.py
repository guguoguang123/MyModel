"""Read-only audit of functional equivalence on the original full-data run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.evaluate import RUN_DIR, check_pair_alignment, digest_file, validate_splits
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import collate_circuits
from triwave_gnn.utils.equivalence_audit import circuit_macro, pair_metrics, similarity_distribution
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model


def pair_scores(hf, pairs, valid, chunk_size):
    """Cast embeddings before float64 cosine; keep the original float32 score path."""
    indices = torch.nonzero(valid, as_tuple=True)[0]
    raw, high = [], []
    for part in indices.split(chunk_size):
        left, right = pairs[:, part]
        a, b = hf[left], hf[right]
        cosine32 = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
        original = 1 - (1 - cosine32)
        cosine64 = F.cosine_similarity(a.double(), b.double(), dim=-1, eps=1e-8)
        raw.append(original.cpu())
        high.append(cosine64.cpu())
    return torch.cat(raw) if raw else torch.empty(0), torch.cat(high) if high else torch.empty(0, dtype=torch.float64)


def all_pair_roles(dataset):
    graph, labels = dataset.graph, dataset.label
    pairs = labels["hf_tt_pair_node_index"]
    mask = labels["hf_tt_pair_mask"]
    target = labels["hf_tt_distance"]
    valid = mask & np.isfinite(target)
    ptr = graph["node_ptr"]
    pair_ptr = labels["hf_tt_pair_ptr"]
    node_roles = graph["node_role"]
    roles = Counter()
    bad_boundary = []
    for i in range(len(dataset)):
        check_pair_alignment(dataset, i, dataset[i])
        f0, f1 = map(int, pair_ptr[i:i + 2])
        selected = pairs[:, f0:f1][:, valid[f0:f1]]
        if selected.size == 0:
            continue
        n0, n1 = map(int, ptr[i:i + 2])
        if selected.min() < n0 or selected.max() >= n1:
            bad_boundary.append(dataset.ids[i])
            continue
        for role_id, count in zip(*np.unique(node_roles[selected], return_counts=True)):
            roles[dataset.role_vocab[int(role_id)]] += int(count)
    return {"all_pairs": int(pairs.shape[1]), "valid_pairs": int(valid.sum()),
            "invalid_or_masked_pairs": int((~valid).sum()),
            "valid_endpoints_by_role": dict(roles),
            "all_valid_endpoints_cell_output": roles == Counter({"cell_output": 2 * int(valid.sum())}),
            "circuits_with_boundary_errors": bad_boundary}


def choose_batch_cases(dataset, names, indices):
    candidates = sorted(((int(dataset.graph["num_nodes"][indices[name]]), name) for name in names
                         if 100 <= int(dataset.graph["num_nodes"][indices[name]]) <= 12000))
    if len(candidates) < 3:
        candidates = sorted((int(dataset.graph["num_nodes"][indices[name]]), name) for name in names)
    return [candidates[k][1] for k in (0, len(candidates) // 2, -1)]


def batch_check(model, dataset, names, indices, device, chunk_size):
    originals = [dataset[indices[name]] for name in names]
    singles = []
    with torch.inference_mode():
        for item in originals:
            single = move(item, device)
            singles.append(model(single, include_behavior=False)[1].cpu())
        batch = move(collate_circuits(originals), device)
        combined = model(batch, include_behavior=False)[1].cpu()
    rows, offset = [], 0
    for item, hf_single in zip(originals, singles):
        size = item["pin"].num_nodes
        hf_batch = combined[offset:offset + size]
        valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
        s32, s64 = pair_scores(hf_single, item["hf_pairs"], valid, chunk_size)
        b32, b64 = pair_scores(hf_batch, item["hf_pairs"], valid, chunk_size)
        rows.append({"circuit": item.name, "node_count": size, "valid_pairs": int(valid.sum()),
                     "max_abs_hf_difference": float((hf_single - hf_batch).abs().max()),
                     "max_abs_score32_difference": float((s32 - b32).abs().max()) if s32.numel() else None,
                     "max_abs_score64_difference": float((s64 - b64).abs().max()) if s64.numel() else None})
        offset += size
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    checkpoint_path = args.run / "best.pt"
    split_path = args.run / "split.json"
    checkpoint_hash, split_hash = digest_file(checkpoint_path), digest_file(split_path)
    splits = json.loads(split_path.read_text())
    print(f"Loading full dataset {args.data} and original checkpoint {checkpoint_path}", flush=True)
    dataset = CircuitDataset(args.data, cache_mode="none")
    model, checkpoint = load_model(checkpoint_path, dataset, args.device)
    if checkpoint.get("source_sha256") != dataset.source_sha256:
        raise ValueError("Checkpoint and full dataset hashes differ")
    val_key = validate_splits(splits, checkpoint)
    indices = {name: i for i, name in enumerate(dataset.ids)}
    role_audit = all_pair_roles(dataset)
    print("Pair roles:", role_audit, flush=True)
    if role_audit["circuits_with_boundary_errors"]:
        raise ValueError("Valid functional pair crosses circuit boundary")
    selected = choose_batch_cases(dataset, splits[val_key], indices)
    batch_rows = batch_check(model, dataset, selected, indices, args.device, args.chunk_size)
    print("Single versus batch:", batch_rows, flush=True)

    score32, score64, labels, offsets = [], [], [], [0]
    with torch.inference_mode():
        for number, name in enumerate(splits[val_key], 1):
            index = indices[name]
            item = dataset[index]
            check_pair_alignment(dataset, index, item)
            valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
            target = item["hf_tt_target"][valid]
            if bool((target < 0).any()):
                raise ValueError(f"Negative TT distance in {name}")
            item_device = move(item, args.device)
            hf = model(item_device, include_behavior=False)[1]
            s32, s64 = pair_scores(hf, item_device["hf_pairs"],
                                   valid.to(args.device), args.chunk_size)
            score32.append(s32)
            score64.append(s64)
            labels.append(torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0))
            offsets.append(offsets[-1] + s32.numel())
            if number % 100 == 0 or number == len(splits[val_key]):
                print(f"Validation {number}/{len(splits[val_key])} circuits", flush=True)
            del item_device, hf
    score32, score64, labels = map(torch.cat, (score32, score64, labels))
    if not score32.numel() or not bool(torch.isfinite(score32).all() and torch.isfinite(score64).all()):
        raise ValueError("Empty or nonfinite validation scores")
    output = {"metadata": {"checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                           "split_sha256": split_hash, "source_sha256": dataset.source_sha256,
                           "validation_circuits": len(splits[val_key]), "device": args.device,
                           "positive_definition": "isclose(TT distance, 0, atol=1e-8, rtol=0)",
                           "decision_rule": "cosine similarity >= validation-only threshold"},
              "pair_alignment": {"validated_dataset_slices": len(splits[val_key]), **role_audit},
              "batch_consistency": batch_rows, "precision_comparison": {}, "validation": {}}
    delta = (score32.double() - score64).abs()
    output["precision_comparison"] = {
        "same_hf": True, "score_count": score32.numel(), "max_abs_score_difference": float(delta.max()),
        "mean_abs_score_difference": float(delta.mean()),
        "float32_equal_one": int((score32 == 1).sum()), "float64_equal_one": int((score64 == 1).sum()),
        "float32_unique_scores": int(torch.unique(score32).numel()),
        "float64_unique_scores": int(torch.unique(score64).numel())}
    for name, scores in (("original_float32", score32), ("hf_cast_float64_before_cosine", score64)):
        threshold = select_threshold(scores, labels)
        pooled = pair_metrics(scores, labels, threshold)
        rows = [pair_metrics(scores[start:stop], labels[start:stop], threshold)
                for start, stop in zip(offsets, offsets[1:])]
        macro = circuit_macro(rows)
        distribution = similarity_distribution(scores, labels)
        output["validation"][name] = {"pooled": pooled,
                                       "macro_auc": macro["roc_auc"],
                                       "macro_auc_contributing_circuits": macro["contributing_circuits"]["roc_auc"],
                                       "both_class_circuits": macro["both_class_circuit_count"],
                                       "score_quantiles": {key: entry["quantiles"] for key, entry in distribution.items()
                                                           if key in ("positive", "negative")}}
        print(f"{name}: pooled P={pooled['precision']} R={pooled['recall']} F1={pooled['f1']} "
              f"AUC={pooled['roc_auc']} threshold={threshold}; macro AUC={macro['roc_auc']} "
              f"({macro['contributing_circuits']['roc_auc']} circuits)", flush=True)
        print(f"{name}: positive/negative score quantiles: {output['validation'][name]['score_quantiles']}", flush=True)
    if digest_file(checkpoint_path) != checkpoint_hash or digest_file(split_path) != split_hash:
        raise RuntimeError("Checkpoint or split changed during audit")
    destination = args.output or args.run / "equivalence_precision_audit.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved diagnostic report: {destination}", flush=True)
    print("Conclusion: pair endpoints all cell_output =",
          role_audit["all_valid_endpoints_cell_output"],
          "; label zero is positive; higher cosine means more likely equivalent.", flush=True)


if __name__ == "__main__":
    main()
