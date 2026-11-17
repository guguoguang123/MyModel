"""Manual, cache-only circuit-balanced sampling control for the frozen-hf head."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash, within_circuit_auc
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.functional_equivalence import grouped_curve, select_threshold
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

DEFAULT_SOURCE = original.EXP / "frozen_equivalence_head_dev_20261113"
CAP, FP_BUDGET = 128, 69
OBJECTIVE = ("equal circuit mean; within a both-class circuit: "
             "0.5*mean(positive BCE)+0.5*mean(negative BCE); "
             "single-class circuit: mean(available-class BCE)")


def circuit_loss(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    class_means = [losses[labels == value].mean() for value in (True, False)
                   if bool((labels == value).any())]
    if not class_means:
        raise ValueError("Cannot compute circuit loss without valid samples")
    return torch.stack(class_means).mean()


def sample_circuit(cache, generator, cap=CAP):
    selected = []
    for value in (True, False):
        available = torch.nonzero(cache["labels"] == value).flatten()
        selected.append(available[torch.randperm(available.numel(), generator=generator)[:cap]])
    columns = torch.cat(selected)
    left, right = cache["pairs"][:, columns]
    return {"features": original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right]).detach(),
            "labels": cache["labels"][columns], "cache_pair_columns": columns,
            "original_pair_columns": cache["original_pair_columns"][columns],
            "positive_count": selected[0].numel(), "negative_count": selected[1].numel(),
            "circuit": cache["metadata"]["circuit"],
            "pair_count_group": original.group_name(cache["labels"].numel())}


def circuit_schedule(count, batch_size, steps, seed):
    """Uniform shuffle per sweep; no duplicate circuit in an update.

    A fresh uniform permutation gives every circuit equal inclusion probability
    at each position. Finish a sweep before starting another so all circuits
    receive training, rather than leaving some unseen in a short run.
    """
    generator = torch.Generator().manual_seed(seed)
    batches = []
    while len(batches) < steps:
        batches.extend(torch.randperm(count, generator=generator).split(batch_size))
    return batches[:steps]


def budget_threshold(scores, labels, budget=FP_BUDGET):
    thresholds, tp, fp = grouped_curve(scores, labels)
    allowed = torch.nonzero(fp <= budget).flatten()
    if not allowed.numel() or int(tp[allowed].max()) == 0:
        return math.nextafter(float(scores.max()), math.inf)
    allowed = allowed[tp[allowed] == tp[allowed].max()]
    allowed = allowed[fp[allowed] == fp[allowed].min()]
    return float(thresholds[allowed[0]])  # highest threshold on remaining ties


def auc_summary(scores, labels, offsets, circuits):
    rows = []
    for name, start, stop in zip(circuits, offsets, offsets[1:]):
        rows.append({"circuit": name, **pair_metrics(scores[start:stop], labels[start:stop], None)})
    return within_circuit_auc(rows), rows


def score_summary(scores, labels, offsets, circuits, score_kind):
    threshold = select_threshold(scores, labels)
    budget = budget_threshold(scores, labels)
    aucs, rows = auc_summary(scores, labels, offsets, circuits)
    result = {"score_kind": score_kind, "maximum_f1": pair_metrics(scores, labels, threshold),
              **aucs, "fp_le_69": pair_metrics(scores, labels, budget),
              "per_circuit_auc": [{"circuit": row["circuit"], "roc_auc": row["roc_auc"],
                                   "positive_count": row["positive_count"], "negative_count": row["negative_count"]}
                                  for row in rows], "groups_by_original_valid_pairs": {}}
    for group in original.GROUPS:
        indices = [i for i, (a, b) in enumerate(zip(offsets, offsets[1:]))
                   if b > a and original.group_name(b - a) == group]
        local_s = torch.cat([scores[offsets[i]:offsets[i+1]] for i in indices]) if indices else scores[:0]
        local_y = torch.cat([labels[offsets[i]:offsets[i+1]] for i in indices]) if indices else labels[:0]
        group_aucs = within_circuit_auc([rows[i] for i in indices])
        result["groups_by_original_valid_pairs"][group] = {
            "circuit_count": len(indices), "sample_count": local_s.numel(),
            "at_global_max_f1_threshold": pair_metrics(local_s, local_y, threshold),
            **group_aucs, "at_global_fp_budget_threshold": pair_metrics(local_s, local_y, budget)}
    return result


def print_summary(method, result):
    for tag, entry in [("pooled", result), *result["groups_by_original_valid_pairs"].items()]:
        best = entry["maximum_f1"] if tag == "pooled" else entry["at_global_max_f1_threshold"]
        low = entry["fp_le_69"] if tag == "pooled" else entry["at_global_fp_budget_threshold"]
        info(f"{method} / {tag}: max-F1 P/R/F1={best['precision']}/{best['recall']}/{best['f1']} | "
             f"pooled/macro/within_auc={best['roc_auc']}/{entry['circuit_macro_roc_auc']}/{entry['within_circuit_weighted_auc']} | "
             f"AUC circuits={entry['macro_participating_circuits']} | low-FP TP/FP={low['tp']}/{low['fp']}")
    info(f"{method}: max-F1 threshold={result['maximum_f1']['threshold']} | "
         f"FP<=69 threshold={result['fp_le_69']['threshold']}")


def check_cache(cache, row, hidden):
    required = ("hf_l2", "pairs", "labels", "original_pair_columns", "metadata")
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"缓存缺失 {row['circuit']}: {missing}；停止，不补跑推理")
    hf, pairs, labels = (cache[key] for key in ("hf_l2", "pairs", "labels"))
    if (cache["metadata"]["circuit"] != row["circuit"] or cache["metadata"]["split"] != row["split"]
            or hf.ndim != 2 or hf.shape[1] != hidden or pairs.shape != (2, labels.numel())
            or pairs.dtype != torch.long or labels.ndim != 1 or labels.dtype != torch.bool
            or cache["original_pair_columns"].shape != labels.shape
            or labels.numel() != row["original_valid_pairs"]):
        raise ValueError(f"缓存索引/标签/电路映射不一致：{row['circuit']}；停止")
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= hf.shape[0]):
        raise ValueError(f"缓存配对越界：{row['circuit']}；停止")


def original_steps(run, sampling, history):
    selected = sampling["sampling"]["selected_positive"] + sampling["sampling"]["selected_negative"]
    actual = sum(len(row["cache_pair_columns"]) for row in sampling["selections"])
    epochs = run["completed_epochs"]
    if actual != selected or [row["epoch"] for row in history] != list(range(1, epochs + 1)):
        raise ValueError("Original sample count or completed epoch records disagree")
    return epochs * math.ceil(selected / run["batch_size"])


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    source = args.source.resolve()
    required = ("run.json", "selection.json", "cache_manifest.json", "training_pairs.pt",
                "history.json", "last_head.pt", "validation_scores.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"缺失原试验记录：{missing}；停止，不补跑推理")
    previous = json.loads((source / "run.json").read_text())
    manifest = json.loads((source / "cache_manifest.json").read_text())
    selection = json.loads((source / "selection.json").read_text())
    if previous["status"] != "completed" or previous["functional_net_identity"]:
        raise ValueError("Require completed original-propagation head experiment")
    if previous["source_sha256"] != file_hash(Path(original.__file__)):
        raise ValueError("Original entry changed; cannot reconstruct identical initialization recipe")
    if previous["seed"] != 42 or previous["lr"] != 1e-3 or previous["optimizer"] != "Adam":
        raise ValueError("Unexpected original seed/optimizer/lr; stopping without alternative")
    if args.device != previous["device"]:
        raise ValueError("Use the original experiment device to keep this sampling control fixed")
    counts = Counter(row["split"] for row in manifest)
    if counts != {"train": 64, "val": 32} or len({row["circuit"] for row in manifest}) != 96:
        raise ValueError("Require exactly the original 64 train and 32 val cache entries")
    if {(row["split"], row["circuit"]) for row in manifest} != {
            (row["split"], row["circuit"]) for row in selection["circuits"]}:
        raise ValueError("Cache manifest and original subset selection differ")
    for row in manifest:
        path = Path(row["cache_file"])
        if not path.is_file():
            raise FileNotFoundError(f"缺失缓存：{path}；停止，不补跑推理")
    old_head = torch.load(source / "last_head.pt", map_location="cpu", weights_only=False)
    hidden = old_head["hidden"]
    prior_sampling = torch.load(source / "training_pairs.pt", map_location="cpu", weights_only=False)
    steps = original_steps(previous, prior_sampling, json.loads((source / "history.json").read_text()))
    circuits_per_step = max(1, previous["batch_size"] // (2 * CAP))  # four circuits, <=1024 pairs
    saved = torch.load(source / "validation_scores.pt", map_location="cpu", weights_only=False)
    required_scores = ("cosine_scores64", "head_logits", "labels", "offsets", "circuits")
    missing = [key for key in required_scores if key not in saved]
    if missing:
        raise ValueError(f"缺失已存验证分数/电路 ID：{missing}；停止，不补跑推理")
    val_rows = [row for row in manifest if row["split"] == "val"]
    offsets = list(map(int, saved["offsets"]))
    if (saved["circuits"] != [row["circuit"] for row in val_rows]
            or len(offsets) != 33 or offsets[0] != 0
            or offsets[-1] != saved["labels"].numel()
            or any(b - a != row["original_valid_pairs"] for row, a, b in zip(val_rows, offsets, offsets[1:]))
            or any(saved[key].shape != saved["labels"].shape for key in ("cosine_scores64", "head_logits"))):
        raise ValueError("Saved validation scores and original circuit mapping disagree")

    # Cache contents are checked before any optimizer is constructed or stepped.
    generator = torch.Generator().manual_seed(previous["seed"])
    train, val = [], []
    group_counts = {group: {"both_class_circuits": 0, "single_class_circuits": 0,
                            "empty_circuits": 0, "positive_samples": 0, "negative_samples": 0}
                    for group in original.GROUPS}
    for row in manifest:
        cache = torch.load(row["cache_file"], map_location="cpu", weights_only=False)
        check_cache(cache, row, hidden)
        if row["split"] == "val":
            i = len(val)
            if not torch.equal(cache["labels"], saved["labels"][offsets[i]:offsets[i+1]]):
                raise ValueError(f"Validation label order mismatch: {row['circuit']}")
            # Retain only the existing hf and indices, not unused raw hf.
            val.append({key: cache[key] for key in ("hf_l2", "pairs")})
            continue
        sampled = sample_circuit(cache, generator)
        entry = group_counts[sampled["pair_count_group"]]
        pos, neg = sampled["positive_count"], sampled["negative_count"]
        entry["positive_samples"] += pos
        entry["negative_samples"] += neg
        entry["both_class_circuits" if pos and neg else "single_class_circuits" if pos or neg else "empty_circuits"] += 1
        if pos or neg:
            train.append(sampled)
    if not train:
        raise ValueError("No cached training circuit has valid samples")
    schedule = circuit_schedule(len(train), circuits_per_step, steps, previous["seed"])
    args.output.mkdir(parents=True, exist_ok=False)
    with direct_log(args.output / "console.log"):
        record = {"status": "training_head", "notice": original.NOTICE, "source_run": str(source),
                  "source_entry_sha256": previous["source_sha256"],
                  "entry_sha256": file_hash(Path(__file__)), "seed": previous["seed"],
                  "encoder_actual_epoch": previous["checkpoint_actual_epoch"],
                  "encoder_checkpoint_sha256": previous["checkpoint_sha256"],
                  "encoder_forward_count": 0, "encoder_loaded": False,
                  "functional_net_identity": False, "test_evaluated": False,
                  "train_circuits": counts["train"], "active_train_circuits": len(train), "val_circuits": counts["val"],
                  "original_optimizer_steps": steps, "planned_optimizer_steps": steps,
                  "lr": previous["lr"], "optimizer": "Adam", "cap_per_circuit_per_class": CAP,
                  "circuits_per_step": circuits_per_step, "objective": OBJECTIVE,
                  "sampling": "fixed no-replacement class samples per circuit; no hard negatives or rare-sample repetition",
                  "circuit_draw_rule": "uniform permutation per sweep; batches of four distinct circuits; truncate to original optimizer step count",
                  "group_counts": group_counts,
                  "group_balance_notice": "按电路和类别计算目标，不保证各组总体标签比例为1:1",
                  "initialization": "reconstructed from unchanged original make_head and seed=42; original initial weights were not saved",
                  "device": args.device, "feature": previous["feature"],
                  "threshold_rules": "score >= threshold; complete equal-score groups; max-F1 ties: higher threshold; FP<=69: max TP, then min FP, then higher threshold",
                  "group_threshold_rule": "global validation maximum-F1 and global FP<=69 operating points; no per-group threshold tuning"}
        write_json(args.output / "run.json", record)
        info(original.NOTICE)
        info(OBJECTIVE)
        info(record["group_balance_notice"])
        for group, entry in group_counts.items():
            info(f"Training group={group}: {entry}")
        info(f"Original/new optimizer steps={steps}/{steps}; circuits per step={circuits_per_step}")
        # Original source resets RNG immediately before constructing this same head.
        set_seed(previous["seed"])
        head = original.make_head(hidden).to(args.device)
        initial_digest = original.state_digest(head)
        atomic_save({"head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                     "seed": previous["seed"], "hidden": hidden, "state_sha256": initial_digest,
                     "reconstruction": record["initialization"]}, args.output / "initial_head.pt")
        optimizer = torch.optim.Adam(head.parameters(), lr=previous["lr"])
        atomic_save({"objective": OBJECTIVE, "group_counts": group_counts,
                     "circuits": [{key: value for key, value in row.items() if key not in ("features", "labels")} for row in train],
                     "circuit_batches": [[train[i]["circuit"] for i in batch.tolist()] for batch in schedule]},
                    args.output / "training_samples.pt")
        history = []
        head.train()
        for step, batch in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            local_losses = []
            for index in batch.tolist():
                row = train[index]
                logits = head(row["features"].to(args.device)).squeeze(-1)
                local_losses.append(circuit_loss(logits, row["labels"].to(args.device)))
            # Do not flatten pair losses: each circuit contributes one mean.
            loss = torch.stack(local_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite circuit-balanced loss")
            loss.backward()
            optimizer.step()
            history.append({"step": step, "circuit_mean_bce": float(loss.detach()),
                            "circuits": [train[i]["circuit"] for i in batch.tolist()]})
            if step == 1 or step % 10 == 0 or step == steps:
                info(f"Head step={step}/{steps} circuit-mean BCE={float(loss.detach()):.6f}")
        write_json(args.output / "history.json", history)
        atomic_save({"head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                     "hidden": hidden, "optimizer_steps": len(history), "objective": OBJECTIVE,
                     "initial_state_sha256": initial_digest, "encoder_loaded": False,
                     "source_run": str(source), "notice": original.NOTICE}, args.output / "last_head.pt")
        head.eval()
        parts = []
        with torch.no_grad():
            for cache in val:
                chunks = []
                for pairs in cache["pairs"].split(previous["batch_size"], dim=1):
                    left, right = pairs
                    features = original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right])
                    chunks.append(head(features.to(args.device)).squeeze(-1).cpu())
                parts.append(torch.cat(chunks) if chunks else torch.empty(0))
        balanced_logits = torch.cat(parts)
        methods = {"cosine_baseline": saved["cosine_scores64"], "original_head": saved["head_logits"],
                   "circuit_balanced_head": balanced_logits}
        report = {"notice": original.NOTICE, "scope": "same saved 32 validation circuits; all original valid pairs",
                  "objective": OBJECTIVE, "group_counts": group_counts,
                  "optimizer_steps": steps, "initial_state_sha256": initial_digest,
                  "group_threshold_rule": record["group_threshold_rule"], "methods": {}}
        for method, scores in methods.items():
            result = score_summary(scores, saved["labels"], offsets, saved["circuits"],
                                   "float64 cosine" if method == "cosine_baseline" else "logit")
            report["methods"][method] = result
            print_summary(method, result)
        write_json(args.output / "comparison.json", report)
        atomic_save({**saved, "circuit_balanced_head_logits": balanced_logits}, args.output / "validation_scores.pt")
        record.update(status="completed", completed_optimizer_steps=len(history), initial_state_sha256=initial_digest)
        write_json(args.output / "run.json", record)
        info(original.NOTICE)
        info(f"Completed circuit-balanced head control: {args.output / 'comparison.json'}")


def self_check():
    # Synthetic samples only; exercise the unequal-class and single-class objectives.
    logits = torch.tensor([0.1, -0.7, 1.0, 0.8], requires_grad=True)
    labels = torch.tensor([True, False, False, False])
    expected = .5 * F.softplus(-logits[0]) + .5 * F.softplus(logits[1:]).mean()
    loss = circuit_loss(logits, labels)
    assert torch.allclose(loss, expected)
    repeated = torch.cat((logits[:1], logits[1:].repeat(3)))
    repeated_labels = torch.tensor([True] + [False] * 9)
    assert torch.allclose(loss, circuit_loss(repeated, repeated_labels))
    assert torch.allclose(circuit_loss(logits, torch.ones(4, dtype=torch.bool)), F.softplus(-logits).mean())
    # Unequal circuit sizes must have equal loss and gradient weights.
    other = torch.tensor([-0.4], requires_grad=True)
    objective = torch.stack((loss, circuit_loss(other, torch.tensor([True])))).mean()
    grad = torch.autograd.grad(objective, logits, retain_graph=True)[0]
    local_grad = torch.autograd.grad(loss, logits)[0]
    assert torch.allclose(grad, local_grad / 2)
    toy = {"hf_l2": F.normalize(torch.arange(24, dtype=torch.float32).reshape(6, 4) + 1, dim=-1),
           "pairs": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]), "labels": labels,
           "original_pair_columns": torch.arange(4), "metadata": {"circuit": "synthetic"}}
    sample = sample_circuit(toy, torch.Generator().manual_seed(42), cap=2)
    assert (sample["positive_count"], sample["negative_count"]) == (1, 2)
    assert sample["cache_pair_columns"].unique().numel() == 3
    schedule = circuit_schedule(64, 4, 40, 42)
    assert len(schedule) == 40 and all(row.unique().numel() == 4 for row in schedule)
    exposures = torch.bincount(torch.cat(schedule), minlength=64)
    assert int(exposures.min()) == 2 and int(exposures.max()) == 3
    set_seed(42)
    reference = original.make_head(4)
    set_seed(42)
    head = original.make_head(4)
    assert original.state_digest(reference) == original.state_digest(head)
    before = original.state_digest(head)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    circuit_loss(head(sample["features"]).squeeze(-1), sample["labels"]).backward()
    optimizer.step()
    assert original.state_digest(head) != before
    scores = torch.tensor([.9, .8, .8, .7])
    y = torch.tensor([True, True, False, False])
    assert budget_threshold(scores, y, budget=0) == float(scores[0])
    assert budget_threshold(scores, y, budget=1) == float(scores[1])
    scores = torch.tensor([.9, .8, .8])
    y = torch.tensor([True, False, False])
    assert budget_threshold(scores, y, budget=1) == float(scores[0])
    print("PASS: class/circuit mean objective and gradients; no-replacement cap; 40-step uniform-sweep schedule; identical seeded initialization; head-only update; grouped-score FP budgets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--self-check", action="store_true")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.output is None:
        parser.error("--run requires a new --output directory")
    else:
        run(args)


if __name__ == "__main__":
    main()
