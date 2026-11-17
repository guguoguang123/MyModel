"""Manual cache-only hard-negative control, starting from ordinary step 500."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import compare_circuit_balanced_head as control
from examples import continue_circuit_balanced_head as continuation
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

START_RUN = original.EXP / "frozen_equivalence_head_continue500_20261113"
ORDINARY_RUN = original.EXP / "frozen_equivalence_head_continue5000_20261113"
START_STEP, END_STEP, REFRESH_INTERVAL, CHUNK_SIZE = 500, 5000, 500, 4096
WALL_SECONDS, VALIDATION_RESERVE = 300, 20
MINING_SEED = 42 + 2_000_003


class TrainingTimeLimit(Exception):
    pass


def training_time_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TrainingTimeLimit("time_limit; reserved final validation time within 300 seconds")


def features_at(cache, columns):
    left, right = cache["pairs"][:, columns]
    return original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right]).detach()


@torch.no_grad()
def choose_negatives(head, cache, candidates, budget, generator, device,
                     chunk_size=CHUNK_SIZE, training_deadline=None):
    """Stream top scores; random half is uniform over remaining known negatives.

    Candidates are in ascending cache/original column order. Stable descending
    score sorting resolves hard-score ties by the earlier original column.
    Only a chunk of pair features and O(budget) best scores are on the device.
    """
    head.eval()
    training_time_check(training_deadline)
    count = min(budget, candidates.numel())
    if candidates.numel() <= count:
        return candidates.clone(), {"policy": "all_candidates", "hard_count": 0,
                                    "uniform_count": 0, "all_candidate_count": count}
    hard_count = (count + 1) // 2
    uniform_count = count - hard_count
    best_scores = torch.empty(0, device=device)
    best_positions = torch.empty(0, dtype=torch.long, device=device)
    for start in range(0, candidates.numel(), chunk_size):
        training_time_check(training_deadline)
        chunk = candidates[start:start + chunk_size]
        logits = head(features_at(cache, chunk).to(device)).squeeze(-1)
        continuation.finite(logits, "训练候选负样本 logits")
        scores = torch.cat((best_scores, logits))
        positions = torch.cat((best_positions, torch.arange(start, start + chunk.numel(), device=device)))
        keep = torch.argsort(scores, descending=True, stable=True)[:hard_count]
        best_scores, best_positions = scores[keep], positions[keep]
    hard_positions = best_positions.cpu()
    remaining_mask = torch.ones(candidates.numel(), dtype=torch.bool)
    remaining_mask[hard_positions] = False
    remaining = torch.nonzero(remaining_mask).flatten()
    random_positions = remaining[torch.randperm(remaining.numel(), generator=generator)[:uniform_count]]
    selected = torch.cat((candidates[hard_positions], candidates[random_positions]))
    return selected, {"policy": "top_half_plus_uniform_remaining", "hard_count": hard_count,
                      "uniform_count": uniform_count, "all_candidate_count": 0,
                      "hard_cache_columns": candidates[hard_positions], "hard_logits": best_scores.cpu(),
                      "uniform_cache_columns": candidates[random_positions]}


def load_training_pools(samples, manifest, hidden):
    lookup = {row["circuit"]: row for row in manifest if row["split"] == "train"}
    if len(lookup) != 64 or set(lookup) != {row["circuit"] for row in samples["circuits"]}:
        raise ValueError("Require exactly the original 64 training circuits")
    pools, counts = [], {group: {"circuits": 0, "cached_negatives": 0,
                                "original_negative_budget": 0, "larger_pool_circuits": 0}
                         for group in original.GROUPS}
    for row in samples["circuits"]:
        meta = lookup[row["circuit"]]
        path = Path(meta["cache_file"])
        if not path.is_file():
            raise FileNotFoundError(f"缺失缓存 {path}；停止，不补跑推理")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("hf_l2", "pairs", "labels", "original_pair_columns"):
            if key not in loaded:
                raise ValueError(f"缺失缓存字段 {row['circuit']}/{key}；停止")
        cache = {key: loaded[key] for key in ("hf_l2", "pairs", "labels", "original_pair_columns")}
        if (cache["hf_l2"].shape[1] != hidden or cache["labels"].dtype != torch.bool
                or cache["pairs"].shape != (2, cache["labels"].numel())
                or cache["labels"].numel() != meta["original_valid_pairs"]):
            raise ValueError(f"Cached pair/label dimensions disagree: {row['circuit']}")
        columns = row["cache_pair_columns"]
        if not torch.equal(cache["original_pair_columns"][columns], row["original_pair_columns"]):
            raise ValueError(f"Original fixed pair mapping changed: {row['circuit']}")
        fixed_labels = cache["labels"][columns]
        positives = columns[fixed_labels]
        candidates = torch.nonzero(~cache["labels"]).flatten()
        if (positives.numel() != row["positive_count"]
                or int((~fixed_labels).sum()) != row["negative_count"]
                or candidates.numel() != meta["negative_count"]):
            raise ValueError(f"Cached class counts changed: {row['circuit']}")
        if row["negative_count"] > 128:
            raise ValueError("Original per-circuit negative budget unexpectedly exceeds 128")
        entry = counts[row["pair_count_group"]]
        entry["circuits"] += 1
        entry["cached_negatives"] += candidates.numel()
        entry["original_negative_budget"] += row["negative_count"]
        entry["larger_pool_circuits"] += int(candidates.numel() > row["negative_count"])
        pools.append({"record": row, "cache": cache, "positive_columns": positives,
                      "negative_candidates": candidates, "negative_budget": row["negative_count"]})
    if not any(entry["larger_pool_circuits"] for entry in counts.values()):
        raise ValueError("缓存只含原固定少量负样本，缺少更大候选池；停止，不扩大数据")
    return pools, counts


@torch.no_grad()
def refresh_samples(head, pools, generator, step, device, deadline):
    active, records = [], []
    before_rng = generator.get_state().clone()
    for pool in pools:
        training_time_check(deadline)
        row, cache = pool["record"], pool["cache"]
        negatives, mining = choose_negatives(head, cache, pool["negative_candidates"],
                                            pool["negative_budget"], generator, device,
                                            training_deadline=deadline)
        columns = torch.cat((pool["positive_columns"], negatives))
        active.append({"features": features_at(cache, columns).to(device),
                       "labels": cache["labels"][columns].to(device), "circuit": row["circuit"]})
        records.append({"circuit": row["circuit"], "pair_count_group": row["pair_count_group"],
                        "positive_cache_columns": pool["positive_columns"],
                        "negative_cache_columns": negatives,
                        "negative_original_columns": cache["original_pair_columns"][negatives],
                        "negative_candidate_count": pool["negative_candidates"].numel(),
                        "negative_budget": pool["negative_budget"], **mining})
    return active, {"refresh_total_step": step, "first_update_total_step": step+1,
                    "negative_rng_before": before_rng, "negative_rng_after": generator.get_state().clone(),
                    "circuits": records}


def fixed_reference(pools, device):
    return [{**pool["record"], "features": features_at(pool["cache"], pool["record"]["cache_pair_columns"]).to(device),
             "labels": pool["cache"]["labels"][pool["record"]["cache_pair_columns"]].to(device)} for pool in pools]


def print_comparison(methods):
    for method, result in methods.items():
        control.print_summary(method, result)
        for tag, metrics in [("pooled", result["maximum_f1"]),
                             *((group, row["at_global_max_f1_threshold"])
                               for group, row in result["groups_by_original_valid_pairs"].items())]:
            info(f"{method} / {tag}: max-F1 TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
    new = methods["hard_negative_final"]
    for baseline in ("cosine_baseline", "ordinary_step500", "ordinary_step5000"):
        ref = methods[baseline]
        a = new["groups_by_original_valid_pairs"][">=10000"]["at_global_max_f1_threshold"]
        b = ref["groups_by_original_valid_pairs"][">=10000"]["at_global_max_f1_threshold"]
        info(f"Hard-negative minus {baseline}: pooled F1 delta={new['maximum_f1']['f1']-ref['maximum_f1']['f1']:+.8f}; "
             f"dense max-F1 FP/recall={a['fp']}/{a['recall']} vs {b['fp']}/{b['recall']}; "
             f"FP<=69 TP/actualFP={new['fp_le_69']['tp']}/{new['fp_le_69']['fp']} vs {ref['fp_le_69']['tp']}/{ref['fp_le_69']['fp']}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    source, ordinary = args.source.resolve(), args.ordinary_run.resolve()
    source_run = json.loads((source / "run.json").read_text())
    ordinary_run = json.loads((ordinary / "run.json").read_text())
    state = torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    if (state["optimizer_steps"] != START_STEP or source_run["status"] != "completed"
            or state["objective"] != control.OBJECTIVE):
        raise ValueError("Require the original ordinary step-500 checkpoint")
    if not state.get("optimizer"):
        raise ValueError("第500步缺失优化器状态，无法按要求公平恢复；停止")
    checkpoint_hash = file_hash(source / "last.pt")
    if (ordinary_run["status"] != "completed" or ordinary_run["completed_total_step"] != END_STEP
            or ordinary_run["best_training_step"] != END_STEP
            or ordinary_run["source_checkpoint_sha256"] != checkpoint_hash):
        raise ValueError("Ordinary 5000-step comparison must have the same step-500 start and final-step validation")
    fixed_source = Path(source_run.get("fixed_sample_run", source_run["source_run"]))
    previous = json.loads((fixed_source / "run.json").read_text())
    if previous["device"] != args.device or previous["lr"] != 1e-3:
        raise ValueError("Keep the original device and lr=1e-3")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "preparing", "notice": original.NOTICE, "source_run": str(source),
              "ordinary_control_run": str(ordinary), "source_checkpoint_sha256": checkpoint_hash,
              "entry_sha256": file_hash(Path(__file__)), "hidden": state["hidden"],
              "optimizer_restore": "restored ordinary step-500 Adam state", "lr": previous["lr"],
              "objective": control.OBJECTIVE, "start_step": START_STEP, "end_step": END_STEP,
              "additional_steps": END_STEP-START_STEP, "wall_clock_seconds": WALL_SECONDS,
              "validation_time_reserve_seconds": VALIDATION_RESERVE,
              "refresh_interval_updates": REFRESH_INTERVAL, "scoring_chunk_size": CHUNK_SIZE,
              "negative_rng_seed": MINING_SEED, "positive_policy": "exact original fixed positive columns",
              "negative_policy": "same original per-circuit budget <=128; ceil(half) top logits, remainder uniform without replacement from other negatives; use all if candidate count <= budget",
              "hard_score_ties": "earlier original pair column first",
              "circuit_order": "exact saved ordinary step501..5000 circuit batches; negative RNG independent",
              "selection": "final complete finite step only; no best-by-validation or training-objective selection",
              "training_fit_reference": "optional fit values use ONLY the original fixed pairs, not refreshed samples",
              "encoder_loaded": False, "encoder_forward_count": 0, "test_evaluated": False,
              "validation_passes_started": 0, "validation_passes_completed": 0,
              "device": args.device, "group_threshold_rule": previous["group_threshold_rule"]}
    write_json(args.output / "run.json", record)
    atomic_save(state, args.output / "source_step500.pt")
    safe_state, active_refresh, safe_refresh = None, None, None
    updates, fixed_fit = [], []
    with direct_log(args.output / "console.log"):
        info(original.NOTICE)
        info(record["optimizer_restore"])
        try:
            with continuation.wall_limit(started + WALL_SECONDS):
                set_seed(previous["seed"])
                head = original.make_head(state["hidden"]).to(args.device)
                head.load_state_dict(state["head"], strict=True)
                optimizer = torch.optim.Adam(head.parameters(), lr=previous["lr"])
                optimizer.load_state_dict(state["optimizer"])
                if any(group["lr"] != previous["lr"] for group in optimizer.param_groups):
                    raise ValueError("Restored Adam learning rate differs")
                safe_state = continuation.snapshot(head, optimizer, START_STEP, record)
                atomic_save(safe_state, args.output / "last.pt")
                samples = torch.load(fixed_source / "training_samples.pt", map_location="cpu", weights_only=False)
                manifest = json.loads((Path(previous["source_run"]) / "cache_manifest.json").read_text())
                pools, counts = load_training_pools(samples, manifest, state["hidden"])
                record["candidate_pool_counts"] = counts
                write_json(args.output / "run.json", record)
                for group, entry in counts.items():
                    info(f"Train-only candidate pool group={group}: {entry}")
                schedule = json.loads((ordinary / "circuit_schedule.json").read_text())
                batches = schedule["batches"]
                if schedule["start_total_step"] != START_STEP+1 or len(batches) != END_STEP-START_STEP:
                    raise ValueError("Missing exact ordinary step501..5000 circuit sequence")
                index = {pool["record"]["circuit"]: i for i, pool in enumerate(pools)}
                # All draws come directly from the recorded control, never the mining RNG.
                indexed_batches = [[index[name] for name in batch] for batch in batches]
                write_json(args.output / "circuit_schedule.json", schedule)
                fixed = fixed_reference(pools, args.device)
                fixed_fit.append(continuation.training_fit(head, fixed, START_STEP))
                write_json(args.output / "fixed_pair_training_fit.json", fixed_fit)
                rng = torch.Generator().manual_seed(MINING_SEED)
                training_deadline = started + WALL_SECONDS - VALIDATION_RESERVE
                (args.output / "sampling").mkdir()
                stop_reason = "step_limit"
                try:
                    for step in range(START_STEP+1, END_STEP+1):
                        training_time_check(training_deadline)
                        if (step-START_STEP-1) % REFRESH_INTERVAL == 0:
                            active, fresh = refresh_samples(head, pools, rng, step-1, args.device, training_deadline)
                            sampling_path = args.output / "sampling" / f"refresh_step_{step-1}.pt"
                            atomic_save(fresh, sampling_path)
                            active_refresh = {"refresh_total_step": step-1, "sampling_file": str(sampling_path.resolve()),
                                              "negative_rng_state": rng.get_state().clone()}
                            info(f"Refreshed training negatives at total_step={step-1}; chunk_size={CHUNK_SIZE}")
                        head.train()
                        optimizer.zero_grad(set_to_none=True)
                        local = []
                        for i in indexed_batches[step-START_STEP-1]:
                            row = active[i]
                            logits = head(row["features"]).squeeze(-1)
                            continuation.finite(logits, "训练 logits")
                            local.append(control.circuit_loss(logits, row["labels"]))
                        loss = torch.stack(local).mean()
                        continuation.finite(loss, "训练损失")
                        loss.backward()
                        for parameter in head.parameters():
                            if parameter.grad is not None:
                                continuation.finite(parameter.grad, "分类头梯度")
                        optimizer.step()
                        for parameter in head.parameters():
                            continuation.finite(parameter, "分类头参数")
                        candidate = continuation.snapshot(head, optimizer, step, record)
                        candidate["hard_negative_sampling"] = active_refresh
                        safe_state = candidate
                        safe_refresh = active_refresh
                        updates.append({"total_step": step, "refreshed_sample_minibatch_bce": float(loss.detach()),
                                        "refresh_total_step": active_refresh["refresh_total_step"]})
                        if step % 500 == 0:
                            info(f"Head total_step={step}/{END_STEP}; changing-sample minibatch BCE={float(loss.detach()):.6f} (not convergence evidence)")
                            atomic_save(safe_state, args.output / "last.pt")
                    training_time_check(training_deadline)
                except TrainingTimeLimit as error:
                    stop_reason = "time_limit"
                    info(str(error))
                # The main result always uses the FINAL finite weights, never a selected earlier head.
                head.load_state_dict(safe_state["head"], strict=True)
                atomic_save(safe_state, args.output / "last.pt")
                atomic_save(safe_state, args.output / "final_head.pt")
                write_json(args.output / "update_history.json", updates)
                final_step = safe_state["optimizer_steps"]
                fixed_fit.append(continuation.training_fit(head, fixed, final_step))
                write_json(args.output / "fixed_pair_training_fit.json", fixed_fit)
                record.update(status="final_validation", stop_reason=stop_reason, completed_total_step=final_step,
                              equal_step_budget_completed=final_step == END_STEP)
                write_json(args.output / "run.json", record)
                # Validation information is first opened after all training and mining have ended.
                saved = torch.load(ordinary / "validation_scores.pt", map_location="cpu", weights_only=False)
                record["validation_passes_started"] = 1
                write_json(args.output / "run.json", record)
                scores = continuation.validation_once(head, manifest, saved, args.device, original.BATCH_SIZE)
                record["validation_passes_completed"] = 1
                atomic_save({"hard_negative_final_logits": scores, "labels": saved["labels"],
                             "offsets": saved["offsets"], "circuits": saved["circuits"],
                             "final_total_step": final_step}, args.output / "validation_scores.pt")
                old = json.loads((ordinary / "comparison.json").read_text())
                if old["selected_total_step"] != END_STEP or old["last_total_step"] != END_STEP:
                    raise ValueError("Saved ordinary validation does not represent final step5000")
                methods = {"cosine_baseline": old["methods"]["cosine_baseline"],
                           "ordinary_step500": old["methods"]["circuit_balanced_head_step500"],
                           "ordinary_step5000": old["methods"]["continued_best_train_objective"],
                           "hard_negative_final": control.score_summary(scores, saved["labels"], saved["offsets"], saved["circuits"], "logit")}
                report = {"notice": original.NOTICE, "scope": old["scope"], "final_total_step": final_step,
                          "equal_step_budget_completed": final_step == END_STEP, "stop_reason": stop_reason,
                          "selected_by": "final finite checkpoint; no validation-based checkpoint selection",
                          "group_threshold_rule": previous["group_threshold_rule"], "methods": methods}
                write_json(args.output / "comparison.json", report)
                print_comparison(methods)
                record["status"] = "completed"
                info(f"Completed hard-negative control: {args.output / 'comparison.json'}")
        except (continuation.WallLimit, continuation.NonFinite) as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if safe_state is not None:
                atomic_save(safe_state, args.output / "last.pt")
                atomic_save(safe_state, args.output / "final_head.pt")
            write_json(args.output / "update_history.json", updates)
            record.update(status="stopped_wall_clock" if isinstance(error, continuation.WallLimit) else "stopped_nonfinite",
                          stop_reason="time_limit" if isinstance(error, continuation.WallLimit) else "nonfinite",
                          stop_detail=str(error), completed_total_step=safe_state["optimizer_steps"] if safe_state else START_STEP,
                          validation_not_restarted=True)
            info(str(error))
        record.update(elapsed_wall_seconds=time.monotonic()-started,
                      last_completed_refresh=safe_refresh["refresh_total_step"] if safe_refresh else None)
        write_json(args.output / "run.json", record)
        info(f"status={record['status']} | stop_reason={record.get('stop_reason')} | total_step={record.get('completed_total_step')} | elapsed={record['elapsed_wall_seconds']:.1f}s")
        info(original.NOTICE)


def self_check():
    # Tiny synthetic cached embeddings: rank is known without any graph model.
    class ToyHead(torch.nn.Module):
        def forward(self, features):
            return features[:, :1]
    hf = torch.zeros(10, 2)
    hf[:, 0] = torch.arange(10, dtype=torch.float32)
    cache = {"hf_l2": hf, "pairs": torch.stack((torch.zeros(10, dtype=torch.long), torch.arange(10)))}
    candidates = torch.arange(10)
    generator = torch.Generator().manual_seed(123)
    global_rng = torch.get_rng_state().clone()
    selected, audit = choose_negatives(ToyHead(), cache, candidates, 5, generator, "cpu", chunk_size=2)
    assert torch.equal(audit["hard_cache_columns"], torch.tensor([9, 8, 7]))
    assert selected.numel() == selected.unique().numel() == 5
    assert audit["uniform_count"] == 2 and not bool(torch.isin(audit["uniform_cache_columns"], audit["hard_cache_columns"]).any())
    assert torch.equal(global_rng, torch.get_rng_state())
    small, _ = choose_negatives(ToyHead(), cache, candidates[:2], 128, generator, "cpu", chunk_size=1)
    assert torch.equal(small, candidates[:2])
    # Identical scores use earlier original columns, independent of chunk boundaries.
    tied_cache = {"hf_l2": torch.zeros_like(hf), "pairs": cache["pairs"]}
    _, audit = choose_negatives(ToyHead(), tied_cache, candidates, 5, generator, "cpu", chunk_size=2)
    assert torch.equal(audit["hard_cache_columns"], torch.tensor([0, 1, 2]))
    again, _ = choose_negatives(ToyHead(), cache, candidates, 5, torch.Generator().manual_seed(123), "cpu", chunk_size=3)
    assert torch.equal(selected, again)
    assert [step-1 for step in range(501, 5001) if (step-501) % 500 == 0] == list(range(500, 5000, 500))
    print("PASS: streamed top-half; uniform disjoint remainder; no replacement; insufficient-pool handling; deterministic ties/chunks; isolated RNG; refresh timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--self-check", action="store_true")
    parser.add_argument("--source", type=Path, default=START_RUN)
    parser.add_argument("--ordinary-run", type=Path, default=ORDINARY_RUN)
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
