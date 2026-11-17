"""I0/I1 dense-group ranking-loss contrast from the saved G1 step43438 head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import compare_circuit_balanced_head as control
from examples import continue_circuit_balanced_head as continuation
from examples import continue_train512_budget as budget
from examples import expand_training_frozen_head as coverage
from examples import expanded_frozen_head_validation as reuse
from examples import final_frozen_head_evaluation as final
from examples import finetune_hard_negative_head as hard
from examples import frozen_equivalence_head as original
from examples import group_score_calibration_cv as saved_val_scores
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save, random_state, restore_random_state
from triwave_gnn.utils.functional_equivalence import grouped_curve, select_threshold
from triwave_gnn.utils.equivalence_audit import pair_count_group_metrics
from triwave_gnn.utils.logger import info

SOURCE = original.EXP / "frozen_head_dense_sampling_G1_20261113"
FINAL_VAL = original.EXP / "final_B_head33438_20261113"
TRAIN_CACHE = original.EXP / "frozen_head_train512_cache_20261113"
START, ADDITIONAL, WALL, VAL_RESERVE = 43438, 10000, 300, 45
RANK_WEIGHT, RANK_NEGATIVE_CAP = 0.1, 16
DENSE_REFERENCE_TP, OVERALL_REFERENCE_TP = 219, 8025
NOTICE = ("I0/I1均从原G1 seed=20261113 step43438恢复分类头与Adam；仅I1增加密集双类电路组内排序损失；"
          "完整val开发评价，不是测试成绩；不读取test、不运行图编码器；I1不是原G1正式版本")


def read(path):
    return final.read(path)


def load(path):
    return final.load(path)


def preflight(args):
    state = load(SOURCE / "final_head.pt")
    source_run = read(SOURCE / "run.json")
    source_recipe = read(SOURCE / "common_recipe.json")
    source_comparison = read(SOURCE / "comparison.json")
    config = read(FINAL_VAL / "frozen_config.json")
    calibration = read(FINAL_VAL / "calibration_val1663.json")
    if (source_run.get("status") != "completed" or not source_comparison.get("complete_val_evaluation")
            or state.get("optimizer_steps") != START or not state.get("optimizer")
            or state.get("negative_rng_state") is None or state.get("circuit_rng_state") is None
            or state.get("global_random_state") is None or state.get("objective") != control.OBJECTIVE
            or source_recipe.get("source_step") != 33438 or args.device != config["device"]
            or config.get("functional_net_identity") is not False
            or not calibration.get("complete_val_calibration")):
        raise ValueError("Need completed original G1 step43438 with Adam/RNG, frozen original-propagation B config, and full-val calibration")
    if (source_recipe.get("encoder_sha256") != config["encoder_sha256"]
            or file_hash(Path(config["encoder_checkpoint"])) != config["encoder_sha256"]
            or calibration.get("frozen_config_sha256") != file_hash(FINAL_VAL / "frozen_config.json")):
        raise ValueError("B encoder/config provenance changed")
    train_plan = read(TRAIN_CACHE / "plan.json")
    manifest = read(TRAIN_CACHE / "cache_manifest.json")
    samples_file = load(state["fixed_samples_file"])
    samples = samples_file["circuits"]
    splits = read(original.B_RUN / "split.json")
    active_names = [row["circuit"] for row in manifest if row["original_valid_pairs"]]
    if (len(manifest) != 512 or len(samples) != 426
            or [row["circuit"] for row in samples] != active_names
            or not {row["circuit"] for row in manifest} <= set(splits["train"])
            or set(active_names) & set(config["val_ids"])
            or train_plan["cache_contract"] != config["cache_contract"]):
        raise ValueError("512 train circuit/cache identity or split contract changed")
    lookup = {row["circuit"]: row for row in manifest}
    missing_train = [row["cache_file"] for row in manifest
                     if row["original_valid_pairs"] and not Path(row["cache_file"]).is_file()]
    if missing_train:
        raise FileNotFoundError(f"Training caches missing; will not run encoder: {missing_train[:8]}")
    sampling = load(state["sampling_file"])
    if (sampling["refresh_total_step"] != 43000
            or not torch.equal(sampling["negative_rng_after"], state["negative_rng_state"])
            or [row["circuit"] for row in sampling["circuits"]] != active_names):
        raise ValueError("Saved G1 negative sampling/RNG state differs")
    baseline = load(FINAL_VAL / "val_scores.pt")
    if (baseline["circuits"] != config["val_ids"] or len(set(baseline["circuits"])) != 1663
            or file_hash(FINAL_VAL / "val_scores.pt") != calibration["val_scores_sha256"]):
        raise ValueError("Full val index/hash changed")
    val_plan = {"selected_ids": config["val_ids"], "cache_contract": config["cache_contract"],
                "external_cache_manifests": [str(reuse.DEV / "cache_manifest.json"),
                    str(coverage.EXTENDED / "cache_manifest.json"),
                    str(final.REMAINING / "cache_manifest.json")]}
    val_cache = reuse.external_cache_index(val_plan)
    missing_val = sorted(set(config["val_ids"]) - set(val_cache))
    if missing_val:
        raise FileNotFoundError(f"Val hf caches missing; will not run encoder: {missing_val[:8]}")
    g1 = source_comparison["methods"]["G1"]["maximum_f1"]["at_fixed_val_threshold"]
    dense_ref = source_comparison["methods"]["G1"]["maximum_f1"]["groups_by_original_valid_pairs"][">=10000"]["at_fixed_val_threshold"]
    if (g1["tp"] != OVERALL_REFERENCE_TP or dense_ref["tp"] != DENSE_REFERENCE_TP
            or dense_ref["fp"] != 8906):
        raise ValueError(f"Saved G1 references differ from requested TP/FP targets: overall={g1['tp']}, dense={dense_ref['tp']}/{dense_ref['fp']}")
    return state, source_run, source_recipe, source_comparison, config, train_plan, manifest, samples_file, sampling, lookup, baseline, val_cache


def restore_active(samples, sampling, lookup, plan):
    active = []
    for sample, saved in zip(samples, sampling["circuits"]):
        cache = coverage.load_train_cache(lookup[sample["circuit"]], plan)
        budget.check_saved_samples(cache, sample)
        columns = sample["cache_pair_columns"]
        positives = columns[cache["labels"][columns]]
        negatives = saved["negative_cache_columns"]
        if (not torch.equal(saved["positive_cache_columns"], positives)
                or negatives.numel() != sample["negative_count"]
                or negatives.unique().numel() != negatives.numel()
                or bool(cache["labels"][negatives].any())
                or not torch.equal(cache["original_pair_columns"][negatives], saved["negative_original_columns"])):
            raise ValueError(f"Saved G1 sampled pair alignment changed: {sample['circuit']}")
        chosen = torch.cat((positives, negatives))
        labels = cache["labels"][chosen].to(plan["device"])
        active.append({"circuit": sample["circuit"],
                       "dense": int(lookup[sample["circuit"]]["original_valid_pairs"]) >= 10000,
                       "features": hard.features_at(cache, chosen).to(plan["device"]),
                       "labels": labels})
        del cache
    return active


def batch_loss(head, rows, branch):
    bce_terms, rank_terms, mixed_terms, rank_circuits = [], [], [], 0
    for row in rows:
        logits = head(row["features"]).squeeze(-1)
        continuation.finite(logits, "训练 logits")
        bce = control.circuit_loss(logits, row["labels"])
        rank = logits.new_zeros(())
        if (branch == "I1" and row["dense"] and bool(row["labels"].any())
                and bool((~row["labels"]).any())):
            positive_logits = logits[row["labels"]]
            negative_logits = logits[~row["labels"]]
            hardest = torch.topk(negative_logits, k=min(RANK_NEGATIVE_CAP, negative_logits.numel()),
                                 largest=True, sorted=False).values
            rank = F.softplus(hardest[:, None] - positive_logits[None, :]).mean()
            continuation.finite(rank, "排序损失")
            rank_circuits += 1
        bce_terms.append(bce)
        rank_terms.append(rank)
        mixed_terms.append(bce + RANK_WEIGHT * rank if branch == "I1" else bce)
    bce_mean = torch.stack(bce_terms).mean()
    rank_mean = torch.stack(rank_terms).mean()
    loss = torch.stack(mixed_terms).mean()
    continuation.finite(loss, "训练总损失")
    return loss, bce_mean, rank_mean, rank_circuits


@torch.inference_mode()
def score_head(head, cache, device):
    pieces = []
    for pairs in cache["pairs"].split(hard.CHUNK_SIZE, dim=1):
        left, right = pairs
        features = original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right]).to(device)
        logits = head(features).squeeze(-1)
        continuation.finite(logits, "val logits")
        pieces.append(logits.cpu())
    return torch.cat(pieces) if pieces else torch.empty(0)


def minimum_fp_at_tp(scores, labels, required_tp):
    thresholds, tp, fp = grouped_curve(scores, labels)
    feasible = torch.nonzero(tp >= required_tp).flatten()
    if not feasible.numel():
        raise ValueError(f"Required TP {required_tp} is not achievable")
    feasible = feasible[fp[feasible] == fp[feasible].min()]
    threshold = float(thresholds[feasible[0]])
    return reuse.fixed_metrics(scores, labels, threshold)


def evaluate_branch(args, head, config, baseline, val_cache, record):
    head.eval()
    scores_by_circuit = []
    for name, start, stop in zip(baseline["circuits"], baseline["offsets"], baseline["offsets"][1:]):
        cache = load(val_cache[name])
        reuse.check_cache(cache, name, config["cache_contract"], "cache_contract" in cache)
        if not torch.equal(cache["labels"], baseline["labels"][start:stop]):
            raise ValueError(f"Val cache label/index mismatch for {name}")
        scores_by_circuit.append(score_head(head, cache, args.device))
    scores = torch.cat(scores_by_circuit)
    labels = baseline["labels"].bool()
    threshold = select_threshold(scores, labels)
    maximum = final.equivalence_summary(scores, labels, baseline["offsets"], baseline["circuits"],
                                        threshold, "val", "logit")
    dense_mask = torch.cat([torch.full((b-a,), original.group_name(b-a) == ">=10000", dtype=torch.bool)
                            for a, b in zip(baseline["offsets"], baseline["offsets"][1:])])
    dense_point = minimum_fp_at_tp(scores[dense_mask], labels[dense_mask], DENSE_REFERENCE_TP)
    overall_point = minimum_fp_at_tp(scores, labels, OVERALL_REFERENCE_TP)
    report = {"maximum_f1": maximum,
              "dense_group_min_fp_at_tp219": dense_point,
              "overall_min_fp_at_tp8025": overall_point,
              "score_kind": "original classifier-head logit; no C2 calibration"}
    info("VAL %s max-F1 P/R/F1=%s/%s/%s TP/FP=%s/%s threshold=%s" %
         (args.branch, maximum["at_fixed_val_threshold"]["precision"],
          maximum["at_fixed_val_threshold"]["recall"], maximum["at_fixed_val_threshold"]["f1"],
          maximum["at_fixed_val_threshold"]["tp"], maximum["at_fixed_val_threshold"]["fp"], threshold))
    info("VAL %s dense TP>=219 min-FP: TP/FP=%d/%d R/P=%s/%s threshold=%s" %
         (args.branch, dense_point["tp"], dense_point["fp"], dense_point["recall"],
          dense_point["precision"], dense_point["threshold"]))
    info("VAL %s overall TP>=8025 min-FP: TP/FP=%d/%d R/P=%s/%s threshold=%s" %
         (args.branch, overall_point["tp"], overall_point["fp"], overall_point["recall"],
          overall_point["precision"], overall_point["threshold"]))
    for group, row in maximum["groups_by_original_valid_pairs"].items():
        entry = row["at_fixed_val_threshold"]
        info("VAL %s group=%s maxF1 P/R/F1=%s/%s/%s TP/FP/FN=%d/%d/%d" %
             (args.branch, group, entry["precision"], entry["recall"], entry["f1"],
              entry["tp"], entry["fp"], entry["fn"]))
    info("VAL %s AUC pooled/macro/within=%s/%s/%s" %
         (args.branch, maximum["at_fixed_val_threshold"]["roc_auc"],
          maximum["circuit_macro_roc_auc"], maximum["within_circuit_weighted_auc"]))
    write_json(args.output / "thresholds.json", {"score_kind": "logit", "global_max_f1": threshold,
                "dense_group_min_fp_tp219": dense_point["threshold"],
                "overall_min_fp_tp8025": overall_point["threshold"]})
    record.update(status="completed", completed_val_evaluation=True)
    write_json(args.output / "comparison.json", {"notice": NOTICE, "branch": args.branch,
                "complete_val_evaluation": True, "test_evaluated": False,
                "completed_additional_steps": record["completed_additional_steps"],
                "completed_total_step": record["completed_total_step"], "stop_reason": record["stop_reason"],
                "method": report})
    write_json(args.output / "run.json", record)
    return report


def combine_peer(args, current, source_comparison):
    if args.peer_output is None or not (args.peer_output / "comparison.json").is_file():
        info("另一I分支完整val结果尚未保存；暂不作I0/I1汇总。")
        return
    peer = read(args.peer_output / "comparison.json")
    current = read(args.output / "comparison.json")
    if not peer.get("complete_val_evaluation") or peer["branch"] == args.branch:
        raise ValueError("Peer must be the other completed I branch")
    pair = {peer["branch"]: peer, current["branch"]: current}
    if set(pair) != {"I0", "I1"}:
        raise ValueError("Expected one completed I0 and one completed I1")
    histories = {branch: read(Path(args.peer_output if branch == peer["branch"] else args.output) /
                              "update_history.json") for branch in ("I0", "I1")}
    schedules = {branch: [row["circuits"] for row in histories[branch]] for branch in histories}
    if schedules["I0"] != schedules["I1"]:
        raise ValueError("I0/I1 circuit schedules differ; comparison is not paired")
    schedule_hash = hashlib.sha256(json.dumps(schedules["I0"], separators=(",", ":")).encode()).hexdigest()
    val = saved_val_scores.load_saved_scores()
    dense_mask = val["pair_group_codes"] == len(original.GROUPS) - 1
    g1_dense = minimum_fp_at_tp(val["logits"][dense_mask], val["labels"][dense_mask], DENSE_REFERENCE_TP)
    g1_overall = minimum_fp_at_tp(val["logits"], val["labels"], OVERALL_REFERENCE_TP)
    methods = {"original_G1_step43438": {
                   "maximum_f1": source_comparison["methods"]["G1"]["maximum_f1"],
                   "dense_group_min_fp_at_tp219": g1_dense,
                   "overall_min_fp_at_tp8025": g1_overall,
                   "score_kind": "saved original G1 logits; no model forward"},
               "I0": pair["I0"]["method"], "I1": pair["I1"]["method"]}
    summary = {"notice": NOTICE, "test_evaluated": False, "methods": methods,
               "branch_steps": {name: pair[name]["completed_additional_steps"] for name in ("I0", "I1")},
               "circuit_schedules_identical": True, "circuit_schedule_sha256": schedule_hash,
               "equal_10000_step_budget": all(pair[name]["completed_additional_steps"] == ADDITIONAL
                                               for name in ("I0", "I1"))}
    write_json(args.output / "comparison_I0_I1.json", summary)
    for name in ("original_G1_step43438", "I0", "I1"):
        m = methods[name]
        if name == "original_G1_step43438":
            maxf = m["maximum_f1"]["at_fixed_val_threshold"]
            dense_point = m["dense_group_min_fp_at_tp219"]
            overall_point = m["overall_min_fp_at_tp8025"]
            auc = m["maximum_f1"]
        else:
            maxf = m["maximum_f1"]["at_fixed_val_threshold"]
            dense_point = m["dense_group_min_fp_at_tp219"]
            overall_point = m["overall_min_fp_at_tp8025"]
            auc = m["maximum_f1"]
        info("COMPARE %s maxF1 P/R/F1=%s/%s/%s TP/FP=%s/%s" %
             (name, maxf["precision"], maxf["recall"], maxf["f1"], maxf["tp"], maxf["fp"]))
        info("COMPARE %s dense TP>=219 minFP TP/FP/R/P=%s/%s/%s/%s; overall TP>=8025 minFP TP/FP=%s/%s" %
             (name, dense_point.get("tp"), dense_point.get("fp"), dense_point.get("recall"),
              dense_point.get("precision"), overall_point.get("tp"), overall_point.get("fp")))
        info("COMPARE %s AUC pooled/macro/within=%s/%s/%s" %
             (name, auc.get("roc_auc", auc.get("at_fixed_val_threshold", {}).get("roc_auc")),
              auc.get("circuit_macro_roc_auc"), auc.get("within_circuit_weighted_auc")))
    i0, i1 = methods["I0"], methods["I1"]
    for key, label in (("dense_group_min_fp_at_tp219", "dense TP>=219 FP"),
                       ("overall_min_fp_at_tp8025", "overall TP>=8025 FP")):
        a, b = i0[key], i1[key]
        info("I1 minus I0 %s delta=%d; TP I0/I1=%d/%d" % (label, b["fp"]-a["fp"], a["tp"], b["tp"]))
    for group in original.GROUPS:
        a = i0["maximum_f1"]["groups_by_original_valid_pairs"][group]["at_fixed_val_threshold"]
        b = i1["maximum_f1"]["groups_by_original_valid_pairs"][group]["at_fixed_val_threshold"]
        info("I1 vs I0 max-F1 group=%s P/R/F1=%s/%s/%s vs %s/%s/%s TP/FP/FN=%d/%d/%d vs %d/%d/%d" %
             (group, b["precision"], b["recall"], b["f1"], a["precision"], a["recall"], a["f1"],
              b["tp"], b["fp"], b["fn"], a["tp"], a["fp"], a["fn"]))


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite branch output: {args.output}")
    (state, source_run, source_recipe, source_comparison, config, train_plan, manifest,
     samples_file, sampling, lookup, baseline, val_cache) = preflight(args)
    samples = samples_file["circuits"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "sampling").mkdir()
    recipe = {"source_checkpoint": str((SOURCE / "final_head.pt").resolve()),
              "source_checkpoint_sha256": file_hash(SOURCE / "final_head.pt"),
              "source_step": START, "planned_additional_steps": ADDITIONAL,
              "encoder_checkpoint": config["encoder_checkpoint"], "encoder_sha256": config["encoder_sha256"],
              "functional_net_identity": False, "objective_I0": control.OBJECTIVE,
              "objective_I1": control.OBJECTIVE + f" + {RANK_WEIGHT}*dense_rank_softplus(top{RANK_NEGATIVE_CAP})",
              "rank_scope": "original_valid_pairs>=10000 and current sampled labels contain both classes",
              "rank_formula": "mean(softplus(s_negative_top - s_positive)) over selected negatives x sampled positives",
              "dense_negative_topk": RANK_NEGATIVE_CAP, "rank_weight": RANK_WEIGHT,
              "lr": 0.001, "fixed_samples_sha256": file_hash(Path(state["fixed_samples_file"])),
              "circuit_rng": "restored source G1 state; same independent copy in I0/I1",
              "negative_rng": "restored source G1 state; independent copy per branch; mining refresh unchanged",
              "refresh_interval": hard.REFRESH_INTERVAL, "wall_seconds_per_branch": WALL,
              "validation_reserve_seconds": VAL_RESERVE}
    recipe_hash = reuse.digest_json(recipe)
    write_json(args.output / "recipe.json", recipe)
    atomic_save(state, args.output / "source_G1_step43438.pt")
    record = {"status": "preparing", "notice": NOTICE, "branch": args.branch,
              "hidden": state["hidden"],
              "source_run": str(SOURCE), "source_checkpoint_sha256": recipe["source_checkpoint_sha256"],
              "source_step": START, "planned_additional_steps": ADDITIONAL,
              "completed_additional_steps": 0, "completed_total_step": START,
              "common_recipe_sha256": recipe_hash, "optimizer_restore": "G1 step43438 Adam restored exactly",
              "circuit_rng_restore": "G1 step43438 circuit RNG state copied independently",
              "negative_rng_restore": "G1 step43438 negative mining RNG state copied independently",
              "circuit_sampler": "exact original G1 dense1+other3 schedule",
              "fixed_positive_policy": "exact original G1 fixed positive columns",
              "hard_negative_policy": "exact original G1 top-half+uniform refresh every 500 global steps",
              "rank_loss": "disabled" if args.branch == "I0" else recipe["objective_I1"],
              "group_counts": source_run["group_counts"], "empty_train_circuits": source_run["empty_train_circuits"],
              "encoder_loaded": False, "encoder_forward_count": 0,
              "test_evaluated": False, "test_results_read": False,
              "stop_reason": None, "device": args.device}
    safe = None
    updates = []
    with direct_log(args.output / "console.log"):
        write_json(args.output / "run.json", record)
        try:
            with continuation.wall_limit(started + WALL):
                head = original.make_head(state["hidden"]).to(args.device)
                head.load_state_dict(state["head"], strict=True)
                head.requires_grad_(True)
                optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
                optimizer.load_state_dict(state["optimizer"])
                if any(group["lr"] != 0.001 for group in optimizer.param_groups):
                    raise ValueError("Source G1 Adam lr differs from 0.001")
                restore_random_state(state["global_random_state"])
                circuit_rng = torch.Generator()
                circuit_rng.set_state(state["circuit_rng_state"])
                negative_rng = torch.Generator()
                negative_rng.set_state(state["negative_rng_state"])
                sampling_file = state["sampling_file"]
                active = restore_active(samples, sampling, lookup, train_plan)
                # G1 schedule: each step contains one distinct dense and three distinct other circuits.
                groups = {"dense_pairs": torch.tensor([i for i, row in enumerate(samples)
                          if int(lookup[row["circuit"]]["original_valid_pairs"]) >= 10000]),
                          "other_pairs": torch.tensor([i for i, row in enumerate(samples)
                          if int(lookup[row["circuit"]]["original_valid_pairs"]) < 10000])}
                safe = continuation.snapshot(head, optimizer, START, record)
                safe.update(global_random_state=random_state(), circuit_rng_state=circuit_rng.get_state().clone(),
                            negative_rng_state=negative_rng.get_state().clone(), sampling_file=sampling_file,
                            fixed_samples_file=state["fixed_samples_file"], branch=args.branch,
                            additional_optimizer_steps=0, common_recipe_sha256=recipe_hash)
                atomic_save(safe, args.output / "last.pt")
                deadline = started + WALL - VAL_RESERVE
                stop_reason = "step_limit"
                try:
                    for step in range(START + 1, START + ADDITIONAL + 1):
                        hard.training_time_check(deadline)
                        if (step - 1) % hard.REFRESH_INTERVAL == 0:
                            active, sampling_file = coverage.refresh(head, samples, lookup, train_plan,
                                                                     negative_rng, step - 1, deadline, args.output)
                            for row in active:
                                row["dense"] = int(lookup[row["circuit"]]["original_valid_pairs"]) >= 10000
                            info(f"{args.branch} unchanged negative refresh at total_step={step-1}")
                        batch = torch.cat((groups["dense_pairs"][torch.randperm(len(groups["dense_pairs"]), generator=circuit_rng)[:1]],
                                           groups["other_pairs"][torch.randperm(len(groups["other_pairs"]), generator=circuit_rng)[:3]])).tolist()
                        head.train()
                        optimizer.zero_grad(set_to_none=True)
                        rows = [active[i] for i in batch]
                        loss, bce, rank, rank_circuits = batch_loss(head, rows, args.branch)
                        loss.backward()
                        for parameter in head.parameters():
                            if parameter.grad is not None:
                                continuation.finite(parameter.grad, "分类头梯度")
                        optimizer.step()
                        for parameter in head.parameters():
                            continuation.finite(parameter, "分类头参数")
                        safe = continuation.snapshot(head, optimizer, step, record)
                        safe.update(global_random_state=random_state(), circuit_rng_state=circuit_rng.get_state().clone(),
                                    negative_rng_state=negative_rng.get_state().clone(), sampling_file=sampling_file,
                                    fixed_samples_file=state["fixed_samples_file"], branch=args.branch,
                                    additional_optimizer_steps=step-START, common_recipe_sha256=recipe_hash)
                        updates.append({"total_step": step, "additional_step": step-START,
                                        "circuits": [samples[i]["circuit"] for i in batch],
                                        "bce_loss": float(bce.detach()), "rank_loss": float(rank.detach()),
                                        "rank_active_circuits": rank_circuits,
                                        "weighted_rank_loss": float((RANK_WEIGHT * rank).detach()),
                                        "total_loss": float(loss.detach())})
                        if (step - START) % 500 == 0:
                            atomic_save(safe, args.output / "last.pt")
                            info(f"{args.branch} step={step-START}/{ADDITIONAL} BCE={float(bce.detach()):.6f} "
                                 f"rank={float(rank.detach()):.6f} active_dense_circuits={rank_circuits} "
                                 f"weighted_rank={RANK_WEIGHT*float(rank.detach()):.6f} total={float(loss.detach()):.6f}")
                except hard.TrainingTimeLimit:
                    stop_reason = "time_limit"
                if args.branch == "I0" and any(update["rank_loss"] != 0.0 for update in updates):
                    raise AssertionError("I0 path must have zero ranking loss")
                head.load_state_dict(safe["head"], strict=True)
                atomic_save(safe, args.output / "last.pt")
                atomic_save(safe, args.output / "final_head.pt")
                record.update(status="evaluating_val", completed_total_step=safe["optimizer_steps"],
                              completed_additional_steps=safe["optimizer_steps"]-START,
                              stop_reason=stop_reason, training_elapsed_wall_seconds=time.monotonic()-started)
                write_json(args.output / "update_history.json", updates)
                write_json(args.output / "run.json", record)
                report = evaluate_branch(args, head, config, baseline, val_cache, record)
                combine_peer(args, report, source_comparison)
        except (continuation.WallLimit, continuation.NonFinite) as error:
            complete = (args.output / "comparison.json").is_file()
            record.update(status="completed" if complete else "incomplete_timeout" if isinstance(error, continuation.WallLimit) else "stopped_nonfinite",
                          complete_val_evaluation=complete, stop_detail=str(error))
            if record["stop_reason"] is None:
                record["stop_reason"] = "time_limit" if isinstance(error, continuation.WallLimit) else "nonfinite"
            info("保存最后完整有限更新；未完成的训练/val不作为完整对照。")
        finally:
            if safe is not None:
                atomic_save(safe, args.output / "last.pt")
                atomic_save(safe, args.output / "final_head.pt")
                record.update(completed_total_step=safe["optimizer_steps"],
                              completed_additional_steps=safe["optimizer_steps"]-START,
                              branch_circuit_visits=(safe["optimizer_steps"]-START)*4)
            write_json(args.output / "update_history.json", updates)
            record["elapsed_wall_seconds"] = time.monotonic() - started
            write_json(args.output / "run.json", record)
            info(f"{args.branch} status={record['status']}; steps={record['completed_additional_steps']}/{ADDITIONAL}; "
                 f"stop={record['stop_reason']}; elapsed={record['elapsed_wall_seconds']:.1f}s")
            info(NOTICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--branch", choices=("I0", "I1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer-output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
