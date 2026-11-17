"""Fixed-threshold extended validation; manual execution, resumable per circuit."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import frozen_equivalence_head as original
from examples.continue_circuit_balanced_head import WallLimit, wall_limit
from examples.compare_functional_stages import cosine_scores64, file_hash, within_circuit_auc
from examples.evaluate import validate_splits
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.trainer import move
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.net_identity import checkpoint_net_identity
from triwave_gnn.utils.random_seed import set_seed

EXP = original.EXP
DEV = EXP / "frozen_equivalence_head_dev_20261113"
STEP500 = EXP / "frozen_equivalence_head_continue500_20261113"
HARD5000 = EXP / "frozen_equivalence_head_hard_negative_from500_20261113"
NOTICE = "验证划分中的128电路扩展检查；阈值固定自原32电路开发子集；不是最终测试成绩"
SEED, SIZE, WALL_SECONDS, CHUNK = 20261113, 128, 1800, 4096
METHODS = ("B_cosine", "balanced_head_step500", "hard_negative_head_step5000")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def input_stats(data):
    return {name: {"size": (data / name).stat().st_size, "mtime_ns": (data / name).stat().st_mtime_ns}
            for name in ("graphs.npz", "labels.npz")}


def cache_key(name):
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared run: {args.output}")
    dev_run = json.loads((DEV / "run.json").read_text())
    checkpoint = original.B_RUN / "best_equiv.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["epoch"] != 5 or checkpoint_net_identity(state):
        raise ValueError("Require B best_equiv epoch5 with original propagation")
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != dev_run["checkpoint_sha256"]:
        raise ValueError("B checkpoint differs from the encoder used for the development caches")
    splits = json.loads((original.B_RUN / "split.json").read_text())
    val_key = validate_splits(splits, state)
    selection = json.loads((DEV / "selection.json").read_text())
    excluded = sorted(row["circuit"] for row in selection["circuits"] if row["split"] == "val")
    if len(excluded) != 32 or len(set(excluded)) != 32 or not set(excluded).issubset(splits[val_key]):
        raise ValueError("Expected exactly the original 32 development validation IDs")
    # Selection precedes ANY reading of pair statistics, labels or scores.
    eligible = sorted(set(splits[val_key]) - set(excluded))
    if len(eligible) < SIZE:
        raise ValueError("Fewer than 128 remaining val IDs; no replacement or expansion")
    chosen = random.Random(SEED).sample(eligible, SIZE)

    dev_result_path = DEV / "comparison.json"
    step_result_path = STEP500 / "comparison.json"
    hard_result_path = HARD5000 / "comparison.json"
    dev_result = json.loads(dev_result_path.read_text())
    step_result = json.loads(step_result_path.read_text())
    hard_result = json.loads(hard_result_path.read_text())
    if step_result["selected_total_step"] != 500 or hard_result["final_total_step"] != 5000:
        raise ValueError("Calibration results do not match requested fixed head steps")
    head_paths = {"balanced_head_step500": STEP500 / "last.pt",
                  "hard_negative_head_step5000": HARD5000 / "final_head.pt"}
    step_run = json.loads((STEP500 / "run.json").read_text())
    fixed_run_path = Path(step_run.get("fixed_sample_run", step_run["source_run"])) / "run.json"
    fixed_run = json.loads(fixed_run_path.read_text())
    hard_run = json.loads((HARD5000 / "run.json").read_text())
    if (fixed_run["encoder_checkpoint_sha256"] != checkpoint_hash
            or fixed_run["functional_net_identity"] is not False
            or hard_run["source_checkpoint_sha256"] != file_hash(head_paths["balanced_head_step500"])):
        raise ValueError("Fixed heads do not share the required B encoder and ordinary step500 lineage")
    for method, path in head_paths.items():
        head_state = torch.load(path, map_location="cpu", weights_only=False)
        if head_state["optimizer_steps"] != (500 if method == "balanced_head_step500" else 5000):
            raise ValueError(f"Unexpected head checkpoint step: {method}")
        if head_state["hidden"] != state["config"]["hidden"]:
            raise ValueError("Head/encoder hidden width mismatch")
    thresholds = {
        "B_cosine": {"main_threshold": dev_result["cosine_baseline"]["threshold"],
                     "old_low_fp_threshold": step_result["methods"]["cosine_baseline"]["fp_le_69"]["threshold"],
                     "main_source": str(dev_result_path.resolve()), "low_fp_source": str(step_result_path.resolve()),
                     "score_kind": "original float64 cosine"},
        "balanced_head_step500": {
            "main_threshold": step_result["methods"]["continued_best_train_objective"]["maximum_f1"]["threshold"],
            "old_low_fp_threshold": step_result["methods"]["continued_best_train_objective"]["fp_le_69"]["threshold"],
            "main_source": str(step_result_path.resolve()), "low_fp_source": str(step_result_path.resolve()), "score_kind": "logit"},
        "hard_negative_head_step5000": {
            "main_threshold": hard_result["methods"]["hard_negative_final"]["maximum_f1"]["threshold"],
            "old_low_fp_threshold": hard_result["methods"]["hard_negative_final"]["fp_le_69"]["threshold"],
            "main_source": str(hard_result_path.resolve()), "low_fp_source": str(hard_result_path.resolve()), "score_kind": "logit"}}
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           for row in thresholds.values() for value in (row["main_threshold"], row["old_low_fp_threshold"])):
        raise ValueError("A required old threshold is missing or nonfinite; no recalibration")
    if args.device != dev_run["device"]:
        raise ValueError("Use the original development inference device")
    stats = input_stats(args.data)
    if stats != dev_run["input_files"]:
        raise ValueError("Bundle file size/mtime differs from development input; no full-data audit attempted")
    contract = {"encoder_checkpoint_sha256": checkpoint_hash, "encoder_epoch": 5,
                "functional_net_identity": False, "input_files": stats,
                "hidden": state["config"]["hidden"], "hf_dtype": "float32", "l2_eps": 1e-12,
                "pair_policy": "all original hf_tt_mask & finite target in original column order",
                "positive_policy": "float32 target isclose(0,atol=1e-8,rtol=0)",
                "cosine_policy": "raw float32 hf endpoints converted to float64 before cosine eps=1e-8",
                "device": args.device}
    locked = [checkpoint, *head_paths.values(), dev_result_path, step_result_path, hard_result_path,
              DEV / "selection.json", DEV / "run.json", STEP500 / "run.json", HARD5000 / "run.json", fixed_run_path,
              original.B_RUN / "split.json", Path(__file__), Path(original.__file__)]
    plan = {"notice": NOTICE, "seed": SEED, "requested_circuits": SIZE, "selected_ids": chosen,
            "excluded_development_ids": excluded, "eligible_val_circuits": len(eligible),
            "selection_rule": "random.Random(20261113).sample(sorted(original val IDs minus original 32 dev IDs),128); uniform without replacement; no label/pair-count/prediction filtering or replacements",
            "data": str(args.data.resolve()), "device": args.device, "wall_seconds": WALL_SECONDS,
            "encoder_checkpoint": str(checkpoint.resolve()),
            "head_checkpoints": {key: str(path.resolve()) for key, path in head_paths.items()},
            "thresholds": thresholds, "cache_contract": contract,
            "locked_files": {str(path.resolve()): file_hash(path) for path in locked},
            "external_cache_manifests": [str(path.resolve()) for path in
                                         [DEV / "cache_manifest.json", *(args.cache_manifest or [])]],
            "full_bundle_hash_verification": False, "full_data_audit": False,
            "training": False, "test_evaluated": False}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "selection.json", {key: plan[key] for key in
               ("seed", "selection_rule", "selected_ids", "excluded_development_ids", "eligible_val_circuits")})
    write_json(args.output / "thresholds.json", thresholds)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "run.json", {"status": "prepared", "notice": NOTICE,
               "complete_128_circuit_result": False, "completed_score_circuits": 0,
               "training": False, "test_evaluated": False})
    info(f"Prepared {SIZE} uniformly selected val IDs, seed={SEED}; excluded development IDs=32; no inference")
    for method, row in thresholds.items():
        info(f"Frozen {method}: main={row['main_threshold']}, old-low-FP={row['old_low_fp_threshold']}")


def external_cache_index(plan):
    result = {}
    selected = set(plan["selected_ids"])
    for name in plan["external_cache_manifests"]:
        path = Path(name)
        if not path.is_file():
            continue
        owner_plan = path.parent / "plan.json"
        owner_run = path.parent / "run.json"
        compatible = False
        if owner_plan.is_file():
            compatible = json.loads(owner_plan.read_text()).get("cache_contract") == plan["cache_contract"]
        if not compatible and owner_run.is_file():
            run = json.loads(owner_run.read_text())
            compatible = (run.get("checkpoint_sha256") == plan["cache_contract"]["encoder_checkpoint_sha256"]
                          and run.get("functional_net_identity") is False
                          and run.get("input_files") == plan["cache_contract"]["input_files"]
                          and run.get("source_sha256") == file_hash(Path(original.__file__))
                          and run.get("encoder_state_unchanged_after_extraction") is True)
        if not compatible:
            info(f"Cache manifest lacks matching encoder/input provenance; skipping: {path}")
            continue
        rows = json.loads(path.read_text())
        for row in rows:
            if row["circuit"] in selected and Path(row["cache_file"]).is_file():
                result.setdefault(row["circuit"], Path(row["cache_file"]))
    return result


def check_cache(cache, name, contract, own):
    if own and cache.get("cache_contract") != contract:
        raise ValueError(f"Saved cache contract differs: {name}; refusing to overwrite")
    if cache.get("metadata", {}).get("circuit") != name:
        raise ValueError(f"Cached circuit identity differs: {name}")
    required = ("pairs", "labels", "original_pair_columns", "hf_l2", "cosine_scores64")
    if any(key not in cache for key in required):
        raise ValueError(f"Compatible cache lacks full pairs/labels/embedding fields: {name}")
    count = cache["labels"].numel()
    if (cache["pairs"].shape != (2, count) or cache["labels"].dtype != torch.bool
            or cache["original_pair_columns"].shape != (count,)
            or cache["cosine_scores64"].shape != (count,)
            or cache["hf_l2"].ndim != 2 or cache["hf_l2"].shape[1] != contract["hidden"]):
        raise ValueError(f"Cached shapes are inconsistent: {name}")


@torch.inference_mode()
def extract_cache(reader, encoder, name, contract, device):
    item = reader.item(name)
    target = item["hf_tt_target"]
    valid = item["hf_tt_mask"].bool() & torch.isfinite(target)
    pairs = item["hf_pairs"][:, valid].cpu().clone()
    labels = torch.isclose(target[valid], torch.zeros_like(target[valid]), atol=1e-8, rtol=0).cpu()
    columns = torch.nonzero(valid).flatten().cpu()
    _, hf, _ = encoder(move(item, device), include_behavior=False)
    hf = hf.cpu()
    if not bool(torch.isfinite(hf).all()):
        raise ValueError(f"Nonfinite frozen encoder hf: {name}")
    cosine = cosine_scores64(hf, pairs, torch.ones(labels.numel(), dtype=torch.bool), CHUNK)
    return {"hf_raw": hf, "hf_l2": F.normalize(hf, p=2, dim=-1, eps=1e-12),
            "pairs": pairs, "labels": labels, "original_pair_columns": columns,
            "cosine_scores64": cosine, "cache_contract": contract,
            "metadata": {"circuit": name, "node_count": hf.shape[0], "original_valid_pairs": labels.numel(),
                         "encoder_forward_count": 1, "split": "val_extended"}}


@torch.inference_mode()
def score_cache(cache, heads, device, plan_hash):
    values = {"B_cosine": cache["cosine_scores64"]}
    for method, head in heads.items():
        chunks = []
        for pairs in cache["pairs"].split(CHUNK, dim=1):
            left, right = pairs
            features = original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right])
            chunks.append(head(features.to(device)).squeeze(-1).cpu())
        values[method] = torch.cat(chunks) if chunks else torch.empty(0)
    if any(not bool(torch.isfinite(scores).all()) for scores in values.values()):
        raise ValueError("Nonfinite fixed candidate scores")
    return {"scores": values, "labels": cache["labels"], "metadata": cache["metadata"],
            "original_pair_columns": cache["original_pair_columns"], "plan_sha256": plan_hash}


def fixed_metrics(scores, labels, threshold):
    row = pair_metrics(scores, labels, threshold)
    row["false_positive_rate"] = row["fp"] / row["negative_count"] if row["negative_count"] and row["fp"] is not None else None
    if row["false_positive_rate"] is None:
        row["undefined_reasons"]["false_positive_rate"] = "no valid negative pairs"
    return row


def summarize(scores, labels, offsets, names, thresholds):
    main = thresholds["main_threshold"]
    low = thresholds["old_low_fp_threshold"]
    per_circuit = [{"circuit": name, "original_valid_pairs": b-a,
                    **fixed_metrics(scores[a:b], labels[a:b], main),
                    "at_old_low_fp_threshold": fixed_metrics(scores[a:b], labels[a:b], low)}
                   for name, a, b in zip(names, offsets, offsets[1:])]
    result = {"score_kind": thresholds["score_kind"], "thresholds_reselected": False,
              "at_old_dev_max_f1_threshold": fixed_metrics(scores, labels, main),
              "at_old_low_fp_threshold": fixed_metrics(scores, labels, low),
              **within_circuit_auc(per_circuit), "circuit_count": len(names),
              "zero_valid_pair_circuits": sum(a == b for a, b in zip(offsets, offsets[1:])),
              "per_circuit": per_circuit, "groups_by_original_valid_pairs": {}}
    for group in original.GROUPS:
        indices = [i for i, (a, b) in enumerate(zip(offsets, offsets[1:]))
                   if b > a and original.group_name(b-a) == group]
        s = torch.cat([scores[offsets[i]:offsets[i+1]] for i in indices]) if indices else scores[:0]
        y = torch.cat([labels[offsets[i]:offsets[i+1]] for i in indices]) if indices else labels[:0]
        result["groups_by_original_valid_pairs"][group] = {
            "circuit_count": len(indices), "sample_count": y.numel(),
            "at_old_dev_max_f1_threshold": fixed_metrics(s, y, main),
            "at_old_low_fp_threshold": fixed_metrics(s, y, low),
            **within_circuit_auc([per_circuit[i] for i in indices])}
    return result


def print_result(method, result):
    for tag, row in [("pooled", result), *result["groups_by_original_valid_pairs"].items()]:
        main = row["at_old_dev_max_f1_threshold"]
        low = row["at_old_low_fp_threshold"]
        info(f"{method} / {tag}: FIXED old-max-F1 threshold P/R/F1={main['precision']}/{main['recall']}/{main['f1']} | "
             f"TP/FP/FN={main['tp']}/{main['fp']}/{main['fn']} | "
             f"pooled/macro/within AUC={main['roc_auc']}/{row['circuit_macro_roc_auc']}/{row['within_circuit_weighted_auc']} | "
             f"AUC circuits={row['macro_participating_circuits']} | "
             f"FIXED old-low-FP threshold actual FP/FPR/Recall={low['fp']}/{low['false_positive_rate']}/{low['recall']}")


def run(args):
    started = time.monotonic()
    plan_path = args.output / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("Run --prepare first; no implicit subset selection at inference time")
    plan = json.loads(plan_path.read_text())
    plan_hash = digest_json(plan)
    names = plan["selected_ids"]
    if len(names) != SIZE or len(set(names)) != SIZE or set(names) & set(plan["excluded_development_ids"]):
        raise ValueError("Prepared selection changed")
    if args.device != plan["device"]:
        raise ValueError("Use the prepared device")
    if input_stats(Path(plan["data"])) != plan["cache_contract"]["input_files"]:
        raise ValueError("Prepared input file metadata changed; no automatic full audit")
    for name, expected in plan["locked_files"].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f"Prepared model/calibration/code input changed: {name}")
    final_path = args.output / "comparison.json"
    if final_path.is_file():
        completed = json.loads(final_path.read_text())
        if completed.get("plan_sha256") == plan_hash and completed.get("complete_128_circuit_result") is True:
            info(f"Already completed all128; reusing result without inference: {final_path}")
            return
        raise ValueError("Existing comparison belongs to another plan or is incomplete")
    hf_dir, score_dir = args.output / "hf_cache", args.output / "score_cache"
    hf_dir.mkdir(exist_ok=True)
    score_dir.mkdir(exist_ok=True)
    record = {"status": "running", "notice": NOTICE, "plan_sha256": plan_hash,
              "requested_circuits": SIZE, "completed_score_circuits": 0, "complete_128_circuit_result": False,
              "encoder_forwards_this_attempt": 0, "reused_hf_circuits_this_attempt": 0,
              "reused_score_circuits_this_attempt": 0, "wall_seconds": WALL_SECONDS,
              "thresholds_reselected": False, "training": False, "test_evaluated": False}
    scores_by_name = {}
    manifest_path = args.output / "cache_manifest.json"
    cache_manifest = ({row["circuit"]: row for row in json.loads(manifest_path.read_text())}
                      if manifest_path.is_file() else {})
    with direct_log(args.output / "console.log"), ExitStack() as stack:
        info(NOTICE)
        write_json(args.output / "run.json", record)
        try:
            with wall_limit(started + WALL_SECONDS):
                external = external_cache_index(plan)
                info(f"Compatible existing hf candidates among selected128: {len(external)}")
                heads = {}
                for method, path in plan["head_checkpoints"].items():
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    head = original.make_head(plan["cache_contract"]["hidden"])
                    head.load_state_dict(state["head"], strict=True)
                    heads[method] = head.to(args.device).eval().requires_grad_(False)
                reader, encoder = None, None
                # Archive-order processing is only an I/O optimization AFTER IDs
                # are fixed, unrelated to labels, counts, or model predictions.
                graph = original.NpzSlices(Path(plan["data"]) / "graphs.npz")
                stack.callback(graph.close)
                archive_ids = [str(x) for x in graph.read("design_ids")]
                archive_order = {name: i for i, name in enumerate(archive_ids)}
                process_order = sorted(names, key=archive_order.__getitem__)
                set_seed(original.SEED)
                for index, name in enumerate(process_order, 1):
                    key = cache_key(name)
                    hf_path, score_path = hf_dir / f"{key}.pt", score_dir / f"{key}.pt"
                    if score_path.is_file():
                        scored = torch.load(score_path, map_location="cpu", weights_only=False)
                        if scored.get("plan_sha256") != plan_hash or scored["metadata"]["circuit"] != name:
                            raise ValueError(f"Existing score cache differs: {name}")
                        record["reused_score_circuits_this_attempt"] += 1
                    else:
                        source = hf_path if hf_path.is_file() else external.get(name)
                        if source is not None:
                            cache = torch.load(source, map_location="cpu", weights_only=False)
                            check_cache(cache, name, plan["cache_contract"], source == hf_path)
                            record["reused_hf_circuits_this_attempt"] += 1
                        else:
                            if reader is None:
                                label = original.NpzSlices(Path(plan["data"]) / "labels.npz")
                                stack.callback(label.close)
                                state = torch.load(plan["encoder_checkpoint"], map_location="cpu", weights_only=False)
                                reader = original.SelectedReader(graph, label, state)
                                encoder, _ = load_model(Path(plan["encoder_checkpoint"]), reader, "cpu")
                                original.freeze_encoder(encoder)
                                encoder.to(args.device)
                            cache = extract_cache(reader, encoder, name, plan["cache_contract"], args.device)
                            # Persist hf before scoring, so a later interruption
                            # never requires repeating completed cached inference.
                            atomic_save(cache, hf_path)
                            source = hf_path
                            record["encoder_forwards_this_attempt"] += 1
                        cache_manifest[name] = {"circuit": name, "cache_file": str(source.resolve()),
                                                "original_valid_pairs": cache["labels"].numel()}
                        write_json(manifest_path, list(cache_manifest.values()))
                        scored = score_cache(cache, heads, args.device, plan_hash)
                        atomic_save(scored, score_path)
                        del cache
                    scores_by_name[name] = scored
                    record["completed_score_circuits"] = len(scores_by_name)
                    write_json(args.output / "run.json", record)
                    info(f"Extended val completed {index}/{SIZE}: {name} | original valid pairs={scored['labels'].numel()}")
                offsets = [0]
                for name in names:
                    offsets.append(offsets[-1] + scores_by_name[name]["labels"].numel())
                labels = torch.cat([scores_by_name[name]["labels"] for name in names])
                report = {"notice": NOTICE, "plan_sha256": plan_hash, "complete_128_circuit_result": True,
                          "requested_circuits": SIZE, "completed_circuits": len(scores_by_name),
                          "selection_seed": SEED, "selected_ids": names,
                          "zero_valid_pair_circuits": sum(a == b for a, b in zip(offsets, offsets[1:])),
                          "thresholds_reselected": False, "test_evaluated": False,
                          "low_fp_note": "Old development low-FP thresholds are fixed; actual FP on this larger subset need not be <=69",
                          "methods": {}}
                for method in METHODS:
                    scores = torch.cat([scores_by_name[name]["scores"][method] for name in names])
                    result = summarize(scores, labels, offsets, names, plan["thresholds"][method])
                    report["methods"][method] = result
                    print_result(method, result)
                write_json(final_path, report)
                record.update(status="completed", complete_128_circuit_result=True,
                              zero_valid_pair_circuits=report["zero_valid_pair_circuits"])
                info(f"Completed extended validation: all128 circuits; zero-valid-pair circuits={report['zero_valid_pair_circuits']}; {final_path}")
        except WallLimit:
            record.update(status="incomplete_timeout", complete_128_circuit_result=False,
                          stop_reason="30 minute hard wall limit", partial_result_not_a_complete_128_result=True)
            info("未完成：达到30分钟上限。已完成电路缓存保留；不能当作完整128电路结果。用同一命令复用缓存继续。")
        record.update(elapsed_wall_seconds=time.monotonic()-started,
                      completed_score_cache_files=sum((score_dir / f"{cache_key(name)}.pt").is_file() for name in names))
        write_json(args.output / "run.json", record)
        info(f"status={record['status']} | complete128={record['complete_128_circuit_result']} | cached completed circuits={record['completed_score_cache_files']}/{SIZE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(original.__file__).resolve().parent / "data/train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-manifest", type=Path, action="append")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
