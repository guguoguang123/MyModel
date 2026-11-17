"""Manually continue the cached circuit-balanced head, with a fixed fit budget."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import compare_circuit_balanced_head as control
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash, within_circuit_auc
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

SOURCE = original.EXP / "frozen_equivalence_head_circuit_balanced_20261113"
CHECK_STEPS = (40, 100, 200, 300, 400, 500)
MAX_STEP, WALL_SECONDS = 500, 15 * 60
TRAIN_NOTICE = "固定训练配对的训练拟合指标，不是泛化成绩；训练阈值不用于验证"


class WallLimit(Exception):
    pass


class NonFinite(Exception):
    pass


@contextmanager
def wall_limit(deadline):
    def expired(*_):
        raise WallLimit("本次墙钟预算已耗尽，停止并保存最后一次完整有限更新")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def finite(value, what):
    if not bool(torch.isfinite(value).all()):
        raise NonFinite(f"非有限 {what}，立即停止；不运行验证")


def save_training_curve(directory, history):
    """Dependency-free plot of the same recorded fixed-training-set checks."""
    if not history:
        return
    series = [("Training objective", "#c44", [r["circuit_balanced_objective"] for r in history]),
              ("Training pooled AUC", "#167", [r["training_maximum_f1"]["roc_auc"] for r in history]),
              ("Training circuit-macro AUC", "#282", [r["circuit_macro_roc_auc"] for r in history])]
    steps = [r["total_step"] for r in history]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="650" viewBox="0 0 800 650">',
           '<rect width="800" height="650" fill="white"/>',
           '<text x="70" y="22" font-size="15">Fixed training pairs only; not generalization metrics</text>']
    for i, (label, color, values) in enumerate(series):
        valid = [(s, v) for s, v in zip(steps, values) if v is not None]
        if not valid:
            continue
        lo, hi = min(v for _, v in valid), max(v for _, v in valid)
        span = max(hi - lo, 1e-6)
        top = 55 + i * 195
        points = " ".join(f"{75 + 650*(s-steps[0])/max(1,steps[-1]-steps[0]):.2f},{top+125-115*(v-lo)/span:.2f}"
                          for s, v in valid)
        svg += [f'<text x="75" y="{top-8}" font-size="14">{label} ({lo:.6f} .. {hi:.6f})</text>',
                f'<path d="M75 {top} V{top+130} H740" stroke="#888" fill="none"/>',
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="75" y="{top+150}" font-size="12">step {steps[0]}</text>',
                f'<text x="670" y="{top+150}" font-size="12">step {steps[-1]}</text>']
    svg.append('</svg>')
    temporary = directory / "training_curve.svg.tmp"
    temporary.write_text("\n".join(svg))
    temporary.replace(directory / "training_curve.svg")


def snapshot(head, optimizer, step, record):
    return {"head": cpu_copy(head.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
            "hidden": record["hidden"], "optimizer_steps": step,
            "sampler_position": step, "objective": control.OBJECTIVE,
            "optimizer_restore": record["optimizer_restore"],
            "source_run": record["source_run"], "encoder_loaded": False,
            "notice": original.NOTICE}


def load_fixed_train(samples, manifest, hidden, device):
    lookup = {row["circuit"]: row for row in manifest if row["split"] == "train"}
    train = []
    if set(lookup) != {row["circuit"] for row in samples["circuits"]}:
        raise ValueError("Saved fixed samples and original train circuit IDs disagree")
    for row in samples["circuits"]:
        meta = lookup[row["circuit"]]
        path = Path(meta["cache_file"])
        if not path.is_file():
            raise FileNotFoundError(f"缺失缓存：{path}；停止，不补跑编码器")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("hf_l2", "pairs", "labels", "original_pair_columns"):
            if key not in cache:
                raise ValueError(f"缺失缓存字段 {row['circuit']}/{key}；停止")
        columns = row["cache_pair_columns"]
        if not torch.equal(cache["original_pair_columns"][columns], row["original_pair_columns"]):
            raise ValueError(f"Fixed pair identity mismatch: {row['circuit']}")
        left, right = cache["pairs"][:, columns]
        labels = cache["labels"][columns]
        if (int(labels.sum()) != row["positive_count"]
                or labels.numel() - int(labels.sum()) != row["negative_count"]):
            raise ValueError(f"Fixed pair label count mismatch: {row['circuit']}")
        features = original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right]).detach()
        if features.shape != (labels.numel(), 2 * hidden):
            raise ValueError("Cached hf width does not match head")
        train.append({**row, "features": features.to(device), "labels": labels.to(device)})
    return train


@torch.no_grad()
def training_fit(head, train, step):
    head.eval()
    scores, labels, losses, offsets, auc_rows = [], [], [], [0], []
    for row in train:
        logits = head(row["features"]).squeeze(-1)
        finite(logits, "训练 logits")
        loss = control.circuit_loss(logits, row["labels"])
        finite(loss, "完整训练目标")
        losses.append(float(loss))
        scores.append(logits.cpu())
        labels.append(row["labels"].cpu())
        offsets.append(offsets[-1] + logits.numel())
        auc_rows.append({"circuit": row["circuit"], **pair_metrics(scores[-1], labels[-1], None)})
    all_scores, all_labels = torch.cat(scores), torch.cat(labels)
    threshold = select_threshold(all_scores, all_labels)
    result = {"notice": TRAIN_NOTICE, "total_step": step,
              "circuit_balanced_objective": math.fsum(losses) / len(losses),
              "training_maximum_f1": pair_metrics(all_scores, all_labels, threshold),
              **within_circuit_auc(auc_rows), "groups_by_original_valid_pairs": {}}
    for group in original.GROUPS:
        selected = [i for i, row in enumerate(train) if row["pair_count_group"] == group]
        s = torch.cat([scores[i] for i in selected]) if selected else all_scores[:0]
        y = torch.cat([labels[i] for i in selected]) if selected else all_labels[:0]
        local_threshold = select_threshold(s, y) if s.numel() else None
        result["groups_by_original_valid_pairs"][group] = {
            "circuit_count": len(selected), "sample_count": s.numel(),
            "circuit_balanced_objective": math.fsum(losses[i] for i in selected) / len(selected) if selected else None,
            "at_global_train_max_f1_threshold": pair_metrics(s, y, threshold),
            "group_training_maximum_f1": pair_metrics(s, y, local_threshold),
            **within_circuit_auc([auc_rows[i] for i in selected])}
    pooled = result["training_maximum_f1"]
    info(f"TRAIN FIT step={step}: objective={result['circuit_balanced_objective']:.8f} | "
         f"P/R/F1={pooled['precision']}/{pooled['recall']}/{pooled['f1']} | "
         f"pooled/macro AUC={pooled['roc_auc']}/{result['circuit_macro_roc_auc']}")
    for group, entry in result["groups_by_original_valid_pairs"].items():
        row = entry["group_training_maximum_f1"]
        info(f"TRAIN FIT step={step} group={group}: objective={entry['circuit_balanced_objective']} | "
             f"group-train max-F1 P/R/F1={row['precision']}/{row['recall']}/{row['f1']} | "
             f"pooled/macro AUC={row['roc_auc']}/{entry['circuit_macro_roc_auc']}")
    return result


@torch.no_grad()
def validation_once(head, manifest, saved, device, batch_size):
    head.eval()
    rows = [row for row in manifest if row["split"] == "val"]
    if [row["circuit"] for row in rows] != saved["circuits"]:
        raise ValueError("Validation cache and saved scores circuit order differ")
    parts = []
    for i, row in enumerate(rows):
        cache = torch.load(row["cache_file"], map_location="cpu", weights_only=False)
        a, b = saved["offsets"][i:i+2]
        if not torch.equal(cache["labels"], saved["labels"][a:b]):
            raise ValueError(f"Validation cache label order mismatch: {row['circuit']}")
        chunks = []
        for pairs in cache["pairs"].split(batch_size, dim=1):
            left, right = pairs
            features = original.pair_features(cache["hf_l2"][left], cache["hf_l2"][right])
            logits = head(features.to(device)).squeeze(-1)
            finite(logits, "验证 logits")
            chunks.append(logits.cpu())
        parts.append(torch.cat(chunks) if chunks else torch.empty(0))
    return torch.cat(parts)


def run(args):
    started = time.monotonic()
    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    source_record = json.loads((source / "run.json").read_text())
    fixed_source = source
    visited = set()
    while not (fixed_source / "training_samples.pt").is_file():
        if fixed_source in visited:
            raise ValueError("Cycle in saved source_run lineage")
        visited.add(fixed_source)
        fixed_source = Path(json.loads((fixed_source / "run.json").read_text())["source_run"]).resolve()
    previous = json.loads((fixed_source / "run.json").read_text())
    history = json.loads((fixed_source / "history.json").read_text())
    checkpoint_path = args.checkpoint or source / ("last.pt" if (source / "last.pt").is_file() else "last_head.pt")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    start_step = int(state["optimizer_steps"])
    max_step, wall_seconds = args.max_total_step, args.wall_seconds
    if (source_record["status"] != "completed" or state["objective"] != control.OBJECTIVE
            or (args.start_step is not None and start_step != args.start_step)):
        raise ValueError("Require completed circuit-balanced source at the specified starting step")
    if max_step <= start_step:
        raise ValueError("Maximum total step must exceed the source step")
    check_steps = ([start_step] + list(range((start_step // args.check_interval + 1) * args.check_interval,
                                           max_step + 1, args.check_interval)) if args.check_interval else
                   [step for step in CHECK_STEPS if start_step <= step <= max_step])
    check_steps = sorted(set([start_step, max_step, *check_steps]))
    if previous["entry_sha256"] != file_hash(Path(control.__file__)):
        raise ValueError("Circuit-balanced source changed since the original run")
    if previous["device"] != args.device or previous["lr"] != 1e-3:
        raise ValueError("Keep original device and lr=1e-3")
    optimizer_key = next((key for key in ("optimizer", "optimizer_state_dict") if key in state), None)
    trend = [{"steps": [i + 1, min(i + 10, len(history))],
              "mean_minibatch_loss": math.fsum(row["circuit_mean_bce"] for row in history[i:i+10]) / len(history[i:i+10])}
             for i in range(0, len(history), 10)]
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "preparing", "notice": original.NOTICE, "training_notice": TRAIN_NOTICE,
              "source_run": str(source), "fixed_sample_run": str(fixed_source),
              "source_checkpoint": str(checkpoint_path.resolve()), "source_checkpoint_sha256": file_hash(checkpoint_path),
              "entry_sha256": file_hash(Path(__file__)), "hidden": state["hidden"],
              "start_step": start_step, "maximum_additional_steps": max_step-start_step, "maximum_total_step": max_step,
              "wall_clock_seconds": wall_seconds, "check_steps": check_steps,
              "validation_time_reserve_seconds": min(20, wall_seconds * 0.1),
              "min_effective_objective_decrease": args.min_improvement, "patience_checkpoints": args.patience,
              "objective": control.OBJECTIVE, "lr": previous["lr"], "optimizer": "Adam",
              "optimizer_restore": "restored saved optimizer state" if optimizer_key else
                  "optimizer state not saved; Adam reinitialized at lr=1e-3; NOT an exact continuous resume",
              "old_loss_trend": trend,
              "trend_interpretation": "historical minibatches use different circuits; full fixed-sample objective measured at source step before updates",
              "best_selection": "strictly lowest full fixed-training-pair objective at scheduled checks; source step included; no validation selection",
              "effective_improvement_rule": "absolute decrease from previous best objective >= min_improvement; a smaller new minimum is saved but does not reset patience",
              "early_stop": "patience consecutive scheduled checkpoints without an effective training-objective decrease",
              "sampling": "reuse exact saved columns; continue original uniform circuit shuffle at source step+1",
              "budget_notice": "Reaching a step/time budget does not establish convergence",
              "encoder_loaded": False, "encoder_forward_count": 0, "test_evaluated": False,
              "validation_passes_started": 0, "validation_passes_completed": 0,
              "device": args.device, "group_threshold_rule": previous["group_threshold_rule"]}
    write_json(args.output / "run.json", record)
    atomic_save(state, args.output / f"source_step{start_step}.pt")
    safe_state = None
    fit_history, updates = [], []
    best_value, best_step, stale = math.inf, start_step, 0
    with direct_log(args.output / "console.log"):
        info(original.NOTICE)
        info(f"Old ten-step minibatch loss means: {trend}")
        info(record["optimizer_restore"])
        info(TRAIN_NOTICE)
        try:
            with wall_limit(started + wall_seconds):
                set_seed(previous["seed"])
                head = original.make_head(state["hidden"]).to(args.device)
                head.load_state_dict(state["head"], strict=True)
                optimizer = torch.optim.Adam(head.parameters(), lr=previous["lr"])
                if optimizer_key:
                    optimizer.load_state_dict(state[optimizer_key])
                    if any(group["lr"] != previous["lr"] for group in optimizer.param_groups):
                        raise ValueError("Saved optimizer learning rate differs from original")
                safe_state = snapshot(head, optimizer, start_step, record)
                atomic_save(safe_state, args.output / "last.pt")
                samples = torch.load(fixed_source / "training_samples.pt", map_location="cpu", weights_only=False)
                if samples["objective"] != control.OBJECTIVE:
                    raise ValueError("Saved objective differs")
                cache_source = Path(previous["source_run"])
                manifest = json.loads((cache_source / "cache_manifest.json").read_text())
                train = load_fixed_train(samples, manifest, state["hidden"], args.device)
                schedule = control.circuit_schedule(len(train), previous["circuits_per_step"], max_step, previous["seed"])
                schedule_names = [[train[i]["circuit"] for i in batch.tolist()] for batch in schedule]
                if schedule_names[:40] != samples["circuit_batches"]:
                    raise ValueError("Circuit draw prefix differs; refusing to change sampling")
                # A continuation's stored schedule is also checked through the
                # source step, preserving the partially completed shuffle sweep.
                if source != fixed_source:
                    old_schedule = json.loads((source / "circuit_schedule.json").read_text())
                    first = old_schedule["start_total_step"]
                    old_batches = old_schedule["batches"][:start_step-first+1]
                    if schedule_names[first-1:start_step] != old_batches:
                        raise ValueError("Continuation circuit schedule prefix differs")
                write_json(args.output / "circuit_schedule.json", {"start_total_step": start_step+1, "batches": schedule_names[start_step:]})

                def inspect_fit(step):
                    nonlocal best_value, best_step, stale
                    result = training_fit(head, train, step)
                    value = result["circuit_balanced_objective"]
                    improved = value < best_value
                    effective = improved and (best_value-value >= args.min_improvement)
                    if improved:
                        candidate = {**safe_state, "training_objective": value}
                        atomic_save(candidate, args.output / "best_train.pt")
                        best_value, best_step = value, step
                    stale = 0 if effective else stale+1
                    result.update(new_best=improved, effective_improvement=effective, no_improvement_intervals=stale)
                    fit_history.append(result)
                    write_json(args.output / f"train_step_{step}.json", result)
                    write_json(args.output / "training_fit_history.json", fit_history)
                    save_training_curve(args.output, fit_history)
                    atomic_save(safe_state, args.output / "last.pt")

                inspect_fit(start_step)
                stop_reason = "step_limit"
                for step in range(start_step+1, max_step + 1):
                    if time.monotonic() >= started + wall_seconds - record["validation_time_reserve_seconds"]:
                        stop_reason = "time_limit"
                        break
                    head.train()
                    optimizer.zero_grad(set_to_none=True)
                    local_losses = []
                    for i in schedule[step-1].tolist():
                        row = train[i]
                        logits = head(row["features"]).squeeze(-1)
                        finite(logits, "训练 logits")
                        local_losses.append(control.circuit_loss(logits, row["labels"]))
                    loss = torch.stack(local_losses).mean()
                    finite(loss, "训练损失")
                    loss.backward()
                    for parameter in head.parameters():
                        if parameter.grad is not None:
                            finite(parameter.grad, "分类头梯度")
                    optimizer.step()
                    for parameter in head.parameters():
                        finite(parameter, "分类头参数")
                    # Atomic assignment retains a complete prior snapshot if the
                    # wall timer interrupts an update or device-to-host copy.
                    safe_state = snapshot(head, optimizer, step, record)
                    updates.append({"total_step": step, "minibatch_circuit_mean_bce": float(loss.detach())})
                    if step % 20 == 0:
                        info(f"Head total_step={step}/{max_step} minibatch BCE={float(loss.detach()):.6f}")
                    if step in check_steps:
                        inspect_fit(step)
                        if stale >= args.patience:
                            stop_reason = "plateau"
                            break
                atomic_save(safe_state, args.output / "last.pt")
                write_json(args.output / "update_history.json", updates)
                record.update(status="final_validation", stop_reason=stop_reason,
                              completed_total_step=safe_state["optimizer_steps"],
                              best_training_step=best_step, best_training_objective=best_value)
                write_json(args.output / "run.json", record)
                best = torch.load(args.output / "best_train.pt", map_location="cpu", weights_only=False)
                head.load_state_dict(best["head"], strict=True)
                saved = torch.load(source / "validation_scores.pt", map_location="cpu", weights_only=False)
                record["validation_passes_started"] = 1
                write_json(args.output / "run.json", record)
                new_scores = validation_once(head, manifest, saved, args.device, original.BATCH_SIZE)
                record["validation_passes_completed"] = 1
                atomic_save({"continued_best_train_logits": new_scores, "labels": saved["labels"],
                             "offsets": saved["offsets"], "circuits": saved["circuits"],
                             "selected_total_step": best_step}, args.output / "validation_scores.pt")
                old_comparison = json.loads((source / "comparison.json").read_text())
                baseline_key = "circuit_balanced_head" if source == fixed_source else "continued_best_train_objective"
                if source != fixed_source and old_comparison["selected_total_step"] != start_step:
                    raise ValueError("Saved validation baseline is not the requested source step; no additional evaluation")
                methods = {"cosine_baseline": old_comparison["methods"]["cosine_baseline"],
                           f"circuit_balanced_head_step{start_step}": old_comparison["methods"][baseline_key]}
                methods["continued_best_train_objective"] = control.score_summary(
                    new_scores, saved["labels"], saved["offsets"], saved["circuits"], "logit")
                report = {"notice": original.NOTICE, "scope": old_comparison["scope"],
                          "selected_by": "fixed training objective only; validation evaluated once at end",
                          "selected_total_step": best_step, "last_total_step": safe_state["optimizer_steps"],
                          "stop_reason": stop_reason, "budget_notice": record["budget_notice"],
                          "optimizer_restore": record["optimizer_restore"],
                          "group_threshold_rule": previous["group_threshold_rule"], "methods": methods}
                write_json(args.output / "comparison.json", report)
                for method, result in methods.items():
                    control.print_summary(method, result)
                    for tag, metrics in [("pooled", result["maximum_f1"]),
                                         *((group, entry["at_global_max_f1_threshold"])
                                           for group, entry in result["groups_by_original_valid_pairs"].items())]:
                        info(f"{method} / {tag}: max-F1 TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
                record["status"] = "completed"
                info(f"Completed bounded head continuation: {args.output / 'comparison.json'}")
        except (WallLimit, NonFinite) as error:
            # Disarm before saving; finalization never restarts training/evaluation.
            signal.setitimer(signal.ITIMER_REAL, 0)
            if safe_state is not None:
                atomic_save(safe_state, args.output / "last.pt")
            write_json(args.output / "update_history.json", updates)
            record.update(status="stopped_wall_clock" if isinstance(error, WallLimit) else "stopped_nonfinite",
                          stop_reason="time_limit" if isinstance(error, WallLimit) else "nonfinite",
                          stop_detail=str(error), completed_total_step=safe_state["optimizer_steps"] if safe_state else start_step,
                          best_training_step=best_step if math.isfinite(best_value) else None,
                          best_training_objective=best_value if math.isfinite(best_value) else None,
                          validation_not_restarted=True)
            info(str(error))
        record["elapsed_wall_seconds"] = time.monotonic() - started
        write_json(args.output / "run.json", record)
        info(f"status={record['status']} | stop_reason={record.get('stop_reason')} | last total step={record.get('completed_total_step', start_step)} | "
             f"best training step={record.get('best_training_step')} | elapsed={record['elapsed_wall_seconds']:.1f}s")
        info(record["budget_notice"])
        info(original.NOTICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--max-total-step", type=int, default=MAX_STEP)
    parser.add_argument("--wall-seconds", type=int, default=WALL_SECONDS)
    parser.add_argument("--check-interval", type=int)
    parser.add_argument("--min-improvement", type=float, default=0.)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_total_step <= 5000 or not 1 <= args.wall_seconds <= WALL_SECONDS:
        parser.error("total step must be 1..5000 and wall budget must be 1..900 seconds")
    if ((args.check_interval is not None and args.check_interval < 1) or args.patience < 1
            or not math.isfinite(args.min_improvement) or args.min_improvement < 0):
        parser.error("check interval and patience must be positive; min improvement must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
