"""Prepare or run isolated stage-3 functional pair sampling experiments.

No action is implicit: --prepare writes plans, --self-check uses temporary
models, and only --mode starts up to five fine-tuning epochs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.compare_functional_stages import cosine_scores64, within_circuit_auc
from examples.evaluate import RUN_DIR, digest_file, validate_splits
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.model import TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS, collate_circuits
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint, random_state
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.equivalence_audit import pair_metrics, pair_count_group_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import AverageMeter, BatchProgress, info
from triwave_gnn.utils.losses import BEHAVIOR, STAGE_WEIGHTS, loss_terms
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.pair_sampling import MODES, STRATA, sample_functional_pairs, sampling_summary, strata_masks
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.utils.shared_validation import baseline_lock


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class Tee:
    """Write terminal output and a directly flushed log in the same process."""
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, value):
        self.terminal.write(value)
        self.stream.write(value)
        self.stream.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.stream.flush()

    def isatty(self):
        # Preserve normal terminal output; the log also captures the panel.
        return self.terminal.isatty()

    def __getattr__(self, key):
        return getattr(self.terminal, key)


@contextmanager
def direct_log(path):
    stdout, stderr = sys.stdout, sys.stderr
    with path.open("a", buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
        try:
            yield
        except BaseException as error:
            traceback.print_exc()
            record_path = path.parent / "run.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
                record.update(status="failed", error=repr(error))
                write_json(record_path, record)
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def epoch_orders(names, seed, epochs):
    generator = random.Random(seed)
    result = []
    for _ in range(epochs):
        order = list(names)
        generator.shuffle(order)
        result.append(order)
    return result


def make_optimizer(model, source, lr):
    # Match original Adam hyperparameters, but never restore optimizer moments.
    group = source["optimizer"]["param_groups"][0]
    options = {key: group[key] for key in ("betas", "eps", "weight_decay", "amsgrad") if key in group}
    return torch.optim.Adam(model.parameters(), lr=lr, **options)


def data_distribution(data, splits):
    """Count current labels only; no regeneration or endpoint/source re-audit."""
    with np.load(data / "labels.npz", allow_pickle=False) as archive:
        ids = [str(name) for name in archive["design_ids"]]
        ptr, distance, mask = archive["hf_tt_pair_ptr"], archive["hf_tt_distance"], archive["hf_tt_pair_mask"]
    lookup = {name: index for index, name in enumerate(ids)}
    result = {}
    for split in ("train", "val"):
        names = splits["train"] if split == "train" else splits.get("val", splits.get("validation"))
        counts = dict.fromkeys(STRATA, 0)
        for name in names:
            i = lookup[name]
            start, stop = map(int, ptr[i:i + 2])
            t = distance[start:stop]
            t = t[mask[start:stop] & np.isfinite(t)]
            positive = np.isclose(t, 0, atol=1e-8, rtol=0)
            near = ~positive & (t > 0) & (t <= .2)
            far = ~positive & (t >= .8)
            for key, values in zip(STRATA, (positive, near, far, ~(positive | near | far))):
                counts[key] += int(values.sum())
        total = sum(counts.values())
        result[split] = {"counts": counts, "valid_pairs": total,
                         "ratios": {key: value / total if total else None for key, value in counts.items()}}
    return result


def experiment_plan(args, source, checkpoint_hash, distribution, mode):
    config = source["config"]
    return {
        "status": "prepared", "mode": mode, "data": str(args.data.resolve()),
        "initial_checkpoint": str(args.checkpoint.resolve()), "initial_checkpoint_sha256": checkpoint_hash,
        "source_config": config, "source_epoch": source["epoch"],
        "inherit": ["architecture", "weights", "vocabularies", "split_ids", "feature_normalization"],
        "restore_optimizer_or_resume_progress": False, "first_epoch": 1,
        "lr": args.lr, "epochs": args.epochs, "batch_size": config["batch_size"],
        "seed": config["seed"], "num_workers": config["num_workers"],
        "lr_step": config.get("lr_step", 20),
        "sampler_seed": config["seed"] + 1_000_003, "sampler_epoch_seed": "sampler_seed + epoch - 1",
        "cap": args.cap, "stratum_target_ratio": {"zero": 1, "near": 1, "far": 2},
        "other_distance_policy": "reserve rounded original fraction of per-circuit budget, at least one; retain every nonempty stratum; remaining main strata weights 1:1:2 renormalized over nonempty strata",
        "replacement_policy": "stratified only: cover every available pair in an undersupplied stratum before replacement draws",
        "stratum_definitions": {"zero": "isclose(TT, 0, atol=1e-8, rtol=0)",
                                "near": "not zero and 0 < TT <= 0.2", "far": "not zero and TT >= 0.8",
                                "other": "all remaining valid distances (normally 0.2 < TT < 0.8)"},
        "loss": "unchanged triwave_gnn.utils.losses.loss_terms(output, loss_view, stage=3)",
        "functional_loss_chunking": "cosine chunks recomputed in backward, concatenated before original whole-batch zero_normalize; no gradient isolation or per-chunk normalization",
        "unique_pair_definition": "unique original hf pair columns selected within each circuit; duplicate_rate = (sampled - unique) / sampled",
        "task_weights": STAGE_WEIGHTS[3], "gradient_clip_norm": 1.,
        "validation": "all original pairs, float64 embeddings before cosine, one validation-only global maximum-F1 threshold per epoch",
        "validation_loss": "original float32 loss on full validation pairs with original batch size and order",
        "best_loss_selection": "minimum full-pair validation mean loss; baseline epoch 0 included",
        "best_equiv_selection": "maximum pooled validation F1; baseline epoch 0 included; keep earlier epoch on ties",
        "evaluation_metrics": ["precision", "recall", "f1", "roc_auc", "circuit_macro_roc_auc",
                               "within_circuit_weighted_auc", "prob_mae", "rc_accuracy", "tt_distance_mae",
                               "hb_tp_mae", "hb_first_mean_mae", "hb_stable_mean_mae", "hb_glitch_rate_mae"],
        "data_distribution": distribution, "test_evaluation": False,
    }


def prepare(args, source):
    splits = source["split_ids"]
    distribution = data_distribution(args.data, splits)
    checkpoint_hash = digest_file(args.checkpoint)
    orders = epoch_orders(splits["train"], source["config"]["seed"], args.epochs)
    for mode in MODES:
        directory = args.experiment_root / mode
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite prepared/results directory: {directory}")
    for mode in MODES:
        directory = args.experiment_root / mode
        directory.mkdir(parents=True, exist_ok=True)
        plan = experiment_plan(args, source, checkpoint_hash, distribution, mode)
        write_json(directory / "plan.json", plan)
        write_json(directory / "split.json", splits)
        write_json(directory / "epoch_orders.json", orders)
        info(f"Prepared {mode}: {directory}")
    write_json(args.experiment_root / "pair_distribution.json", distribution)
    for split, row in distribution.items():
        info(f"{split} original valid pair distribution: {row}")


def raw_metric_sums(output, item, scores):
    """Original prediction transforms and masks; CPU float64 accumulation."""
    result = {}

    def add(key, prediction, target, mask):
        valid = mask.bool() & torch.isfinite(target)
        if bool(valid.any()):
            error = (prediction[valid].double() - target[valid].double()).abs()
            if not bool(torch.isfinite(error).all()):
                raise ValueError(f"Nonfinite {key} prediction")
            result[key] = (float(error.sum()), error.numel())

    pins = item["pin"]
    add("prob_mae", output["hf_prob_logits"].sigmoid(), pins.hf_prob, torch.ones_like(pins.hf_prob, dtype=torch.bool))
    mask = item["hs_mask"].bool()
    if bool(mask.any()):
        correct = (output["hs_rc_logits"][mask].sigmoid() >= .5) == (item["hs_target"][mask] >= .5)
        result["rc_accuracy"] = (int(correct.sum()), correct.numel())
    for j, (key, mask_key, kind) in enumerate(BEHAVIOR):
        raw = output["hb_raw"][:, j]
        prediction = raw.sigmoid() if kind == "probability" else torch.expm1(raw).clamp_min(0)
        add(key + "_mae", prediction, pins[key], pins[mask_key])
    valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
    target = item["hf_tt_target"][valid].cpu().double()
    result["tt_distance_mae"] = (float(((1 - scores) - target).abs().sum()), target.numel())
    return result


def large_circuit_metrics(rows):
    eligible = [row for row in rows if row["positive_count"] + row["negative_count"] >= 10_000]
    positive = sum(row["positive_count"] for row in eligible)
    negative = sum(row["negative_count"] for row in eligible)
    available = bool(eligible) and all(row["tp"] is not None for row in eligible)
    tp = sum(row["tp"] for row in eligible) if available else None
    fp = sum(row["fp"] for row in eligible) if available else None
    return {"definition": "circuits with >=10000 original valid functional pairs; pooled group confusion counts at global validation threshold",
            "circuit_count": len(eligible), "positive_count": positive, "negative_count": negative,
            "tp": tp, "fp": fp, "false_positive_rate": fp / negative if available and negative else None,
            "recall": tp / positive if available and positive else None}


def evaluate_validation(model, dataset, names, indices, args, epoch, include_pair_count_groups=False):
    model.eval()
    loader = DataLoader(Subset(dataset, [indices[name] for name in names]), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(args.seed))
    score_parts, label_parts, offsets, rows = [], [], [0], []
    sums, counts, terms, loss_meter = {}, {}, {}, AverageMeter()
    with torch.inference_mode(), BatchProgress("val", 3, epoch, len(loader), args.progress_interval,
                                               epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            item = move(batch, args.device)
            output = model_outputs(model, item)
            loss, local_terms = loss_terms(output, item, 3, functional_chunk_size=args.chunk_size)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite full-pair validation loss")
            names_batch = item.name
            loss_meter.update(float(loss), len(names_batch))
            for key, value in local_terms.items():
                terms.setdefault(key, AverageMeter()).update(value, len(names_batch))
            valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
            scores = cosine_scores64(output["hf"], item["hf_pairs"], valid, args.chunk_size)
            target = item["hf_tt_target"][valid].cpu()
            labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
            groups = item["pin"].batch[item["hf_pairs"][0, valid]].cpu()
            for circuit, name in enumerate(names_batch):
                group = groups == circuit
                local = scores[group]
                score_parts.append(local)
                label_parts.append(labels[group])
                offsets.append(offsets[-1] + local.numel())
                rows.append({"circuit": name, "original_valid_pairs": local.numel()})
            for key, (value, count) in raw_metric_sums(output, item, scores).items():
                sums[key] = sums.get(key, 0.) + value
                counts[key] = counts.get(key, 0) + count
            progress.update(number, loss_meter.avg, {}, {key: sums[key] / counts[key]
                                                        for key in sums if counts[key]})
            del item, batch, output, loss
    scores, labels = torch.cat(score_parts), torch.cat(label_parts)
    threshold = select_threshold(scores, labels)
    pooled = pair_metrics(scores, labels, threshold)
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        row.update(pair_metrics(scores[start:stop], labels[start:stop], threshold))
    grouped = {"groups_by_original_valid_pairs": pair_count_group_metrics(scores, labels, offsets, threshold)} \
        if include_pair_count_groups else {}
    return {"epoch": epoch, "mean_loss": loss_meter.avg, **grouped,
            "terms": {key: meter.avg for key, meter in terms.items()},
            "metrics": {key: sums[key] / counts[key] if counts[key] else None for key in sums},
            "metric_counts": counts, "functional_equivalence": pooled,
            **within_circuit_auc(rows), "large_pair_circuits": large_circuit_metrics(rows),
            "validation_sampling": "none: all original valid functional pairs", "per_circuit": rows}


def print_validation(result, label):
    pooled = result["functional_equivalence"]
    fields = {key: pooled[key] for key in ("precision", "recall", "f1", "roc_auc", "positive_count", "negative_count", "similarity_threshold")}
    fields.update({key: result[key] for key in ("circuit_macro_roc_auc", "macro_participating_circuits", "within_circuit_weighted_auc")})
    info(label + " equivalence: " + " | ".join(f"{key}={value}" for key, value in fields.items()))
    info(label + " metrics: " + " | ".join(f"{key}={value}" for key, value in result["metrics"].items()))
    info(label + f" full-pair loss={result['mean_loss']} | >=10000 pair circuits: {result['large_pair_circuits']}")


def train_epoch(model, optimizer, dataset, order, indices, args, epoch, run_dir, sampling_reference=None):
    model.train()
    generator = torch.Generator().manual_seed(args.sampler_seed + epoch - 1)
    loader = DataLoader(Subset(dataset, [indices[name] for name in order]), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(args.seed + epoch))
    all_rows, loss_meter, terms = [], AverageMeter(), {}
    def expected_batches():
        with sampling_reference.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["epoch"] == epoch:
                    yield record
    expected = iter(expected_batches()) if sampling_reference is not None else None
    with (run_dir / "sampling.jsonl").open("a", buffering=1) as stats_stream, \
         BatchProgress("train", 3, epoch, len(loader), args.progress_interval, epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            view, _, rows = sample_functional_pairs(batch, args.mode, generator, args.cap)
            if expected is not None:
                reference = next(expected, None)
                if (reference is None or reference["batch"] != number or rows != reference["circuits"]):
                    raise ValueError(f"B graph/pair sampling differs at epoch={epoch}, batch={number}")
            # Only the functional loss view is sampled. All readouts use full graphs.
            item = move(batch, args.device)
            loss_item = copy(item)
            for key in ("hf_pairs", "hf_tt_target", "hf_tt_mask"):
                loss_item[key] = view[key].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model_outputs(model, item)
            total, local_terms = loss_terms(output, loss_item, 3, functional_chunk_size=args.chunk_size)
            if not model.training or not total.requires_grad or not bool(torch.isfinite(total)):
                raise RuntimeError("Fine-tuning must execute a finite differentiable train-mode loss")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_meter.update(float(total.detach()), len(batch.name))
            for key, value in local_terms.items():
                terms.setdefault(key, AverageMeter()).update(value, len(batch.name))
            record = {"epoch": epoch, "batch": number, "lr": [group["lr"] for group in optimizer.param_groups],
                      "loss": float(total.detach()), "terms": local_terms, "gradient_norm": float(grad_norm),
                      "sampling": sampling_summary(rows), "circuits": rows}
            if expected is not None:
                record["B_pair_columns_match"] = True
            stats_stream.write(json.dumps(record, allow_nan=False) + "\n")
            all_rows.extend(rows)
            progress.update(number, loss_meter.avg, {key: meter.avg for key, meter in terms.items()}, {})
            del output, total, item, loss_item, batch, view
    if expected is not None and next(expected, None) is not None:
        raise ValueError("Fewer training batches than B sampling reference")
    return {"mean_loss": loss_meter.avg, "terms": {key: meter.avg for key, meter in terms.items()},
            "sampling": sampling_summary(all_rows), "optimizer_steps": len(loader),
            "actual_lr": [group["lr"] for group in optimizer.param_groups],
            "sampler_state": generator.get_state()}


def save_checkpoint(path, model, optimizer, source, args, epoch, validation, sampler_state=None):
    state = {key: deepcopy(source[key]) for key in
             ("architecture_id", "logic_vocab", "role_vocab", "pin_vocab", "source_sha256", "split_ids", "feature_normalization")}
    config = dict(source["config"], lr=args.lr, output=str(args.experiment_root / args.mode), resume=None)
    state.update({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                  "stage": 3, "epoch": epoch, "stage_complete": False,
                  "finetune_mode": args.mode, "finetune_epochs": args.epochs,
                  "initial_checkpoint": str(args.checkpoint), "sampler_state": sampler_state,
                  "validation_loss": validation["mean_loss"],
                  "validation_f1": validation["functional_equivalence"]["f1"],
                  "validation_threshold": validation["functional_equivalence"]["similarity_threshold"],
                  "random_state": random_state()})
    atomic_save(state, path)


def run(args):
    run_dir = args.experiment_root / args.mode
    plan = json.loads((run_dir / "plan.json").read_text())
    if plan["status"] != "prepared" or (run_dir / "run.json").exists():
        raise FileExistsError(f"This experiment has already been started: {run_dir}")
    if digest_file(args.checkpoint) != plan["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint differs from the prepared plan")
    if args.data.resolve() != Path(plan["data"]) or args.checkpoint.resolve() != Path(plan["initial_checkpoint"]):
        raise ValueError("Data/checkpoint paths differ from the prepared plan")
    for key in ("lr", "epochs", "cap"):
        if getattr(args, key) != plan[key]:
            raise ValueError(f"{key} differs from the prepared plan")
    args.batch_size, args.seed = plan["batch_size"], plan["seed"]
    args.num_workers, args.sampler_seed = plan["num_workers"], plan["sampler_seed"]
    with direct_log(run_dir / "console.log"):
        record = dict(plan, status="running", device=args.device, evaluation_chunk_size=args.chunk_size)
        write_json(run_dir / "run.json", record)
        info(f"Fine-tuning {args.mode} | output={run_dir} | lr={args.lr} | batch_size={args.batch_size} | seed={args.seed}")
        info("Fresh optimizer and epoch 1; initialization loads model weights only; no resume-stage logic")
        set_seed(args.seed)
        dataset = CircuitDataset(args.data, cache_mode="small")
        model, source = load_model(args.checkpoint, dataset, args.device)
        if source["source_sha256"] != dataset.source_sha256:
            raise ValueError("Data files differ from the source experiment")
        splits = json.loads((run_dir / "split.json").read_text())
        val_key = validate_splits(splits, source)
        orders = json.loads((run_dir / "epoch_orders.json").read_text())
        if orders != epoch_orders(splits["train"], args.seed, args.epochs):
            raise ValueError("Prepared training orders differ")
        indices = {name: i for i, name in enumerate(dataset.ids)}
        optimizer = make_optimizer(model, source, args.lr)
        if optimizer.state:
            raise RuntimeError("Fresh optimizer unexpectedly has moments")
        baseline_path = args.experiment_root / "baseline_stage3.json"
        baseline_identity = {"checkpoint_sha256": plan["initial_checkpoint_sha256"],
                             "source_sha256": dataset.source_sha256,
                             "validation_ids_sha256": hashlib.sha256(json.dumps(splits[val_key]).encode()).hexdigest(),
                             "batch_size": args.batch_size, "device": args.device,
                             "evaluation_code_sha256": digest_file(Path(__file__)),
                             "float64_helper_sha256": digest_file(Path(__file__).with_name("compare_functional_stages.py"))}
        with baseline_lock(args.experiment_root / ".baseline.lock", notify=info):
            if baseline_path.exists():
                shared = json.loads(baseline_path.read_text())
                if shared["identity"] != baseline_identity:
                    raise ValueError("Shared baseline identity differs; use a new experiment root")
                baseline = shared["result"]
                info(f"Reusing identical pre-finetune full-validation baseline: {baseline_path}")
            else:
                baseline = evaluate_validation(model, dataset, splits[val_key], indices, args, 0)
                write_json(baseline_path, {"identity": baseline_identity, "result": baseline})
        write_json(run_dir / "baseline.json", baseline)
        print_validation(baseline, "BASELINE")
        best_loss, best_f1 = baseline["mean_loss"], baseline["functional_equivalence"]["f1"]
        for filename in ("best_loss.pt", "best_equiv.pt"):
            save_checkpoint(run_dir / filename, model, optimizer, source, args, 0, baseline)
        history = []
        for epoch, order in enumerate(orders, 1):
            info(f"Epoch {epoch}/{args.epochs} | actual_lr={[group['lr'] for group in optimizer.param_groups]}")
            train = train_epoch(model, optimizer, dataset, order, indices, args, epoch, run_dir)
            validation = evaluate_validation(model, dataset, splits[val_key], indices, args, epoch)
            write_json(run_dir / f"val_epoch_{epoch}.json", validation)
            sampler_state = train.pop("sampler_state")
            if validation["mean_loss"] < best_loss:
                best_loss = validation["mean_loss"]
                save_checkpoint(run_dir / "best_loss.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            f1 = validation["functional_equivalence"]["f1"]
            if f1 is not None and (best_f1 is None or f1 > best_f1):
                best_f1 = f1
                save_checkpoint(run_dir / "best_equiv.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            save_checkpoint(run_dir / "last.pt", model, optimizer, source, args, epoch, validation, sampler_state)
            history.append({"epoch": epoch, "train": train,
                            "val": {key: value for key, value in validation.items() if key != "per_circuit"}})
            write_json(run_dir / "history.json", history)
            info(f"Epoch {epoch} sampling: {train['sampling']}")
            print_validation(validation, f"VAL epoch={epoch}")
            lr_step = plan["lr_step"]
            if lr_step and epoch % lr_step == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
        record.update(status="completed", completed_epochs=args.epochs, best_loss=best_loss, best_f1=best_f1)
        write_json(run_dir / "run.json", record)
        info(f"Completed {args.mode}; best_loss.pt / best_equiv.pt / history.json | test not evaluated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=RUN_DIR / "stage3_best.pt")
    parser.add_argument("--experiment-root", type=Path, default=RUN_DIR.parent / "functional_pair_sampling_stage3")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cap", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if not (args.prepare or args.self_check or args.mode):
        parser.error("Choose --prepare, --self-check or --mode; training never starts implicitly")
    if args.mode and (args.prepare or args.self_check):
        parser.error("Run mode cannot be combined with preparation/self-check")
    if not 1 <= args.epochs <= 5 or args.lr <= 0 or min(args.cap, args.chunk_size) < 1 or args.progress_interval < 0:
        parser.error("Require 1..5 epochs, positive lr/cap/chunk-size, nonnegative progress interval")
    if args.mode:
        run(args)
        return
    source = load_checkpoint(args.checkpoint, "cpu")
    if source.get("stage") != 3:
        raise ValueError("Experiments require the source stage3_best.pt")
    if args.self_check:
        from triwave_gnn.tests.test_pair_finetune import self_check
        info("SELF-CHECK only: temporary in-memory models; no formal experiment training")
        source_hash = digest_file(args.checkpoint)
        result = self_check(source, args.device)
        if digest_file(args.checkpoint) != source_hash:
            raise RuntimeError("Source checkpoint changed during self-check")
        result["initial_checkpoint_sha256"] = source_hash
        args.experiment_root.mkdir(parents=True, exist_ok=True)
        write_json(args.experiment_root / "self_check.json", result)
        info(f"Self-check passed: {result}")
    if args.prepare:
        prepare(args, source)


if __name__ == "__main__":
    main()
