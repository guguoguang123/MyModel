"""K2 direct H_f cosine metric fine-tuning.

This file is deliberately separate from the historical K0/K1 script.  It
never creates an auxiliary classification head and it never evaluates test.
Training starts only when ``--run`` is supplied explicitly; ``--dry-run``
loads the source checkpoint and optimizer state without a forward pass.
"""

from __future__ import annotations

from copy import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.compare_functional_stages import cosine_scores64, within_circuit_auc
from examples.finetune_functional_pairs import large_circuit_metrics, raw_metric_sums
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint, random_state
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.equivalence_audit import pair_count_group_metrics, pair_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.losses import loss_terms
from triwave_gnn.utils.logger import AverageMeter, BatchProgress, info
from triwave_gnn.utils.metric_loss import direct_metric_loss_from_embeddings
from triwave_gnn.utils.metric_pair_sampling import sample_metric_indices
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.pair_sampling import sample_functional_pairs, sampling_summary
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.model import TriWaveGNN
from triwave_gnn.utils.net_identity import checkpoint_net_identity


SOURCE_CHECKPOINT_SHA256 = "6c38b549173a9d3bf4099c2665ab4898433496c7631e722f5390ad6a807f548a"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "exp/functional_pair_sampling_stage3/circuit_cap/best_equiv.pt"
DEFAULT_ORDERS = Path(__file__).resolve().parent / "exp/functional_pair_sampling_stage3/circuit_cap/epoch_orders.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "exp/functional_equivalence_metric_K2_20261117"
GROUP_QUOTA = {"1-999": 8, "1000-9999": 8, ">=10000": 16}
LAMBDA_BY_EPOCH = {1: 0.01, 2: 0.03, 3: 0.05, 4: 0.05, 5: 0.05}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def restore_source_optimizer(model, source, device):
    """Restore the source Adam moments for exactly the original model parameters."""
    if source.get("auxiliary_head") is not None:
        raise ValueError("K2 source checkpoint unexpectedly contains an auxiliary head")
    source_group = source["optimizer"]["param_groups"][0]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=source_group["lr"],
        betas=tuple(source_group["betas"]),
        eps=source_group["eps"],
        weight_decay=source_group["weight_decay"],
        amsgrad=source_group.get("amsgrad", False),
    )
    optimizer.load_state_dict(source["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
    if len(list(model.parameters())) != len(source_group["params"]):
        raise ValueError("source optimizer parameter layout differs from K2 model")
    if not optimizer.state:
        raise ValueError("source checkpoint has no Adam state to restore")
    return optimizer


def validate_orders(orders, train_names, epochs: int) -> None:
    if len(orders) != epochs or any(len(order) != len(train_names) for order in orders):
        raise ValueError("epoch_orders.json does not match the fixed train split or epoch count")
    expected = set(train_names)
    if any(set(order) != expected for order in orders):
        raise ValueError("epoch order contains IDs outside the fixed train split")


def minimum_fp(scores, labels, target_tp, mask=None):
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    scores, labels = scores[mask], labels[mask].bool()
    positives = int(labels.sum())
    if positives < target_tp:
        return {"target_tp": target_tp, "status": "undefined: insufficient positives",
                "available_positives": positives}
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores, ordered_labels = scores[order], labels[order].to(torch.int64)
    tp = ordered_labels.cumsum(0)
    fp = (1 - ordered_labels).cumsum(0)
    ends = torch.ones_like(ordered_labels, dtype=torch.bool)
    if ends.numel() > 1:
        ends[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    candidates = torch.nonzero(ends & (tp >= target_tp), as_tuple=False).flatten()
    choice = candidates[torch.argmin(fp[candidates])]
    tpi, fpi = int(tp[choice]), int(fp[choice])
    return {"target_tp": target_tp, "tp": tpi, "fp": fpi, "fn": positives - tpi,
            "recall": tpi / positives, "precision": tpi / (tpi + fpi) if tpi + fpi else None,
            "threshold": float(ordered_scores[choice])}


def evaluate_full_val(model, dataset, val_names, indices, args, epoch, config_hash):
    """Full fixed val protocol; no sampling and no test access."""
    model.eval()
    loader = DataLoader(Subset(dataset, [indices[name] for name in val_names]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                        exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(args.seed))
    score_parts, label_parts, offsets, rows = [], [], [0], []
    sums, counts, terms, loss_meter = {}, {}, {}, AverageMeter()
    with torch.inference_mode(), BatchProgress("K2 val", 3, epoch, len(loader), args.progress_interval,
                                               epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            item = move(batch, args.device)
            output = model_outputs(model, item)
            loss, local_terms = loss_terms(output, item, 3, functional_chunk_size=args.chunk_size)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite full-validation base loss")
            loss_meter.update(float(loss), len(batch.name))
            for key, value in local_terms.items():
                terms.setdefault(key, AverageMeter()).update(value, len(batch.name))
            valid = item["hf_tt_mask"].bool() & torch.isfinite(item["hf_tt_target"])
            scores = cosine_scores64(output["hf"], item["hf_pairs"], valid, args.chunk_size)
            target = item["hf_tt_target"][valid].cpu()
            labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
            owners = item["pin"].batch[item["hf_pairs"][0, valid]].cpu()
            for circuit, name in enumerate(batch.name):
                local = owners == circuit
                local_scores, local_labels = scores[local], labels[local]
                score_parts.append(local_scores); label_parts.append(local_labels)
                offsets.append(offsets[-1] + local_scores.numel())
                rows.append({"circuit": name, "original_valid_pairs": int(local_scores.numel())})
            for key, (value, count) in raw_metric_sums(output, item, scores).items():
                sums[key] = sums.get(key, 0.) + value
                counts[key] = counts.get(key, 0) + count
            progress.update(number, loss_meter.avg, {}, {key: sums[key] / counts[key] for key in sums if counts[key]})
            del batch, item, output, loss
    scores, labels = torch.cat(score_parts), torch.cat(label_parts)
    threshold = select_threshold(scores, labels)
    pooled = pair_metrics(scores, labels, threshold)
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        row.update(pair_metrics(scores[start:stop], labels[start:stop], threshold))
    groups = pair_count_group_metrics(scores, labels, offsets, threshold)
    dense_mask = torch.zeros(scores.numel(), dtype=torch.bool)
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        if row["original_valid_pairs"] >= 10000:
            dense_mask[start:stop] = True
    result = {
        "epoch": epoch, "config_sha256": config_hash, "mean_loss": loss_meter.avg,
        "terms": {key: meter.avg for key, meter in terms.items()},
        "metrics": {key: sums[key] / counts[key] if counts[key] else None for key in sums},
        "metric_counts": counts, "functional_equivalence": pooled,
        "groups_by_original_valid_pairs": groups,
        "dense_group_auc": groups[">=10000"]["roc_auc"],
        "minimum_fp_dense_tp219": minimum_fp(scores, labels, 219, dense_mask),
        "minimum_fp_overall_tp8025": minimum_fp(scores, labels, 8025),
        **within_circuit_auc(rows), "large_pair_circuits": large_circuit_metrics(rows),
        "validation_sampling": "none: all original valid functional pairs; H_f cosine score",
        "test_read": False, "per_circuit": rows,
    }
    return result


def train_epoch(model, optimizer, dataset, order, indices, args, epoch, output_dir, config_hash):
    model.train()
    base_rng = torch.Generator().manual_seed(args.sampler_seed + epoch - 1)
    metric_rng = torch.Generator().manual_seed(args.metric_seed + epoch - 1)
    loader = DataLoader(Subset(dataset, [indices[name] for name in order]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                        exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(args.seed + epoch))
    loss_meter, metric_meter, term_meters = AverageMeter(), AverageMeter(), {}
    rows = []
    lambda_metric = LAMBDA_BY_EPOCH[epoch]
    hard_fraction = 0.0 if epoch == 1 else args.hard_negative_fraction
    with (output_dir / "sampling.jsonl").open("a", buffering=1) as stream, \
         BatchProgress("K2 train", 3, epoch, len(loader), args.progress_interval,
                       epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            base_view, _, base_rows = sample_functional_pairs(batch, "circuit_cap", base_rng, args.cap)
            item = move(batch, args.device)
            loss_item = copy(item)
            for key in ("hf_pairs", "hf_tt_target", "hf_tt_mask"):
                loss_item[key] = base_view[key].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model_outputs(model, item)
            output["hf"].retain_grad()
            original, local_terms = loss_terms(output, loss_item, 3, functional_chunk_size=args.chunk_size)

            valid_cpu = item["hf_tt_mask"].detach().cpu().bool() & torch.isfinite(item["hf_tt_target"].detach().cpu())
            pairs_cpu = item["hf_pairs"].detach().cpu()
            target_cpu = item["hf_tt_target"].detach().cpu()
            owner_cpu = item["pin"].batch.detach().cpu()
            circuit_cpu = owner_cpu[pairs_cpu[0]]
            train_scores = None
            if epoch >= 2:
                candidate_scores = cosine_scores64(output["hf"], item["hf_pairs"], valid_cpu.to(args.device), args.chunk_size)
                train_scores = torch.zeros(target_cpu.numel(), dtype=torch.float64)
                train_scores[torch.nonzero(valid_cpu, as_tuple=True)[0]] = candidate_scores
            sampled = sample_metric_indices(
                pairs_cpu, target_cpu, valid_cpu, circuit_cpu, metric_rng,
                group_quota=GROUP_QUOTA, per_circuit_cap=args.metric_per_circuit_cap,
                scores=train_scores, hard_negative_fraction=hard_fraction,
                split_name="train", node_owner=owner_cpu,
            )
            positive_pairs = item["hf_pairs"][:, sampled.positive_indices.to(args.device)]
            negative_pairs = item["hf_pairs"][:, sampled.negative_indices.to(args.device)]
            metric_loss = direct_metric_loss_from_embeddings(
                output["hf"], positive_pairs, negative_pairs, args.margin, args.temperature)
            total = original + lambda_metric * metric_loss
            if not total.requires_grad or not bool(torch.isfinite(total)) or not bool(torch.isfinite(metric_loss)):
                raise FloatingPointError(f"non-finite K2 loss at epoch={epoch}, batch={number}")
            total.backward()
            hf_grad = output["hf"].grad
            if hf_grad is None or not bool(torch.isfinite(hf_grad).all()) or float(hf_grad.abs().sum()) == 0.0:
                raise RuntimeError("K2 metric loss produced no finite nonzero H_f gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm,
                                                       error_if_nonfinite=True)
            optimizer.step()
            loss_meter.update(float(total.detach()), len(batch.name)); metric_meter.update(float(metric_loss.detach()), len(batch.name))
            for key, value in local_terms.items():
                term_meters.setdefault(key, AverageMeter()).update(value, len(batch.name))
            record = {
                "epoch": epoch, "batch": number, "loss_total": float(total.detach()),
                "loss_original": float(original.detach()), "loss_metric": float(metric_loss.detach()),
                "lambda_metric": lambda_metric, "margin": args.margin, "temperature": args.temperature,
                "gradient_norm": float(grad_norm), "terms": local_terms,
                "base_sampling": sampling_summary(base_rows), "metric_sampling": sampled.stats,
                "config_sha256": config_hash,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            rows.append(record)
            progress.update(number, loss_meter.avg,
                            {**{key: meter.avg for key, meter in term_meters.items()},
                             "metric": metric_meter.avg}, {})
            del output, total, original, metric_loss, item, loss_item, batch
    if len(loader) != 243:
        raise RuntimeError(f"K2 requires exactly 243 optimizer steps per epoch, got {len(loader)}")
    aggregate_sampling = {
        "batches": len(rows),
        "comparison_count": sum(row["metric_sampling"]["comparison_count"] for row in rows),
        "positive_count": sum(row["metric_sampling"]["positive_count"] for row in rows),
        "negative_count": sum(row["metric_sampling"]["negative_count"] for row in rows),
        "candidate_count": sum(row["metric_sampling"]["candidate_count"] for row in rows),
        "fallback_count": sum(row["metric_sampling"]["fallback_count"] for row in rows),
        "unique_pair_count": sum(row["metric_sampling"]["unique_pair_count"] for row in rows),
        "duplicate_pair_count": sum(row["metric_sampling"]["duplicate_pair_count"] for row in rows),
        "duplicate_rate": (sum(row["metric_sampling"]["duplicate_pair_count"] for row in rows) /
                           max(sum(row["metric_sampling"]["positive_count"] + row["metric_sampling"]["negative_count"]
                                   for row in rows), 1)),
        "hard_negative_fraction": (sum(row["metric_sampling"]["hard_negative_fraction"] * row["metric_sampling"]["comparison_count"]
                                       for row in rows) / max(sum(row["metric_sampling"]["comparison_count"] for row in rows), 1)),
    }
    return {"total_loss": loss_meter.avg, "metric_loss": metric_meter.avg,
            "terms": {key: meter.avg for key, meter in term_meters.items()},
            "lambda_metric": lambda_metric, "optimizer_steps": len(loader),
            "sampling": aggregate_sampling}


def save_checkpoint(path, model, optimizer, source, args, epoch, validation, config_hash):
    state = {key: source[key] for key in
             ("architecture_id", "logic_vocab", "role_vocab", "pin_vocab", "source_sha256",
              "split_ids", "feature_normalization")}
    state.update({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "config": dict(source["config"], output=str(args.output), resume=None),
                  "stage": 3, "epoch": epoch, "stage_complete": False,
                  "finetune_mode": "K2_direct_hf_metric", "metric_config_sha256": config_hash,
                  "validation_loss": validation["mean_loss"],
                  "validation_f1": validation["functional_equivalence"]["f1"],
                  "validation_threshold": validation["functional_equivalence"]["similarity_threshold"],
                  "random_state": random_state()})
    if "auxiliary_head" in state:
        raise RuntimeError("K2 checkpoint would contain an auxiliary head")
    atomic_save(state, path)


def build_config(args, source, dataset, orders):
    return {
        "experiment": "K2_direct_H_f_cosine_metric",
        "status": "prepared_for_manual_start",
        "initial_checkpoint": str(args.checkpoint.resolve()),
        "initial_checkpoint_sha256": digest_file(args.checkpoint),
        "data": str(args.data.resolve()), "data_sha256": dataset.source_sha256,
        "orders_file": str(args.orders.resolve()), "orders_sha256": digest_file(args.orders),
        "train_ids": len(source["split_ids"]["train"]), "val_ids": len(source["split_ids"]["val"]),
        "test_read": False, "split_source": "source checkpoint split_ids; train and val only",
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer_steps_per_epoch": 243,
        "cap": args.cap,
        "seed": args.seed, "sampler_seed": args.sampler_seed, "metric_seed": args.metric_seed,
        "learning_rate": args.learning_rate, "gradient_clip_norm": args.gradient_clip_norm,
        "margin": args.margin, "temperature": args.temperature,
        "lambda_metric_by_epoch": LAMBDA_BY_EPOCH,
        "base_loss": "triwave_gnn.utils.losses.loss_terms(output, loss_item, stage=3)",
        "base_task_weights": {"prob": 3.0, "rc": 1.0, "func": 2.0, "behavior": 2.0},
        "metric_score": "cosine_similarity(output[hf][u], output[hf][v], eps=1e-8)",
        "metric_loss": "mean(softplus((score_negative-score_positive+margin)/temperature))",
        "metric_group_quota": GROUP_QUOTA, "metric_per_circuit_cap": args.metric_per_circuit_cap,
        "hard_negative": {"epoch_1_fraction": 0.0, "epoch_2_plus_fraction": args.hard_negative_fraction,
                           "source": "current train batch scores only; no val/test scores"},
        "pair_policy": "positive isclose(TT,0,atol=1e-8,rtol=0); negative finite nonzero TT",
        "evaluation": "full val, H_f cosine only, original group and threshold protocol; test not iterated",
        "auxiliary_head": False, "optimizer_restore": "source checkpoint Adam moments restored",
    }


def dry_run(args):
    source = load_checkpoint(args.checkpoint, "cpu")
    if digest_file(args.checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("initial checkpoint SHA256 does not match K2 specification")
    if source.get("stage") != 3 or source.get("auxiliary_head") is not None:
        raise ValueError("source checkpoint is not a stage-3 model without an auxiliary head")
    model = TriWaveGNN(len(source["logic_vocab"]), len(source["role_vocab"]),
                       len(source["pin_vocab"]), source["config"]["hidden"],
                       functional_net_identity=checkpoint_net_identity(source)).to(args.device)
    model.load_state_dict(source["model"], strict=True)
    loaded = source
    optimizer = restore_source_optimizer(model, loaded, args.device)
    orders = json.loads(args.orders.read_text())
    validate_orders(orders, loaded["split_ids"]["train"], args.epochs)
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Adam state count does not match model parameter count")
    print(json.dumps({"status": "dry_run_ok", "checkpoint_sha256": digest_file(args.checkpoint),
                      "source_data_sha256_from_checkpoint": loaded["source_sha256"],
                      "optimizer_state_entries": len(optimizer.state),
                      "train_ids": len(loaded["split_ids"]["train"]), "val_ids": len(loaded["split_ids"]["val"]),
                      "steps_per_epoch": len(loaded["split_ids"]["train"]) // args.batch_size +
                      int(len(loaded["split_ids"]["train"]) % args.batch_size != 0),
                      "auxiliary_head": False, "test_read": False}, indent=2))


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty K2 output: {args.output}")
    if digest_file(args.checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("initial checkpoint SHA256 does not match K2 specification")
    source = load_checkpoint(args.checkpoint, "cpu")
    if source.get("auxiliary_head") is not None:
        raise ValueError("K2 cannot start from a checkpoint containing an auxiliary head")
    set_seed(args.seed)
    dataset = CircuitDataset(args.data, cache_mode="small")
    if source["source_sha256"] != dataset.source_sha256:
        raise ValueError("data bundle hash differs from source checkpoint")
    model, loaded = load_model(args.checkpoint, dataset, args.device)
    optimizer = restore_source_optimizer(model, loaded, args.device)
    splits = loaded["split_ids"]
    if splits.get("train") != loaded["split_ids"].get("train") or splits.get("val") != loaded["split_ids"].get("val"):
        raise ValueError("train/val split differs from the source checkpoint")
    orders = json.loads(args.orders.read_text())
    validate_orders(orders, splits["train"], args.epochs)
    indices = {name: i for i, name in enumerate(dataset.ids)}
    config = build_config(args, loaded, dataset, orders)
    config_hash = json_sha256(config)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", {**config, "config_sha256": config_hash})
    write_json(args.output / "run.json", {"status": "running", "config_sha256": config_hash,
                                           "test_read": False, "initial_checkpoint_sha256": digest_file(args.checkpoint)})
    try:
        history = []
        for epoch, order in enumerate(orders, 1):
            train = train_epoch(model, optimizer, dataset, order, indices, args, epoch, args.output, config_hash)
            validation = evaluate_full_val(model, dataset, splits["val"], indices, args, epoch, config_hash)
            checkpoint_path = args.output / f"epoch_{epoch}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, loaded, args, epoch, validation, config_hash)
            checkpoint_hash = digest_file(checkpoint_path)
            validation["checkpoint_sha256"] = checkpoint_hash
            write_json(args.output / f"val_epoch_{epoch}.json", validation)
            history.append({"epoch": epoch, "train": train, "val": validation})
            write_json(args.output / "history.json", history)
        write_json(args.output / "run.json", {"status": "completed", "config_sha256": config_hash,
                                               "completed_epochs": args.epochs, "test_read": False,
                                               "initial_checkpoint_sha256": digest_file(args.checkpoint)})
    except Exception as error:
        write_json(args.output / "run.json", {"status": "failed", "config_sha256": config_hash,
                                               "error": repr(error), "test_read": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly start K2 training")
    parser.add_argument("--dry-run", action="store_true", help="load checkpoint/Adam state only")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--orders", type=Path, default=DEFAULT_ORDERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cap", type=int, default=1024,
                        help="base functional-pair sampling cap; fixed to the K0/K1 protocol")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--metric-per-circuit-cap", type=int, default=32)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=1_000_045)
    parser.add_argument("--metric-seed", type=int, default=20261117)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.run == args.dry_run:
        parser.error("choose exactly one of --run or --dry-run")
    if (args.epochs != 5 or args.batch_size != 32 or args.learning_rate != 3e-5
            or args.cap != 1024):
        parser.error("K2 fixed protocol requires epochs=5, batch-size=32, learning-rate=3e-5 and cap=1024")
    if args.temperature <= 0 or args.margin < 0 or args.gradient_clip_norm <= 0:
        parser.error("invalid metric hyperparameters")
    if args.run:
        run(args)
    else:
        dry_run(args)


if __name__ == "__main__":
    main()
