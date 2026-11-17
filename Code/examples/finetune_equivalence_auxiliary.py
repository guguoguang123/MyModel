"""Paired end-to-end K0/K1 development experiment for H_f equivalence supervision."""

from __future__ import annotations

import argparse
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.compare_functional_stages import cosine_scores64, within_circuit_auc
from examples.evaluate import RUN_DIR, digest_file, validate_splits
from examples.finetune_functional_pairs import (
    large_circuit_metrics,
    print_validation,
    raw_metric_sums,
    train_epoch as base_train_epoch,
    write_json,
)
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint, random_state
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.equivalence_audit import pair_count_group_metrics, pair_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import AverageMeter, BatchProgress, info
from triwave_gnn.utils.losses import loss_terms
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.pair_sampling import sample_functional_pairs, sampling_summary
from triwave_gnn.utils.random_seed import set_seed


class EquivalenceAuxiliaryHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, hf: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        left, right = hf[pairs[0]], hf[pairs[1]]
        features = torch.cat(((left - right).abs(), left * right), dim=-1)
        return self.layers(features).squeeze(-1)


def copy_encoder_optimizer_state(optimizer, model, checkpoint_optimizer):
    """Restore the source Adam moments for encoder/task parameters; K1 head starts fresh."""
    source_group = checkpoint_optimizer["param_groups"][0]
    live_params = list(model.parameters())
    source_ids = source_group["params"]
    if len(live_params) != len(source_ids):
        raise ValueError("Source Adam parameter layout does not match the encoder")
    for parameter, source_id in zip(live_params, source_ids):
        if source_id in checkpoint_optimizer["state"]:
            state = deepcopy(checkpoint_optimizer["state"][source_id])
            optimizer.state[parameter] = {
                key: value.to(parameter.device) if torch.is_tensor(value) and key != "step" else value
                for key, value in state.items()
            }


def valid_pair_view(item):
    mask = item["hf_tt_mask"].bool() & torch.isfinite(item["hf_tt_target"])
    pairs = item["hf_pairs"][:, mask]
    target = item["hf_tt_target"][mask]
    if pairs.shape[1] != target.numel():
        raise ValueError("Functional pair indices and labels are misaligned")
    labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0).to(torch.float32)
    return pairs, labels


def sample_auxiliary_pairs(item, generator: torch.Generator, cap: int):
    pairs, labels = valid_pair_view(item)
    node_batch = item["pin"].batch
    circuit_of_pair = node_batch[pairs[0]]
    if not torch.equal(circuit_of_pair, node_batch[pairs[1]]):
        raise ValueError("Cross-circuit equivalence pair found in a disconnected batch")
    selected_pairs, selected_labels, sampled_rows = [], [], []
    counts = {"1-999": {"positive": 0, "negative": 0},
              "1000-9999": {"positive": 0, "negative": 0},
              ">=10000": {"positive": 0, "negative": 0}}
    for circuit, name in enumerate(item.name):
        columns = torch.nonzero(circuit_of_pair == circuit, as_tuple=False).flatten()
        local_labels = labels[columns]
        positive = columns[local_labels > 0.5]
        negative = columns[local_labels <= 0.5]
        budget = min(cap, columns.numel())
        if positive.numel() and negative.numel():
            each = min(positive.numel(), negative.numel(), budget // 2)
            n_pos = n_neg = each
        elif positive.numel():
            n_pos, n_neg = min(positive.numel(), budget), 0
        else:
            n_pos, n_neg = 0, min(negative.numel(), budget)
        chosen = []
        if n_pos:
            chosen.append(positive[torch.randperm(positive.numel(), generator=generator)[:n_pos]])
        if n_neg:
            chosen.append(negative[torch.randperm(negative.numel(), generator=generator)[:n_neg]])
        if chosen:
            local = torch.cat(chosen)
            local = local[torch.randperm(local.numel(), generator=generator)]
            selected_pairs.append(pairs[:, local])
            selected_labels.append(labels[local])
        valid_count = int(columns.numel())
        group = "1-999" if valid_count < 1000 else "1000-9999" if valid_count < 10000 else ">=10000"
        counts[group]["positive"] += n_pos
        counts[group]["negative"] += n_neg
        sampled_rows.append({"circuit": name, "valid_pairs": valid_count,
                             "positive_candidates": int(positive.numel()),
                             "negative_candidates": int(negative.numel()),
                             "sampled_positive": n_pos, "sampled_negative": n_neg})
    if not selected_pairs:
        raise ValueError("No valid functional pairs in auxiliary batch")
    result_pairs = torch.cat(selected_pairs, dim=1)
    result_labels = torch.cat(selected_labels)
    if result_pairs.shape[1] != result_labels.numel():
        raise ValueError("Sampled auxiliary pair indices and labels are misaligned")
    return result_pairs, result_labels, counts, sampled_rows


def restore_model_and_optimizer(checkpoint_path, dataset, device, with_head):
    model, source = load_model(checkpoint_path, dataset, device)
    head = EquivalenceAuxiliaryHead(source["config"]["hidden"]).to(device) if with_head else None
    optimizer_params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.Adam(optimizer_params, lr=source["optimizer"]["param_groups"][0]["lr"],
                                 betas=tuple(source["optimizer"]["param_groups"][0]["betas"]),
                                 eps=source["optimizer"]["param_groups"][0]["eps"],
                                 weight_decay=source["optimizer"]["param_groups"][0]["weight_decay"],
                                 amsgrad=source["optimizer"]["param_groups"][0]["amsgrad"])
    if head is None:
        optimizer.load_state_dict(source["optimizer"])
    else:
        copy_encoder_optimizer_state(optimizer, model, source["optimizer"])
    return model, head, optimizer, source


def train_k1_epoch(model, head, optimizer, dataset, order, indices, args, epoch, run_dir):
    model.train()
    head.train()
    base_rng = torch.Generator().manual_seed(args.sampler_seed + epoch - 1)
    aux_rng = torch.Generator().manual_seed(args.aux_seed + epoch - 1)
    loader = DataLoader(Subset(dataset, [indices[name] for name in order]), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, exclude_keys=SCHEDULE_KEYS,
                        generator=torch.Generator().manual_seed(args.seed + epoch))
    loss_meter, term_meters, eq_meter = AverageMeter(), {}, AverageMeter()
    group_totals = {"1-999": {"positive": 0, "negative": 0},
                    "1000-9999": {"positive": 0, "negative": 0},
                    ">=10000": {"positive": 0, "negative": 0}}
    with (run_dir / "sampling.jsonl").open("a", buffering=1) as samples, \
         BatchProgress("K1 train", 3, epoch, len(loader), args.progress_interval,
                       epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            view, _, base_rows = sample_functional_pairs(batch, "circuit_cap", base_rng, args.cap)
            item = move(batch, args.device)
            loss_item = copy(item)
            for key in ("hf_pairs", "hf_tt_target", "hf_tt_mask"):
                loss_item[key] = view[key].to(args.device)
            aux_pairs, aux_labels, sampled_counts, sampled_rows = sample_auxiliary_pairs(
                item, aux_rng, args.cap)
            aux_pairs = aux_pairs.to(args.device)
            aux_labels = aux_labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model_outputs(model, item)
            original, local_terms = loss_terms(output, loss_item, 3, functional_chunk_size=args.chunk_size)
            eq_logits = head(output["hf"], aux_pairs)
            eq_loss = F.binary_cross_entropy_with_logits(eq_logits, aux_labels)
            total = original + args.lambda_eq * eq_loss
            if not bool(torch.isfinite(total)) or not bool(torch.isfinite(eq_loss)):
                raise FloatingPointError(f"Nonfinite K1 objective at epoch={epoch}, batch={number}")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_meter.update(float(total.detach()), len(batch.name))
            eq_meter.update(float(eq_loss.detach()), len(batch.name))
            for key, value in local_terms.items():
                term_meters.setdefault(key, AverageMeter()).update(value, len(batch.name))
            for group, row in sampled_counts.items():
                group_totals[group]["positive"] += row["positive"]
                group_totals[group]["negative"] += row["negative"]
            samples.write(json.dumps({"epoch": epoch, "batch": number,
                                      "base_sampling": sampling_summary(base_rows),
                                      "auxiliary_group_counts": sampled_counts,
                                      "auxiliary_circuits": sampled_rows,
                                      "loss_original": float(original.detach()),
                                      "loss_eq_bce": float(eq_loss.detach()),
                                      "lambda_eq": args.lambda_eq,
                                      "loss_total": float(total.detach()),
                                      "gradient_norm": float(grad_norm)}, allow_nan=False) + "\n")
            progress.update(number, loss_meter.avg,
                            {**{key: meter.avg for key, meter in term_meters.items()},
                             "eq_bce": eq_meter.avg}, {})
            del batch, item, output, loss_item, total, original, eq_loss, eq_logits
    return {"mean_total_loss": loss_meter.avg,
            "original_terms": {key: meter.avg for key, meter in term_meters.items()},
            "eq_bce": eq_meter.avg, "sampled_group_counts": group_totals,
            "optimizer_steps": len(loader), "aux_rng_state": aux_rng.get_state(),
            "base_rng_state": base_rng.get_state()}


def evaluate(model, dataset, names, indices, args, epoch):
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
            if any(value is not None and not bool(torch.isfinite(value).all()) for value in output.values()):
                raise FloatingPointError("Nonfinite validation model output")
            loss, local_terms = loss_terms(output, item, 3, functional_chunk_size=args.chunk_size)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite full-validation original objective")
            loss_meter.update(float(loss), len(batch.name))
            for key, value in local_terms.items():
                terms.setdefault(key, AverageMeter()).update(value, len(batch.name))
            valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
            scores = cosine_scores64(output["hf"], item["hf_pairs"], valid, args.chunk_size)
            target = item["hf_tt_target"][valid].cpu()
            labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
            groups = item["pin"].batch[item["hf_pairs"][0, valid]].cpu()
            for circuit, name in enumerate(batch.name):
                local = scores[groups == circuit]
                local_labels = labels[groups == circuit]
                score_parts.append(local)
                label_parts.append(local_labels)
                offsets.append(offsets[-1] + local.numel())
                rows.append({"circuit": name, "original_valid_pairs": local.numel()})
            for key, (value, count) in raw_metric_sums(output, item, scores).items():
                sums[key] = sums.get(key, 0.) + value
                counts[key] = counts.get(key, 0) + count
            progress.update(number, loss_meter.avg, {},
                            {key: sums[key] / counts[key] for key in sums if counts[key]})
            del batch, item, output, loss
    scores, labels = torch.cat(score_parts), torch.cat(label_parts)
    threshold = select_threshold(scores, labels)
    pooled = pair_metrics(scores, labels, threshold)
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        row.update(pair_metrics(scores[start:stop], labels[start:stop], threshold))
    pair_groups = pair_count_group_metrics(scores, labels, offsets, threshold)

    def minimum_fp(target_tp, mask):
        local_scores, local_labels = scores[mask], labels[mask]
        positives = int(local_labels.sum())
        if positives < target_tp:
            return {"target_tp": target_tp, "status": "undefined: insufficient positives",
                    "available_positives": positives}
        order = torch.argsort(local_scores, descending=True, stable=True)
        sorted_scores, sorted_labels = local_scores[order], local_labels[order].to(torch.int64)
        tp = sorted_labels.cumsum(0)
        fp = (1 - sorted_labels).cumsum(0)
        ends = torch.ones_like(sorted_labels, dtype=torch.bool)
        ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        candidates = torch.nonzero(ends & (tp >= target_tp), as_tuple=False).flatten()
        choice = candidates[torch.argmin(fp[candidates])]
        tpi, fpi = int(tp[choice]), int(fp[choice])
        return {"target_tp": target_tp, "tp": tpi, "fp": fpi,
                "fn": positives - tpi, "recall": tpi / positives,
                "precision": tpi / (tpi + fpi) if tpi + fpi else None,
                "threshold": float(sorted_scores[choice])}

    dense_mask = torch.zeros(scores.numel(), dtype=torch.bool)
    for row, start, stop in zip(rows, offsets, offsets[1:]):
        if row["original_valid_pairs"] >= 10_000:
            dense_mask[start:stop] = True
    return {"epoch": epoch, "mean_loss": loss_meter.avg,
            "terms": {key: meter.avg for key, meter in terms.items()},
            "metrics": {key: sums[key] / counts[key] if counts[key] else None for key in sums},
            "metric_counts": counts, "functional_equivalence": pooled,
            "groups_by_original_valid_pairs": pair_groups,
            "minimum_fp_dense_tp219": minimum_fp(219, dense_mask),
            "minimum_fp_overall_tp8025": minimum_fp(8025, torch.ones_like(labels, dtype=torch.bool)),
            **within_circuit_auc(rows), "large_pair_circuits": large_circuit_metrics(rows),
            "per_circuit": rows,
            "validation_sampling": "none: all original valid functional pairs; H_f cosine score"}


def save_branch(path, model, head, optimizer, source, args, epoch, branch):
    state = {key: deepcopy(source[key]) for key in
             ("architecture_id", "logic_vocab", "role_vocab", "pin_vocab", "source_sha256",
              "split_ids", "feature_normalization")}
    state.update({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "config": dict(source["config"], output=str(path.parent), resume=None),
                  "stage": 3, "epoch": epoch, "stage_complete": False,
                  "auxiliary_head": head.state_dict() if head is not None else None,
                  "branch": branch, "lambda_eq": args.lambda_eq if head is not None else 0.,
                  "random_state": random_state()})
    atomic_save(state, path)


def run(args):
    source = load_checkpoint(args.checkpoint, "cpu")
    expected_hash = args.checkpoint_sha256
    if digest_file(args.checkpoint) != expected_hash:
        raise ValueError("Candidate checkpoint hash differs from the inspected source")
    if source.get("stage") != 3 or not source.get("stage_complete") is False:
        raise ValueError("Expected the completed B stage-3 fine-tuning checkpoint")
    root = args.output.resolve()
    if not args.only_k1 and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment root: {root}")
    if args.only_k1 and not (root / "K0" / "final_val.json").exists():
        raise FileNotFoundError("--only-k1 requires an existing completed K0 final_val.json")
    root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dataset = CircuitDataset(args.data, cache_mode="small")
    if source["source_sha256"] != dataset.source_sha256:
        raise ValueError("Data bundle hashes differ from the B checkpoint")
    splits = source["split_ids"]
    val_key = validate_splits(splits, source)
    train_names, val_names = splits["train"], splits[val_key]
    indices = {name: i for i, name in enumerate(dataset.ids)}
    orders = json.loads(args.orders.read_text())
    if len(orders) != args.epochs or any(len(order) != len(train_names) for order in orders):
        raise ValueError("Training schedule differs from the fixed B development schedule")
    if any(set(order) != set(train_names) for order in orders):
        raise ValueError("Training order contains IDs outside the original train split")
    branches = ("K1",) if args.only_k1 else ("K0", "K1")
    for branch in branches:
        directory = root / branch
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", {
            "branch": branch, "initial_checkpoint": str(args.checkpoint.resolve()),
            "initial_checkpoint_sha256": expected_hash, "train_ids": len(train_names),
            "val_ids": len(val_names), "test_read": False, "seed": args.seed,
            "epochs": args.epochs, "optimizer_steps_per_epoch": len(train_names) // args.batch_size +
                int(len(train_names) % args.batch_size != 0), "batch_size": args.batch_size,
            "lr": source["optimizer"]["param_groups"][0]["lr"], "cap": args.cap,
            "lambda_eq": args.lambda_eq if branch == "K1" else 0.,
            "base_training_path": "triwave_gnn.utils.pair_sampling.sample_functional_pairs(mode=circuit_cap) and triwave_gnn.utils.losses.loss_terms(stage=3)",
            "aux_pair_sampling": "within-circuit; valid mask and finite TT distance; label zero-distance positive; balanced per circuit when both classes exist; per-circuit cap; independent RNG",
            "evaluation_score": "cosine similarity of H_f; auxiliary head is training-only",
            "task_weights": {"prob": 3., "rc": 1., "func": 2., "behavior": 2.},
            "split": "checkpoint train/val IDs; test IDs not iterated",
            "nonfinite_policy": "immediate stop on nonfinite objective, output, or gradient"})
    # K0 and K1 use distinct model instances initialized from the same source state.
    for branch in branches:
        with (root / branch / "console.log").open("w", buffering=1) as log:
            info(f"Starting {branch}: same B initialization, Adam state, schedule and 5-epoch budget")
            set_seed(args.seed)
            model, head, optimizer, branch_source = restore_model_and_optimizer(
                args.checkpoint, dataset, args.device, with_head=False)
            if branch == "K1":
                set_seed(args.aux_seed)
                head = EquivalenceAuxiliaryHead(branch_source["config"]["hidden"]).to(args.device)
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(head.parameters()),
                    lr=branch_source["optimizer"]["param_groups"][0]["lr"],
                    betas=tuple(branch_source["optimizer"]["param_groups"][0]["betas"]),
                    eps=branch_source["optimizer"]["param_groups"][0]["eps"],
                    weight_decay=branch_source["optimizer"]["param_groups"][0]["weight_decay"],
                    amsgrad=branch_source["optimizer"]["param_groups"][0]["amsgrad"])
                copy_encoder_optimizer_state(optimizer, model, branch_source["optimizer"])
            history = []
            for epoch, order in enumerate(orders, 1):
                info(f"{branch} epoch {epoch}/{args.epochs}")
                if branch == "K0":
                    train = base_train_epoch(model, optimizer, dataset, order, indices, args,
                                             epoch, root / branch)
                else:
                    train = train_k1_epoch(model, head, optimizer, dataset, order, indices,
                                           args, epoch, root / branch)
                if train["optimizer_steps"] != 243:
                    raise RuntimeError(f"Unexpected optimizer step count in {branch}: {train['optimizer_steps']}")
                history_train = {key: value for key, value in train.items()
                                 if not key.endswith("_state")}
                history.append({"epoch": epoch, "train": history_train})
                save_branch(root / branch / f"epoch_{epoch}.pt", model, head, optimizer,
                            branch_source, args, epoch, branch)
                write_json(root / branch / "history.json", history)
                info(f"{branch} epoch={epoch} train={train}")
            final = evaluate(model, dataset, val_names, indices, args, args.epochs)
            write_json(root / branch / "final_val.json", final)
            save_branch(root / branch / "final.pt", model, head, optimizer,
                        branch_source, args, args.epochs, branch)
            print_validation(final, f"{branch} final full-val H_f cosine")
            info(f"{branch} cosine maximum-F1 counts: TP/FP/FN/TN="
                 f"{final['functional_equivalence']['tp']}/{final['functional_equivalence']['fp']}/"
                 f"{final['functional_equivalence']['fn']}/{final['functional_equivalence']['tn']}; "
                 f"dense TP>=219 minimum-FP={final['minimum_fp_dense_tp219']}; "
                 f"overall TP>=8025 minimum-FP={final['minimum_fp_overall_tp8025']}")
            info(f"{branch} pair-count groups: {final['groups_by_original_valid_pairs']}")
            info(f"Completed {branch}; validation only, test not evaluated")
            log.flush()
    k0 = json.loads((root / "K0" / "final_val.json").read_text())
    k1 = json.loads((root / "K1" / "final_val.json").read_text())
    info("K1 minus K0 full-val H_f cosine comparison: "
         f"F1={k1['functional_equivalence']['f1'] - k0['functional_equivalence']['f1']}; "
         f"dense TP>=219 FP={k0['minimum_fp_dense_tp219'].get('fp')} -> "
         f"{k1['minimum_fp_dense_tp219'].get('fp')}; "
         f"overall TP>=8025 FP={k0['minimum_fp_overall_tp8025'].get('fp')} -> "
         f"{k1['minimum_fp_overall_tp8025'].get('fp')}")
    for key in ("prob_mae", "rc_accuracy", "tt_distance_mae", "hb_tp_mae",
                "hb_first_mean_mae", "hb_stable_mean_mae", "hb_glitch_rate_mae"):
        info(f"K1 minus K0 {key}: {k1['metrics'].get(key)} vs {k0['metrics'].get(key)}")
    info(f"K0/K1 development experiment complete: {root}")


def probe(args):
    source = load_checkpoint(args.checkpoint, "cpu")
    if digest_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Candidate checkpoint hash differs from the inspected source")
    set_seed(args.seed)
    dataset = CircuitDataset(args.data, cache_mode="small")
    names = source["split_ids"]["train"]
    indices = {name: i for i, name in enumerate(dataset.ids)}
    selected_index = None
    for name in names:
        item = dataset[indices[name]]
        _, labels = valid_pair_view(item)
        if bool((labels == 1).any()) and bool((labels == 0).any()):
            selected_index = indices[name]
            break
    if selected_index is None:
        raise ValueError("No train circuit with both valid functional classes")
    model, _, _, _ = restore_model_and_optimizer(args.checkpoint, dataset, args.device, False)
    batch = next(iter(DataLoader(Subset(dataset, [selected_index]), batch_size=1,
                                 shuffle=False, num_workers=0, exclude_keys=SCHEDULE_KEYS)))
    item = move(batch, args.device)
    set_seed(args.aux_seed)
    head = EquivalenceAuxiliaryHead(source["config"]["hidden"]).to(args.device)
    generator = torch.Generator().manual_seed(args.aux_seed)
    pairs, labels, counts, sampled = sample_auxiliary_pairs(item, generator, args.cap)
    model.train()
    output = model_outputs(model, item)
    logits = head(output["hf"], pairs.to(args.device))
    eq_bce = F.binary_cross_entropy_with_logits(logits, labels.to(args.device))
    valid = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
    target = item["hf_tt_target"][valid]
    from triwave_gnn.utils.losses import functional_distance, zero_normalize
    dist = functional_distance(output["hf"], item["hf_pairs"])[valid]
    func = F.l1_loss(zero_normalize(dist), zero_normalize(target))
    swapped = head(output["hf"], pairs.flip(0).to(args.device))
    if not torch.allclose(logits, swapped, atol=1e-6, rtol=1e-6):
        raise AssertionError("Auxiliary pair feature is not exchange invariant")
    (loss_terms(output, item, 3)[0] + 0.1 * eq_bce).backward()
    gradients = [parameter.grad for parameter in list(model.parameters()) + list(head.parameters())
                 if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("Auxiliary BCE backward produced missing or nonfinite gradients")
    print(json.dumps({"checkpoint": str(args.checkpoint.resolve()), "circuit": batch.name[0],
                      "sampled_pairs": sampled, "group_counts": counts,
                      "original_L_func": float(func), "auxiliary_BCE": float(eq_bce),
                      "lambda_eq_candidates_not_searched": [0.1],
                      "proposed_fixed_lambda_eq": 0.1,
                      "weighted_auxiliary_term": float(eq_bce) * 0.1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path,
                        default=RUN_DIR.parent / "functional_pair_sampling_stage3" / "circuit_cap" / "best_equiv.pt")
    parser.add_argument("--checkpoint-sha256", default="6c38b549173a9d3bf4099c2665ab4898433496c7631e722f5390ad6a807f548a")
    parser.add_argument("--orders", type=Path,
                        default=RUN_DIR.parent / "functional_pair_sampling_stage3" / "circuit_cap" / "epoch_orders.json")
    parser.add_argument("--output", type=Path,
                        default=RUN_DIR.parent / "functional_equivalence_aux_K0K1_20261114")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-seed", type=int, default=20261114)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cap", type=int, default=1024)
    parser.add_argument("--lambda-eq", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--sampler-seed", type=int, default=1_000_045)
    parser.add_argument("--mode", default="circuit_cap")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only-k1", action="store_true",
                        help="reuse an existing completed K0 in --output and run only K1")
    args = parser.parse_args()
    if args.probe == args.run:
        parser.error("Choose exactly one of --probe or --run")
    if args.only_k1 and not args.run:
        parser.error("--only-k1 is valid only with --run")
    if args.epochs != 5 or args.batch_size != 32 or args.cap != 1024 or args.lambda_eq <= 0:
        parser.error("This fixed experiment requires 5 epochs, batch-size 32, cap 1024 and positive lambda")
    if args.probe:
        probe(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
