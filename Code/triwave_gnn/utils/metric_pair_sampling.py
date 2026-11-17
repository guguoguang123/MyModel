"""Train-only, group-balanced positive/negative metric-pair sampling for K2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


GROUPS = ("1-999", "1000-9999", ">=10000")


def pair_count_group(valid_count: int) -> str:
    if valid_count < 0:
        raise ValueError("valid pair count cannot be negative")
    if valid_count < 1000:
        return "1-999"
    if valid_count < 10000:
        return "1000-9999"
    return ">=10000"


def validate_metric_inputs(
    pair_index: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    circuit_ids: torch.Tensor,
    node_owner: Optional[torch.Tensor] = None,
    split_name: str = "train",
) -> None:
    if split_name != "train":
        raise ValueError("K2 metric sampling is train-only; val/test sampling is forbidden")
    if pair_index.ndim != 2 or pair_index.shape[0] != 2:
        raise ValueError("pair_index must have shape [2, pair_count]")
    n = pair_index.shape[1]
    if target.ndim != 1 or valid_mask.ndim != 1 or circuit_ids.ndim != 1:
        raise ValueError("target, valid_mask and circuit_ids must be one-dimensional")
    if not (target.numel() == valid_mask.numel() == circuit_ids.numel() == n):
        raise ValueError("pair index, target, mask and circuit ids are misaligned")
    if pair_index.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("sampling requires CPU tensors and a private CPU generator")
    if node_owner is not None:
        if node_owner.ndim != 1:
            raise ValueError("node_owner must be one-dimensional")
        valid = valid_mask.bool() & torch.isfinite(target)
        if valid.any():
            left, right = pair_index[:, valid]
            if int(left.max()) >= node_owner.numel() or int(right.max()) >= node_owner.numel():
                raise ValueError("pair endpoint index is outside node_owner")
            if not torch.equal(node_owner[left], node_owner[right]):
                raise ValueError("cross-circuit functional pair found")


def _draw(pool: torch.Tensor, count: int, generator: torch.Generator) -> tuple[torch.Tensor, int]:
    """Draw without replacement where possible; return selected indices and fallback draws."""
    if count <= 0 or pool.numel() == 0:
        return pool[:0], max(count, 0)
    if count <= pool.numel():
        return pool[torch.randperm(pool.numel(), generator=generator)[:count]], 0
    extra = pool[torch.randint(pool.numel(), (count - pool.numel(),), generator=generator)]
    return torch.cat((pool, extra)), count - pool.numel()


def _draw_negative(
    pool: torch.Tensor,
    count: int,
    generator: torch.Generator,
    scores: Optional[torch.Tensor],
    hard_fraction: float,
) -> tuple[torch.Tensor, int, int]:
    if not 0 <= hard_fraction <= 1:
        raise ValueError("hard_negative_fraction must be in [0, 1]")
    if scores is None or hard_fraction == 0 or pool.numel() == 0:
        selected, fallback = _draw(pool, count, generator)
        return selected, 0, fallback
    hard_count = min(count, int(round(count * hard_fraction)))
    order = torch.argsort(scores[pool], descending=True, stable=True)
    hard_pool = pool[order[: min(hard_count, pool.numel())]]
    hard, replacement = _draw(hard_pool, hard_count, generator)
    remaining = pool[~torch.isin(pool, hard.unique())]
    random_count = count - hard_count
    random, random_fallback = _draw(remaining if remaining.numel() else pool, random_count, generator)
    selected = torch.cat((hard, random))
    if selected.numel():
        selected = selected[torch.randperm(selected.numel(), generator=generator)]
    return selected, int(hard_count), int(replacement + random_fallback)


def _allocate_evenly(total: int, count: int, cap: int) -> Dict[int, int]:
    if total <= 0 or count <= 0:
        return {}
    active = list(range(count))
    result = {key: 0 for key in active}
    remaining = min(total, count * cap)
    while remaining:
        progressed = False
        for key in active:
            if result[key] < cap and remaining:
                result[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return result


@dataclass
class MetricSample:
    positive_indices: torch.Tensor
    negative_indices: torch.Tensor
    groups: torch.Tensor
    stats: dict


def sample_metric_indices(
    pair_index: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    circuit_ids: torch.Tensor,
    generator: torch.Generator,
    group_quota: Optional[dict[str, int]] = None,
    per_circuit_cap: int = 32,
    scores: Optional[torch.Tensor] = None,
    hard_negative_fraction: float = 0.0,
    split_name: str = "train",
    node_owner: Optional[torch.Tensor] = None,
) -> MetricSample:
    """Select equal positive/negative pair examples for each size group.

    ``scores`` is only an optional current-train-batch score tensor. The
    function rejects every split other than ``train`` so validation/test
    scores cannot accidentally enter hard-negative selection.
    """
    validate_metric_inputs(pair_index, target, valid_mask, circuit_ids, node_owner, split_name)
    if generator.device.type != "cpu":
        raise ValueError("metric sampler generator must be a CPU generator")
    if per_circuit_cap < 1:
        raise ValueError("per_circuit_cap must be positive")
    if scores is not None:
        if scores.shape != target.shape or not torch.isfinite(scores).all():
            raise ValueError("scores must align with target and be finite")
    quota = {group: 0 for group in GROUPS}
    quota.update(group_quota or {"1-999": 8, "1000-9999": 8, ">=10000": 16})
    if any(value < 0 for value in quota.values()):
        raise ValueError("group quotas must be nonnegative")
    valid = valid_mask.bool() & torch.isfinite(target)
    labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
    unique_circuits = torch.unique(circuit_ids[valid]).tolist()
    circuit_counts = {int(c): int((valid & (circuit_ids == c)).sum()) for c in unique_circuits}
    circuit_group = {c: pair_count_group(n) for c, n in circuit_counts.items()}
    positives_by_group: dict[str, list[torch.Tensor]] = defaultdict(list)
    negatives_by_group: dict[str, list[torch.Tensor]] = defaultdict(list)
    for circuit in unique_circuits:
        c = int(circuit)
        local = valid & (circuit_ids == c)
        positives_by_group[circuit_group[c]].append(torch.nonzero(local & labels, as_tuple=True)[0])
        negatives_by_group[circuit_group[c]].append(torch.nonzero(local & ~labels, as_tuple=True)[0])

    selected_positive, selected_negative, selected_groups = [], [], []
    group_stats = {}
    for group in GROUPS:
        quota_count = int(quota[group])
        pos_pools = positives_by_group[group]
        neg_pools = negatives_by_group[group]
        both = [(p, n) for p, n in zip(pos_pools, neg_pools) if p.numel() and n.numel()]
        pos_parts, neg_parts = [], []
        same_circuit_count = 0
        hard_selected = 0
        fallback_count = 0
        if both and quota_count:
            allocation = _allocate_evenly(quota_count, len(both), per_circuit_cap)
            for local_id, (pos_pool, neg_pool) in enumerate(both):
                count = allocation.get(local_id, 0)
                if not count:
                    continue
                pos, pos_fallback = _draw(pos_pool, count, generator)
                neg, hard, neg_fallback = _draw_negative(
                    neg_pool, count, generator, scores, hard_negative_fraction)
                used = min(pos.numel(), neg.numel())
                if used:
                    pos_parts.append(pos[:used]); neg_parts.append(neg[:used])
                    same_circuit_count += used
                    hard_selected += hard
                    fallback_count += pos_fallback + neg_fallback
        current = sum(part.numel() for part in pos_parts)
        remaining = max(0, quota_count - current)
        if remaining:
            pos_pool = torch.cat(pos_pools) if pos_pools else torch.empty(0, dtype=torch.long)
            neg_pool = torch.cat(neg_pools) if neg_pools else torch.empty(0, dtype=torch.long)
            pos, pos_fallback = _draw(pos_pool, remaining, generator)
            neg, hard, neg_fallback = _draw_negative(
                neg_pool, remaining, generator, scores, hard_negative_fraction)
            used = min(pos.numel(), neg.numel())
            if used:
                pos_parts.append(pos[:used]); neg_parts.append(neg[:used])
                hard_selected += hard
                fallback_count += pos_fallback + neg_fallback
        if pos_parts:
            pos = torch.cat(pos_parts); neg = torch.cat(neg_parts)
            # Pair examples are shuffled but remain aligned one-to-one.
            order = torch.randperm(pos.numel(), generator=generator)
            pos, neg = pos[order], neg[order]
            selected_positive.append(pos); selected_negative.append(neg)
            selected_groups.append(torch.full((pos.numel(),), GROUPS.index(group), dtype=torch.long))
        selected_count = sum(x.numel() for x in selected_positive[-1:]) if pos_parts else 0
        group_stats[group] = {
            "requested_comparisons": quota_count,
            "selected_comparisons": selected_count,
            "same_circuit_comparisons": same_circuit_count,
            "fallback_comparisons": max(0, selected_count - same_circuit_count),
            "candidate_positive_count": int(sum(x.numel() for x in pos_pools)),
            "candidate_negative_count": int(sum(x.numel() for x in neg_pools)),
            "hard_negative_count": hard_selected,
            "hard_negative_fraction": hard_selected / selected_count if selected_count else 0.0,
            "fallback_count": fallback_count,
        }

    if selected_positive:
        positive = torch.cat(selected_positive)
        negative = torch.cat(selected_negative)
        groups = torch.cat(selected_groups)
    else:
        positive = negative = groups = torch.empty(0, dtype=torch.long)
    selected_all = torch.cat((positive, negative)) if positive.numel() else positive
    unique_count = int(torch.unique(selected_all).numel())
    duplicate_count = int(selected_all.numel() - unique_count)
    stats = {
        "split": split_name,
        "group_quota": quota,
        "comparison_count": int(positive.numel()),
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "groups": group_stats,
        "hard_negative_fraction": (sum(v["hard_negative_count"] for v in group_stats.values()) / positive.numel()
                                    if positive.numel() else 0.0),
        "candidate_count": sum(v["candidate_negative_count"] for v in group_stats.values()),
        "fallback_count": sum(v["fallback_count"] for v in group_stats.values()),
        "unique_pair_count": unique_count,
        "duplicate_pair_count": duplicate_count,
        "duplicate_rate": duplicate_count / selected_all.numel() if selected_all.numel() else 0.0,
    }
    return MetricSample(positive, negative, groups, stats)
