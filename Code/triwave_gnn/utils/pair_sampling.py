"""Training-only functional pair selection, grouped by PyG node ownership."""

from copy import copy
import hashlib

import torch


MODES = ("original", "circuit_cap", "circuit_cap_stratified")
STRATA = ("zero", "near", "far", "other")


def strata_masks(target):
    positive = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0)
    near = ~positive & (target > 0) & (target <= .2)
    far = ~positive & (target >= .8)
    return dict(zero=positive, near=near, far=far, other=~(positive | near | far))


def allocate_strata(counts, budget):
    """Retain other distances proportionally, then allocate main bins 1:1:2.

    Give every nonempty bin at least one place. Empty main bins have their
    weights redistributed among the other nonempty main bins. Replacement is
    allowed later, so available counts do not cap the target allocation.
    """
    allocation = dict.fromkeys(STRATA, 0)
    active = [key for key in STRATA[:3] if counts[key]]
    if not budget:
        return allocation
    if budget < sum(bool(count) for count in counts.values()):
        raise ValueError("Budget too small to retain all nonempty strata")
    if counts["other"]:
        other = max(1, round(budget * counts["other"] / sum(counts.values())))
        allocation["other"] = min(other, budget - len(active))
    remaining = budget - allocation["other"]
    if active:
        weights = {"zero": 1, "near": 1, "far": 2}
        ideal = {key: remaining * weights[key] / sum(weights[k] for k in active) for key in active}
        for key in active:
            allocation[key] = max(1, int(ideal[key]))
        while sum(allocation.values()) > budget:
            key = max((k for k in active if allocation[k] > 1), key=lambda k: allocation[k] - ideal[k])
            allocation[key] -= 1
        while sum(allocation.values()) < budget:
            key = max(active, key=lambda k: ideal[k] - allocation[k])
            allocation[key] += 1
    else:
        allocation["other"] = budget
    return allocation


def _draw(pool, count, generator):
    if not count:
        return pool[:0]
    if not pool.numel():
        raise ValueError("Cannot sample an empty stratum")
    if count <= pool.numel():
        return pool[torch.randperm(pool.numel(), generator=generator)[:count]]
    # Cover all available pairs once before drawing replacement entries.
    extra = pool[torch.randint(pool.numel(), (count - pool.numel(),), generator=generator)]
    selected = torch.cat((pool, extra))
    return selected[torch.randperm(count, generator=generator)]


def sample_functional_pairs(item, mode, generator, cap=1024):
    """Return a shallow loss view and source-column indices; never mutate item.

    The view shares all node/edge/RC/behavior attributes with the full batch.
    Exactly hf_pairs, hf_tt_target and hf_tt_mask are synchronously selected.
    Call on CPU before moving the full graph and the loss view to the GPU.
    """
    if mode not in MODES or cap < 1:
        raise ValueError("Invalid sampling mode or cap")
    pairs, target, mask = item["hf_pairs"], item["hf_tt_target"], item["hf_tt_mask"]
    if pairs.device.type != "cpu" or generator.device.type != "cpu":
        raise ValueError("Functional pair sampling uses its own CPU generator")
    if pairs.shape != (2, target.numel()) or mask.shape != target.shape:
        raise ValueError("Functional pair indices, targets and masks are not aligned")
    valid = mask.bool() & torch.isfinite(target)
    valid_indices = torch.nonzero(valid, as_tuple=True)[0]
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= item["pin"].num_nodes):
        raise ValueError("Functional pair node index is out of range")
    owner = item["pin"].get("batch")
    if owner is None:
        owner = torch.zeros(item["pin"].num_nodes, dtype=torch.long)
    if not torch.equal(owner[pairs[0, valid]], owner[pairs[1, valid]]):
        raise ValueError("Functional pair crosses a PyG circuit boundary")
    names = item.name if isinstance(item.name, list) else [item.name]
    groups = owner[pairs[0, valid]]
    choices, rows = [], []
    for circuit in range(len(names)):
        candidates = valid_indices[groups == circuit]
        local_target = target[candidates]
        masks = strata_masks(local_target)
        source_counts = {key: int(masks[key].sum()) for key in STRATA}
        budget = candidates.numel() if mode == "original" else min(candidates.numel(), cap)
        if mode == "original":
            selected = candidates
        elif mode == "circuit_cap":
            selected = _draw(candidates, budget, generator)
        else:
            allocation = allocate_strata(source_counts, budget)
            selected = torch.cat([_draw(candidates[masks[key]], allocation[key], generator) for key in STRATA])
            selected = selected[torch.randperm(selected.numel(), generator=generator)]
        selected_counts = {key: int(value.sum()) for key, value in strata_masks(target[selected]).items()}
        unique = int(torch.unique(selected).numel())
        rows.append({"circuit": names[circuit], "valid_pairs": candidates.numel(), "budget": budget,
                     "sampled_pairs": selected.numel(), "unique_pairs": unique,
                     "duplicate_pairs": selected.numel() - unique,
                     "duplicate_rate": 1 - unique / selected.numel() if selected.numel() else 0.,
                     "source_strata": source_counts, "sampled_strata": selected_counts,
                     "sampled_strata_ratio": {key: selected_counts[key] / selected.numel()
                                              if selected.numel() else 0. for key in STRATA},
                     "selected_columns_sha256": hashlib.sha256(selected.numpy().tobytes()).hexdigest()})
        choices.append(selected)
    selected = valid_indices if mode == "original" else torch.cat(choices)
    result = copy(item)
    for key, value in (("hf_pairs", pairs[:, selected]), ("hf_tt_target", target[selected]),
                       ("hf_tt_mask", mask[selected])):
        result[key] = value
    if not torch.equal(result["hf_pairs"], item["hf_pairs"][:, selected]):
        raise RuntimeError("Selected functional indices lost alignment")
    if not torch.equal(result["hf_tt_target"], item["hf_tt_target"][selected]):
        raise RuntimeError("Selected functional labels lost alignment")
    if not torch.equal(result["hf_tt_mask"], item["hf_tt_mask"][selected]):
        raise RuntimeError("Selected functional masks lost alignment")
    return result, selected, rows


def sampling_summary(rows):
    result = {key: sum(row[key] for row in rows)
              for key in ("valid_pairs", "sampled_pairs", "unique_pairs", "duplicate_pairs")}
    result["circuits"] = len(rows)
    result["duplicate_rate"] = result["duplicate_pairs"] / max(result["sampled_pairs"], 1)
    for field in ("source_strata", "sampled_strata"):
        result[field] = {key: sum(row[field][key] for row in rows) for key in STRATA}
    result["sampled_strata_ratio"] = {key: result["sampled_strata"][key] / max(result["sampled_pairs"], 1)
                                       for key in STRATA}
    return result
