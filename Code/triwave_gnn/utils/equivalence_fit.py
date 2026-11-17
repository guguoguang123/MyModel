"""Losses and fixed subset selection for the train-only equivalence diagnostic.

This module is never imported by the production trainer.
"""
from copy import copy
import random

import torch
from torch.nn import functional as F

from .functional_equivalence import grouped_curve
from .losses import functional_distance, zero_normalize
from .pair_sampling import strata_masks

NOTICE = '训练子集拟合诊断，阈值在训练子集选择，不是验证或测试成绩'


def select_circuits(rows, seed, count=16, max_nodes=20000):
    eligible = [r for r in rows if r['positive_count'] and r['near_count']]
    bounded = [r for r in eligible if r['node_count'] <= max_nodes]
    bounded.sort(key=lambda r: (r['valid_pairs'], r['circuit']))
    if len(bounded) < count or count % 2:
        raise ValueError('Need at least 16 eligible circuits within the node budget')
    # Equal selection from the lowest and highest quartiles; no circuit overlap.
    quarter = max(count // 2, len(bounded) // 4)
    generator = random.Random(seed)
    chosen = generator.sample(bounded[:quarter], count // 2)
    chosen += generator.sample(bounded[-quarter:], count // 2)
    generator.shuffle(chosen)
    return chosen, dict(eligible_both_class=len(eligible), eligible_within_node_budget=len(bounded),
                        quartile_pool_size=quarter, max_nodes=max_nodes,
                        rule='8 uniformly without replacement from lowest and 8 from highest valid-pair-count quartile of eligible train circuits within node budget; independently seeded final order')


def balanced_columns(target, mask, generator, cap=128):
    valid = mask.bool() & torch.isfinite(target)
    strata = strata_masks(target)
    pos = torch.nonzero(valid & strata['zero'], as_tuple=True)[0]
    neg = torch.nonzero(valid & strata['near'], as_tuple=True)[0]
    count = min(cap, pos.numel(), neg.numel())
    if not count:
        raise ValueError('Each selected circuit must contain both classes')
    columns = torch.cat([pos[torch.randperm(pos.numel(), generator=generator)[:count]],
                         neg[torch.randperm(neg.numel(), generator=generator)[:count]]])
    return columns[torch.randperm(columns.numel(), generator=generator)]


def pair_view(item, columns):
    result = copy(item)
    for key in ('hf_pairs', 'hf_tt_target', 'hf_tt_mask'):
        result[key] = item[key][:, columns] if key == 'hf_pairs' else item[key][columns]
    return result


def pair_owners(item):
    pairs = item['hf_pairs']
    if pairs.shape != (2, item['hf_tt_target'].numel()) or item['hf_tt_mask'].shape != item['hf_tt_target'].shape:
        raise ValueError('Pair index, target and mask must align')
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= item['pin'].num_nodes):
        raise ValueError('Pair endpoint outside graph')
    owner = item['pin'].get('batch')
    if owner is None:
        owner = torch.zeros(item['pin'].num_nodes, dtype=torch.long, device=pairs.device)
    left, right = owner[pairs[0]], owner[pairs[1]]
    if not torch.equal(left, right):
        raise ValueError('Functional pair crosses circuit boundary')
    return left


def diagnostic_loss(hf, item, mode, margin=.02, chunk_size=100000):
    """D0 is the exact current functional term, without other task terms/weights."""
    valid = item['hf_tt_mask'].bool() & torch.isfinite(item['hf_tt_target'])
    target = item['hf_tt_target'][valid]
    distance = functional_distance(hf, item['hf_pairs'], chunk_size)[valid]
    if target.numel() < 2:
        raise ValueError('Diagnostic requires at least two valid pairs')
    if mode == 'D0':
        return F.l1_loss(zero_normalize(distance), zero_normalize(target))
    if mode != 'D1' or margin <= 0:
        raise ValueError('Require D0 or D1 and a positive margin')
    owners = pair_owners(item)[valid]
    positive = strata_masks(target)['zero']
    losses = []
    for circuit in owners.unique(sorted=True):
        local = owners == circuit
        pos, neg = local & positive, local & ~positive
        if not bool(pos.any()) or not bool(neg.any()):
            raise ValueError('D1 requires both classes inside every circuit')
        losses.append(.5 * distance[pos].mean() + .5 * F.relu(margin - distance[neg]).mean())
    return torch.stack(losses).mean()


def training_threshold(scores, labels):
    """Same grouped-score max-F1 implementation, with no validation log label."""
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError('Fixed training subset needs both classes')
    thresholds, tp, fp = grouped_curve(scores, labels)
    values = 2 * tp.double() / (int(labels.sum()) + tp + fp)
    return float(thresholds[int(torch.argmax(values))])


def batch_schedule(circuit_names, steps, batch_size):
    # Repeat the saved fixed order. An optimizer step is one complete mini-batch.
    batches = [circuit_names[i:i + batch_size] for i in range(0, len(circuit_names), batch_size)]
    return [batches[step % len(batches)] for step in range(steps)]
