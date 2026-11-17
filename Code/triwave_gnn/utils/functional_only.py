"""E ablation: retain the functional contribution to the original stage-3 total.

Only the standalone E experiment uses this helper; production loss_terms and
the completed B experiment are unchanged.
"""
import torch
from torch.nn import functional as F

from .losses import STAGE_WEIGHTS, functional_distance, loss_terms, zero_normalize


def functional_only_objective(output, item, chunk_size=100000):
    # Use the original code to discover active tasks and its original divisor.
    # These reference values have no path into backward, including other heads.
    with torch.no_grad():
        reference, terms = loss_terms(output, item, 3, functional_chunk_size=chunk_size)
    weights = STAGE_WEIGHTS[3]
    active = [key for key in weights if key in terms]
    denominator = sum(weights[key] for key in active)
    coefficient = weights['func'] / denominator if 'func' in terms else 0.
    mask = item['hf_tt_mask'].bool() & torch.isfinite(item['hf_tt_target'])
    if 'func' in terms:
        predicted = functional_distance(output['hf'], item['hf_pairs'], chunk_size)[mask]
        target = item['hf_tt_target'][mask]
        raw = F.l1_loss(zero_normalize(predicted), zero_normalize(target))
        objective = coefficient * raw
    else:
        # A zero functional gradient; retain the same optimizer-step budget as B.
        raw = output['hf'].sum() * 0.
        objective = raw
    return objective, dict(
        training_functional_objective=float(objective.detach()),
        training_raw_functional_loss=float(raw.detach()),
        original_multitask_reference_loss=float(reference),
        original_active_weight_sum=denominator,
        functional_coefficient_before=coefficient,
        functional_coefficient_after=coefficient,
        original_reference_terms=terms,
        functional_term_active='func' in terms,
    )
