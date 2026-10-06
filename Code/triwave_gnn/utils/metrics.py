"""Masked, interpretable batch metrics for staged representation learning."""

import torch

from .losses import BEHAVIOR, functional_distance
from .utils import get_function_acc


@torch.no_grad()
def batch_metrics(output, item, stage):
    """Return metric name -> (sum, count), so epoch means weight actual labels."""
    result = {}

    def add(name, values):
        if values.numel():
            result[name] = (float(values.sum()), int(values.numel()))

    add("prob_mae", (output["hf_prob_logits"].sigmoid() - item["hf_prob"]).abs())
    mask = item["hs_mask"]
    if bool(mask.any()):
        prediction = output["hs_rc_logits"][mask].sigmoid() >= 0.5
        add("rc_accuracy", (prediction == (item["hs_target"][mask] >= 0.5)).float())
    if stage >= 2:
        mask = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
        if bool(mask.any()):
            distance = functional_distance(output["hf"], item["hf_pairs"])
            add("tt_distance_mae", (distance[mask] - item["hf_tt_target"][mask]).abs())
            rank, count = get_function_acc(output["hf"], item["hf_pairs"],
                                           item["hf_tt_target"], mask)
            if count:
                result["tt_rank_accuracy"] = (rank * count, count)
    if stage >= 3:
        for index, (name, mask_name, kind) in enumerate(BEHAVIOR):
            target = item[name]
            mask = item[mask_name] & torch.isfinite(target)
            if bool(mask.any()):
                raw = output["hb_raw"][:, index][mask]
                prediction = raw.sigmoid() if kind == "probability" else torch.expm1(raw).clamp_min(0)
                add(name + "_mae", (prediction - target[mask]).abs())
    return result
