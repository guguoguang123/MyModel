"""Stage-specific structure, function and transient behavior objectives."""

import torch
from torch.nn import functional as F

BEHAVIOR = (
    ("hb_tp", "hb_tp_mask", "probability"),
    ("hb_first_mean", "hb_first_train_mask", "positive"),
    ("hb_stable_mean", "hb_stable_train_mask", "positive"),
    ("hb_glitch_rate", "hb_glitch_rate_train_mask", "probability"),
)
STAGE_WEIGHTS = {
    1: {"prob": 3.0, "rc": 1.0},
    2: {"prob": 3.0, "rc": 1.0, "func": 2.0},
    3: {"prob": 3.0, "rc": 1.0, "func": 2.0, "behavior": 2.0},
}


def functional_distance(hf, pair_index):
    if pair_index.shape[1] == 0:
        return hf.new_empty(0)
    return 1 - F.cosine_similarity(hf[pair_index[0]], hf[pair_index[1]], dim=-1, eps=1e-8)


def zero_normalize(x):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)


def loss_terms(output, item, stage):
    pins = item["pin"]
    terms = {}
    terms["prob"] = F.l1_loss(output["hf_prob_logits"].sigmoid(), pins.hf_prob)
    rc_mask = item["hs_mask"]
    if bool(rc_mask.any()):
        terms["rc"] = F.binary_cross_entropy_with_logits(output["hs_rc_logits"][rc_mask], item["hs_target"][rc_mask])
    if stage >= 2:
        pair_mask = item["hf_tt_mask"] & torch.isfinite(item["hf_tt_target"])
        if int(pair_mask.sum()) >= 2:
            predicted = functional_distance(output["hf"], item["hf_pairs"])[pair_mask]
            target = item["hf_tt_target"][pair_mask]
            terms["func"] = F.l1_loss(zero_normalize(predicted), zero_normalize(target))
    if stage >= 3:
        behavior_terms = []
        for index, (name, mask_name, kind) in enumerate(BEHAVIOR):
            target = pins[name]
            mask = pins[mask_name] & torch.isfinite(target)
            if not bool(mask.any()):
                continue
            prediction = output["hb_raw"][:, index][mask]
            if kind == "probability":
                value = F.binary_cross_entropy_with_logits(prediction, target[mask])
            else:
                value = F.smooth_l1_loss(prediction, torch.log1p(target[mask].clamp_min(0)))
            terms[name] = value
            behavior_terms.append(value)
        if behavior_terms:
            terms["behavior"] = torch.stack(behavior_terms).mean()
    weights = STAGE_WEIGHTS[stage]
    active = [key for key in weights if key in terms]
    total = sum(terms[key] * weights[key] for key in active) / sum(weights[key] for key in active)
    return total, {key: float(value.detach()) for key, value in terms.items()}
