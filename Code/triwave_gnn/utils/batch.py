"""Batch complete pin graphs as one disconnected graph."""

import torch


NODE_KEYS = (
    "node_role", "node_logic", "node_pin", "depth", "hf_prob",
    "hb_tp", "hb_tp_01", "hb_tp_10", "hb_first_mean", "hb_stable_mean",
    "hb_rise_mean", "hb_fall_mean", "hb_glitch_rate", "hb_conditional_extra",
    "hb_unstable_window", "hb_tp_mask", "hb_tp_01_mask", "hb_tp_10_mask",
    "hb_first_train_mask", "hb_stable_train_mask", "hb_rise_train_mask",
    "hb_fall_train_mask", "hb_glitch_rate_train_mask",
    "hb_glitch_severity_train_mask", "hb_unstable_window_mask",
)
PAIR_KEYS = (
    "hs_target", "hs_mask", "hf_tt_target", "hf_tt_mask",
)


def collate_circuits(items):
    if not items:
        raise ValueError("cannot batch zero circuits")
    if len(items) == 1:
        return items[0]
    role = items[0]["pi_role_index"]
    if any(item["pi_role_index"] != role for item in items):
        raise ValueError("circuits use different role vocabularies")
    sizes = [int(item["node_role"].numel()) for item in items]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    batch = {key: torch.cat([item[key] for item in items], dim=0) for key in NODE_KEYS + PAIR_KEYS}
    batch["name"] = [item["name"] for item in items]
    batch["graph_ptr"] = torch.tensor(offsets, dtype=torch.long)
    batch["pi_role_index"] = role
    batch["max_depth"] = max(item["max_depth"] for item in items)
    batch["topological_order"] = torch.cat([
        item["topological_order"] + offset for item, offset in zip(items, offsets[:-1])
    ])
    for key in ("hs_pairs", "hf_pairs"):
        batch[key] = torch.cat([item[key] + offset for item, offset in zip(items, offsets[:-1])], dim=1)
    for relation in ("net", "cell"):
        edges = torch.cat([item[relation + "_edges"] + offset
                           for item, offset in zip(items, offsets[:-1])], dim=1)
        features = torch.cat([item[relation + "_features"] for item in items], dim=0)
        order = torch.argsort(batch["depth"][edges[1]], stable=True)
        batch[relation + "_edges"] = edges[:, order]
        batch[relation + "_features"] = features[order]
        counts = torch.bincount(batch["depth"][edges[1]], minlength=batch["max_depth"] + 1)
        batch[relation + "_ptr"] = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return batch
