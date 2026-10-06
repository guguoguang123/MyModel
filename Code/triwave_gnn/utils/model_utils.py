"""Checkpoint inspection for published or resumable runs."""

from .checkpoint import load


def model_outputs(model, item, include_behavior=True):
    """Apply all readouts to the three embeddings for evaluation/debugging."""
    hs, hf, hb = model(item, include_behavior=include_behavior)
    return {
        "hs": hs, "hf": hf, "hb": hb,
        "hs_rc_logits": model.pred_rc_logits(hs, item["hs_pairs"]),
        "hf_prob_logits": model.pred_prob_logits(hf),
        "hb_raw": model.pred_behavior_raw(hb) if include_behavior else None,
    }


def checkpoint_summary(path):
    state = load(path, "cpu")
    return {
        "stage": state.get("stage"), "epoch": state.get("epoch"),
        "hidden": state.get("config", {}).get("hidden"),
        "has_optimizer": "optimizer" in state,
        "parameter_count": sum(value.numel() for value in state["model"].values()),
        "design_count": sum(map(len, state.get("split_ids", {}).values())),
    }
