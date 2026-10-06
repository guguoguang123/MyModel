"""Load a trained representation model with its data vocabulary and scaling."""

from __future__ import annotations

from .checkpoint import load as load_checkpoint
from ..model import TriWaveGNN


def load_model(checkpoint_path, dataset, device="cpu", model=None):
    state = load_checkpoint(checkpoint_path, device)
    for name, current in (
        ("logic_vocab", dataset.logic_vocab),
        ("role_vocab", dataset.role_vocab),
        ("pin_vocab", dataset.pin_vocab),
    ):
        if state[name] != current:
            raise ValueError(f"checkpoint {name} differs from the supplied graphs.npz")
    if model is None:
        model = TriWaveGNN(
            len(dataset.logic_vocab), len(dataset.role_vocab), len(dataset.pin_vocab),
            state["config"]["hidden"],
        )
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    for relation, values in state["feature_normalization"].items():
        setattr(dataset, relation + "_mean", values["mean"])
        setattr(dataset, relation + "_std", values["std"])
    dataset._cache.clear()
    return model, state
