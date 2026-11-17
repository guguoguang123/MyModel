"""Differentiable direct H_f cosine ranking loss used by K2."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def cosine_scores(hf: torch.Tensor, pairs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return endpoint cosine scores without detaching H_f or the pair scores."""
    if hf.ndim != 2 or pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("hf must be [nodes, hidden] and pairs must be [2, pairs]")
    if pairs.dtype not in (torch.int32, torch.int64):
        raise ValueError("pair indices must be integer tensors")
    if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= hf.shape[0]):
        raise ValueError("pair endpoint index is out of range")
    if not torch.isfinite(hf).all():
        raise ValueError("H_f contains non-finite values")
    return F.cosine_similarity(hf[pairs[0]], hf[pairs[1]], dim=-1, eps=eps)


def direct_metric_loss(
    score_positive: torch.Tensor,
    score_negative: torch.Tensor,
    margin: float = 0.02,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute mean softplus((negative - positive + margin) / temperature)."""
    if score_positive.ndim != 1 or score_negative.ndim != 1:
        raise ValueError("positive and negative scores must be one-dimensional")
    if score_positive.shape != score_negative.shape:
        raise ValueError("positive and negative score counts must match")
    if margin < 0 or temperature <= 0:
        raise ValueError("margin must be nonnegative and temperature must be positive")
    if not torch.isfinite(score_positive).all() or not torch.isfinite(score_negative).all():
        raise ValueError("metric scores must be finite")
    if score_positive.numel() == 0:
        # Keep an empty comparison differentiable when the caller has H_f scores.
        return (score_positive.sum() + score_negative.sum()) * 0.0
    argument = (score_negative - score_positive + margin) / temperature
    return F.softplus(argument).mean()


def direct_metric_loss_from_embeddings(
    hf: torch.Tensor,
    positive_pairs: torch.Tensor,
    negative_pairs: torch.Tensor,
    margin: float = 0.02,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute the direct loss from endpoint pairs, preserving its H_f graph."""
    positive = cosine_scores(hf, positive_pairs)
    negative = cosine_scores(hf, negative_pairs)
    return direct_metric_loss(positive, negative, margin, temperature)
