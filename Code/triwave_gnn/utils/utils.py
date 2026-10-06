"""Functional distance ranking metric for embedding evaluation."""

import torch

from .losses import functional_distance


def get_function_acc(hf, pairs, distances, mask, samples=100, min_gap=0.05):
    """Compare the order of truth-table distances for sampled pair pairs."""
    valid = torch.nonzero(mask & torch.isfinite(distances), as_tuple=True)[0]
    if valid.numel() < 2:
        return None, 0
    predicted = functional_distance(hf, pairs)[valid]
    target = distances[valid]
    generator = torch.Generator(device="cpu").manual_seed(208)
    sampled = torch.randint(valid.numel(), (samples * 10, 2), generator=generator)
    left, right = sampled[:, 0].to(target.device), sampled[:, 1].to(target.device)
    informative = (left != right) & ((target[left] - target[right]).abs() >= min_gap)
    left, right = left[informative][:samples], right[informative][:samples]
    if not left.numel():
        return None, 0
    correct = ((predicted[left] - predicted[right]) * (target[left] - target[right]) > 0)
    return float(correct.float().mean().item()), int(left.numel())
