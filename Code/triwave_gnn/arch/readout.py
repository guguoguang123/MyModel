"""Pairwise structural features supplied to the model's readout MLP."""

import torch


def pair_features(state, pairs):
    a, b = state[pairs[0]], state[pairs[1]]
    return torch.cat((a + b, torch.abs(a - b), a * b), dim=-1)
