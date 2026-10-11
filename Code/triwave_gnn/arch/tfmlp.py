"""Attention-weighted aggregation of incoming cell-edge messages."""

import torch
from torch import nn

from .mlp import MLP


class TFMlpAggr(nn.Module):
    """Normalize learned scores among incoming cell arcs of each target pin."""

    def __init__(self, message_dim: int, context_dim: int):
        super().__init__()
        self.score = MLP(context_dim, message_dim, 1, activation="tanh")

    def forward(self, messages, context, target, target_nodes):
        local = torch.searchsorted(target_nodes, target)
        score = self.score(context).squeeze(-1)
        maximum = score.new_full((target_nodes.numel(),), -torch.inf)
        maximum.scatter_reduce_(0, local, score, reduce="amax", include_self=True)
        weight = torch.exp(score - maximum[local])
        denominator = score.new_zeros(target_nodes.numel()).index_add_(0, local, weight)
        weighted = messages * (weight / denominator[local].clamp_min(1e-9)).unsqueeze(-1)
        return messages.new_zeros((target_nodes.numel(), messages.shape[-1])).index_add_(0, local, weighted)
