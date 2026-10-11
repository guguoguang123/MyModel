"""Relation-specific messages from predecessor states and optional physics."""

import torch
from torch import nn

from .mlp import MLP
from .tfmlp import TFMlpAggr


class RelationMessage(nn.Module):
    def __init__(self, source_dim: int, hidden_dim: int, physical_dim: int, cell_attention: bool):
        super().__init__()
        self.cell_attention = cell_attention
        self.message = MLP(source_dim + physical_dim, hidden_dim, hidden_dim)
        if cell_attention:
            self.aggregate = TFMlpAggr(hidden_dim, source_dim + hidden_dim + physical_dim)

    def forward(self, source_state, target_state, edges, physical, target_nodes):
        source, target = edges
        if target.numel() == 0:
            return target_state.new_zeros((target_nodes.numel(), target_state.shape[-1]))
        # z_u^k and the allowed edge conditions form the edge message.
        message = self.message(torch.cat((source_state[source], physical), dim=-1))
        if self.cell_attention:
            # The score reads z_u^k, the target's pre-update h_v^k, and e_uv.
            attention_context = torch.cat((source_state[source], target_state[target], physical), dim=-1)
            return self.aggregate(message, attention_context, target, target_nodes)
        local = torch.searchsorted(target_nodes, target)
        total = message.new_zeros((target_nodes.numel(), message.shape[-1])).index_add_(0, local, message)
        count = message.new_zeros(target_nodes.numel()).index_add_(0, local, torch.ones_like(local, dtype=message.dtype))
        return total / count.clamp_min(1).unsqueeze(-1)
