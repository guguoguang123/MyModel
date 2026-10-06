"""Relation-specific, context-conditioned edge messages."""

import torch
from torch import nn

from .mlp import MLP
from .tfmlp import TFMlpAggr


class RelationMessage(nn.Module):
    def __init__(self, state_dim: int, physical_dim: int, logic_dim: int, cell_attention: bool):
        super().__init__()
        self.cell_attention = cell_attention
        self.message = MLP(state_dim + 2 * logic_dim + physical_dim, state_dim, state_dim)
        if cell_attention:
            self.aggregate = TFMlpAggr(state_dim, state_dim + 2 * logic_dim + physical_dim)

    def forward(self, state, edges, physical, logic_embedding, target_nodes):
        source, target = edges
        if target.numel() == 0:
            return state.new_zeros((target_nodes.numel(), state.shape[-1]))
        context = torch.cat((logic_embedding[source], logic_embedding[target], physical), dim=-1)
        message = self.message(torch.cat((state[source], context), dim=-1))
        if self.cell_attention:
            attention_context = torch.cat((state[target], context), dim=-1)
            return self.aggregate(message, attention_context, target, target_nodes)
        local = torch.searchsorted(target_nodes, target)
        total = message.new_zeros((target_nodes.numel(), message.shape[-1])).index_add_(0, local, message)
        count = message.new_zeros(target_nodes.numel()).index_add_(0, local, torch.ones_like(local, dtype=message.dtype))
        return total / count.clamp_min(1).unsqueeze(-1)
