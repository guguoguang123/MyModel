"""Relation-specific messages from predecessor states and optional physics."""

import torch
from torch_geometric.nn import MessagePassing

from .mlp import MLP
from .tfmlp import TFMlpAggr


class RelationMessage(MessagePassing):
    def __init__(self, source_dim: int, hidden_dim: int, physical_dim: int, cell_attention: bool):
        super().__init__(aggr="add" if cell_attention else "mean", node_dim=0)
        self.cell_attention = cell_attention
        self.message_mlp = MLP(source_dim + physical_dim, hidden_dim, hidden_dim)
        if cell_attention:
            self.attention = TFMlpAggr(hidden_dim, source_dim + hidden_dim + physical_dim)

    def forward(self, source_state, target_state, edges, physical, target_nodes):
        if edges.shape[1] == 0:
            return target_state.new_zeros((target_nodes.numel(), target_state.shape[-1]))
        # Propagate into just this level's targets while source IDs stay global.
        edge_index = torch.stack((edges[0], torch.searchsorted(target_nodes, edges[1])))
        return self.propagate(edge_index, z=source_state,
                              h=(None, target_state[target_nodes]), physical=physical,
                              size=(source_state.shape[0], target_nodes.numel()))

    def message(self, z_j, h_i, physical, index, ptr, size_i):
        messages = self.message_mlp(torch.cat((z_j, physical), dim=-1))
        if self.cell_attention:
            # h_i is the target's pre-update state in this branch only.
            context = torch.cat((z_j, h_i, physical), dim=-1)
            return self.attention(messages, context, index, ptr, size_i)
        return messages
