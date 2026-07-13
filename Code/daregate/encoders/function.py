"""DeepGate2-style forward functional message passing."""

from __future__ import annotations
import torch

from torch import Tensor, nn

from ..layers.common import AdditiveAttentionAggregator, ordered_levels
from ..config import DAREGateConfig


class FunctionalEncoder(nn.Module):
    """Propagate functional state from fanins to gates in topological order."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        self.config = config
        if config.structure_dim != config.function_dim:
            raise ValueError(
                "the current DeepGate2-style functional encoder requires "
                "structure_dim == function_dim"
            )
        dim = config.function_dim
        self.and_attention = AdditiveAttentionAggregator(
            config.structure_dim + dim,
            dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
        )
        self.not_attention = AdditiveAttentionAggregator(
            dim,
            dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
        )
        self.and_update = nn.GRUCell(dim, dim)
        self.not_update = nn.GRUCell(dim, dim)

    def forward(
        self,
        structure: Tensor,
        edge_index: Tensor,
        gate_type: Tensor,
        forward_level: Tensor,
    ) -> Tensor:
        function = structure.new_zeros((structure.size(0), self.config.function_dim))
        for level in ordered_levels(forward_level):
            if level == 0:
                continue
            level_edges = edge_index[:, forward_level[edge_index[1]] == level]
            if level_edges.numel() == 0:
                continue

            and_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.and_gate
            ]
            if and_edges.numel() > 0:
                joint_state = torch.cat((structure, function), dim=-1)
                message = self.and_attention(joint_state, and_edges)
                nodes = torch.unique(and_edges[1])
                function = function.index_copy(
                    0, nodes, self.and_update(message[nodes], function[nodes])
                )

            not_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.not_gate
            ]
            if not_edges.numel() > 0:
                message = self.not_attention(function, not_edges)
                nodes = torch.unique(not_edges[1])
                function = function.index_copy(
                    0, nodes, self.not_update(message[nodes], function[nodes])
                )
        return function

