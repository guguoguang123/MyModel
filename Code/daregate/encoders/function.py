"""DeepGate2-style forward functional message passing."""

from __future__ import annotations
import torch

from torch import Tensor, nn

from ..layers.common import (
    AdditiveAttentionAggregator,
    ordered_levels,
    run_checkpointed_levels,
)
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
        self.pi_embedding = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.pi_embedding, mean=0.0, std=0.02)
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
        pi_nodes = torch.nonzero(
            gate_type == self.config.pi_gate, as_tuple=False
        ).flatten()
        if pi_nodes.numel() > 0:
            initial_pi = self.pi_embedding.unsqueeze(0).expand(pi_nodes.numel(), -1)
            function = function.index_copy(
                0, pi_nodes, initial_pi.to(function.dtype)
            )
        levels = [level for level in ordered_levels(forward_level) if level != 0]

        def update_level(function: Tensor, level: int) -> Tensor:
            level_edges = edge_index[:, forward_level[edge_index[1]] == level]
            if level_edges.numel() == 0:
                return function

            and_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.and_gate
            ]
            if and_edges.numel() > 0:
                message = self.and_attention.forward_parts(
                    (structure, function), and_edges
                )
                nodes = torch.unique(and_edges[1])
                updated = self.and_update(message[nodes], function[nodes])
                function = function.index_copy(
                    0, nodes, updated.to(function.dtype)
                )

            not_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.not_gate
            ]
            if not_edges.numel() > 0:
                message = self.not_attention(function, not_edges)
                nodes = torch.unique(not_edges[1])
                updated = self.not_update(message[nodes], function[nodes])
                function = function.index_copy(
                    0, nodes, updated.to(function.dtype)
                )
            return function

        return run_checkpointed_levels(
            function,
            levels,
            update_level,
            enabled=self.training and self.config.topology_checkpointing,
            segment_size=self.config.topology_checkpoint_segment_size,
        )

