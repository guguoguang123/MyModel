"""Bidirectional structural encoder for AIG nodes."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers.common import AdditiveAttentionAggregator, GateGRUUpdater, make_mlp, ordered_levels
from ..config import DAREGateConfig


class DirectionalStructurePass(nn.Module):
    """One topological attention pass in either graph direction."""

    def __init__(self, config: DAREGateConfig, *, reverse: bool) -> None:
        super().__init__()
        self.reverse = reverse
        gate_ids = (config.pi_gate, config.and_gate, config.not_gate)
        self.attention = nn.ModuleDict(
            {
                str(gate_id): AdditiveAttentionAggregator(
                    config.structure_dim,
                    config.structure_dim,
                    num_heads=config.attention_heads,
                    dropout=config.attention_dropout,
                )
                for gate_id in gate_ids
            }
        )
        self.updater = GateGRUUpdater(config.structure_dim, gate_ids)

    def forward(
        self,
        initial_state: Tensor,
        edge_index: Tensor,
        gate_type: Tensor,
        forward_level: Tensor,
    ) -> Tensor:
        directed_edges = edge_index.flip(0) if self.reverse else edge_index
        state = initial_state
        for level in ordered_levels(forward_level, descending=self.reverse):
            level_nodes = torch.nonzero(forward_level == level, as_tuple=False).flatten()
            if level_nodes.numel() == 0:
                continue
            level_edges = directed_edges[
                :, forward_level[directed_edges[1]] == level
            ]
            if level_edges.numel() == 0:
                continue

            message = state.new_zeros(state.shape)
            for gate_id, attention in self.attention.items():
                gate_edges = level_edges[
                    :, gate_type[level_edges[1]] == int(gate_id)
                ]
                if gate_edges.numel() == 0:
                    continue
                gate_message = attention(state, gate_edges)
                target_nodes = torch.unique(gate_edges[1])
                message = message.index_copy(0, target_nodes, gate_message[target_nodes])
            active_nodes = torch.unique(level_edges[1])
            state = self.updater(state, message, gate_type, active_nodes)
        return state


class StructuralEncoder(nn.Module):
    """Encode predecessor and successor context in independent directions."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        self.config = config
        self.gate_embedding = nn.Embedding(config.num_gate_types, config.structure_dim)
        self.level_encoder = make_mlp(
            2,
            config.structure_dim,
            config.structure_dim,
            dropout=config.mlp_dropout,
        )
        self.node_feature_projection = (
            nn.Linear(config.node_feature_dim, config.structure_dim)
            if config.node_feature_dim > 0
            else None
        )
        self.predecessor_pass = DirectionalStructurePass(config, reverse=False)
        self.successor_pass = DirectionalStructurePass(config, reverse=True)
        self.fusion_gate = nn.Linear(2 * config.structure_dim, config.structure_dim)
        self.normalization = nn.LayerNorm(config.structure_dim)

    @staticmethod
    def _normalize_level(level: Tensor) -> Tensor:
        level = level.float()
        return level / level.max().clamp_min(1.0)

    def forward(
        self,
        gate_type: Tensor,
        edge_index: Tensor,
        forward_level: Tensor,
        backward_level: Tensor | None = None,
        node_features: Tensor | None = None,
    ) -> Tensor:
        if backward_level is None:
            backward_level = forward_level.max() - forward_level
        level_features = torch.stack(
            (
                self._normalize_level(forward_level),
                self._normalize_level(backward_level),
            ),
            dim=-1,
        )
        initial = self.gate_embedding(gate_type) + self.level_encoder(level_features)
        if self.node_feature_projection is not None:
            if node_features is None:
                raise ValueError(
                    "node_features are required when node_feature_dim is non-zero"
                )
            initial = initial + self.node_feature_projection(node_features)
        elif node_features is not None:
            raise ValueError(
                "node_features were supplied but config.node_feature_dim is zero"
            )

        predecessor = self.predecessor_pass(
            initial, edge_index, gate_type, forward_level
        )
        successor = self.successor_pass(initial, edge_index, gate_type, forward_level)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((predecessor, successor), dim=-1))
        )
        fused = gate * predecessor + (1.0 - gate) * successor
        return self.normalization(initial + fused)

