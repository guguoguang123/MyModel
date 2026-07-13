"""Probe-wise behavior response propagation and decoding."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers.common import make_mlp, ordered_levels, segment_softmax
from ..config import DAREGateConfig


class ProbeFaninAttention(nn.Module):
    """Attention over fanins independently for every fixed workload probe."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError("dim_out must be divisible by num_heads")
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.query = nn.Linear(dim_in, dim_out)
        self.key = nn.Linear(dim_in, dim_out)
        self.value = nn.Linear(dim_in, dim_out)
        self.attention_vector = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.output = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.query, self.key, self.value, self.output):
            layer.reset_parameters()
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(self, node_state: Tensor, edge_index: Tensor) -> Tensor:
        if node_state.ndim != 3:
            raise ValueError("node_state must have shape [N, K, D]")
        num_nodes, num_probes, _ = node_state.shape
        if edge_index.numel() == 0:
            return node_state.new_zeros((num_nodes, num_probes, self.dim_out))
        source, target = edge_index
        query = self.query(node_state).view(
            num_nodes, num_probes, self.num_heads, self.head_dim
        )
        key = self.key(node_state).view(
            num_nodes, num_probes, self.num_heads, self.head_dim
        )
        value = self.value(node_state).view(
            num_nodes, num_probes, self.num_heads, self.head_dim
        )
        pair = torch.cat((query[target], key[source]), dim=-1)
        scores = torch.nn.functional.leaky_relu(
            (pair * self.attention_vector).sum(dim=-1), negative_slope=0.2
        )
        weights = self.dropout(segment_softmax(scores, target, num_nodes))
        messages = value[source] * weights.unsqueeze(-1)
        aggregated = node_state.new_zeros(
            (num_nodes, num_probes, self.num_heads, self.head_dim)
        )
        aggregated.index_add_(0, target, messages)
        return self.output(aggregated.reshape(num_nodes, num_probes, self.dim_out))


class ANDPairInteraction(nn.Module):
    """Permutation-invariant interaction between the fanins of an AND gate."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = make_mlp(2 * dim, 2 * dim, dim, dropout=dropout)

    def forward(self, behavior_state: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes, num_probes, dim = behavior_state.shape
        if edge_index.numel() == 0:
            return behavior_state.new_zeros((num_nodes, num_probes, dim))
        source, target = edge_index
        source_state = behavior_state[source]
        summed = behavior_state.new_zeros((num_nodes, num_probes, dim))
        squared = behavior_state.new_zeros((num_nodes, num_probes, dim))
        summed.index_add_(0, target, source_state)
        squared.index_add_(0, target, source_state.square())
        pairwise_product = 0.5 * (summed.square() - squared)
        return self.network(torch.cat((summed, pairwise_product), dim=-1))


class ReconvergenceCorrection(nn.Module):
    """Direct behavior correction from a known reconvergence source."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        dim = config.behavior_dim
        context_dim = config.structure_dim + config.function_dim
        self.context_projection = nn.Linear(context_dim, dim)
        self.candidate = make_mlp(
            4 * dim + 1, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.gate = make_mlp(3 * dim, 2 * dim, dim, dropout=config.mlp_dropout)

    def forward(
        self,
        behavior_state: Tensor,
        local_message: Tensor,
        context: Tensor,
        forward_level: Tensor,
        rc_source: Tensor,
        target_nodes: Tensor,
    ) -> Tensor:
        correction = local_message.new_zeros(local_message.shape)
        if target_nodes.numel() == 0:
            return correction
        source_index = rc_source[target_nodes]
        valid_mask = source_index >= 0
        nodes = target_nodes[valid_mask]
        sources = source_index[valid_mask]
        if nodes.numel() == 0:
            return correction

        sink_context = self.context_projection(context[nodes])
        source_context = self.context_projection(context[sources])
        num_probes = behavior_state.size(1)
        sink_context = sink_context.unsqueeze(1).expand(-1, num_probes, -1)
        source_context = source_context.unsqueeze(1).expand(-1, num_probes, -1)
        gap = (forward_level[nodes] - forward_level[sources]).abs().float()
        gap = torch.log1p(gap).view(-1, 1, 1).expand(-1, num_probes, 1)
        candidate = self.candidate(
            torch.cat(
                (
                    behavior_state[sources],
                    local_message[nodes],
                    source_context,
                    sink_context,
                    gap,
                ),
                dim=-1,
            )
        )
        gate = torch.sigmoid(
            self.gate(torch.cat((local_message[nodes], candidate, sink_context), dim=-1))
        )
        return correction.index_copy(0, nodes, gate * candidate)


class BehaviorEncoder(nn.Module):
    """Topological, probe-wise behavior response encoder."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.behavior_dim
        context_dim = config.structure_dim + config.function_dim
        self.gate_embedding = nn.Embedding(
            config.num_gate_types, config.gate_embedding_dim
        )
        self.probe_embedding = nn.Parameter(
            torch.empty(config.num_probes, config.probe_embedding_dim)
        )
        initialization_dim = (
            context_dim + config.gate_embedding_dim + config.probe_embedding_dim
        )
        self.pi_initializer = make_mlp(
            initialization_dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.gate_initializer = make_mlp(
            initialization_dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        attention_input_dim = dim + context_dim
        self.and_attention = ProbeFaninAttention(
            attention_input_dim,
            dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
        )
        self.not_attention = ProbeFaninAttention(
            attention_input_dim,
            dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
        )
        self.and_pair = ANDPairInteraction(dim, config.mlp_dropout)
        self.and_local_fusion = make_mlp(
            2 * dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.not_local_fusion = make_mlp(
            dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.reconvergence = ReconvergenceCorrection(config)

        update_input_dim = dim + context_dim + config.probe_embedding_dim
        self.and_update_projection = make_mlp(
            update_input_dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.not_update_projection = make_mlp(
            update_input_dim, 2 * dim, dim, dropout=config.mlp_dropout
        )
        self.and_update = nn.GRUCell(dim, dim)
        self.not_update = nn.GRUCell(dim, dim)
        self.normalization = nn.LayerNorm(dim)
        nn.init.normal_(self.probe_embedding, mean=0.0, std=0.02)

    def _initial_state(self, context: Tensor, gate_type: Tensor) -> Tensor:
        num_nodes = context.size(0)
        num_probes = self.config.num_probes
        expanded_context = context.unsqueeze(1).expand(-1, num_probes, -1)
        gate = self.gate_embedding(gate_type).unsqueeze(1).expand(-1, num_probes, -1)
        probe = self.probe_embedding.unsqueeze(0).expand(num_nodes, -1, -1)
        features = torch.cat((expanded_context, gate, probe), dim=-1)
        state = self.gate_initializer(features)
        pi_nodes = torch.nonzero(
            gate_type == self.config.pi_gate, as_tuple=False
        ).flatten()
        if pi_nodes.numel() > 0:
            state = state.index_copy(0, pi_nodes, self.pi_initializer(features[pi_nodes]))
        return self.normalization(state)

    def _update_nodes(
        self,
        state: Tensor,
        message: Tensor,
        context: Tensor,
        nodes: Tensor,
        *,
        gate: str,
    ) -> Tensor:
        if nodes.numel() == 0:
            return state
        num_probes = self.config.num_probes
        context_expanded = context[nodes].unsqueeze(1).expand(-1, num_probes, -1)
        probe = self.probe_embedding.unsqueeze(0).expand(nodes.numel(), -1, -1)
        update_features = torch.cat((message[nodes], context_expanded, probe), dim=-1)
        if gate == "and":
            projected = self.and_update_projection(update_features)
            cell = self.and_update
        elif gate == "not":
            projected = self.not_update_projection(update_features)
            cell = self.not_update
        else:
            raise ValueError(f"unsupported behavior update gate: {gate}")
        updated = cell(
            projected.reshape(-1, self.config.behavior_dim),
            state[nodes].reshape(-1, self.config.behavior_dim),
        ).view(nodes.numel(), num_probes, self.config.behavior_dim)
        return state.index_copy(0, nodes, self.normalization(updated))

    def forward(
        self,
        structure: Tensor,
        function: Tensor,
        edge_index: Tensor,
        gate_type: Tensor,
        forward_level: Tensor,
        rc_source: Tensor | None = None,
    ) -> Tensor:
        context = torch.cat((structure, function), dim=-1)
        state = self._initial_state(context, gate_type)
        if rc_source is None:
            rc_source = torch.full_like(gate_type, -1)

        for level in ordered_levels(forward_level):
            if level == 0:
                continue
            level_edges = edge_index[:, forward_level[edge_index[1]] == level]
            if level_edges.numel() == 0:
                continue
            context_expanded = context.unsqueeze(1).expand(
                -1, self.config.num_probes, -1
            )
            attention_state = torch.cat((state, context_expanded), dim=-1)

            and_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.and_gate
            ]
            if and_edges.numel() > 0:
                attention_message = self.and_attention(attention_state, and_edges)
                pair_message = self.and_pair(state, and_edges)
                local = self.and_local_fusion(
                    torch.cat((attention_message, pair_message), dim=-1)
                )
                nodes = torch.unique(and_edges[1])
                correction = self.reconvergence(
                    state, local, context, forward_level, rc_source, nodes
                )
                state = self._update_nodes(
                    state, local + correction, context, nodes, gate="and"
                )

            not_edges = level_edges[
                :, gate_type[level_edges[1]] == self.config.not_gate
            ]
            if not_edges.numel() > 0:
                attention_message = self.not_attention(attention_state, not_edges)
                local = self.not_local_fusion(attention_message)
                nodes = torch.unique(not_edges[1])
                correction = self.reconvergence(
                    state, local, context, forward_level, rc_source, nodes
                )
                state = self._update_nodes(
                    state, local + correction, context, nodes, gate="not"
                )
        return state


class CrossProbeEncoder(nn.Module):
    """Compare fixed probe slots and produce one response embedding."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        self.response_token = nn.Parameter(torch.empty(1, 1, config.behavior_dim))
        self.probe_projection = nn.Linear(
            config.probe_embedding_dim, config.behavior_dim
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.behavior_dim,
            nhead=config.cross_probe_heads,
            dim_feedforward=config.cross_probe_ff_dim,
            dropout=config.mlp_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.cross_probe_layers)
        self.normalization = nn.LayerNorm(config.behavior_dim)
        nn.init.normal_(self.response_token, mean=0.0, std=0.02)

    def forward(
        self, probe_state: Tensor, probe_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_nodes = probe_state.size(0)
        position = self.probe_projection(probe_embedding).unsqueeze(0)
        response = self.response_token.expand(num_nodes, -1, -1)
        encoded = self.encoder(torch.cat((response, probe_state + position), dim=1))
        encoded = self.normalization(encoded)
        return encoded[:, 0], encoded[:, 1:]


class CenteredBehaviorDecoder(nn.Module):
    """Decode two values per probe and enforce the response centering law."""

    def __init__(self, config: DAREGateConfig) -> None:
        super().__init__()
        self.decoder = make_mlp(
            config.behavior_dim,
            config.behavior_dim,
            config.behavior_channels,
            dropout=config.mlp_dropout,
        )

    def forward(self, probe_state: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.decoder(probe_state)
        centered = raw - raw.mean(dim=1, keepdim=True)
        return centered, centered.reshape(centered.size(0), -1)

