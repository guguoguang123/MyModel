"""Small neural building blocks shared by the model encoders."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


def make_mlp(
    dim_in: int,
    dim_hidden: int,
    dim_out: int,
    *,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Create the two-layer MLP used by the implemented model blocks."""
    layers: list[nn.Module] = [nn.Linear(dim_in, dim_hidden), nn.GELU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


def segment_softmax(scores: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Softmax over entries that share the same one-dimensional segment id."""
    if scores.size(0) != index.numel():
        raise ValueError("scores and index must have the same leading size")
    if scores.size(0) == 0:
        return scores
    if index.dtype != torch.long:
        raise TypeError("segment index must have dtype torch.long")

    expanded_index = index.view(-1, *([1] * (scores.ndim - 1))).expand_as(scores)
    segment_shape = (num_segments, *scores.shape[1:])
    maxima = torch.full(
        segment_shape, -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, expanded_index, scores, reduce="amax", include_self=True)
    exponentials = (scores - maxima.index_select(0, index)).exp()
    denominators = torch.zeros(segment_shape, dtype=scores.dtype, device=scores.device)
    denominators.scatter_add_(0, expanded_index, exponentials)
    eps = torch.finfo(scores.dtype).eps
    return exponentials / denominators.index_select(0, index).clamp_min(eps)


class AdditiveAttentionAggregator(nn.Module):
    """Multi-head additive attention over source-to-target graph edges."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
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
        if node_state.ndim != 2:
            raise ValueError("node_state must have shape [num_nodes, dim]")
        num_nodes = node_state.size(0)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() == 0:
            return node_state.new_zeros((num_nodes, self.dim_out))

        source, target = edge_index
        query = self.query(node_state).view(num_nodes, self.num_heads, self.head_dim)
        key = self.key(node_state).view(num_nodes, self.num_heads, self.head_dim)
        value = self.value(node_state).view(num_nodes, self.num_heads, self.head_dim)
        pair = torch.cat((query[target], key[source]), dim=-1)
        scores = torch.nn.functional.leaky_relu(
            (pair * self.attention_vector).sum(dim=-1), negative_slope=0.2
        )
        weights = self.dropout(segment_softmax(scores, target, num_nodes))
        messages = value[source] * weights.unsqueeze(-1)
        aggregated = node_state.new_zeros((num_nodes, self.num_heads, self.head_dim))
        aggregated.index_add_(0, target, messages)
        return self.output(aggregated.reshape(num_nodes, self.dim_out))


class GateGRUUpdater(nn.Module):
    """Gate-type-specific GRU updates with a common interface."""

    def __init__(self, dim: int, gate_ids: Iterable[int]) -> None:
        super().__init__()
        self.cells = nn.ModuleDict(
            {str(gate_id): nn.GRUCell(dim, dim) for gate_id in gate_ids}
        )

    def forward(
        self,
        state: Tensor,
        message: Tensor,
        gate_type: Tensor,
        nodes: Tensor,
    ) -> Tensor:
        updated_state = state
        for gate_id, cell in self.cells.items():
            selected = nodes[gate_type[nodes] == int(gate_id)]
            if selected.numel() == 0:
                continue
            values = cell(message[selected], updated_state[selected])
            updated_state = updated_state.index_copy(0, selected, values)
        return updated_state


def ordered_levels(forward_level: Tensor, *, descending: bool = False) -> list[int]:
    """Return the present topological levels in deterministic order."""
    levels = torch.unique(forward_level.detach()).cpu().tolist()
    return sorted((int(level) for level in levels), reverse=descending)

