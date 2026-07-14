"""Readout heads for the three DeepGate2 representation tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ..layers.common import make_mlp


def _check_pairs(pair_index: Tensor, num_nodes: int) -> None:
    if pair_index.ndim != 2 or pair_index.size(0) != 2:
        raise ValueError("pair_index must have shape [2, P]")
    if pair_index.dtype != torch.long:
        raise TypeError("pair_index must have dtype torch.long")
    if pair_index.numel() and (pair_index.min() < 0 or pair_index.max() >= num_nodes):
        raise ValueError("pair_index contains an invalid node index")


class ProbabilityHead(nn.Module):
    """Predict node logic-1 probability from the functional embedding."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = make_mlp(dim, dim, 1, dropout=dropout)

    def forward(self, function: Tensor) -> Tensor:
        return torch.sigmoid(self.network(function)).squeeze(-1)


class ReconvergenceHead(nn.Module):
    """Predict the symmetric reconvergence relation between node pairs."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = make_mlp(3 * dim, dim, 1, dropout=dropout)

    def forward(self, structure: Tensor, pair_index: Tensor) -> Tensor:
        _check_pairs(pair_index, structure.size(0))
        first, second = structure[pair_index[0]], structure[pair_index[1]]
        features = torch.cat(
            (first + second, (first - second).abs(), first * second), dim=-1
        )
        return torch.sigmoid(self.network(features)).squeeze(-1)


class FunctionalDistanceHead(nn.Module):
    """Measure pairwise functional distance as in DeepGate2."""

    def forward(self, function: Tensor, pair_index: Tensor) -> Tensor:
        _check_pairs(pair_index, function.size(0))
        return 1.0 - functional.cosine_similarity(
            function[pair_index[0]], function[pair_index[1]], dim=-1, eps=1e-8
        )
