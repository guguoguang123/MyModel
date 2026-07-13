"""Topology utilities required by the model's graph adapter."""

from __future__ import annotations

import torch
from torch import Tensor


def topological_levels(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute longest-path levels for a directed acyclic graph.

    The utility is intended for preprocessing or for convenience on graph
    objects that do not already store levels. Training datasets should cache
    its result instead of recomputing it every epoch.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)

    source, target = edge_index
    indegree = torch.bincount(target, minlength=num_nodes)
    remaining = indegree.clone()
    level = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    frontier = torch.nonzero(remaining == 0, as_tuple=False).flatten()
    processed = 0

    while frontier.numel() > 0:
        processed += frontier.numel()
        edge_mask = torch.isin(source, frontier)
        selected_source = source[edge_mask]
        selected_target = target[edge_mask]
        if selected_target.numel() == 0:
            frontier = frontier.new_empty(0)
            continue

        candidate = level[selected_source] + 1
        candidate_level = torch.zeros_like(level)
        candidate_level.scatter_reduce_(
            0, selected_target, candidate, reduce="amax", include_self=True
        )
        touched = torch.unique(selected_target)
        level[touched] = torch.maximum(level[touched], candidate_level[touched])
        remaining -= torch.bincount(selected_target, minlength=num_nodes)
        frontier = touched[remaining[touched] == 0]

    if processed != num_nodes:
        raise ValueError("the graph contains a cycle or invalid directed edges")
    return level


def bidirectional_levels(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Return PI-to-node and PO-to-node longest-path levels."""
    forward = topological_levels(edge_index, num_nodes)
    backward = topological_levels(edge_index.flip(0), num_nodes)
    return forward, backward

