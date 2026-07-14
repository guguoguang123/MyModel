"""Training objectives for response reconstruction and representation shaping."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def _sample_pairs(
    num_nodes: int,
    max_pairs: int,
    device: torch.device,
    batch_index: Tensor | None,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    if num_nodes < 2 or max_pairs == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    if batch_index is None:
        first = torch.randint(num_nodes, (max_pairs,), device=device, generator=generator)
        offset = torch.randint(1, num_nodes, (max_pairs,), device=device, generator=generator)
        return first, (first + offset) % num_nodes

    first_parts: list[Tensor] = []
    second_parts: list[Tensor] = []
    graph_ids, counts = torch.unique(batch_index, return_counts=True)
    eligible = counts >= 2
    graph_ids = graph_ids[eligible]
    counts = counts[eligible]
    if graph_ids.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    total_eligible = int(counts.sum())
    remaining = max_pairs
    for position, graph_id in enumerate(graph_ids):
        nodes = torch.nonzero(batch_index == graph_id, as_tuple=False).flatten()
        if position == graph_ids.numel() - 1:
            pair_count = remaining
        else:
            pair_count = max(1, round(max_pairs * nodes.numel() / total_eligible))
            pair_count = min(pair_count, remaining)
        remaining -= pair_count
        first_local = torch.randint(
            nodes.numel(), (pair_count,), device=device, generator=generator
        )
        offset = torch.randint(
            1, nodes.numel(), (pair_count,), device=device, generator=generator
        )
        first_parts.append(nodes[first_local])
        second_parts.append(nodes[(first_local + offset) % nodes.numel()])
        if remaining == 0:
            break
    return torch.cat(first_parts), torch.cat(second_parts)


def response_similarity_loss(
    behavior: Tensor,
    signature: Tensor,
    *,
    max_pairs: int,
    minimum_response_norm: float,
    batch_index: Tensor | None,
    generator: torch.Generator | None,
) -> Tensor:
    """Match pairwise behavior cosine similarity to response similarity."""
    first, second = _sample_pairs(
        signature.size(0), max_pairs, signature.device, batch_index, generator
    )
    if first.numel() == 0:
        return signature.new_zeros(())
    first_norm = signature[first].norm(dim=-1)
    second_norm = signature[second].norm(dim=-1)
    valid = (first_norm > minimum_response_norm) & (
        second_norm > minimum_response_norm
    )
    if not torch.any(valid):
        return signature.new_zeros(())
    first = first[valid]
    second = second[valid]
    target_similarity = functional.cosine_similarity(
        signature[first], signature[second], dim=-1
    )
    behavior_similarity = functional.cosine_similarity(
        behavior[first], behavior[second], dim=-1
    )
    return functional.mse_loss(behavior_similarity, target_similarity)


def decoupling_loss(
    function_embedding: Tensor,
    behavior_embedding: Tensor,
    *,
    detach_function: bool,
) -> Tensor:
    """Penalize batch-level cross-covariance without node-wise orthogonality."""
    if function_embedding.size(0) < 2:
        return behavior_embedding.new_zeros(())
    function_state = (
        function_embedding.detach() if detach_function else function_embedding
    )
    function_state = functional.layer_norm(function_state, function_state.shape[1:])
    behavior_state = functional.layer_norm(
        behavior_embedding, behavior_embedding.shape[1:]
    )
    cross_covariance = function_state.transpose(0, 1) @ behavior_state
    cross_covariance = cross_covariance / function_embedding.size(0)
    return cross_covariance.square().mean()

