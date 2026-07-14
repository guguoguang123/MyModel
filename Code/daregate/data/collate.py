"""Batch disconnected variable-size AIGs without cross-circuit edges."""

from __future__ import annotations

import torch
from torch import Tensor

from ..training.batch import DAREGateBatch


def _cat_optional(items: list[Tensor | None], *, dim: int = 0) -> Tensor | None:
    if all(item is None for item in items):
        return None
    if any(item is None for item in items):
        raise ValueError("an optional batch field must be present for every graph or none")
    return torch.cat([item for item in items if item is not None], dim=dim)


def _offset_pairs(
    batches: list[DAREGateBatch], field: str, offsets: list[int]
) -> Tensor | None:
    values = [getattr(batch, field) for batch in batches]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field} must be present for every graph or none")
    return torch.cat(
        [value + offset for value, offset in zip(values, offsets) if value is not None],
        dim=1,
    )


def collate_daregate(batches: list[DAREGateBatch]) -> DAREGateBatch:
    if not batches:
        raise ValueError("cannot collate an empty batch")
    offsets: list[int] = []
    total = 0
    for batch in batches:
        offsets.append(total)
        total += batch.num_nodes

    edges = torch.cat(
        [batch.edge_index + offset for batch, offset in zip(batches, offsets)], dim=1
    )
    rc_sources: list[Tensor | None] = []
    for batch, offset in zip(batches, offsets):
        if batch.rc_source is None:
            rc_sources.append(None)
        else:
            shifted = batch.rc_source.clone()
            valid = shifted >= 0
            shifted[valid] += offset
            rc_sources.append(shifted)
    names: list[str] = []
    for batch in batches:
        names.extend(batch.names or ())
    result = DAREGateBatch(
        gate_type=torch.cat([batch.gate_type for batch in batches]),
        edge_index=edges,
        forward_level=torch.cat([batch.forward_level for batch in batches]),
        backward_level=_cat_optional([batch.backward_level for batch in batches]),
        rc_source=_cat_optional(rc_sources),
        signature=torch.cat([batch.signature for batch in batches]),
        response=_cat_optional([batch.response for batch in batches]),
        probability=_cat_optional([batch.probability for batch in batches]),
        tt_pair_index=_offset_pairs(batches, "tt_pair_index", offsets),
        tt_distance=_cat_optional([batch.tt_distance for batch in batches]),
        min_tt_distance=_cat_optional([batch.min_tt_distance for batch in batches]),
        rc_pair_index=_offset_pairs(batches, "rc_pair_index", offsets),
        is_reconvergent=_cat_optional([batch.is_reconvergent for batch in batches]),
        batch_index=torch.cat(
            [torch.full((batch.num_nodes,), i, dtype=torch.long) for i, batch in enumerate(batches)]
        ),
        names=tuple(names),
    )
    result.validate(expected_signature_dim=batches[0].signature.size(1))
    return result
