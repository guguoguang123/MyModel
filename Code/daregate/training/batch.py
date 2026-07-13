"""Explicit batch contract between graph datasets and the trainer."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass
class DAREGateBatch:
    """One or more disconnected AIGs and their node response signatures."""

    gate_type: Tensor
    edge_index: Tensor
    forward_level: Tensor
    signature: Tensor
    backward_level: Tensor | None = None
    rc_source: Tensor | None = None
    node_features: Tensor | None = None
    batch_index: Tensor | None = None
    names: tuple[str, ...] | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.gate_type.numel())

    def validate(self, *, expected_signature_dim: int | None = None) -> None:
        num_nodes = self.num_nodes
        if self.gate_type.shape != (num_nodes,) or self.gate_type.dtype != torch.long:
            raise TypeError("gate_type must be torch.long with shape [N]")
        if self.edge_index.ndim != 2 or self.edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if self.forward_level.shape != (num_nodes,):
            raise ValueError("forward_level must have shape [N]")
        if self.signature.ndim != 2 or self.signature.size(0) != num_nodes:
            raise ValueError("signature must have shape [N, signature_dim]")
        if not self.signature.is_floating_point():
            raise TypeError("signature must be floating point")
        if expected_signature_dim is not None:
            if self.signature.size(1) != expected_signature_dim:
                raise ValueError(
                    f"expected signature dimension {expected_signature_dim}, "
                    f"got {self.signature.size(1)}"
                )
        for name, value in (
            ("backward_level", self.backward_level),
            ("rc_source", self.rc_source),
            ("batch_index", self.batch_index),
        ):
            if value is not None and value.shape != (num_nodes,):
                raise ValueError(f"{name} must have shape [N]")
        if self.rc_source is not None and self.rc_source.dtype != torch.long:
            raise TypeError("rc_source must have dtype torch.long")
        if self.batch_index is not None and self.batch_index.dtype != torch.long:
            raise TypeError("batch_index must have dtype torch.long")
        if self.node_features is not None and self.node_features.size(0) != num_nodes:
            raise ValueError("node_features must have N rows")
        if not torch.isfinite(self.signature).all():
            raise ValueError("signature contains NaN or infinity")

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "DAREGateBatch":
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return DAREGateBatch(**values)

    def model_kwargs(self) -> dict[str, Tensor | None]:
        return {
            "gate_type": self.gate_type,
            "edge_index": self.edge_index,
            "forward_level": self.forward_level,
            "backward_level": self.backward_level,
            "rc_source": self.rc_source,
            "node_features": self.node_features,
        }


def as_daregate_batch(batch: DAREGateBatch | Mapping[str, Any] | Any) -> DAREGateBatch:
    """Normalize dataclass, mapping or attribute-based loader outputs."""
    if isinstance(batch, DAREGateBatch):
        return batch
    names = {field.name for field in fields(DAREGateBatch)}
    if isinstance(batch, Mapping):
        values = {name: batch[name] for name in names if name in batch}
    else:
        values = {name: getattr(batch, name) for name in names if hasattr(batch, name)}
    missing = {"gate_type", "edge_index", "forward_level", "signature"} - values.keys()
    if missing:
        raise TypeError(f"training batch is missing required fields: {sorted(missing)}")
    return DAREGateBatch(**values)

