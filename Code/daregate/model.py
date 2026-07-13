"""Top-level DARE-Gate model composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .encoders.behavior import BehaviorEncoder, CenteredBehaviorDecoder, CrossProbeEncoder
from .config import DAREGateConfig
from .encoders.function import FunctionalEncoder
from .encoders.structure import StructuralEncoder
from .utils.topology import bidirectional_levels


@dataclass
class DAREGateOutput:
    """Named node representations and predictions returned by DARE-Gate."""

    structure: Tensor
    function: Tensor
    behavior: Tensor
    probe_behavior: Tensor
    response: Tensor
    signature: Tensor


class DAREGateModel(nn.Module):
    """Distribution-aware and reconvergence-enhanced AIG representation model."""

    def __init__(self, config: DAREGateConfig | None = None) -> None:
        super().__init__()
        self.config = config or DAREGateConfig()
        self.structure_encoder = StructuralEncoder(self.config)
        self.function_encoder = FunctionalEncoder(self.config)
        self.behavior_encoder = BehaviorEncoder(self.config)
        self.cross_probe_encoder = CrossProbeEncoder(self.config)
        self.behavior_decoder = CenteredBehaviorDecoder(self.config)

    def _validate_inputs(
        self,
        gate_type: Tensor,
        edge_index: Tensor,
        forward_level: Tensor,
        backward_level: Tensor | None,
        rc_source: Tensor | None,
    ) -> None:
        if gate_type.ndim != 1 or gate_type.dtype != torch.long:
            raise TypeError("gate_type must be a one-dimensional torch.long tensor")
        num_nodes = gate_type.numel()
        if num_nodes == 0:
            raise ValueError("a graph must contain at least one node")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if forward_level.shape != (num_nodes,):
            raise ValueError("forward_level must have shape [num_nodes]")
        if backward_level is not None and backward_level.shape != (num_nodes,):
            raise ValueError("backward_level must have shape [num_nodes]")
        if gate_type.min() < 0 or gate_type.max() >= self.config.num_gate_types:
            raise ValueError("gate_type contains an unsupported gate id")
        if edge_index.numel() > 0:
            if edge_index.min() < 0 or edge_index.max() >= num_nodes:
                raise ValueError("edge_index contains an out-of-range node index")
            source, target = edge_index
            if torch.any(forward_level[source] >= forward_level[target]):
                raise ValueError(
                    "edge_index is not consistent with the forward topological levels"
                )
        if rc_source is not None:
            if rc_source.shape != (num_nodes,) or rc_source.dtype != torch.long:
                raise TypeError(
                    "rc_source must be a torch.long tensor with shape [num_nodes]"
                )
            valid = rc_source >= 0
            if torch.any(rc_source[valid] >= num_nodes):
                raise ValueError("rc_source contains an out-of-range node index")
            node_index = torch.arange(num_nodes, device=gate_type.device)
            if torch.any(
                forward_level[rc_source[valid]] >= forward_level[node_index[valid]]
            ):
                raise ValueError("each reconvergence source must precede its sink")

    def forward(
        self,
        gate_type: Tensor,
        edge_index: Tensor,
        forward_level: Tensor,
        *,
        backward_level: Tensor | None = None,
        rc_source: Tensor | None = None,
        node_features: Tensor | None = None,
    ) -> DAREGateOutput:
        if self.config.validate_graph:
            self._validate_inputs(
                gate_type, edge_index, forward_level, backward_level, rc_source
            )

        structure = self.structure_encoder(
            gate_type,
            edge_index,
            forward_level,
            backward_level=backward_level,
            node_features=node_features,
        )
        function = self.function_encoder(
            structure, edge_index, gate_type, forward_level
        )
        propagated_probes = self.behavior_encoder(
            structure,
            function,
            edge_index,
            gate_type,
            forward_level,
            rc_source=rc_source,
        )
        behavior, probe_behavior = self.cross_probe_encoder(
            propagated_probes, self.behavior_encoder.probe_embedding
        )
        response, signature = self.behavior_decoder(probe_behavior)
        return DAREGateOutput(
            structure=structure,
            function=function,
            behavior=behavior,
            probe_behavior=probe_behavior,
            response=response,
            signature=signature,
        )

    @staticmethod
    def _gate_type_from_graph(graph: Any) -> Tensor:
        if hasattr(graph, "gate") and graph.gate is not None:
            return graph.gate.reshape(-1).long()
        if hasattr(graph, "x") and graph.x is not None:
            if graph.x.ndim == 2 and graph.x.size(1) > 1:
                return graph.x.argmax(dim=1).long()
            return graph.x.reshape(-1).long()
        raise AttributeError("graph must provide either 'gate' or 'x'")

    def forward_graph(self, graph: Any) -> DAREGateOutput:
        """Run a PyG-like graph, inferring levels when they are absent."""
        gate_type = self._gate_type_from_graph(graph)
        forward_level = getattr(graph, "forward_level", None)
        backward_level = getattr(graph, "backward_level", None)
        if forward_level is None:
            forward_level, inferred_backward = bidirectional_levels(
                graph.edge_index, gate_type.numel()
            )
            if backward_level is None:
                backward_level = inferred_backward
        return self(
            gate_type,
            graph.edge_index,
            forward_level,
            backward_level=backward_level,
            rc_source=getattr(graph, "rc_source", None),
            node_features=getattr(graph, "node_features", None),
        )
