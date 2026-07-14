"""Top-level DARE-Gate model composition."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .encoders.behavior import BehaviorEncoder, CenteredBehaviorDecoder, CrossProbeEncoder
from .config import DAREGateConfig
from .encoders.function import FunctionalEncoder
from .encoders.structure import StructuralEncoder
from .heads import FunctionalDistanceHead, ProbabilityHead, ReconvergenceHead


@dataclass
class DAREGateOutput:
    """Named node representations and predictions returned by DARE-Gate."""

    structure: Tensor
    function: Tensor
    behavior: Tensor
    probe_behavior: Tensor
    response: Tensor
    signature: Tensor
    probability: Tensor
    reconvergence_probability: Tensor | None
    functional_distance: Tensor | None


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
        self.probability_head = ProbabilityHead(
            self.config.function_dim, self.config.mlp_dropout
        )
        self.reconvergence_head = ReconvergenceHead(
            self.config.structure_dim, self.config.mlp_dropout
        )
        self.functional_distance_head = FunctionalDistanceHead()

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
        tt_pair_index: Tensor | None = None,
        rc_pair_index: Tensor | None = None,
        compute_behavior: bool = True,
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
        )
        function = self.function_encoder(
            structure, edge_index, gate_type, forward_level
        )
        if compute_behavior:
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
        else:
            num_nodes = gate_type.numel()
            behavior = structure.new_zeros(
                (num_nodes, self.config.behavior_dim)
            )
            probe_behavior = structure.new_zeros(
                (num_nodes, self.config.num_probes, self.config.behavior_dim)
            )
            response = structure.new_zeros(
                (num_nodes, self.config.num_probes, self.config.behavior_channels)
            )
            signature = response.reshape(num_nodes, -1)
        probability = self.probability_head(function)
        reconvergence_probability = (
            self.reconvergence_head(structure, rc_pair_index)
            if rc_pair_index is not None
            else None
        )
        functional_distance = (
            self.functional_distance_head(function, tt_pair_index)
            if tt_pair_index is not None
            else None
        )
        return DAREGateOutput(
            structure=structure,
            function=function,
            behavior=behavior,
            probe_behavior=probe_behavior,
            response=response,
            signature=signature,
            probability=probability,
            reconvergence_probability=reconvergence_probability,
            functional_distance=functional_distance,
        )
