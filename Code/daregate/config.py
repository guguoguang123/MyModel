"""Configuration for the DARE-Gate model.

Only options used by the current architecture live here. New experimental
branches should add options when they are implemented.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DAREGateConfig:
    """Hyper-parameters that define the current DARE-Gate architecture."""

    num_gate_types: int = 3
    pi_gate: int = 0
    and_gate: int = 1
    not_gate: int = 2

    structure_dim: int = 128
    function_dim: int = 128
    behavior_dim: int = 128
    gate_embedding_dim: int = 16
    probe_embedding_dim: int = 32

    num_probes: int = 8
    behavior_channels: int = 2
    attention_heads: int = 4
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.1

    cross_probe_heads: int = 4
    cross_probe_layers: int = 1
    cross_probe_ff_dim: int = 256
    cross_probe_node_chunk_size: int = 16384
    validate_graph: bool = True
    topology_checkpointing: bool = True
    topology_checkpoint_segment_size: int = 8

    def __post_init__(self) -> None:
        positive = {
            "num_gate_types": self.num_gate_types,
            "structure_dim": self.structure_dim,
            "function_dim": self.function_dim,
            "behavior_dim": self.behavior_dim,
            "gate_embedding_dim": self.gate_embedding_dim,
            "probe_embedding_dim": self.probe_embedding_dim,
            "num_probes": self.num_probes,
            "behavior_channels": self.behavior_channels,
            "attention_heads": self.attention_heads,
            "cross_probe_heads": self.cross_probe_heads,
            "cross_probe_layers": self.cross_probe_layers,
            "cross_probe_ff_dim": self.cross_probe_ff_dim,
            "cross_probe_node_chunk_size": self.cross_probe_node_chunk_size,
            "topology_checkpoint_segment_size": self.topology_checkpoint_segment_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for dim_name, dim in (
            ("structure_dim", self.structure_dim),
            ("function_dim", self.function_dim),
            ("behavior_dim", self.behavior_dim),
        ):
            if dim % self.attention_heads != 0:
                raise ValueError(f"{dim_name} must be divisible by attention_heads")
        if self.behavior_dim % self.cross_probe_heads != 0:
            raise ValueError("behavior_dim must be divisible by cross_probe_heads")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 <= self.mlp_dropout < 1.0:
            raise ValueError("mlp_dropout must be in [0, 1)")

        gate_ids = (self.pi_gate, self.and_gate, self.not_gate)
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("PI, AND and NOT gate ids must be distinct")
        if min(gate_ids) < 0 or max(gate_ids) >= self.num_gate_types:
            raise ValueError("gate ids must be valid indices into num_gate_types")

