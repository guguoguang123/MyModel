"""Configuration objects for DARE-Gate optimization and training control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights and sampling settings for the implemented training objective."""

    probability_weight: float = 3.0
    reconvergence_weight: float = 1.0
    functionality_weight: float = 2.0
    functionality_start_epoch: int = 0
    behavior_start_epoch: int = 0
    reconstruction_weight: float = 1.0
    similarity_weight: float = 0.1
    decoupling_weight: float = 0.01
    auxiliary_warmup_epochs: int = 5
    max_similarity_pairs: int = 4096
    minimum_response_norm: float = 1e-6
    smooth_l1_beta: float = 0.02
    detach_function_for_decoupling: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("probability_weight", self.probability_weight),
            ("reconvergence_weight", self.reconvergence_weight),
            ("functionality_weight", self.functionality_weight),
            ("reconstruction_weight", self.reconstruction_weight),
            ("similarity_weight", self.similarity_weight),
            ("decoupling_weight", self.decoupling_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.functionality_start_epoch < 0:
            raise ValueError("functionality_start_epoch cannot be negative")
        if self.behavior_start_epoch < self.functionality_start_epoch:
            raise ValueError("behavior_start_epoch cannot precede functionality_start_epoch")
        if self.auxiliary_warmup_epochs < 0:
            raise ValueError("auxiliary_warmup_epochs cannot be negative")
        if self.max_similarity_pairs < 0:
            raise ValueError("max_similarity_pairs cannot be negative")
        if self.minimum_response_norm <= 0.0:
            raise ValueError("minimum_response_norm must be positive")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive")


@dataclass(frozen=True)
class TrainerConfig:
    """Runtime and lifecycle settings for :class:`DAREGateTrainer`."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float | None = 1.0
    use_amp: bool = True
    device: str | None = None
    checkpoint_dir: str | None = None
    output_dir: str | None = None
    save_every: int = 5
    monitor: str = "mae"
    monitor_mode: str = "min"
    early_stopping_patience: int | None = None
    log_interval: int = 20
    show_progress: bool = True
    progress_min_interval: float = 0.5
    seed: int = 3407

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be either 'min' or 'max'")
        if self.early_stopping_patience is not None:
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive")
        if self.log_interval < 0:
            raise ValueError("log_interval cannot be negative")
        if self.progress_min_interval <= 0.0:
            raise ValueError("progress_min_interval must be positive")

