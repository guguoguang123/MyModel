"""Public training API for DARE-Gate."""

from .batch import DAREGateBatch, as_daregate_batch
from .config import ObjectiveConfig, TrainerConfig
from .losses import BehaviorObjective, LossBreakdown
from .trainer import DAREGateTrainer

__all__ = [
    "BehaviorObjective",
    "DAREGateBatch",
    "DAREGateTrainer",
    "LossBreakdown",
    "ObjectiveConfig",
    "TrainerConfig",
    "as_daregate_batch",
]

