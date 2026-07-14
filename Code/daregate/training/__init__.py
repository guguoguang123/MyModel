"""Public training API for DARE-Gate."""

from .batch import DAREGateBatch, as_daregate_batch
from .config import ObjectiveConfig, TrainerConfig
from .joint_losses import JointLossBreakdown, JointObjective
from .trainer import DAREGateTrainer

__all__ = [
    "JointLossBreakdown",
    "JointObjective",
    "DAREGateBatch",
    "DAREGateTrainer",
    "ObjectiveConfig",
    "TrainerConfig",
    "as_daregate_batch",
]

