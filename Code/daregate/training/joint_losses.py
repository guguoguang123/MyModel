"""Joint DeepGate2 and behavior-response training objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ..model import DAREGateOutput
from .batch import DAREGateBatch
from .config import ObjectiveConfig
from .losses import decoupling_loss, response_similarity_loss


@dataclass
class JointLossBreakdown:
    total: Tensor
    probability: Tensor
    reconvergence: Tensor
    functionality: Tensor
    reconstruction: Tensor
    similarity: Tensor
    decoupling: Tensor
    stage: str
    behavior_auxiliary_scale: float

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "probability_loss": float(self.probability.detach()),
            "reconvergence_loss": float(self.reconvergence.detach()),
            "functionality_loss": float(self.functionality.detach()),
            "reconstruction_loss": float(self.reconstruction.detach()),
            "similarity_loss": float(self.similarity.detach()),
            "decoupling_loss": float(self.decoupling.detach()),
            "behavior_auxiliary_scale": self.behavior_auxiliary_scale,
        }


def _required(value: Tensor | None, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"joint objective requires batch field '{name}'")
    return value


def _zscore(value: Tensor) -> Tensor:
    if value.numel() < 2:
        return torch.zeros_like(value)
    standard_deviation = value.std(unbiased=False).clamp_min(1e-8)
    return (value - value.mean()) / standard_deviation


class JointObjective(nn.Module):
    """Train structural, functional and behavior embeddings in one manager."""

    def __init__(self, config: ObjectiveConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectiveConfig()

    def stage_for_epoch(self, epoch: int) -> str:
        if epoch < self.config.functionality_start_epoch:
            return "structure_function_basic"
        if epoch < self.config.behavior_start_epoch:
            return "structure_function_full"
        return "joint"

    def _behavior_auxiliary_scale(self, epoch: int) -> float:
        if self.config.auxiliary_warmup_epochs == 0:
            return 1.0
        progress = epoch - self.config.behavior_start_epoch
        return min(1.0, max(0.0, progress / self.config.auxiliary_warmup_epochs))

    def forward(
        self,
        output: DAREGateOutput,
        batch: DAREGateBatch,
        *,
        epoch: int,
        generator: torch.Generator | None = None,
    ) -> JointLossBreakdown:
        zero = output.signature.new_zeros(())
        probability_loss = zero
        if self.config.probability_weight > 0.0:
            target_probability = _required(batch.probability, "probability").reshape(-1)
            probability_loss = functional.l1_loss(
                output.probability, target_probability
            )

        reconvergence_loss = zero
        if self.config.reconvergence_weight > 0.0:
            prediction = _required(
                output.reconvergence_probability, "rc_pair_index"
            ).reshape(-1)
            target = _required(batch.is_reconvergent, "is_reconvergent").reshape(-1)
            if prediction.numel() > 0:
                # Probability BCE is excluded from AMP because float16 backward
                # can overflow.
                with torch.amp.autocast(
                    device_type=prediction.device.type, enabled=False
                ):
                    reconvergence_loss = functional.binary_cross_entropy(
                        prediction.float().clamp(1e-7, 1.0 - 1e-7), target.float()
                    )

        functionality_active = epoch >= self.config.functionality_start_epoch
        functionality_loss = zero
        if functionality_active and self.config.functionality_weight > 0.0:
            prediction = _required(
                output.functional_distance, "tt_pair_index"
            ).reshape(-1)
            target = _required(batch.tt_distance, "tt_distance").reshape(-1)
            if prediction.numel() > 0:
                functionality_loss = functional.l1_loss(
                    _zscore(prediction), _zscore(target)
                )

        behavior_active = epoch >= self.config.behavior_start_epoch
        reconstruction_loss = zero
        similarity_loss = zero
        decoupling = zero
        auxiliary_scale = self._behavior_auxiliary_scale(epoch)
        if behavior_active:
            reconstruction_loss = functional.smooth_l1_loss(
                output.signature,
                batch.signature,
                beta=self.config.smooth_l1_beta,
            )
            if self.config.similarity_weight > 0.0:
                similarity_loss = response_similarity_loss(
                    output.behavior,
                    batch.signature,
                    max_pairs=self.config.max_similarity_pairs,
                    minimum_response_norm=self.config.minimum_response_norm,
                    batch_index=batch.batch_index,
                    generator=generator,
                )
            if self.config.decoupling_weight > 0.0:
                decoupling = decoupling_loss(
                    output.function,
                    output.behavior,
                    detach_function=self.config.detach_function_for_decoupling,
                )

        total = self.config.probability_weight * probability_loss
        total = total + self.config.reconvergence_weight * reconvergence_loss
        if functionality_active:
            total = total + self.config.functionality_weight * functionality_loss
        if behavior_active:
            total = total + self.config.reconstruction_weight * reconstruction_loss
            total = total + auxiliary_scale * (
                self.config.similarity_weight * similarity_loss
                + self.config.decoupling_weight * decoupling
            )
        return JointLossBreakdown(
            total=total,
            probability=probability_loss,
            reconvergence=reconvergence_loss,
            functionality=functionality_loss,
            reconstruction=reconstruction_loss,
            similarity=similarity_loss,
            decoupling=decoupling,
            stage=self.stage_for_epoch(epoch),
            behavior_auxiliary_scale=auxiliary_scale,
        )
