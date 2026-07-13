"""Streaming validation metrics that do not retain full circuit predictions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor


class ResponseMetrics:
    """Accumulate response errors and cosine agreement over node batches."""

    def __init__(self) -> None:
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.element_count = 0
        self.cosine_sum = 0.0
        self.cosine_count = 0

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        difference = prediction.float() - target.float()
        self.absolute_error += float(difference.abs().sum())
        self.squared_error += float(difference.square().sum())
        self.element_count += difference.numel()
        valid = target.float().norm(dim=-1) > 1e-8
        if torch.any(valid):
            cosine = functional.cosine_similarity(
                prediction.float()[valid], target.float()[valid], dim=-1
            )
            self.cosine_sum += float(cosine.sum())
            self.cosine_count += cosine.numel()

    def compute(self) -> dict[str, float]:
        if self.element_count == 0:
            raise RuntimeError("cannot compute metrics from an empty loader")
        return {
            "mae": self.absolute_error / self.element_count,
            "rmse": math.sqrt(self.squared_error / self.element_count),
            "cosine": (
                self.cosine_sum / self.cosine_count
                if self.cosine_count > 0
                else float("nan")
            ),
        }

