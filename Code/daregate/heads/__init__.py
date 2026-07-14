"""Task-specific supervision heads."""

from .deepgate2 import (
    FunctionalDistanceHead,
    ProbabilityHead,
    ReconvergenceHead,
)

__all__ = ["FunctionalDistanceHead", "ProbabilityHead", "ReconvergenceHead"]
