"""Leakage-safe circuit splits grouped by original design family."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CircuitSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    validation_ratio: float
    test_ratio: float

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = asdict(self)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CircuitSplit":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            test=tuple(payload["test"]),
            seed=int(payload["seed"]),
            validation_ratio=float(payload["validation_ratio"]),
            test_ratio=float(payload["test_ratio"]),
        )

    def validate_against(self, available_names: Sequence[str]) -> None:
        available = set(available_names)
        groups = (set(self.train), set(self.validation), set(self.test))
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("saved split contains overlapping circuit names")
        selected = groups[0] | groups[1] | groups[2]
        missing = selected - available
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"saved split references missing circuits: {preview}")
        if not self.train or not self.validation or not self.test:
            raise ValueError("train, validation and test splits must all be non-empty")


def design_family(name: str) -> str:
    """Map synthesized variants such as ``c1335_syn_000`` to ``c1335``."""
    return name.rsplit("_syn_", 1)[0] if "_syn_" in name else name


def split_by_design_family(
    names: Sequence[str],
    *,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
) -> CircuitSplit:
    if not names:
        raise ValueError("cannot split an empty circuit collection")
    if validation_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than one")
    families = sorted({design_family(name) for name in names})
    if len(families) < 3:
        raise ValueError("at least three design families are required")
    random.Random(seed).shuffle(families)

    validation_count = max(1, round(len(families) * validation_ratio))
    test_count = max(1, round(len(families) * test_ratio))
    while validation_count + test_count >= len(families):
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("ratios leave no design family for training")
    test_families = set(families[:test_count])
    validation_families = set(
        families[test_count : test_count + validation_count]
    )
    train_families = set(families) - test_families - validation_families

    split = CircuitSplit(
        train=tuple(name for name in names if design_family(name) in train_families),
        validation=tuple(
            name for name in names if design_family(name) in validation_families
        ),
        test=tuple(name for name in names if design_family(name) in test_families),
        seed=seed,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    split.validate_against(names)
    return split
