"""Shared dataset, split and DataLoader construction for training CLIs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ..data import (
    CircuitSplit,
    DAREGateNPZDataset,
    collate_daregate,
    split_by_design_family,
)


def save_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def create_or_load_split(
    dataset: DAREGateNPZDataset,
    split_path: str | Path,
    *,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    require_existing: bool,
) -> CircuitSplit:
    path = Path(split_path)
    if path.is_file():
        split = CircuitSplit.load(path)
    elif require_existing:
        raise FileNotFoundError(
            f"resume requires the original split file: {path}"
        )
    else:
        split = split_by_design_family(
            dataset.names,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        split.save(path)
    split.validate_against(dataset.names)
    return split


def build_loaders(
    dataset: DAREGateNPZDataset,
    split: CircuitSplit,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    index = {name: position for position, name in enumerate(dataset.names)}

    def subset(names: tuple[str, ...]) -> Subset:
        return Subset(dataset, [index[name] for name in names])

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_daregate,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        subset(split.train), shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(
        subset(split.validation), shuffle=False, **common
    )
    test_loader = DataLoader(subset(split.test), shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def print_data_summary(dataset: DAREGateNPZDataset, split: CircuitSplit) -> None:
    print(
        "aligned circuits: "
        f"total={len(dataset)} train={len(split.train)} "
        f"validation={len(split.validation)} test={len(split.test)}"
    )
    print(
        "topology ready before training: "
        f"cached={dataset.topology_cache_size} circuits "
        f"seconds={dataset.topology_precompute_seconds:.2f}"
    )
    print(
        "NPZ archives are object dictionaries; num_workers=0 is the safest "
        "setting and avoids worker-level memory duplication."
    )


def infer_output_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    raise ValueError(
        "cannot infer output_dir: checkpoint must be inside output/checkpoints"
    )
