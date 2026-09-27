"""Classified, atomic persistence for training artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from ..model import DAREGateOutput
from .batch import DAREGateBatch


class TrainingOutputStore:
    """Own the stable directory layout below one experiment output root."""

    CATEGORIES = ("checkpoints", "logs", "metrics", "predictions", "embeddings")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.directories = {
            name: self.root / name for name in self.CATEGORIES
        }
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

    def directory(self, category: str) -> Path:
        if category not in self.directories:
            raise KeyError(f"unknown output category: {category}")
        return self.directories[category]

    @staticmethod
    def _save_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def write_epoch(self, record: dict[str, Any], history: list[dict[str, Any]]) -> None:
        self._save_json(self.directory("logs") / "history.json", history)
        epoch = int(record["epoch"])
        self._save_json(
            self.directory("metrics") / f"epoch_{epoch:04d}.json", record
        )

    @staticmethod
    def _atomic_torch_save(value: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def save_batch(
        self,
        split: str,
        batch_number: int,
        batch: DAREGateBatch,
        output: DAREGateOutput,
    ) -> tuple[Path, Path]:
        embedding_dir = self.directory("embeddings") / split
        prediction_dir = self.directory("predictions") / split
        embedding_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        filename = f"batch_{batch_number:06d}.pt"
        embedding_path = embedding_dir / filename
        prediction_path = prediction_dir / filename
        common = {
            "names": batch.names,
            "batch_index": (
                batch.batch_index.detach().cpu()
                if batch.batch_index is not None
                else None
            ),
        }
        embeddings = {
            **common,
            "structure": output.structure.detach().cpu(),
            "function": output.function.detach().cpu(),
            "behavior": output.behavior.detach().cpu(),
            "probe_behavior": output.probe_behavior.detach().cpu(),
        }
        predictions = {
            **common,
            "response": output.response.detach().cpu(),
            "signature": output.signature.detach().cpu(),
            "probability": output.probability.detach().cpu(),
            "reconvergence_probability": (
                output.reconvergence_probability.detach().cpu()
                if output.reconvergence_probability is not None else None
            ),
            "functional_distance": (
                output.functional_distance.detach().cpu()
                if output.functional_distance is not None else None
            ),
        }
        self._atomic_torch_save(embeddings, embedding_path)
        self._atomic_torch_save(predictions, prediction_path)
        return embedding_path, prediction_path
