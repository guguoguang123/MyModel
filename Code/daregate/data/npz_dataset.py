"""Read aligned graph and merged supervision NPZ archives."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional display dependency
    tqdm = None

from ..training.batch import DAREGateBatch
from ..utils.topology import bidirectional_levels, reconvergence_sources


def _load_object_dict(path: Path, member: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        if member not in archive.files:
            raise KeyError(f"{path} does not contain '{member}'")
        value = archive[member]
        if value.shape != ():
            raise ValueError(f"{path}:{member} must be a scalar object dictionary")
        result = value.item()
    if not isinstance(result, dict):
        raise TypeError(f"{path}:{member} is not a dictionary")
    return result


class DAREGateNPZDataset(Dataset):
    """Aligned in-memory view of graphs and merged supervision labels.

    These NPZ files contain object dictionaries, so each archive is fully
    deserialized once. Reuse one dataset object and prefer ``num_workers=0``
    unless the data is first converted to a sharded tensor format.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        names: Sequence[str] | None = None,
        derive_reconvergence_sources: bool = True,
        validate: bool = True,
        precompute_topology: bool = True,
    ) -> None:
        super().__init__()
        root = Path(data_dir)
        self.graphs = _load_object_dict(root / "graphs.npz", "circuits")
        self.labels = _load_object_dict(root / "finalLabels.npz", "labels")
        graph_names = set(self.graphs)
        label_names = set(self.labels)
        if graph_names != label_names:
            missing_labels = graph_names - label_names
            missing_graphs = label_names - graph_names
            raise ValueError(
                "graphs.npz and finalLabels.npz are not aligned: "
                f"missing_labels={len(missing_labels)}, "
                f"missing_graphs={len(missing_graphs)}"
            )
        common = graph_names
        if names is None:
            selected = [name for name in self.graphs if name in common]
        else:
            selected = list(names)
            missing = set(selected) - common
            if missing:
                preview = ", ".join(sorted(missing)[:5])
                raise KeyError(f"requested circuits are missing aligned data: {preview}")
        if not selected:
            raise ValueError("no aligned circuits were found")
        self.names = tuple(selected)
        self.derive_reconvergence_sources = derive_reconvergence_sources
        self.validate_items = validate

        self._topology_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = {}
        self.topology_precompute_seconds = 0.0
        if precompute_topology:
            self.prepare_topology()

    def _graph_tensors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.graphs[name]
        x = np.asarray(graph["x"])
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"{name}: x must have at least two columns")
        gate_type = torch.from_numpy(
            x[:, 1].astype(np.int64, copy=False)
        ).long()
        edge_array = np.asarray(graph["edge_index"], dtype=np.int64)
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError(f"{name}: edge_index must have shape [E, 2]")
        edge_index = torch.from_numpy(edge_array.T.copy()).long()
        return gate_type, edge_index

    def _topology_for(
        self, name: str, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cached = self._topology_cache.get(name)
        if cached is not None:
            return cached
        forward_level, backward_level = bidirectional_levels(
            edge_index, num_nodes
        )
        rc_source = (
            reconvergence_sources(edge_index, forward_level)
            if self.derive_reconvergence_sources
            else None
        )
        topology = (forward_level, backward_level, rc_source)
        self._topology_cache[name] = topology
        return topology

    def prepare_topology(self) -> None:
        """Parse and cache every selected circuit before training starts."""
        started = time.perf_counter()
        names = self.names
        if tqdm is not None:
            names = tqdm(
                names,
                desc=(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} prepare topology"
                ),
                unit="circuit",
                dynamic_ncols=True,
                leave=True,
                disable=None,
            )
        for name in names:
            gate_type, edge_index = self._graph_tensors(name)
            self._topology_for(name, edge_index, gate_type.numel())
        self.topology_precompute_seconds = time.perf_counter() - started

    @property
    def topology_cache_size(self) -> int:
        return len(self._topology_cache)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> DAREGateBatch:
        name = self.names[index]
        gate_type, edge_index = self._graph_tensors(name)
        labels = self.labels[name]

        forward_level, backward_level, rc_source = self._topology_for(
            name, edge_index, gate_type.numel()
        )

        signature = torch.from_numpy(np.asarray(labels["S"], dtype=np.float32))
        response = torch.from_numpy(
            np.asarray(labels["R"], dtype=np.float32).transpose(1, 0, 2).copy()
        )
        batch = DAREGateBatch(
            gate_type=gate_type,
            edge_index=edge_index,
            forward_level=forward_level,
            backward_level=backward_level,
            rc_source=rc_source,
            signature=signature,
            response=response,
            probability=torch.from_numpy(
                np.asarray(labels["prob"], dtype=np.float32)
            ),
            tt_pair_index=torch.from_numpy(
                np.asarray(labels["tt_pair_index"], dtype=np.int64)
                .reshape(-1, 2).T.copy()
            ).long(),
            tt_distance=torch.from_numpy(
                np.asarray(labels["tt_dis"], dtype=np.float32)
            ),
            min_tt_distance=torch.from_numpy(
                np.asarray(labels["min_tt_dis"], dtype=np.float32)
            ),
            rc_pair_index=torch.from_numpy(
                np.asarray(labels["rc_pair_index"], dtype=np.int64)
                .reshape(-1, 2).T.copy()
            ).long(),
            is_reconvergent=torch.from_numpy(
                np.asarray(labels["is_rc"], dtype=np.float32)
            ),
            batch_index=torch.zeros(gate_type.numel(), dtype=torch.long),
            names=(name,),
        )
        if self.validate_items:
            batch.validate(expected_signature_dim=signature.size(1))
        return batch
