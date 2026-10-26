"""Read the two concatenated, pickle-free physical training bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.schema import validate_bundle
from .utils.dag_utils import node_depth
from .utils.data_utils import feature_stats, normalize_features
from .parser_func import PinCircuitData


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class CircuitDataset(Dataset):
    def __init__(self, root: str | Path, cache_mode: str = "auto"):
        root = Path(root)
        self.source_sha256 = {}
        for name in ("graphs.npz", "labels.npz"):
            digest = hashlib.sha256()
            with (root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            self.source_sha256[name] = digest.hexdigest()
        self.graph = _load(root / "graphs.npz")
        self.label = _load(root / "labels.npz")
        g, y = self.graph, self.label
        validate_bundle(g, y)
        self.ids = [str(x) for x in g["design_ids"]]
        self.logic_vocab = [str(x) for x in g["logic_vocab"]]
        self.role_vocab = [str(x) for x in g["role_vocab"]]
        self.pi_role_index = self.role_vocab.index("pi")
        self.pin_vocab = sorted(set(str(x) for x in g["node_pin_name"]))
        pin_lookup = {name: index for index, name in enumerate(self.pin_vocab)}
        self._pin_ids = np.array([pin_lookup[str(x)] for x in g["node_pin_name"]], np.int64)
        self.net_mean, self.net_std = feature_stats(g["net_cause_features"], g["net_cause_valid"])
        self.cell_mean, self.cell_std = feature_stats(g["cell_cause_features"], g["cell_cause_valid"])
        if cache_mode not in ("auto", "all", "small", "none"):
            raise ValueError("cache_mode must be auto, all, small or none")
        self.cache_mode = "all" if cache_mode == "auto" and len(g["node_role"]) <= 1_000_000 else cache_mode
        if self.cache_mode == "auto":
            self.cache_mode = "small"
        self._cache = {}

    def fit_normalization(self, design_indices):
        """Fit physical feature scaling on training designs only."""
        g = self.graph
        for relation in ("net", "cell"):
            ptr = g[relation + "_edge_ptr"]
            selected = np.concatenate([
                np.arange(ptr[i], ptr[i + 1], dtype=np.int64) for i in design_indices
            ])
            mean, std = feature_stats(
                g[relation + "_cause_features"][selected],
                g[relation + "_cause_valid"][selected],
            )
            setattr(self, relation + "_mean", mean)
            setattr(self, relation + "_std", std)
        self._cache.clear()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> PinCircuitData:
        if index in self._cache:
            return self._cache[index]
        g, y = self.graph, self.label
        n0, n1 = map(int, g["node_ptr"][index:index + 2])
        e0, e1 = map(int, g["net_edge_ptr"][index:index + 2])
        c0, c1 = map(int, g["cell_edge_ptr"][index:index + 2])
        h0, h1 = map(int, y["hs_pair_ptr"][index:index + 2])
        f0, f1 = map(int, y["hf_tt_pair_ptr"][index:index + 2])
        def tensor(a, dtype=None):
            return torch.as_tensor(np.array(a, copy=True), dtype=dtype)
        def features(prefix, start, stop, mean, std):
            valid = g[prefix + "_cause_valid"][start:stop]
            raw = g[prefix + "_cause_features"][start:stop]
            return tensor(normalize_features(raw, valid, mean, std), torch.float32)
        item = PinCircuitData()
        item.name = self.ids[index]
        item["pin"].num_nodes = n1 - n0
        for key, value in {
            "node_role": tensor(g["node_role"][n0:n1], torch.long),
            "node_logic": tensor(g["node_logic_type"][n0:n1], torch.long),
            "node_pin": tensor(self._pin_ids[n0:n1], torch.long),
            "topological_order": tensor(g["topological_order"][n0:n1] - n0, torch.long),
            "hf_prob": tensor(y["hf_prob"][n0:n1], torch.float32),
        }.items():
            item["pin"][key] = value
        item["pin"].is_pi = item["pin"].node_role == self.pi_role_index
        item["pin", "net", "pin"].edge_index = tensor(g["net_edge_index"][:, e0:e1] - n0, torch.long)
        item["pin", "cell", "pin"].edge_index = tensor(g["cell_edge_index"][:, c0:c1] - n0, torch.long)
        item["pin", "net", "pin"].edge_attr = features("net", e0, e1, self.net_mean, self.net_std)
        item["pin", "cell", "pin"].edge_attr = features("cell", c0, c1, self.cell_mean, self.cell_std)
        for key, value in {
            "hs_pairs": tensor(y["hs_pair_node_index"][:, h0:h1] - n0, torch.long),
            "hf_pairs": tensor(y["hf_tt_pair_node_index"][:, f0:f1] - n0, torch.long),
            "hs_target": tensor(y["hs_is_rc"][h0:h1], torch.float32),
            "hs_mask": tensor(y["hs_pair_mask"][h0:h1], torch.bool),
            "hf_tt_target": tensor(y["hf_tt_distance"][f0:f1], torch.float32),
            "hf_tt_mask": tensor(y["hf_tt_pair_mask"][f0:f1], torch.bool),
        }.items():
            item[key] = value
        # A topological scan updates every pin once. Group edges by destination
        # depth so all independent pins at a depth run in one tensor operation.
        depth = node_depth(g["topological_order"][n0:n1] - n0,
                           g["net_edge_index"][:, e0:e1] - n0,
                           g["cell_edge_index"][:, c0:c1] - n0, n1 - n0)
        item["pin"].depth = tensor(depth, torch.long)
        for key in ("hb_tp", "hb_tp_01", "hb_tp_10", "hb_first_mean", "hb_stable_mean", "hb_rise_mean", "hb_fall_mean", "hb_glitch_rate", "hb_conditional_extra", "hb_unstable_window"):
            item["pin"][key] = tensor(y[key][n0:n1], torch.float32)
        for key in ("hb_tp_mask", "hb_tp_01_mask", "hb_tp_10_mask", "hb_first_train_mask", "hb_stable_train_mask", "hb_rise_train_mask", "hb_fall_train_mask", "hb_glitch_rate_train_mask", "hb_glitch_severity_train_mask", "hb_unstable_window_mask"):
            item["pin"][key] = tensor(y[key][n0:n1], torch.bool)
        # Large bundles cache small circuits; all graphs stay on the CPU here.
        if self.cache_mode == "all" or (self.cache_mode == "small" and n1 - n0 < 10000):
            self._cache[index] = item
        return item
