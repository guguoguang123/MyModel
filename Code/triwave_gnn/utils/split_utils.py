"""Persist and validate a fixed split of whole circuit designs."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np


SPLIT_FORMAT = "fixed-design-split-1.0"
SPLIT_NAMES = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_identity(data_dir):
    """Read only metadata and hash the NPZs; do not load large feature arrays."""
    data_dir = Path(data_dir)
    keys = ("design_ids", "node_mapping_sha256", "netlist_sha256", "node_ptr")
    with np.load(data_dir / "graphs.npz", allow_pickle=False) as graph, np.load(
        data_dir / "labels.npz", allow_pickle=False
    ) as label:
        for key in keys:
            if not np.array_equal(graph[key], label[key]):
                raise ValueError(f"graph/label mismatch: {key}")
        ids = list(map(str, graph["design_ids"]))
        hashes = list(map(str, graph["netlist_sha256"]))
    sources = {name: file_sha256(data_dir / name) for name in ("graphs.npz", "labels.npz")}
    return ids, hashes, sources


def make_fixed_split(ids, netlist_hashes, sources, seed=42):
    if len(ids) != len(set(ids)) or len(ids) != len(netlist_hashes):
        raise ValueError("design IDs must be unique and aligned with netlist hashes")
    groups = {}
    for name, digest in zip(ids, netlist_hashes):
        if not digest:
            raise ValueError(f"missing netlist hash: {name}")
        groups.setdefault(digest, []).append(name)
    if len(groups) < 3:
        raise ValueError("at least three distinct netlists are required")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    test_count = max(1, round(len(order) * 0.15))
    val_count = max(1, round(len(order) * 0.15))
    group_splits = {
        "test": order[:test_count],
        "val": order[test_count:test_count + val_count],
        "train": order[test_count + val_count:],
    }
    splits = {key: sorted(name for digest in group_splits[key] for name in groups[digest])
              for key in SPLIT_NAMES}
    return {
        "format": SPLIT_FORMAT,
        "seed": seed,
        "source_sha256": sources,
        "policy": {
            "unit": "exact_netlist_sha256",
            "parent_source_grouping": "not_verified",
            "evaluation_scope": "internal_dataset",
        },
        "splits": splits,
    }


def validate_fixed_split(manifest, ids, netlist_hashes, sources):
    """Reject changed inputs, missing IDs, duplicate membership and hash overlap."""
    if manifest.get("format") != SPLIT_FORMAT:
        raise ValueError("unsupported fixed split format")
    if manifest.get("source_sha256") != sources:
        raise ValueError("fixed split NPZ hashes differ; prepare a new split for these inputs")
    splits = manifest.get("splits", {})
    if (not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES)
            or any(not isinstance(splits[key], list) or not splits[key]
                   or any(not isinstance(name, str) for name in splits[key])
                   for key in SPLIT_NAMES)):
        raise ValueError("fixed split must have nonempty train, val and test lists")
    selected = [name for key in SPLIT_NAMES for name in splits[key]]
    if len(selected) != len(set(selected)):
        raise ValueError("fixed split contains duplicate or overlapping design IDs")
    if len(ids) != len(set(ids)) or len(ids) != len(netlist_hashes):
        raise ValueError("dataset IDs and netlist hashes are inconsistent")
    if set(selected) != set(ids):
        raise ValueError("fixed split must cover exactly the current dataset design IDs")
    index = {name: i for i, name in enumerate(ids)}
    seen = {}
    for key in SPLIT_NAMES:
        for name in splits[key]:
            digest = netlist_hashes[index[name]]
            if not digest:
                raise ValueError(f"missing netlist hash: {name}")
            if digest in seen and seen[digest] != key:
                raise ValueError("the same netlist hash appears in multiple splits")
            seen[digest] = key
    return {key: [index[name] for name in splits[key]] for key in SPLIT_NAMES}


def load_fixed_split(path, dataset):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_fixed_split(manifest, dataset.ids,
                                list(map(str, dataset.graph["netlist_sha256"])),
                                dataset.source_sha256)


def resolve_split_file(args):
    """Full training uses an explicit manifest; small checks may sample designs."""
    if args.split_mode == "random":
        if args.split_file is not None:
            raise ValueError("--split-file requires --split-mode fixed")
        return None
    if args.max_nodes or args.max_designs:
        raise ValueError("sampling limits require --split-mode random or train_small.py")
    path = args.split_file or args.data / "split.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed split {path}; run examples/prepare_split.py first")
    return path.resolve()
