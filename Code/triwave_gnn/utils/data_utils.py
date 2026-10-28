"""Physical feature transforms and reproducible circuit-level splitting."""

import random

import numpy as np


def feature_stats(values, valid):
    x = np.log1p(np.maximum(np.where(valid, values, 0), 0))
    count = np.maximum(valid.sum(axis=0), 1)
    mean = (x * valid).sum(axis=0) / count
    var = (((x - mean) * valid) ** 2).sum(axis=0) / count
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)


def normalize_features(values, valid, mean, std):
    transformed = np.log1p(np.maximum(np.where(valid, values, 0), 0))
    transformed = np.where(valid, (transformed - mean) / std, 0)
    return np.concatenate((transformed, valid.astype(np.float32)), axis=1)


def split_designs(dataset, seed, max_nodes=None, max_designs=0):
    if max_nodes is not None and max_nodes < 0:
        raise ValueError("max_nodes must be nonnegative")
    if max_designs < 0 or 0 < max_designs < 3:
        raise ValueError("max_designs must be 0 or at least 3 for train/val/test")
    rng = random.Random(seed)
    eligible = [index for index in range(len(dataset.ids))
                if max_nodes is None or int(dataset.graph["num_nodes"][index]) <= max_nodes]
    if max_designs and len(eligible) > max_designs:
        eligible = sorted(rng.sample(eligible, max_designs))
    groups = {}
    for index in eligible:
        name = dataset.ids[index]
        groups.setdefault(name.split("__", 1)[0], []).append(index)
    if len(groups) > 1 and any(len(group) < 6 for group in groups.values()):
        groups = {"all": [index for group in groups.values() for index in group]}
    splits = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * 0.15))
        n_val = max(1, round(len(group) * 0.15))
        splits["test"] += group[:n_test]
        splits["val"] += group[n_test:n_test + n_val]
        splits["train"] += group[n_test + n_val:]
    if not all(splits.values()):
        raise ValueError("split has empty train, validation or test designs")
    return splits
