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


def split_designs(dataset, seed, max_nodes=None):
    rng = random.Random(seed)
    groups = {}
    for index, name in enumerate(dataset.ids):
        if max_nodes is not None and int(dataset.graph["num_nodes"][index]) > max_nodes:
            continue
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
