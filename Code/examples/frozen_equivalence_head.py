"""Bounded development experiment: frozen B embeddings and a symmetric MLP.

Only --run reads selected real graphs. --self-check uses synthetic tensors.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.compare_functional_stages import cosine_scores64, file_hash
from examples.evaluate import validate_splits
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.model import TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.equivalence_audit import pair_metrics, pair_count_group_metrics
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.net_identity import checkpoint_net_identity
from triwave_gnn.utils.random_seed import set_seed

EXP = Path(__file__).resolve().parent / "exp"
B_RUN = EXP / "functional_pair_sampling_stage3/circuit_cap"
TRAIN_STATS = EXP / "be_full_train_equivalence_20261112_080356_866135/B.json"
NOTICE = "小规模开发试验，不能与完整验证集成绩直接比较，也不是最终测试成绩"
GROUPS = ("1-999", "1000-9999", ">=10000")
SEED, EPOCHS, BATCH_SIZE, MAX_PER_CLASS = 42, 10, 1024, 25000


def pair_features(left, right):
    """Inputs are L2-normalized hf only; symmetric in the two endpoints."""
    return torch.cat(((left - right).abs(), left * right), dim=-1)


def make_head(hidden):
    return nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))


def freeze_encoder(model):
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    model.functional_net_identity = False


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group_name(count):
    return GROUPS[0] if count < 1000 else GROUPS[1] if count < 10000 else GROUPS[2]


def choose_circuits(rows, split_names, size, seed, require_mixed):
    # Read only identity and class counts, never a stored prediction/metric.
    counts = {}
    for row in rows:
        name = row["circuit"]
        if name in counts:
            raise ValueError(f"Duplicate circuit statistics: {name}")
        counts[name] = {"circuit": name, "positive_count": int(row["positive_count"]),
                        "negative_count": int(row["negative_count"])}
    if not set(split_names).issubset(counts):
        raise ValueError("Saved statistics do not cover the requested split")
    pools = {key: [] for key in GROUPS}
    for name in sorted(split_names):
        row = counts[name]
        total = row["positive_count"] + row["negative_count"]
        if total > 0 and (not require_mixed or min(row["positive_count"], row["negative_count"]) > 0):
            pools[group_name(total)].append({**row, "original_valid_pairs": total,
                                              "pair_count_group": group_name(total)})
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    # Round robin gives 22/21/21 (train), 11/11/10 (val) if all pools suffice.
    while len(selected) < size:
        previous = len(selected)
        for key in GROUPS:
            if pools[key] and len(selected) < size:
                selected.append(pools[key].pop())
        if len(selected) == previous:
            raise ValueError(f"Only {len(selected)} eligible circuits; require {size}; stopping")
    if not any(row["positive_count"] for row in selected) or not any(row["negative_count"] for row in selected):
        raise ValueError("Selected subset needs both classes; no adaptive reselection")
    return selected


class NpzSlices:
    """Read only small metadata or selected NPY ranges inside an NPZ archive.

    Compressed ZIP streams may decompress skipped prefixes, but no full array
    is materialized, hashed, validated, or used to inspect unselected graphs.
    Streams stay open and circuits are read in archive order to avoid rewinds.
    """
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        self.members = {}
        self.lanes = {}

    def close(self):
        for stream, *_ in self.members.values():
            stream.close()
        for stream in self.lanes.values():
            stream.close()
        self.archive.close()

    def metadata(self, key):
        if key not in self.members:
            stream = self.archive.open(key + ".npy")
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                stream.close()
                raise ValueError(f"Unsupported NPY header version {version}")
            if dtype.hasobject or len(shape) not in (1, 2):
                stream.close()
                raise ValueError(f"Unsupported bounded slice dtype/shape: {key}")
            self.members[key] = stream, shape, fortran, dtype, stream.tell()
        return self.members[key]

    def read(self, key, start=None, stop=None, axis=0):
        stream, shape, fortran, dtype, offset = self.metadata(key)
        start = 0 if start is None else int(start)
        stop = shape[axis] if stop is None else int(stop)
        if not 0 <= start <= stop <= shape[axis]:
            raise ValueError(f"Invalid selected slice {key}: {start}:{stop}")

        def block(element_offset, count, lane=None):
            source = stream
            if lane is not None:
                # Independent row/column streams keep selected reads monotonic
                # for C-order [2, pairs] indices, avoiding repeated decompression.
                lane_key = key, lane
                if lane_key not in self.lanes:
                    self.lanes[lane_key] = self.archive.open(key + ".npy")
                source = self.lanes[lane_key]
            source.seek(offset + element_offset * dtype.itemsize)
            raw = source.read(count * dtype.itemsize)
            if len(raw) != count * dtype.itemsize:
                raise ValueError(f"Truncated NPZ member: {key}")
            return np.frombuffer(raw, dtype=dtype).copy()

        if len(shape) == 1:
            return block(start, stop - start)
        rows, cols = shape
        if not fortran and axis == 0:
            return block(start * cols, (stop - start) * cols).reshape(stop - start, cols)
        if fortran and axis == 1:
            return block(start * rows, (stop - start) * rows).reshape(rows, stop - start, order="F")
        if axis == 1:
            return np.stack([block(i * cols + start, stop - start, lane=i) for i in range(rows)])
        return np.stack([block(j * rows + start, stop - start, lane=j) for j in range(cols)], axis=1)


class SelectedReader:
    """Reuse CircuitDataset conversion on one selected local bundle at a time."""
    def __init__(self, graph, label, state):
        self.graph, self.label, self.state = graph, label, state
        self.ids = [str(x) for x in graph.read("design_ids")]
        if self.ids != [str(x) for x in label.read("design_ids")]:
            raise ValueError("Graph/label design metadata differ")
        self.lookup = {name: index for index, name in enumerate(self.ids)}
        self.ptr = {key: graph.read(key) for key in ("node_ptr", "net_edge_ptr", "cell_edge_ptr")}
        self.ptr.update({key: label.read(key) for key in ("hs_pair_ptr", "hf_tt_pair_ptr")})
        if not np.array_equal(self.ptr["node_ptr"], label.read("node_ptr")):
            raise ValueError("Graph/label node pointer metadata differ")
        self.logic_vocab = [str(x) for x in graph.read("logic_vocab")]
        self.role_vocab = [str(x) for x in graph.read("role_vocab")]
        self.pin_vocab = state["pin_vocab"]
        # load_model reuses these attributes and applies saved normalization.
        self._cache = {}

    def bounds(self, name, key):
        index = self.lookup[name]
        return tuple(map(int, self.ptr[key][index:index + 2]))

    def item(self, name):
        n0, n1 = self.bounds(name, "node_ptr")
        graph = {"node_ptr": np.array([0, n1 - n0])}
        for key in ("node_role", "node_logic_type", "node_pin_name", "topological_order"):
            graph[key] = self.graph.read(key, n0, n1)
        graph["topological_order"] -= n0
        for relation in ("net", "cell"):
            a, b = self.bounds(name, relation + "_edge_ptr")
            graph[relation + "_edge_ptr"] = np.array([0, b - a])
            graph[relation + "_edge_index"] = self.graph.read(relation + "_edge_index", a, b, axis=1) - n0
            for suffix in ("_cause_features", "_cause_valid"):
                graph[relation + suffix] = self.graph.read(relation + suffix, a, b)
        label = {}
        for prefix, index_key, keys in (
            ("hs", "hs_pair_node_index", ("hs_is_rc", "hs_pair_mask")),
            ("hf_tt", "hf_tt_pair_node_index", ("hf_tt_distance", "hf_tt_pair_mask")),
        ):
            a, b = self.bounds(name, prefix + "_pair_ptr")
            label[prefix + "_pair_ptr"] = np.array([0, b - a])
            label[index_key] = self.label.read(index_key, a, b, axis=1) - n0
            for key in keys:
                label[key] = self.label.read(key, a, b)
        # Only selected node slices; these labels never enter head features.
        for key in ("hf_prob", "hb_tp", "hb_tp_01", "hb_tp_10", "hb_first_mean",
                    "hb_stable_mean", "hb_rise_mean", "hb_fall_mean", "hb_glitch_rate",
                    "hb_conditional_extra", "hb_unstable_window", "hb_tp_mask",
                    "hb_tp_01_mask", "hb_tp_10_mask", "hb_first_train_mask",
                    "hb_stable_train_mask", "hb_rise_train_mask", "hb_fall_train_mask",
                    "hb_glitch_rate_train_mask", "hb_glitch_severity_train_mask", "hb_unstable_window_mask"):
            label[key] = self.label.read(key, n0, n1)
        # Bypass only the full-bundle constructor; conversion is unchanged.
        local = CircuitDataset.__new__(CircuitDataset)
        local.graph, local.label, local.ids = graph, label, [name]
        local.logic_vocab, local.role_vocab = self.logic_vocab, self.role_vocab
        local.pi_role_index = self.role_vocab.index("pi")
        pins = {pin: i for i, pin in enumerate(self.pin_vocab)}
        local._pin_ids = np.array([pins[str(x)] for x in graph["node_pin_name"]], dtype=np.int64)
        local.cache_mode, local._cache = "none", {}
        for relation in ("net", "cell"):
            for suffix in ("_mean", "_std"):
                setattr(local, relation + suffix, getattr(self, relation + suffix))
        item = local[0]
        pairs = item["hf_pairs"]
        if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= n1 - n0):
            raise ValueError(f"Selected functional pair leaves circuit: {name}")
        return item


def resource_plan(reader, rows, hidden, output):
    nodes = pairs = max_graph_bytes = 0
    for row in rows:
        name = row["circuit"]
        a, b = reader.bounds(name, "node_ptr")
        count = b - a
        nodes += count
        pairs += row["original_valid_pairs"]
        net = np.diff(reader.bounds(name, "net_edge_ptr"))[0]
        cell = np.diff(reader.bounds(name, "cell_edge_ptr"))[0]
        raw_pairs = np.diff(reader.bounds(name, "hf_tt_pair_ptr"))[0]
        hs_pairs = np.diff(reader.bounds(name, "hs_pair_ptr"))[0]
        max_graph_bytes = max(max_graph_bytes, int(count * (8 * hidden * 4 + 512)
                              + (net + cell) * 256 + (raw_pairs + hs_pairs) * 64))
    cache_bytes = nodes * hidden * 8 + pairs * 33  # raw + normalized hf; indices, columns, labels, cosine
    pair_buffers = pairs * 128  # metrics sorting, class indices, score concatenation
    train_features = 2 * MAX_PER_CLASS * 2 * hidden * 4
    working_bytes = cache_bytes * 2 + pair_buffers + train_features * 3 + max_graph_bytes * 4 + (256 << 20)
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    for limit_path, used_path in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                  ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        limit, used = Path(limit_path), Path(used_path)
        if limit.exists() and used.exists() and limit.read_text().strip() != "max":
            available = min(available, max(0, int(limit.read_text()) - int(used.read_text())))
    disk = shutil.disk_usage(output).free
    result = dict(selected_nodes=nodes, selected_valid_pairs=pairs,
                  estimated_cache_bytes=cache_bytes, estimated_peak_host_bytes=working_bytes,
                  available_host_bytes=available, free_disk_bytes=disk,
                  estimated_peak_graph_device_bytes=max_graph_bytes * 2 + (256 << 20),
                  policy="stop if cache >2 GiB, estimated host peak >50% available RAM, or disk requirement >50% free; no reselection")
    if cache_bytes > (2 << 30) or working_bytes > available // 2 or cache_bytes * 3 > disk // 2:
        raise MemoryError(f"Selected cache exceeds resource budget; stopping without fallback: {result}")
    return result


def extract_once(reader, model, rows, device, directory):
    directory.mkdir()
    manifest = []
    for number, row in enumerate(sorted(rows, key=lambda r: reader.lookup[r["circuit"]]), 1):
        item = reader.item(row["circuit"])
        target = item["hf_tt_target"]
        valid = item["hf_tt_mask"].bool() & torch.isfinite(target)
        labels = torch.isclose(target[valid], torch.zeros_like(target[valid]), atol=1e-8, rtol=0)
        actual = (int(labels.sum()), labels.numel() - int(labels.sum()))
        if actual != (row["positive_count"], row["negative_count"]):
            raise ValueError(f"Selected circuit counts differ from saved statistics: {row['circuit']}")
        with torch.inference_mode():
            _, hf, _ = model(move(item, device), include_behavior=False)
            hf = hf.cpu()
            if not bool(torch.isfinite(hf).all()):
                raise ValueError(f"Nonfinite hf: {row['circuit']}")
            # Original cosine uses raw float32 hf converted to float64 first.
            pairs = item["hf_pairs"].cpu()[:, valid]
            cosine = cosine_scores64(hf, pairs, torch.ones(labels.numel(), dtype=torch.bool), 65536)
            cached = {"hf_raw": hf, "hf_l2": F.normalize(hf, p=2, dim=-1, eps=1e-12),
                      "pairs": pairs, "labels": labels, "cosine_scores64": cosine,
                      "original_pair_columns": torch.nonzero(valid).flatten(),
                      "metadata": row}
            path = directory / f"circuit_{number:03d}.pt"
            atomic_save(cached, path)
        manifest.append({**row, "cache_file": str(path.resolve()), "node_count": hf.shape[0]})
        info(f"Cached encoder forward {number}/{len(rows)}: {row['circuit']} | positive={actual[0]} negative={actual[1]}")
        del item, hf, cached
    return manifest


def load_caches(manifest, split):
    return [torch.load(row["cache_file"], map_location="cpu", weights_only=False)
            for row in manifest if row["split"] == split]


def fixed_training_pairs(caches):
    totals = [sum(int((cache["labels"] == value).sum()) for cache in caches) for value in (True, False)]
    count = min(MAX_PER_CLASS, *totals)
    if not count:
        raise ValueError("Training subset has no positive or negative pairs")
    rng = np.random.default_rng(SEED)
    parts, labels, selections = [], [], []
    for value, total in zip((True, False), totals):
        # Uniform class ranks over ALL selected training pairs, without replacement.
        ranks = np.sort(rng.choice(total, size=count, replace=False))
        offset = 0
        for cache in caches:
            columns = torch.nonzero(cache["labels"] == value).flatten()
            local_ranks = ranks[(ranks >= offset) & (ranks < offset + columns.numel())] - offset
            selected = columns[torch.from_numpy(local_ranks)]
            offset += columns.numel()
            if not selected.numel():
                continue
            left, right = cache["pairs"][:, selected]
            parts.append(pair_features(cache["hf_l2"][left], cache["hf_l2"][right]))
            labels.append(torch.full((selected.numel(),), float(value)))
            selections.append({"circuit": cache["metadata"]["circuit"], "positive": value,
                               "cache_pair_columns": selected,
                               "original_pair_columns": cache["original_pair_columns"][selected]})
    return torch.cat(parts), torch.cat(labels), selections, {
        "available_positive": totals[0], "available_negative": totals[1],
        "selected_positive": count, "selected_negative": count,
        "sampling": "uniform within each class across selected train circuits; without replacement; seed=42; fixed across epochs"}


def report_scores(scores, labels, offsets, score_kind):
    threshold = select_threshold(scores, labels)
    result = {"score_kind": score_kind, "threshold_selection": "maximum pooled F1 on this validation subset only",
              **pair_metrics(scores, labels, threshold),
              "groups_by_original_valid_pairs": pair_count_group_metrics(scores, labels, offsets, threshold)}
    for tag, row in [("pooled", result), *result["groups_by_original_valid_pairs"].items()]:
        info(f"{score_kind} / {tag}: " + " | ".join(f"{key}={row[key]}" for key in
             ("precision", "recall", "f1", "roc_auc", "tp", "fp", "fn")))
    info(f"{score_kind} validation-subset maximum-F1 threshold={threshold}")
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with direct_log(args.output / "console.log"), ExitStack() as stack:
        record = {"status": "preparing", "notice": NOTICE, "seed": SEED,
                  "test_evaluated": False, "functional_net_identity": False,
                  "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": 1e-3,
                  "head": "Linear(2*hidden,64) -> ReLU -> Linear(64,1) logit", "optimizer": "Adam",
                  "loss": "BCEWithLogitsLoss", "encoder_mode": "all parameters frozen; eval; inference_mode for extraction",
                  "head_selection": "final epoch 10; no validation-based epoch selection",
                  "feature": "concat(abs(L2(hf_u)-L2(hf_v)), L2(hf_u)*L2(hf_v)); hf only",
                  "selection_rule": "seed=42; sorted IDs shuffled independently within original-pair-count groups; round robin groups, redistribute exhausted slots; all train circuits must have both classes; no prediction-based selection",
                  "dataset_check_scope": "metadata and selected circuit slices only; no full-bundle scan, hash, or validate_bundle",
                  "source": str(Path(__file__).resolve()), "source_sha256": file_hash(Path(__file__)),
                  "device": args.device}
        write_json(args.output / "run.json", record)
        info(NOTICE)
        checkpoint_path = B_RUN / "best_equiv.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_net_identity(state):
            raise ValueError("B checkpoint unexpectedly enables functional_net_identity")
        splits = json.loads((B_RUN / "split.json").read_text())
        val_key = validate_splits(splits, state)
        epoch = state["epoch"]
        validation_path = B_RUN / (f"val_epoch_{epoch}.json" if epoch else "baseline.json")
        validation = json.loads(validation_path.read_text())
        if (validation["epoch"] != epoch or validation["functional_equivalence"]["f1"] != state["validation_f1"]
                or validation["functional_equivalence"]["similarity_threshold"] != state["validation_threshold"]):
            raise ValueError("B best_equiv checkpoint epoch does not match its saved validation record")
        train_saved = json.loads(TRAIN_STATS.read_text())
        checkpoint_hash = file_hash(checkpoint_path)
        if train_saved["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Saved train statistics refer to a different B checkpoint")
        selected_train = choose_circuits(train_saved["train"]["per_circuit"], splits["train"], 64, SEED, True)
        selected_val = choose_circuits(validation["per_circuit"], splits[val_key], 32, SEED, False)
        rows = [{**row, "split": split} for split, selected in (("train", selected_train), ("val", selected_val)) for row in selected]
        graph, label = NpzSlices(args.data / "graphs.npz"), NpzSlices(args.data / "labels.npz")
        stack.callback(graph.close)
        stack.callback(label.close)
        reader = SelectedReader(graph, label, state)
        del train_saved, validation
        hidden = state["config"]["hidden"]
        resources = resource_plan(reader, rows, hidden, args.output)
        record.update(checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
                      checkpoint_actual_epoch=epoch, resources=resources,
                      source_bundle_sha256_from_checkpoint_unverified=state["source_sha256"],
                      input_files={name: {"size": (args.data / name).stat().st_size,
                                          "mtime_ns": (args.data / name).stat().st_mtime_ns}
                                   for name in ("graphs.npz", "labels.npz")},
                      statistics_sources={"train": str(TRAIN_STATS.resolve()), "val": str(validation_path.resolve())})
        write_json(args.output / "selection.json", {"rule": record["selection_rule"], "seed": SEED, "circuits": rows,
                   "totals": {split: {"circuits": len(selected),
                                      "positive_count": sum(row["positive_count"] for row in selected),
                                      "negative_count": sum(row["negative_count"] for row in selected),
                                      "groups": {key: sum(row["pair_count_group"] == key for row in selected) for key in GROUPS}}
                              for split, selected in (("train", selected_train), ("val", selected_val))}})
        record["status"] = "extracting"
        write_json(args.output / "run.json", record)
        info(f"B best_equiv actual_epoch={epoch}, original propagation, frozen eval encoder")
        info(f"Resource estimate: {resources}")
        set_seed(SEED)
        model, _ = load_model(checkpoint_path, reader, "cpu")
        freeze_encoder(model)
        before = state_digest(model)
        if args.device.startswith("cuda"):
            free_gpu, _ = torch.cuda.mem_get_info(torch.device(args.device))
            # Conservative graph-state allowance; stop rather than switch device/subset.
            gpu_estimate = resources["estimated_peak_graph_device_bytes"]
            record["gpu_resource_estimate"] = {"estimated_bytes": gpu_estimate, "free_bytes": free_gpu}
            if gpu_estimate > free_gpu // 2:
                raise MemoryError(f"Selected graph estimate exceeds GPU budget; stopping: {record['gpu_resource_estimate']}")
        model.to(args.device)
        manifest = extract_once(reader, model, rows, args.device, args.output / "cache")
        write_json(args.output / "cache_manifest.json", manifest)
        if before != state_digest(model) or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen encoder state changed")
        record["encoder_forward_count"] = len(manifest)
        record["encoder_state_unchanged_after_extraction"] = True
        # Encoder is released before head training: cache is the only input.
        del model, state
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        graph.close()
        label.close()
        train_cache = load_caches(manifest, "train")
        features, labels, selections, sampling = fixed_training_pairs(train_cache)
        atomic_save({"selections": selections, "sampling": sampling}, args.output / "training_pairs.pt")
        record.update(status="training_head", training_pairs=sampling)
        write_json(args.output / "run.json", record)
        info(f"Fixed training pairs: {sampling}")
        del train_cache, selections
        set_seed(SEED)
        head = make_head(hidden).to(args.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(SEED)
        history = []
        for epoch in range(1, EPOCHS + 1):
            head.train()
            order = torch.randperm(labels.numel(), generator=generator)
            total_loss = 0.
            for columns in order.split(BATCH_SIZE):
                optimizer.zero_grad(set_to_none=True)
                logits = head(features[columns].to(args.device)).squeeze(-1)
                loss = criterion(logits, labels[columns].to(args.device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite head loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * columns.numel()
            history.append({"epoch": epoch, "mean_training_bce": total_loss / labels.numel()})
            write_json(args.output / "history.json", history)
            info(f"Head epoch={epoch}/{EPOCHS} BCE={history[-1]['mean_training_bce']:.6f}")
        atomic_save({"head": head.cpu().state_dict(), "epoch": EPOCHS, "hidden": hidden,
                     "encoder_checkpoint": record["checkpoint"], "encoder_checkpoint_sha256": checkpoint_hash,
                     "encoder_actual_epoch": record["checkpoint_actual_epoch"], "functional_net_identity": False,
                     "notice": NOTICE, "feature": record["feature"]}, args.output / "last_head.pt")
        head.to(args.device).eval()
        val_cache = load_caches(manifest, "val")
        cosine, logits_parts, label_parts, offsets = [], [], [], [0]
        with torch.no_grad():
            for cache in val_cache:
                scores = []
                for pairs in cache["pairs"].split(BATCH_SIZE, dim=1):
                    left, right = pairs
                    x = pair_features(cache["hf_l2"][left], cache["hf_l2"][right]).to(args.device)
                    scores.append(head(x).squeeze(-1).cpu())
                logits_parts.append(torch.cat(scores))
                cosine.append(cache["cosine_scores64"])
                label_parts.append(cache["labels"])
                offsets.append(offsets[-1] + cache["labels"].numel())
        cosine, logits, labels = torch.cat(cosine), torch.cat(logits_parts), torch.cat(label_parts)
        report = {"notice": NOTICE, "encoder_actual_epoch": record["checkpoint_actual_epoch"],
                  "head_actual_epoch": EPOCHS, "validation_circuits": len(val_cache),
                  "validation_pair_distribution": "all original valid pairs of the fixed 32 circuits; no balancing",
                  "cosine_baseline": report_scores(cosine, labels, offsets, "B original float64 cosine"),
                  "classification_head": report_scores(logits, labels, offsets, "independent frozen-hf MLP logits")}
        write_json(args.output / "comparison.json", report)
        atomic_save({"cosine_scores64": cosine, "head_logits": logits, "labels": labels,
                     "offsets": offsets, "circuits": [c["metadata"]["circuit"] for c in val_cache]},
                    args.output / "validation_scores.pt")
        if file_hash(checkpoint_path) != checkpoint_hash:
            raise RuntimeError("Source checkpoint changed")
        record.update(status="completed", completed_epochs=EPOCHS,
                      encoder_released_before_head_training=True, source_checkpoint_unchanged=True)
        write_json(args.output / "run.json", record)
        info(NOTICE)
        info(f"Completed frozen equivalence head experiment: {args.output / 'comparison.json'}")


def self_check():
    """Only symmetry, eval/freeze, head gradients/update; no real graphs."""
    set_seed(SEED)
    encoder = TriWaveGNN(2, 4, 3, hidden=4)
    freeze_encoder(encoder)
    before = state_digest(encoder)
    assert not any(module.training for module in encoder.modules())
    assert not encoder.functional_net_identity
    assert all(not p.requires_grad for p in encoder.parameters())
    with torch.no_grad():
        # Synthetic node states through an existing encoder parameter only.
        hf = F.normalize(encoder.initialization.internal["f"](torch.randn(6, 32)), dim=-1)
    left, right = hf[:3], hf[3:]
    features = pair_features(left, right)
    assert torch.equal(features, pair_features(right, left))
    head = make_head(4)
    initial = [p.detach().clone() for p in head.parameters()]
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    assert not ({id(p) for p in encoder.parameters()} &
                {id(p) for group in optimizer.param_groups for p in group["params"]})
    loss = nn.BCEWithLogitsLoss()(head(features).squeeze(-1), torch.tensor([0., 1., 0.]))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert any(bool((p.grad != 0).any()) for p in head.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    optimizer.step()
    assert any(not torch.equal(old, p) for old, p in zip(initial, head.parameters()))
    assert before == state_digest(encoder)
    print("PASS: feature exchange invariance; encoder eval/frozen/unchanged; head gradients and update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/train")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.output is None:
            parser.error("--run requires a new --output directory")
        run(args)


if __name__ == "__main__":
    main()
