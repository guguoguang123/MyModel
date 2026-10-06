"""Validate the concatenated physical graph/label contract before training."""

from __future__ import annotations

import numpy as np


GRAPH_FIELDS = {
    "bundle_format", "schema_version", "design_ids", "num_nodes", "node_ptr",
    "net_edge_ptr", "cell_edge_ptr", "node_mapping_sha256", "netlist_sha256",
    "logic_vocab", "role_vocab", "node_role", "node_logic_type", "node_pin_name",
    "net_edge_index", "cell_edge_index", "topological_order",
    "net_cause_features", "net_cause_valid", "cell_cause_features", "cell_cause_valid",
    "net_cause_names", "cell_cause_names", "net_cause_units", "cell_cause_units",
}
LABEL_FIELDS = {
    "bundle_format", "schema_version", "design_ids", "node_ptr", "hs_pair_ptr",
    "hf_tt_pair_ptr", "target_node_id", "node_mapping_sha256", "netlist_sha256",
    "training_ready", "time_unit", "hf_prob", "hs_pair_node_index", "hs_is_rc", "hs_pair_mask",
    "hf_tt_pair_node_index", "hf_tt_distance", "hf_tt_pair_mask",
    "hb_tp", "hb_tp_01", "hb_tp_10", "hb_first_mean", "hb_stable_mean",
    "hb_rise_mean", "hb_fall_mean", "hb_glitch_rate", "hb_conditional_extra",
    "hb_unstable_window", "hb_tp_mask", "hb_tp_01_mask", "hb_tp_10_mask",
    "hb_first_train_mask", "hb_stable_train_mask", "hb_rise_train_mask",
    "hb_fall_train_mask", "hb_glitch_rate_train_mask", "hb_glitch_severity_train_mask",
    "hb_unstable_window_mask",
}

NET_CAUSES = (
    "path_resistance", "total_net_resistance", "ground_capacitance",
    "coupling_capacitance", "total_net_capacitance", "sink_pin_capacitance",
)
CELL_CAUSES = (
    "input_pin_capacitance", "output_load_capacitance", "drive_strength",
    "threshold_voltage_class", "cell_area", "max_capacitance", "max_transition",
)


def _ptr(name, value, count, end):
    if value.shape != (count + 1,) or value[0] != 0 or value[-1] != end or np.any(value[1:] < value[:-1]):
        raise ValueError(f"invalid {name}: expected {count + 1} monotone offsets ending at {end}")


def _edges(name, edges, ptr, node_ptr, rank):
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError(f"{name} must have shape [2, E] with integer IDs")
    _ptr(name + "_ptr", ptr, len(node_ptr) - 1, edges.shape[1])
    for index in range(len(node_ptr) - 1):
        part = edges[:, ptr[index]:ptr[index + 1]]
        if part.size and (part.min() < node_ptr[index] or part.max() >= node_ptr[index + 1]):
            raise ValueError(f"{name} crosses graph boundary at design {index}")
    if edges.shape[1] and np.any(rank[edges[0]] >= rank[edges[1]]):
        raise ValueError(f"{name} violates supplied topological_order")


def validate_bundle(graph, labels):
    missing_graph = GRAPH_FIELDS - graph.keys()
    missing_labels = LABEL_FIELDS - labels.keys()
    if missing_graph or missing_labels:
        raise ValueError(f"missing graph fields {sorted(missing_graph)}; missing label fields {sorted(missing_labels)}")
    for key in ("bundle_format", "schema_version", "design_ids", "node_ptr", "node_mapping_sha256", "netlist_sha256"):
        if not np.array_equal(graph[key], labels[key]):
            raise ValueError(f"graph/label mismatch: {key}")
    if str(graph["bundle_format"]) != "concatenated-ptr-1.0":
        raise ValueError(f"unsupported bundle format: {graph['bundle_format']}")
    if str(graph["schema_version"]) != "physical-1.0":
        raise ValueError(f"unsupported physical schema: {graph['schema_version']}")
    if tuple(map(str, graph["net_cause_names"])) != NET_CAUSES or tuple(map(str, graph["cell_cause_names"])) != CELL_CAUSES:
        raise ValueError("physical feature names or order differ from the model contract")
    if tuple(map(str, graph["net_cause_units"])) != ("ohm", "ohm", "fF", "fF", "fF", "fF"):
        raise ValueError("net physical feature units differ from the model contract")
    if tuple(map(str, graph["cell_cause_units"])) != ("fF", "fF", "relative_X", "category", "um2", "fF", "ns"):
        raise ValueError("cell physical feature units differ from the model contract")
    count = len(graph["design_ids"])
    if count < 3 or len(set(map(str, graph["design_ids"]))) != count:
        raise ValueError("need at least three uniquely named designs")
    node_count = len(graph["node_role"])
    _ptr("node_ptr", graph["node_ptr"], count, node_count)
    if np.any(np.diff(graph["node_ptr"]) == 0):
        raise ValueError("every design must contain at least one node")
    if not np.array_equal(np.diff(graph["node_ptr"]), graph["num_nodes"]):
        raise ValueError("num_nodes disagrees with node_ptr")
    if labels["training_ready"].shape != (count,) or not np.all(labels["training_ready"]):
        raise ValueError("every design must be training_ready")
    if labels["time_unit"].shape != (count,) or set(map(str, labels["time_unit"])) != {"ns"}:
        raise ValueError("all behavior time labels must use ns")
    if not np.array_equal(labels["target_node_id"], np.arange(node_count)):
        raise ValueError("target_node_id must align with global graph node IDs")
    if graph["node_logic_type"].shape != (node_count,) or graph["node_pin_name"].shape != (node_count,):
        raise ValueError("node feature lengths do not match node_ptr")
    if graph["node_role"].min() < 0 or graph["node_role"].max() >= len(graph["role_vocab"]):
        raise ValueError("node_role contains unknown vocabulary IDs")
    if graph["node_logic_type"].min() < 0 or graph["node_logic_type"].max() >= len(graph["logic_vocab"]):
        raise ValueError("node_logic_type contains unknown vocabulary IDs")
    if "pi" not in list(map(str, graph["role_vocab"])):
        raise ValueError("role_vocab must contain pi")
    order = graph["topological_order"]
    if order.shape != (node_count,) or order.min() < 0 or order.max() >= node_count or len(np.unique(order)) != node_count:
        raise ValueError("topological_order must be a permutation of all nodes")
    rank = np.empty(node_count, np.int64)
    rank[order] = np.arange(node_count)
    for name in ("net", "cell"):
        edges = graph[name + "_edge_index"]
        _edges(name + "_edge_index", edges, graph[name + "_edge_ptr"], graph["node_ptr"], rank)
        features = graph[name + "_cause_features"]
        valid = graph[name + "_cause_valid"]
        expected = 6 if name == "net" else 7
        if features.shape != (edges.shape[1], expected) or valid.shape != features.shape:
            raise ValueError(f"{name} physical feature shape mismatch")
        if not np.isfinite(features[valid]).all():
            raise ValueError(f"{name} physical features marked valid contain nonfinite values")
    for name, pair_key, mask_key, target_key in (
        ("hs", "hs_pair_node_index", "hs_pair_mask", "hs_is_rc"),
        ("hf_tt", "hf_tt_pair_node_index", "hf_tt_pair_mask", "hf_tt_distance"),
    ):
        pairs = labels[pair_key]
        if pairs.ndim != 2 or pairs.shape[0] != 2:
            raise ValueError(f"{pair_key} must have shape [2, P]")
        _ptr(name + "_pair_ptr", labels[name + "_pair_ptr"], count, pairs.shape[1])
        if labels[mask_key].shape != (pairs.shape[1],) or labels[target_key].shape != (pairs.shape[1],):
            raise ValueError(f"{name} pair labels have wrong length")
        for index in range(count):
            part = pairs[:, labels[name + "_pair_ptr"][index]:labels[name + "_pair_ptr"][index + 1]]
            if part.size and (part.min() < graph["node_ptr"][index] or part.max() >= graph["node_ptr"][index + 1]):
                raise ValueError(f"{name} pair crosses design boundary")
        if not np.isfinite(labels[target_key][labels[mask_key]]).all():
            raise ValueError(f"{name} valid pair target is nonfinite")
    for key in LABEL_FIELDS:
        if key.startswith("hb_") and key not in ("hb_glitch_severity_train_mask",) and not key.endswith("mask"):
            if labels[key].shape != (node_count,):
                raise ValueError(f"{key} must have one value per node")
    if labels["hf_prob"].shape != (node_count,) or not np.isfinite(labels["hf_prob"]).all():
        raise ValueError("hf_prob must be finite for every node")
    for key in LABEL_FIELDS:
        if key.startswith("hb_") and key.endswith("mask") and labels[key].shape != (node_count,):
            raise ValueError(f"{key} mask must have one value per node")
    for target, mask in (
        ("hb_tp", "hb_tp_mask"), ("hb_tp_01", "hb_tp_01_mask"),
        ("hb_tp_10", "hb_tp_10_mask"), ("hb_first_mean", "hb_first_train_mask"),
        ("hb_stable_mean", "hb_stable_train_mask"), ("hb_rise_mean", "hb_rise_train_mask"),
        ("hb_fall_mean", "hb_fall_train_mask"), ("hb_glitch_rate", "hb_glitch_rate_train_mask"),
        ("hb_conditional_extra", "hb_glitch_severity_train_mask"),
        ("hb_unstable_window", "hb_unstable_window_mask"),
    ):
        if not np.isfinite(labels[target][labels[mask]]).all():
            raise ValueError(f"{target} contains nonfinite values where {mask} is true")
    for key in ("hf_prob", "hb_tp", "hb_tp_01", "hb_tp_10"):
        values = labels[key]
        if np.nanmin(values) < -1e-6 or np.nanmax(values) > 1 + 1e-6:
            raise ValueError(f"{key} lies outside [0, 1]")
    for key, mask in (("hs_is_rc", "hs_pair_mask"), ("hf_tt_distance", "hf_tt_pair_mask"),
                      ("hb_glitch_rate", "hb_glitch_rate_train_mask")):
        values = labels[key][labels[mask]]
        if values.size and (values.min() < -1e-6 or values.max() > 1 + 1e-6):
            raise ValueError(f"{key} lies outside [0, 1] where supervised")
    if not np.allclose(labels["hb_tp"], labels["hb_tp_01"] + labels["hb_tp_10"], atol=1e-6):
        raise ValueError("hb_tp disagrees with directional transitions")
