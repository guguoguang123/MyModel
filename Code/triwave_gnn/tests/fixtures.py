"""Tiny complete NPZ contract with different graph depths for offline tests."""

import numpy as np

from ..utils.schema import NET_CAUSES, CELL_CAUSES


def make_bundle(directory):
    sizes, node_parts, net_parts, cell_parts, hs_parts, hf_parts = [], [], [], [], [], []
    node_ptr, net_ptr, cell_ptr, hs_ptr, hf_ptr = [0], [0], [0], [0], [0]
    for extra_inverters in range(3):
        roles = [0, 0, 1, 1, 2]
        logic = [0, 0, 1, 1, 1]
        pins = ["PI", "PI", "A", "B", "Y"]
        net = [(0, 2), (1, 3)]
        cell = [(2, 4), (3, 4)]
        previous = 4
        for _ in range(1 + extra_inverters):
            source = len(roles)
            roles += [1, 2]
            logic += [2, 2]
            pins += ["A", "Y"]
            net.append((previous, source))
            cell.append((source, source + 1))
            previous = source + 1
        po = len(roles)
        roles.append(3)
        logic.append(3)
        pins.append("PO")
        net.append((previous, po))
        offset = node_ptr[-1]
        sizes.append(len(roles))
        node_parts.append((roles, logic, pins))
        net_parts.append(np.asarray(net, dtype=np.int64).T + offset)
        cell_parts.append(np.asarray(cell, dtype=np.int64).T + offset)
        hs_parts.append(np.array([[2, 2], [3, 5]], dtype=np.int64) + offset)
        hf_parts.append(np.array([[4, 4, previous], [previous, po, po]], dtype=np.int64) + offset)
        for ptr, count in ((node_ptr, len(roles)), (net_ptr, len(net)),
                           (cell_ptr, len(cell)), (hs_ptr, 2), (hf_ptr, 3)):
            ptr.append(ptr[-1] + count)
    total = node_ptr[-1]
    shared = {
        "bundle_format": np.array("concatenated-ptr-1.0"),
        "schema_version": np.array("physical-1.0"),
        "design_ids": np.array(["fixture__a", "fixture__b", "fixture__c"]),
        "node_ptr": np.array(node_ptr, dtype=np.int64),
        "node_mapping_sha256": np.array(["mapping-a", "mapping-b", "mapping-c"]),
        "netlist_sha256": np.array(["netlist-a", "netlist-b", "netlist-c"]),
    }
    graph = dict(shared, num_nodes=np.array(sizes),
                 net_edge_ptr=np.array(net_ptr), cell_edge_ptr=np.array(cell_ptr),
                 logic_vocab=np.array(["PI", "NAND", "NOT", "PO"]),
                 role_vocab=np.array(["pi", "cell_input", "cell_output", "po"]),
                 node_role=np.concatenate([p[0] for p in node_parts]),
                 node_logic_type=np.concatenate([p[1] for p in node_parts]),
                 node_pin_name=np.concatenate([p[2] for p in node_parts]),
                 net_edge_index=np.concatenate(net_parts, axis=1),
                 cell_edge_index=np.concatenate(cell_parts, axis=1),
                 topological_order=np.arange(total),
                 net_cause_names=np.array(NET_CAUSES), cell_cause_names=np.array(CELL_CAUSES),
                 net_cause_units=np.array(["ohm", "ohm", "fF", "fF", "fF", "fF"]),
                 cell_cause_units=np.array(["fF", "fF", "relative_X", "category", "um2", "fF", "ns"]))
    for relation, width, count in (("net", 6, net_ptr[-1]), ("cell", 7, cell_ptr[-1])):
        graph[relation + "_cause_features"] = np.linspace(0.1, 2, count * width).reshape(count, width).astype(np.float32)
        graph[relation + "_cause_valid"] = np.ones((count, width), dtype=bool)
    label = dict(shared, training_ready=np.ones(3, dtype=bool), time_unit=np.array(["ns"] * 3),
                 target_node_id=np.arange(total), hf_prob=np.full(total, 0.5, np.float32),
                 hs_pair_ptr=np.array(hs_ptr), hf_tt_pair_ptr=np.array(hf_ptr),
                 hs_pair_node_index=np.concatenate(hs_parts, axis=1),
                 hf_tt_pair_node_index=np.concatenate(hf_parts, axis=1),
                 hs_is_rc=np.tile([0., 1.], 3).astype(np.float32), hs_pair_mask=np.ones(6, bool),
                 hf_tt_distance=np.tile([0.8, 0.8, 0.], 3).astype(np.float32),
                 hf_tt_pair_mask=np.ones(9, bool))
    for name in ("tp", "tp_01", "tp_10", "first_mean", "stable_mean", "rise_mean",
                 "fall_mean", "glitch_rate", "conditional_extra", "unstable_window"):
        label["hb_" + name] = np.full(total, 0.4 if name == "tp" else 0.2, np.float32)
    for name in ("tp_mask", "tp_01_mask", "tp_10_mask", "first_train_mask", "stable_train_mask",
                 "rise_train_mask", "fall_train_mask", "glitch_rate_train_mask",
                 "glitch_severity_train_mask", "unstable_window_mask"):
        label["hb_" + name] = np.ones(total, bool)
    np.savez_compressed(directory / "graphs.npz", **graph)
    np.savez_compressed(directory / "labels.npz", **label)
