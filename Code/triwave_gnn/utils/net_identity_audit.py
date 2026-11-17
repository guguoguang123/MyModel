"""Inspect stored graph semantics for the functional net identity experiment."""

from collections import Counter

import numpy as np

from .net_identity import COMBINATIONAL_TYPES


def audit_graph_bundle(graph, labels, notify=None):
    roles = list(map(str, graph["role_vocab"]))
    logic = list(map(str, graph["logic_vocab"]))
    safe_logic_ids = [i for i, name in enumerate(logic) if name in COMBINATIONAL_TYPES]
    totals, role_pairs, used_logic = Counter(), Counter(), Counter()
    failures, rows = Counter(), []
    for index, name in enumerate(map(str, graph["design_ids"])):
        n0, n1 = map(int, graph["node_ptr"][index:index + 2])
        e0, e1 = map(int, graph["net_edge_ptr"][index:index + 2])
        c0, c1 = map(int, graph["cell_edge_ptr"][index:index + 2])
        net = graph["net_edge_index"][:, e0:e1] - n0
        cell = graph["cell_edge_index"][:, c0:c1] - n0
        r, t = graph["node_role"][n0:n1], graph["node_logic_type"][n0:n1]
        order = graph["topological_order"][n0:n1] - n0
        rank = np.empty(n1 - n0, dtype=np.int64)
        rank[order] = np.arange(n1 - n0)
        src, dst = net
        incoming = np.bincount(dst, minlength=n1 - n0)
        cell_incoming = np.bincount(cell[1], minlength=n1 - n0)
        safe = np.isin(t, safe_logic_ids)
        bad = {
            "multiple_net_incoming_edges_nodes": int((incoming > 1).sum()),
            "unsupported_source_role_edges": int((~np.isin(r[src], [roles.index("pi"), roles.index("cell_output")])).sum()),
            "unsupported_receiver_role_edges": int((~np.isin(r[dst], [roles.index("cell_input"), roles.index("po")])).sum()),
            "unsupported_source_logic_edges": int((~safe[src]).sum()),
            "unsupported_receiver_logic_edges": int((~safe[dst]).sum()),
            "net_receivers_with_cell_incoming_edges": int((cell_incoming[dst] > 0).sum()),
            "noncausal_net_edges": int((rank[src] >= rank[dst]).sum()),
            "noncausal_cell_edges": int((rank[cell[0]] >= rank[cell[1]]).sum()),
            "cell_edges_not_input_to_output": int(((r[cell[0]] != roles.index("cell_input")) |
                                                   (r[cell[1]] != roles.index("cell_output"))).sum()),
        }
        cell_id = graph["node_cell_id"][n0:n1]
        bad["cell_edges_crossing_cell_ids"] = int((cell_id[cell[0]] != cell_id[cell[1]]).sum())
        # Count true multiple sources separately from repeated identical edges.
        bad["multiple_distinct_net_driver_nodes"] = 0
        bad["duplicate_net_edges"] = 0
        if bad["multiple_net_incoming_edges_nodes"]:
            unique = np.unique(net, axis=1)
            unique_incoming = np.bincount(unique[1], minlength=n1 - n0)
            bad["multiple_distinct_net_driver_nodes"] = int((unique_incoming > 1).sum())
            bad["duplicate_net_edges"] = net.shape[1] - unique.shape[1]
        # A consistency observation, not a full truth-table proof of wiring.
        one_counts = labels["hf_one_count"][n0:n1]
        mismatch = int((one_counts[src] != one_counts[dst]).sum())
        bad["net_endpoint_static_one_count_mismatch_edges"] = mismatch
        for code, count in enumerate(np.bincount(r[src] * len(roles) + r[dst], minlength=len(roles)**2)):
            if count:
                role_pairs[f"{roles[code // len(roles)]} -> {roles[code % len(roles)]}"] += int(count)
        values, counts = np.unique(t, return_counts=True)
        used_logic.update({logic[int(i)]: int(n) for i, n in zip(values, counts)})
        totals.update(circuits=1, nodes=n1 - n0, net_edges=e1 - e0, cell_edges=c1 - c0,
                      net_receivers=int((incoming > 0).sum()),
                      fanout_drivers=int((np.bincount(src, minlength=n1 - n0) > 1).sum()))
        failures.update(bad)
        if any(bad.values()):
            rows.append({"circuit": name, "failures": bad})
        if notify and ((index + 1) % 2000 == 0 or index + 1 == len(graph["design_ids"])):
            notify(f"Net preflight: {index + 1}/{len(graph['design_ids'])} circuits")
    return {
        "status": "passed" if not any(failures.values()) else "failed",
        "scope": "stored complete graph bundle; no model inference, training, or test metrics",
        "totals": dict(totals), "net_endpoint_roles": dict(role_pairs),
        "used_logic_node_counts": dict(used_logic), "failure_counts": dict(failures),
        "circuits_with_failures": rows,
        "semantics_evidence": {
            "available": ["dataset.py preserves net_edge_index source/destination order",
                          "node_depth and prepare_topology schedule targets after all predecessors",
                          "stored roles, cell IDs, logic types, topological_order",
                          "project graph-contract documentation defines net as wiring and cell as gate logic",
                          "net endpoint static hf_one_count comparison (necessary consistency only)"],
            "generator_source_available": False,
            "limitation": "用户确认生成脚本不在本机；未复核生成脚本。图包未提供逐边反相/解析/状态操作标记，不能独立证明生成端从未折叠此类操作；本实验依据现有 net/cell 图契约及上述实测约束。",
            "inverting_gates": "NOT/NAND/NOR/AOI/OAI are allowed as cell logic; their input-to-output cell edges remain learned; no inversion operation is applied by the net-edge reader",
        },
    }
