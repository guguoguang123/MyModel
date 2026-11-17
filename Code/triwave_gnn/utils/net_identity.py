"""Conservative guards for ordinary combinational net identity propagation."""

import torch

# Clock gates, tri-state drivers, sequential/state elements and unknown types
# require explicit semantics; they are not covered by this experiment.
COMBINATIONAL_TYPES = frozenset((
    "PI", "PO", "LOGIC0", "LOGIC1", "AND", "OR", "NAND", "NOR", "NOT",
    "XOR", "XNOR", "BUF", "CLKBUF", "MUX2", "HA", "FA",
    "AOI21", "AOI211", "AOI22", "AOI221", "AOI222",
    "OAI21", "OAI211", "OAI22", "OAI221", "OAI222", "OAI33",
))


def annotate_net_identity_roles(item, role_vocab, logic_vocab):
    """Derive graph-only eligibility, with no labels or supervision changes."""
    pins = item["pin"]
    safe_logic = torch.zeros_like(pins.node_logic, dtype=torch.bool)
    for index, name in enumerate(logic_vocab):
        if name in COMBINATIONAL_TYPES:
            safe_logic |= pins.node_logic == index
    role = {name: i for i, name in enumerate(role_vocab)}
    driver = torch.zeros_like(safe_logic)
    receiver = torch.zeros_like(safe_logic)
    for name in ("pi", "cell_output"):
        if name in role:
            driver |= pins.node_role == role[name]
    for name in ("cell_input", "po"):
        if name in role:
            receiver |= pins.node_role == role[name]
    pins.functional_net_driver_ok = driver & safe_logic
    pins.functional_net_receiver_ok = receiver & safe_logic
    return item


def validate_net_identity_item(item):
    """Reject ambiguous nets rather than selecting/averaging their sources.

    Called once per F forward, after PyG collation and topology preparation.
    Physical-feature validity masks are not edge-activity masks: every net edge
    in this graph contract is a logical connection, even with missing physics.
    """
    pins = item["pin"]
    edges = item["pin", "net", "pin"].edge_index
    if edges.shape[1] == 0:
        return
    for key in ("functional_net_driver_ok", "functional_net_receiver_ok"):
        if key not in pins:
            raise ValueError("functional_net_identity needs role/type eligibility from the graph reader")
    src, dst = edges
    counts = torch.bincount(dst, minlength=pins.num_nodes)
    failures = {
        "multiple_net_incoming_edges_nodes": int((counts > 1).sum()),
        "unsupported_source_role_or_logic_edges": int((~pins.functional_net_driver_ok[src]).sum()),
        "unsupported_receiver_role_or_logic_edges": int((~pins.functional_net_receiver_ok[dst]).sum()),
        "noncausal_net_depth_edges": int((pins.depth[src] >= pins.depth[dst]).sum()),
    }
    cell = item["pin", "cell", "pin"].edge_index
    cell_incoming = torch.zeros(pins.num_nodes, dtype=torch.bool, device=dst.device)
    cell_incoming[cell[1]] = True
    failures["net_receivers_with_cell_incoming_edges"] = int(cell_incoming[dst].sum())
    if "batch" in pins:
        failures["cross_circuit_net_edges"] = int((pins.batch[src] != pins.batch[dst]).sum())
    if any(failures.values()):
        raise ValueError(f"functional_net_identity refuses unverified net edges: {failures}")


def checkpoint_net_identity(state):
    """Old checkpoints default off; reject conflicting/nonboolean metadata."""
    config = state.get("config", {})
    value = state.get("functional_net_identity", config.get("functional_net_identity", False))
    if not isinstance(value, bool):
        raise ValueError("functional_net_identity checkpoint metadata must be boolean")
    if "functional_net_identity" in config and config["functional_net_identity"] != value:
        raise ValueError("Conflicting functional_net_identity checkpoint metadata")
    return value
