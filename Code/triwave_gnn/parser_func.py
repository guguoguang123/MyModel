"""PyG pin-graph data contract and node-index helpers."""

import torch
from torch_geometric.data import HeteroData


class PinCircuitData(HeteroData):
    """One pin node store, net/cell relations, and graph-level pair labels."""

    def __cat_dim__(self, key, value, store=None, *args, **kwargs):
        if key in ("hs_pairs", "hf_pairs"):
            return 1
        return super().__cat_dim__(key, value, store, *args, **kwargs)

    def __inc__(self, key, value, store=None, *args, **kwargs):
        if key in ("hs_pairs", "hf_pairs", "topological_order"):
            return self["pin"].num_nodes
        return super().__inc__(key, value, store, *args, **kwargs)


def primary_input_indices(item):
    return torch.nonzero(item["pin"].is_pi, as_tuple=True)[0]


def output_pin_indices(item):
    """Return sinks of the directed pin graph in local node numbering."""
    pins = item["pin"]
    has_outgoing = torch.zeros(pins.num_nodes, dtype=torch.bool, device=pins.node_role.device)
    for relation in ("net", "cell"):
        has_outgoing[item["pin", relation, "pin"].edge_index[0]] = True
    return torch.nonzero(~has_outgoing, as_tuple=True)[0]
