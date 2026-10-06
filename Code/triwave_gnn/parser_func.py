"""Small graph conversion helpers used by downstream pin-level tasks."""

import torch


def primary_input_indices(item):
    return torch.nonzero(item["node_role"] == item["pi_role_index"], as_tuple=True)[0]


def output_pin_indices(item):
    """Return sinks of the directed pin graph in local node numbering."""
    count = item["node_role"].numel()
    has_outgoing = torch.zeros(count, dtype=torch.bool, device=item["node_role"].device)
    for relation in ("net_edges", "cell_edges"):
        has_outgoing[item[relation][0]] = True
    return torch.nonzero(~has_outgoing, as_tuple=True)[0]
