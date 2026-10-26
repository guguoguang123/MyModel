"""PyG batching and the depth schedule for a complete pin graph or batch."""

import torch
from torch_geometric.data import Batch


# Derived schedules must be rebuilt after batching graphs of unequal depth.
SCHEDULE_KEYS = ("level_ptr", "max_depth", "ordered_edge_index", "ordered_edge_attr")


def collate_circuits(items):
    """Build a disconnected PyG batch; PyG offsets edges and supervised pairs."""
    if not items:
        raise ValueError("cannot batch zero circuits")
    return Batch.from_data_list(items, exclude_keys=SCHEDULE_KEYS)


def prepare_topology(data):
    """Group each relation's edges by target depth after graph batching."""
    if data.__dict__.get("_topology_ready", False):
        return data
    depth = data["pin"].depth
    data.max_depth = int(depth.max())
    for relation in ("net", "cell"):
        edges = data["pin", relation, "pin"]
        edge_depth = depth[edges.edge_index[1]]
        order = torch.argsort(edge_depth, stable=True)
        # Keep canonical edges intact so Batch.to_data_list() can undo batching.
        edges.ordered_edge_index = edges.edge_index[:, order]
        edges.ordered_edge_attr = edges.edge_attr[order]
        counts = torch.bincount(edge_depth, minlength=data.max_depth + 1)
        edges.level_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    data._topology_ready = True
    return data
