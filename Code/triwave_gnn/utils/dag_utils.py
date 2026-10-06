"""Topological level construction for a pin-level directed acyclic graph."""

import numpy as np


def node_depth(topological_order, net_edges, cell_edges, node_count):
    incoming = [[] for _ in range(node_count)]
    for edges in (net_edges, cell_edges):
        for source, target in edges.T:
            incoming[int(target)].append(int(source))
    depth = np.zeros(node_count, dtype=np.int32)
    for node in topological_order:
        parents = incoming[int(node)]
        if parents:
            depth[int(node)] = 1 + max(depth[parent] for parent in parents)
    return depth
