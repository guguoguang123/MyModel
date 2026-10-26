"""One causal topological scan for Hs, Hf and Hb."""

import torch
from torch import nn
from ..utils.batch import prepare_topology


class TopologicalPropagation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item, states, relations, updates, include_behavior=True):
        item = prepare_topology(item)
        depth = item["pin"].depth
        branches = ("s", "f", "b") if include_behavior else ("s", "f")
        for level in range(1, item.max_depth + 1):
            nodes = torch.nonzero(depth == level, as_tuple=False).flatten()
            if nodes.numel() == 0:
                continue
            for branch in branches:
                if branch == "s":
                    source_state = states["s"]
                elif branch == "f":
                    source_state = torch.cat((states["s"], states["f"]), dim=-1)
                else:
                    source_state = torch.cat((states["s"], states["f"], states["b"]), dim=-1)
                current = states[branch]
                for relation in ("net", "cell"):
                    edge_store = item["pin", relation, "pin"]
                    start, stop = edge_store.level_ptr[level:level + 2].tolist()
                    if stop == start:
                        continue
                    edges = edge_store.ordered_edge_index[:, start:stop]
                    targets = torch.unique(edges[1], sorted=True)
                    features = edge_store.ordered_edge_attr[start:stop]
                    if branch != "b":
                        features = features[:, :0]
                    message = relations[branch + "_" + relation](source_state, current, edges, features, targets)
                    updated = updates[branch + "_" + relation](message, current[targets])
                    current = current.index_copy(0, targets, updated)
                states[branch] = current
        return states
