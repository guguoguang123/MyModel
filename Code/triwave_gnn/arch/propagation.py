"""One causal topological scan for Hs, Hf and Hb."""

import torch
from torch import nn


class TopologicalPropagation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item, logic, states, relations, updates, include_behavior=True):
        depth = item["depth"]
        branches = ("s", "f", "b") if include_behavior else ("s", "f")
        for level in range(1, item["max_depth"] + 1):
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
                    start, stop = item[relation + "_ptr"][level:level + 2].tolist()
                    if stop == start:
                        continue
                    edges = item[relation + "_edges"][:, start:stop]
                    targets = torch.unique(edges[1], sorted=True)
                    features = item[relation + "_features"][start:stop]
                    if branch != "b":
                        features = features[:, :0]
                    message = relations[branch + "_" + relation](source_state, edges, features, logic, targets)
                    updated = updates[branch + "_" + relation](message, current[targets])
                    current = current.index_copy(0, targets, updated)
                states[branch] = current
        return states
