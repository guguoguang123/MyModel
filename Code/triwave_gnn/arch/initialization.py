"""Initialize distinct PI structure states and shared functional states."""

import hashlib
import math

import torch
from torch import nn


class PinInitialization(nn.Module):
    def __init__(self, logic_count, role_count, pin_count, hidden):
        super().__init__()
        self.hidden = hidden
        self.logic_embedding = nn.Embedding(logic_count, 16)
        self.role_embedding = nn.Embedding(role_count, 8)
        self.pin_embedding = nn.Embedding(pin_count, 8)
        self.internal = nn.ModuleDict({k: nn.Linear(32, hidden) for k in ("s", "f", "b")})
        self.pi_function = nn.Parameter(torch.zeros(hidden))
        self.pi_behavior = nn.Parameter(torch.zeros(hidden))

    def forward(self, item):
        logic = self.logic_embedding(item["node_logic"])
        base = torch.cat((logic, self.role_embedding(item["node_role"]), self.pin_embedding(item["node_pin"])), dim=-1)
        states = {k: torch.tanh(self.internal[k](base)) for k in ("s", "f", "b")}
        names = item["name"] if isinstance(item["name"], list) else [item["name"]]
        pointers = item.get("graph_ptr")
        if pointers is None:
            pointers = torch.tensor([0, item["node_role"].numel()], device=item["node_role"].device)
        for index, name in enumerate(names):
            start, stop = int(pointers[index]), int(pointers[index + 1])
            pi = torch.nonzero(item["node_role"][start:stop] == item["pi_role_index"],
                               as_tuple=False).flatten() + start
            if not pi.numel():
                continue
            # Fixed per design, distinct within a circuit, independent of labels.
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            count = pi.numel()
            if count <= self.hidden:
                vectors = torch.linalg.qr(torch.randn(self.hidden, self.hidden, generator=generator), mode="reduced").Q[:count]
            else:
                vectors = torch.randn(count, self.hidden, generator=generator)
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            states["s"] = states["s"].index_copy(0, pi, vectors.to(device=pi.device, dtype=states["s"].dtype))
            states["f"] = states["f"].index_copy(0, pi, self.pi_function.expand(count, -1))
            states["b"] = states["b"].index_copy(0, pi, self.pi_behavior.expand(count, -1))
        return states
