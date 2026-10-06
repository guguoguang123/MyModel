"""Assemble PI initialization, topological propagation and task heads."""

from torch import nn

from .arch import PinInitialization, TopologicalPropagation, pair_features
from .arch.message import RelationMessage
from .arch.mlp import MLP


class TriWaveGNN(nn.Module):
    def __init__(self, logic_count: int, role_count: int, pin_count: int, hidden: int = 48):
        super().__init__()
        self.initialization = PinInitialization(logic_count, role_count, pin_count, hidden)
        # Declare relation-specific aggregators and recurrent updates here;
        # the propagation module schedules them by DAG level.
        self.aggregators = nn.ModuleDict()
        self.updates = nn.ModuleDict()
        for branch, width in (("s", hidden), ("f", 2 * hidden), ("b", 3 * hidden)):
            for relation, physical_width, attention in (("net", 12, False), ("cell", 14, True)):
                name = branch + "_" + relation
                self.aggregators[name] = RelationMessage(
                    width, physical_width if branch == "b" else 0, 16, attention
                )
                self.updates[name] = nn.GRUCell(width, hidden)
        self.readout_rc = MLP(3 * hidden, hidden, 1)
        self.readout_prob = nn.Linear(hidden, 1)
        self.readout_behavior = MLP(hidden, hidden, 10)
        self.propagation = TopologicalPropagation()

    def forward(self, item, include_behavior=True):
        logic, states = self.initialization(item)
        states = self.propagation(item, logic, states, self.aggregators, self.updates,
                                  include_behavior=include_behavior)
        return states["s"], states["f"], states["b"]

    def pred_prob_logits(self, hf):
        return self.readout_prob(hf).squeeze(-1)

    def pred_rc_logits(self, hs, pairs):
        return self.readout_rc(pair_features(hs, pairs)).squeeze(-1)

    def pred_behavior_raw(self, hb):
        return self.readout_behavior(hb)

    def load_pretrained(self, path=None, device="cpu"):
        """Load published weights into a model with matching vocabulary sizes."""
        from .utils.checkpoint import load
        from .pretrained import DEFAULT_MODEL
        checkpoint = load(path or DEFAULT_MODEL, device)
        if checkpoint.get("config", {}).get("hidden") != self.initialization.hidden:
            raise ValueError("pretrained hidden width differs from this model")
        self.load_state_dict(checkpoint["model"])
        return checkpoint
