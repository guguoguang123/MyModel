"""Attention-weighted aggregation of incoming cell-edge messages."""

from torch import nn
from torch_geometric.utils import softmax

from .mlp import MLP


class TFMlpAggr(nn.Module):
    """Normalize learned scores among incoming cell arcs of each target pin."""

    def __init__(self, message_dim: int, context_dim: int):
        super().__init__()
        self.score = MLP(context_dim, message_dim, 1, activation="tanh")

    def forward(self, messages, context, index, ptr=None, num_nodes=None):
        score = self.score(context).squeeze(-1)
        weight = softmax(score, index=index, ptr=ptr, num_nodes=num_nodes)
        # MessagePassing performs the following sum over incoming weighted messages.
        return messages * weight.unsqueeze(-1)
