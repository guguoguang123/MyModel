"""Basic feed-forward network used in messages and readout heads."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), act(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.layers(x)
