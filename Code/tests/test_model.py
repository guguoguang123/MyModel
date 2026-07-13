"""Small graph tests for model contracts and differentiability."""

import unittest
from types import SimpleNamespace

import torch

from Code.daregate import DAREGateConfig, DAREGateModel


def reconvergent_aig() -> tuple[torch.Tensor, ...]:
    # Node 0 branches to nodes 2 and 3, then reconverges at node 4.
    gate_type = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 3], [2, 2, 3, 4, 4]], dtype=torch.long
    )
    forward_level = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    backward_level = torch.tensor([2, 2, 1, 1, 0], dtype=torch.long)
    rc_source = torch.tensor([-1, -1, -1, -1, 0], dtype=torch.long)
    return gate_type, edge_index, forward_level, backward_level, rc_source


class DAREGateModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = DAREGateConfig(
            structure_dim=32,
            function_dim=32,
            behavior_dim=32,
            gate_embedding_dim=8,
            probe_embedding_dim=8,
            attention_heads=4,
            cross_probe_heads=4,
            cross_probe_ff_dim=64,
            mlp_dropout=0.0,
        )
        self.model = DAREGateModel(self.config)
        self.graph = reconvergent_aig()

    def test_forward_shapes_and_centering(self) -> None:
        gate, edge, forward, backward, rc_source = self.graph
        output = self.model(
            gate,
            edge,
            forward,
            backward_level=backward,
            rc_source=rc_source,
        )
        self.assertEqual(output.structure.shape, (5, 32))
        self.assertEqual(output.function.shape, (5, 32))
        self.assertEqual(output.behavior.shape, (5, 32))
        self.assertEqual(output.probe_behavior.shape, (5, 8, 32))
        self.assertEqual(output.response.shape, (5, 8, 2))
        self.assertEqual(output.signature.shape, (5, 16))
        torch.testing.assert_close(
            output.response.sum(dim=1),
            torch.zeros(5, 2),
            atol=1e-6,
            rtol=0.0,
        )

    def test_backward_reaches_all_three_encoders(self) -> None:
        output = self.model(
            self.graph[0],
            self.graph[1],
            self.graph[2],
            backward_level=self.graph[3],
            rc_source=self.graph[4],
        )
        output.signature.square().mean().backward()
        for module in (
            self.model.structure_encoder,
            self.model.function_encoder,
            self.model.behavior_encoder,
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_optional_reconvergence_and_graph_adapter(self) -> None:
        gate, edge, forward, backward, _ = self.graph
        graph = SimpleNamespace(
            gate=gate.float().unsqueeze(1),
            edge_index=edge,
            forward_level=forward,
            backward_level=backward,
        )
        output = self.model.forward_graph(graph)
        self.assertEqual(output.signature.shape, (5, 16))

    def test_invalid_topology_is_rejected(self) -> None:
        gate, edge, forward, backward, rc_source = self.graph
        invalid_level = forward.clone()
        invalid_level[4] = 1
        with self.assertRaisesRegex(ValueError, "topological"):
            self.model(
                gate,
                edge,
                invalid_level,
                backward_level=backward,
                rc_source=rc_source,
            )


if __name__ == "__main__":
    unittest.main()

