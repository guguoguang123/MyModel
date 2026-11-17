import unittest

import torch

from triwave_gnn.utils.metric_loss import (
    cosine_scores,
    direct_metric_loss,
    direct_metric_loss_from_embeddings,
)


class MetricLossTest(unittest.TestCase):
    def test_cosine_score(self):
        hf = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        pairs = torch.tensor([[0, 0], [1, 2]])
        scores = cosine_scores(hf, pairs)
        self.assertTrue(torch.allclose(scores, torch.tensor([1., 0.])))

    def test_margin_and_temperature(self):
        p, n = torch.tensor([.9]), torch.tensor([.8])
        a = direct_metric_loss(p, n, margin=.02, temperature=.05)
        b = direct_metric_loss(p, n, margin=.20, temperature=.05)
        c = direct_metric_loss(p, n, margin=.02, temperature=.10)
        self.assertGreater(float(b), float(a))
        self.assertGreater(float(c), float(a))

    def test_score_direction(self):
        p0, n0 = torch.tensor([.8]), torch.tensor([.7])
        p1, n1 = torch.tensor([.9]), torch.tensor([.95])
        self.assertLess(float(direct_metric_loss(p1, n0)), float(direct_metric_loss(p0, n0)))
        self.assertGreater(float(direct_metric_loss(p0, n1)), float(direct_metric_loss(p0, n0)))

    def test_hf_gradient_and_no_detach(self):
        hf = torch.tensor([[1., .1], [.9, .2], [.1, 1.]], requires_grad=True)
        positive = torch.tensor([[0], [1]])
        negative = torch.tensor([[1], [2]])
        loss = direct_metric_loss_from_embeddings(hf, positive, negative)
        loss.backward()
        self.assertIsNotNone(hf.grad)
        self.assertGreater(float(hf.grad.abs().sum()), 0.)

    def test_empty_and_single(self):
        p = torch.empty(0, requires_grad=True)
        n = torch.empty(0, requires_grad=True)
        loss = direct_metric_loss(p, n)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(p.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(direct_metric_loss(torch.tensor([.9]), torch.tensor([.8]))))

    def test_rejects_nonfinite_and_detach_is_not_used(self):
        with self.assertRaises(ValueError):
            direct_metric_loss(torch.tensor([float('nan')]), torch.tensor([.1]))
        hf = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        with self.assertRaises(ValueError):
            cosine_scores(hf, torch.tensor([[0], [3]]))


if __name__ == '__main__':
    unittest.main()
