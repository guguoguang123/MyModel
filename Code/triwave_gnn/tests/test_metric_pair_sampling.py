import unittest

import torch

from triwave_gnn.utils.metric_pair_sampling import (
    GROUPS,
    pair_count_group,
    sample_metric_indices,
)


def synthetic_pairs():
    # Four circuits: one in each size group plus a second dense circuit.
    counts = [3, 1000, 10000, 10000]
    pairs, targets, circuits = [], [], []
    offset = 0
    for circuit, count in enumerate(counts):
        local = torch.arange(count, dtype=torch.long)
        pairs.append(torch.stack((local + offset, local + offset + 1)))
        # Every circuit has a balanced prefix and a long negative tail.
        targets.append(torch.cat((torch.zeros(min(2, count)), torch.ones(count - min(2, count)))))
        circuits.append(torch.full((count,), circuit, dtype=torch.long))
        offset += count + 1
    return torch.cat(pairs, dim=1), torch.cat(targets), torch.cat(circuits)


class MetricPairSamplingTest(unittest.TestCase):
    def test_group_boundaries(self):
        self.assertEqual(pair_count_group(0), '1-999')
        self.assertEqual(pair_count_group(999), '1-999')
        self.assertEqual(pair_count_group(1000), '1000-9999')
        self.assertEqual(pair_count_group(9999), '1000-9999')
        self.assertEqual(pair_count_group(10000), '>=10000')

    def test_labels_alignment_and_balancing(self):
        pairs, target, circuit = synthetic_pairs()
        mask = torch.ones_like(target, dtype=torch.bool)
        sample = sample_metric_indices(
            pairs, target, mask, circuit, torch.Generator().manual_seed(7),
            group_quota={'1-999': 2, '1000-9999': 2, '>=10000': 4},
            per_circuit_cap=4,
        )
        self.assertEqual(sample.positive_indices.numel(), sample.negative_indices.numel())
        self.assertEqual(sample.positive_indices.numel(), sample.stats['comparison_count'])
        self.assertTrue(torch.all(torch.isclose(target[sample.positive_indices], torch.zeros_like(target[sample.positive_indices]))))
        self.assertTrue(torch.all(target[sample.negative_indices] != 0))
        self.assertEqual(sample.positive_indices.numel(), 8)

    def test_group_quotas_and_dense_not_exclusive(self):
        pairs, target, circuit = synthetic_pairs()
        sample = sample_metric_indices(
            pairs, target, torch.ones_like(target, dtype=torch.bool), circuit,
            torch.Generator().manual_seed(8),
            group_quota={'1-999': 2, '1000-9999': 2, '>=10000': 4},
            per_circuit_cap=2,
        )
        self.assertEqual([sample.stats['groups'][g]['selected_comparisons'] for g in GROUPS], [2, 2, 4])
        # The dense quota is split by the per-circuit cap; no one circuit can own it all.
        self.assertLessEqual(sample.stats['groups']['>=10000']['same_circuit_comparisons'], 4)

    def test_train_only_rejects_val_and_test(self):
        pairs, target, circuit = synthetic_pairs()
        kwargs = dict(pair_index=pairs, target=target, valid_mask=torch.ones_like(target, dtype=torch.bool),
                      circuit_ids=circuit, generator=torch.Generator().manual_seed(1))
        with self.assertRaises(ValueError):
            sample_metric_indices(**kwargs, split_name='val')
        with self.assertRaises(ValueError):
            sample_metric_indices(**kwargs, split_name='test')

    def test_duplicate_rate_and_hard_negative_fallback(self):
        pairs, target, circuit = synthetic_pairs()
        scores = torch.linspace(0, 1, target.numel())
        sample = sample_metric_indices(
            pairs, target, torch.ones_like(target, dtype=torch.bool), circuit,
            torch.Generator().manual_seed(2),
            group_quota={'1-999': 10, '1000-9999': 10, '>=10000': 10},
            per_circuit_cap=32, scores=scores, hard_negative_fraction=.5,
        )
        selected = torch.cat((sample.positive_indices, sample.negative_indices))
        unique = torch.unique(selected).numel()
        self.assertEqual(sample.stats['unique_pair_count'], int(unique))
        self.assertEqual(sample.stats['duplicate_pair_count'], selected.numel() - int(unique))
        self.assertEqual(sample.stats['duplicate_rate'], sample.stats['duplicate_pair_count'] / selected.numel())
        self.assertGreater(sample.stats['fallback_count'], 0)
        self.assertGreater(sample.stats['hard_negative_fraction'], 0.0)

    def test_endpoint_alignment_rejects_cross_circuit(self):
        pairs, target, circuit = synthetic_pairs()
        owner = torch.zeros(int(pairs.max()) + 1, dtype=torch.long)
        owner[pairs[1, 0]] = 1
        with self.assertRaises(ValueError):
            sample_metric_indices(pairs, target, torch.ones_like(target, dtype=torch.bool), circuit,
                                  torch.Generator().manual_seed(3), node_owner=owner)


if __name__ == '__main__':
    unittest.main()
