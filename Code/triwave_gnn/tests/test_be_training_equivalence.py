"""Check training-only threshold diagnostics and original-count grouping."""

import unittest

import torch

from examples.compare_be_training_equivalence import summarize_archive, training_threshold
from triwave_gnn.utils.equivalence_audit import pair_metrics


class TrainingEquivalenceTest(unittest.TestCase):
    def test_tied_scores_cannot_be_split_to_improve_f1(self):
        scores = torch.tensor([.9, .9, .8, .7], dtype=torch.float64)
        labels = torch.tensor([True, False, True, False])
        threshold = training_threshold(scores, labels)
        self.assertEqual(threshold, .8)
        metrics = pair_metrics(scores, labels, threshold)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (2, 1, 0))
        self.assertEqual(metrics['f1'], .8)

    def test_empty_and_single_class_are_not_fabricated(self):
        archive = {'scores': torch.tensor([.8], dtype=torch.float64),
                   'labels': torch.tensor([True]), 'offsets': [0, 0, 1],
                   'circuit_names': ['empty', 'positive_only']}
        result = summarize_archive(archive, .9)
        self.assertIsNone(training_threshold(archive['scores'], archive['labels']))
        self.assertEqual(result['macro_participating_circuits'], 0)
        self.assertIsNone(result['within_circuit_weighted_auc'])
        self.assertIsNone(result['groups_by_original_valid_pairs']['1-999']['precision'])
        self.assertIsNone(result['groups_by_original_valid_pairs']['1000-9999']['roc_auc'])
        self.assertEqual(result['zero_pair_circuits_excluded_from_groups'], 1)

    def test_groups_follow_original_counts_and_auc_excludes_cross_circuit_pairs(self):
        counts = [999, 1000, 10000]
        # Each circuit has one positive scoring above its own negatives. Absolute
        # scores shift across circuits, so pooled AUC must differ from within AUC.
        parts = [torch.full((n,), float(i), dtype=torch.float64) for i, n in enumerate(counts)]
        labels = [torch.zeros(n, dtype=torch.bool) for n in counts]
        for scores, label in zip(parts, labels):
            scores[0] += .5
            label[0] = True
        result = summarize_archive({'scores': torch.cat(parts), 'labels': torch.cat(labels),
                                    'offsets': [0, 999, 1999, 11999],
                                    'circuit_names': ['small', 'medium', 'large']}, .25)
        self.assertEqual(result['circuit_macro_roc_auc'], 1.)
        self.assertEqual(result['within_circuit_weighted_auc'], 1.)
        self.assertLess(result['roc_auc'], 1.)
        for group, count in zip(('1-999', '1000-9999', '>=10000'), counts):
            row = result['groups_by_original_valid_pairs'][group]
            self.assertEqual(row['sample_count'], count)
            self.assertEqual(row['positive_ratio'], 1 / count)
        # Alternative training calibration must not mutate the main threshold.
        training_threshold(torch.cat(parts), torch.cat(labels))
        self.assertEqual(result['similarity_threshold'], .25)


if __name__ == '__main__':
    unittest.main()
