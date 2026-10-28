"""Audit macro denominators, undefined classes, quantiles and input alignment."""

import importlib.util
from pathlib import Path
import unittest

import torch

from triwave_gnn.utils.equivalence_audit import (
    pair_metrics, circuit_macro, similarity_distribution, metric_differences,
)


class AuditStatistics(unittest.TestCase):
    def test_macro_is_not_pooled_and_single_class_auc_is_null(self):
        first_scores = torch.tensor([.9, .1])
        first_labels = torch.tensor([True, False])
        second_scores = torch.tensor([.9, .8, .7, .6])
        second_labels = torch.tensor([False, False, False, False])
        first = pair_metrics(first_scores, first_labels, .5)
        second = pair_metrics(second_scores, second_labels, .5)
        self.assertIsNone(second['roc_auc'])
        self.assertIsNone(second['recall'])
        self.assertEqual(second['f1'], 0.)
        self.assertIn('roc_auc', second['undefined_reasons'])
        macro = circuit_macro([first, second])
        pooled = pair_metrics(torch.cat((first_scores, second_scores)),
                              torch.cat((first_labels, second_labels)), .5)
        self.assertEqual(macro['precision'], .5)
        self.assertEqual(macro['f1'], .5)
        self.assertEqual(macro['recall'], 1.)
        self.assertEqual(macro['contributing_circuits']['recall'], 1)
        self.assertEqual(macro['contributing_circuits']['roc_auc'], 1)
        self.assertAlmostEqual(pooled['precision'], .2)
        self.assertAlmostEqual(pooled['f1'], 1 / 3)
        self.assertEqual(pooled['positive_ratio'], 1 / 6)
        self.assertEqual(first['threshold'], second['threshold'])

    def test_undefined_and_empty_circuits(self):
        result = pair_metrics(torch.tensor([.1]), torch.tensor([False]), .5)
        for metric in ('precision', 'recall', 'f1', 'roc_auc'):
            self.assertIsNone(result[metric])
            self.assertIn(metric, result['undefined_reasons'])
        self.assertEqual(result['accuracy'], 1.)
        empty = pair_metrics(torch.empty(0), torch.empty(0, dtype=torch.bool), .5)
        self.assertEqual(empty['status'], 'no_valid_pairs')
        self.assertIsNone(empty['positive_ratio'])
        macro = circuit_macro([empty])
        self.assertIsNone(macro['roc_auc'])
        self.assertEqual(macro['contributing_circuits']['roc_auc'], 0)
        no_predictions = pair_metrics(torch.tensor([.1, .2]), torch.tensor([True, False]), .5)
        self.assertIsNone(no_predictions['precision'])
        self.assertEqual(no_predictions['f1'], 0.)

    def test_quantiles_and_near_one_are_not_clipped(self):
        scores = torch.tensor([.4, .8, 1., 1. + 5e-7], dtype=torch.float64)
        labels = torch.tensor([True, False, True, False])
        result = similarity_distribution(scores, labels)
        self.assertEqual(result['positive']['quantiles']['0.5'], .7)
        self.assertEqual(result['positive']['equal_one_fraction'], .5)
        self.assertEqual(result['negative']['equal_one_fraction'], 0.)
        self.assertEqual(result['negative']['greater_than_one_fraction'], .5)
        self.assertEqual(result['negative']['near_one_fractions']['1e-06'], .5)
        self.assertEqual(result['negative']['near_one_fractions']['1e-07'], 0.)
        expected = torch.quantile(scores, torch.tensor([.01, .25, .5, .99], dtype=torch.float64))
        for q, v in zip((.01, .25, .5, .99), expected):
            self.assertAlmostEqual(result['all']['quantiles'][str(q)], float(v))
        missing = similarity_distribution(scores, torch.zeros(4, dtype=torch.bool))
        self.assertIsNone(missing['positive']['quantiles'])

    def test_differences_and_nulls(self):
        result = metric_differences({'precision': .25, 'roc_auc': None}, {'precision': .5, 'roc_auc': .8})
        self.assertEqual(result['precision']['delta'], .25)
        self.assertIsNone(result['roc_auc']['delta'])

    def test_split_aliases_overlap_and_fixed_threshold(self):
        entry = Path(__file__).resolve().parents[2] / 'examples/evaluate.py'
        spec = importlib.util.spec_from_file_location('audit_evaluation', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.validate_splits({'train': ['a'], 'validation': ['b'], 'test': ['c']},
                                               {'split_ids': {'train': ['a'], 'val': ['b'], 'test': ['c']}}), 'validation')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            module.validate_splits({'val': ['b'], 'test': ['b']}, {})
        with self.assertRaisesRegex(ValueError, 'differs'):
            module.validate_splits({'val': ['b'], 'test': ['c']}, {'split_ids': {'val': ['c'], 'test': ['b']}})
        results, rows = {}, [{'circuit': 'a'}, {'circuit': 'b'}]
        module.attach_equivalence(results, rows, torch.tensor([.9, .1, .8, .2]),
                                  torch.tensor([True, False, False, True]), [0, 2, 4], .85)
        for row in rows:
            self.assertEqual(row['functional_equivalence']['threshold'], .85)
        self.assertEqual(results['functional_equivalence']['threshold'], .85)


    def test_terminal_undefined_metrics_are_na_and_all_requested_fields_print(self):
        import contextlib
        import io
        entry = Path(__file__).resolve().parents[2] / 'examples/evaluate.py'
        spec = importlib.util.spec_from_file_location('summary_evaluation', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        test = module.legacy_results({}, {})
        test['functional_equivalence'] = pair_metrics(torch.tensor([.1, .2]), torch.tensor([False, False]), None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.print_summary(test)
        text = output.getvalue()
        self.assertIn('N/A (no finite valid labels)', text)
        self.assertIn('N/A (no informative comparisons)', text)
        self.assertIn('roc_auc=N/A (requires positive and negative pairs)', text)
        self.assertIn('TP=N/A (validation threshold unavailable)', text)
        self.assertNotIn('None', text)
        self.assertNotIn('null', text)
        for key in ('prob_mae', 'rc_accuracy', 'tt_distance_mae', 'tt_rank_accuracy', 'hb_tp_mae', 'hb_first_mean_mae', 'hb_stable_mean_mae', 'hb_glitch_rate_mae'):
            self.assertIn(key + '=', text)
        test = module.legacy_results({key: .25 for key in module.LEGACY_METRICS}, {key: 1 for key in module.LEGACY_METRICS})
        test['functional_equivalence'] = pair_metrics(torch.tensor([.9, .1]), torch.tensor([True, False]), .8)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.print_summary(test)
        text = output.getvalue()
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith('[INFO] TEST metrics: prob_mae='))
        self.assertTrue(lines[1].startswith('[INFO] TEST equivalence: precision='))
        self.assertTrue(lines[2].startswith('[INFO] TEST equivalence details: threshold='))
        self.assertIn('prob_mae=0.25000000', text)
        self.assertIn('rc_accuracy=75.0000%', text)
        self.assertIn('tt_rank_accuracy=25.0000%', text)
        self.assertIn('precision=100.0000%', text)
        self.assertIn('roc_auc=1.00000000', text)
        self.assertIn('threshold=0.8', text)
        self.assertIn('TP=1 | FP=0 | TN=1 | FN=0', text)



if __name__ == '__main__':
    unittest.main()
