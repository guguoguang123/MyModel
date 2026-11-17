"""Gradient diagnostic equations, real dependencies, N/A and no-update checks."""
import unittest

import torch

from .test_pair_finetune import tiny_item
from ..model import TriWaveGNN
from ..utils.batch import collate_circuits
from ..utils.losses import loss_terms
from ..utils.model_utils import model_outputs
from ..utils.task_gradient_diagnostic import (module_parameter_groups, weighted_task_losses,
    task_gradients, compare_gradients, gradient_measurements, aggregate, prioritized_findings)


class TaskGradientTest(unittest.TestCase):
    def test_original_loss_values_and_total_gradients_all_mask_combinations(self):
        for rc in (False, True):
            for behavior in (False, True):
                torch.manual_seed(72)
                item = tiny_item('formula', [0., .1, .9, 0.], ['pi', 'cell_input', 'cell_output', 'po'])
                item.hf_pairs = torch.tensor([[4, 4, 6, 2], [6, 7, 7, 6]])
                item.hs_mask.fill_(rc)
                for key in ('hb_tp_mask', 'hb_first_train_mask', 'hb_stable_train_mask', 'hb_glitch_rate_train_mask'):
                    item['pin'][key].fill_(behavior)
                model = TriWaveGNN(1, 4, 1, hidden=8).double().train()
                for relation in ('net', 'cell'):
                    item['pin', relation, 'pin'].edge_attr = item['pin', relation, 'pin'].edge_attr.double()
                output = model_outputs(model, item)
                original, reference = loss_terms(output, item, 3, functional_chunk_size=2)
                weighted, details = weighted_task_losses(output, item, 2)
                self.assertAlmostEqual(float(sum(weighted.values())), float(original), places=12)
                self.assertEqual(details['effective_weights']['func'], 2 / (5 + int(rc) + 2*int(behavior)))
                parameters = tuple(model.parameters())
                expected = torch.autograd.grad(original, parameters, retain_graph=True, allow_unused=True)
                grads, _ = task_gradients(weighted, parameters, verify_total=True)
                for index, e in enumerate(expected):
                    parts = [g[index] for g in grads.values() if g[index] is not None]
                    if e is None:
                        self.assertEqual(parts, [])
                    else:
                        torch.testing.assert_close(e, sum(parts), atol=1e-10, rtol=1e-10)

    def test_actual_dependencies_and_no_parameter_update(self):
        torch.manual_seed(54)
        item = tiny_item('dependencies', [0., .1, .9, 0.], ['pi', 'cell_input', 'cell_output', 'po'])
        item.hf_pairs = torch.tensor([[4, 4, 6, 2], [6, 7, 7, 6]])
        model = TriWaveGNN(1, 4, 1, hidden=8).train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        names, groups = module_parameter_groups(model)
        output = model_outputs(model, collate_circuits([item]))
        losses, _ = weighted_task_losses(output, item, 10000)
        grads, _ = task_gradients(losses, tuple(model.parameters()), verify_total=True)
        for i, name in enumerate(names):
            if groups[name] == 'function_path':
                self.assertIsNone(grads['rc'][i])
            if groups[name] in ('behavior_path', 'task_heads') and name.startswith(('updates.b_', 'aggregators.b_', 'initialization.internal.b', 'initialization.pi_behavior', 'readout_behavior')):
                self.assertIsNone(grads['func'][i])
        rows, norms = gradient_measurements(grads, names, groups)
        self.assertEqual(rows['rc']['function_path']['status'], '不适用')
        self.assertEqual(rows['rc']['function_path']['common_parameter_tensors'], 0)
        self.assertGreater(norms['func']['structure_path']['weighted_norm'], 0)
        self.assertGreater(norms['behavior']['function_path']['weighted_norm'], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_cosine_common_support_and_zero_cases(self):
        left = [torch.tensor([1., 0.], dtype=torch.float64), None, torch.tensor([10.])]
        right = [torch.tensor([-2., 0.], dtype=torch.float64), torch.tensor([100.]), None]
        value = compare_gradients(left, right, range(3))
        self.assertEqual(value['cosine'], -1.)
        self.assertEqual(value['norm_ratio_to_func'], 2.)
        self.assertEqual(value['common_parameter_elements'], 2)
        self.assertEqual(compare_gradients(left, [None]*3, range(3))['status'], '不适用')
        value = compare_gradients([torch.zeros(2)], [torch.ones(2)], [0])
        self.assertIsNone(value['cosine'])
        self.assertIsNone(value['norm_ratio_to_func'])

    def test_aggregation_excludes_na_and_preserves_quantiles(self):
        names = ['a']
        groups = {'a': 'structure_path'}
        rows = []
        for other in (-1., 1., 0.):
            comparisons, _ = gradient_measurements({'func': [torch.tensor([1.])], 'prob': [torch.tensor([other])]}, names, groups)
            rows.append({'comparisons': comparisons})
        result = aggregate(rows)
        prob = result['prob']['structure_path']
        self.assertEqual(prob['applicable_batches'], 2)
        self.assertEqual(prob['negative_cosine_batch_fraction'], .5)
        self.assertEqual(prob['cosine_quantiles']['0.5'], 0.)
        self.assertEqual(result['rc']['all_common']['applicable_batches'], 0)
        findings = prioritized_findings({'stage3': result, 'E': result})
        self.assertEqual(findings['priority_separate_ablations'], [])

    def test_weighted_sum_cancellation_is_not_fake_cosine(self):
        comparisons, _ = gradient_measurements({'func': [torch.tensor([1.])],
            'prob': [torch.tensor([-1.])], 'rc': [torch.tensor([1.])]}, ['x'], {'x': 'shared_initialization'})
        self.assertIsNone(comparisons['other_weighted_sum']['all_common']['cosine'])
        self.assertEqual(comparisons['full_multitask_total']['all_common']['cosine'], 1.)


if __name__ == '__main__':
    unittest.main()
