"""Contracts for the train-only fitting diagnostic, independent of full data."""
import unittest
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .test_pair_finetune import tiny_item, tiny_source
from ..utils.batch import collate_circuits
from ..utils.equivalence_fit import (balanced_columns, batch_schedule, diagnostic_loss,
                                    pair_owners, pair_view, select_circuits, training_threshold)
from ..utils.functional_equivalence import select_threshold
from ..utils.losses import loss_terms

ROLES = ['pi', 'cell_input', 'cell_output', 'po']


class EquivalenceFitTests(unittest.TestCase):
    def test_d0_is_exact_current_functional_term_and_gradient(self):
        torch.manual_seed(2)
        item = tiny_item('d0', [0., .1, .18, 0.], ROLES)
        hf = torch.randn(8, 5, requires_grad=True)
        output = {'hf': hf, 'hf_prob_logits': torch.zeros(8), 'hs_rc_logits': torch.zeros(2)}
        diagnostic = diagnostic_loss(hf, item, 'D0')
        total, terms = loss_terms(output, item, stage=2)
        _, unrelated = loss_terms(output, item, stage=1)
        original = (total * 6 - 3 * torch.tensor(unrelated['prob']) - torch.tensor(unrelated['rc'])) / 2
        self.assertAlmostEqual(float(diagnostic), terms['func'], places=6)
        self.assertTrue(torch.allclose(torch.autograd.grad(diagnostic, hf, retain_graph=True)[0],
                                       torch.autograd.grad(original, hf)[0], atol=1e-6))

    def test_d1_averages_circuits_then_classes_not_pooled_pairs(self):
        a = tiny_item('a', [0., .1], ROLES)
        b = tiny_item('b', [0., 0., 0., .1, .1, .1], ROLES)
        a.hf_pairs = torch.tensor([[0, 0], [1, 1]])
        b.hf_pairs = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        item = collate_circuits([a, b])
        hf = torch.zeros(16, 2, dtype=torch.float64)
        hf[0] = torch.tensor([1., 0.]); hf[1] = torch.tensor([0., 1.])
        hf[8] = torch.tensor([1., 0.]); hf[9] = torch.tensor([1., 0.])
        hf.requires_grad_()
        actual = diagnostic_loss(hf, item, 'D1', margin=.02)
        # Circuit A contributes .5, B contributes .01, independently of pair counts.
        self.assertAlmostEqual(float(actual), .255, places=12)
        actual.backward()
        self.assertTrue(torch.isfinite(hf.grad).all())
        self.assertGreater(float(hf.grad.norm()), 0)
        self.assertAlmostEqual(float(diagnostic_loss(hf, item, 'D1', margin=.04)), .26, places=12)

    def test_fixed_draw_balanced_unique_masked_and_rng_isolated(self):
        target = torch.tensor([0., 0., 0., .1, .15, .9, float('nan'), 0.])
        mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.bool)
        state = torch.get_rng_state().clone()
        a = balanced_columns(target, mask, torch.Generator().manual_seed(5))
        b = balanced_columns(target, mask, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(a.numel(), 4)
        self.assertEqual(a.unique().numel(), 4)
        self.assertEqual(int((target[a] == 0).sum()), 2)
        self.assertTrue(bool((target[a] <= .2).all()))
        self.assertEqual(balanced_columns(target, mask, torch.Generator().manual_seed(5), 1).numel(), 2)

    def test_pyg_selected_pairs_targets_masks_and_owners(self):
        graphs = [tiny_item(str(i), [0., .1, .9, 0.], ROLES) for i in range(2)]
        full = collate_circuits(graphs)
        columns = torch.tensor([0, 1, 4, 5])
        view = pair_view(full, columns)
        self.assertTrue(torch.equal(view.hf_tt_target, torch.tensor([0., .1, 0., .1])))
        self.assertTrue(torch.equal(pair_owners(view), torch.tensor([0, 0, 1, 1])))
        self.assertTrue(torch.equal(view.hf_pairs, full.hf_pairs[:, columns]))
        self.assertTrue(torch.equal(view.hf_tt_mask, full.hf_tt_mask[columns]))
        self.assertEqual(full.hf_pairs.shape[1], 8)
        bad = pair_view(full, columns)
        bad.hf_pairs = bad.hf_pairs.clone(); bad.hf_pairs[1, 0] += 8
        with self.assertRaises(ValueError):
            pair_owners(bad)

    def test_threshold_groups_ties_and_matches_audited_selector(self):
        scores = torch.tensor([.9, .9, .7, .7], dtype=torch.float64)
        labels = torch.tensor([True, False, True, False])
        self.assertEqual(training_threshold(scores, labels), .7)
        self.assertEqual(training_threshold(scores, labels), select_threshold(scores, labels))
        scores = torch.tensor([.9, .8, .8, .7], dtype=torch.float64)
        labels = torch.tensor([True, False, False, True])
        self.assertEqual(training_threshold(scores, labels), .9)

    def test_selection_and_schedule_reproducible(self):
        rows = [dict(circuit=str(i), valid_pairs=i + 2, positive_count=1, near_count=1,
                     node_count=100) for i in range(80)]
        rows += [dict(circuit='excluded', valid_pairs=10000, positive_count=5, near_count=20,
                      node_count=3000)]
        a, _ = select_circuits(rows, 42, max_nodes=2048)
        b, _ = select_circuits(rows, 42, max_nodes=2048)
        self.assertEqual(a, b)
        self.assertEqual(len({r['circuit'] for r in a}), 16)
        self.assertEqual(sum(int(r['circuit']) < 20 for r in a), 8)
        self.assertEqual(sum(int(r['circuit']) >= 60 for r in a), 8)
        names = [r['circuit'] for r in a]
        schedule = batch_schedule(names, 500, 4)
        self.assertEqual(len(schedule), 500)
        self.assertEqual(schedule[:4], schedule[4:8])
        self.assertEqual(sum(schedule[:4], []), names)

    def test_empty_class_not_fabricated(self):
        with self.assertRaises(ValueError):
            balanced_columns(torch.tensor([.1, .2]), torch.ones(2, dtype=torch.bool), torch.Generator())
        item = tiny_item('one_class', [.1, .1], ROLES)
        with self.assertRaises(ValueError):
            diagnostic_loss(torch.randn(8, 2), item, 'D1')

    def test_two_step_disposable_lifecycle_saves_without_resume_skip(self):
        from examples import diagnose_equivalence_fit as entry
        torch.manual_seed(17)
        source = tiny_source()
        source.update(architecture_id='test', feature_normalization={}, split_ids={'train': ['a']},
                      source_sha256={})
        # More than two distinct distances: standardizing just two scalars
        # makes the D0 objective locally constant away from its std floor.
        item = tiny_item('a', [0., .1, .15, 0.], ROLES)
        item.hf_pairs = torch.tensor([[4, 4, 6, 2], [6, 7, 7, 6]])
        subset = {'items': {'a': item}, 'selections': {'a': torch.arange(4)}}
        manifest = dict(seed=42, lr=1e-4, margin=.02, steps=2, batch_size=1,
                        circuit_order=['a'], optimizer_step_batches=[['a'], ['a']],
                        initial_checkpoint_sha256='temporary', subset_sha256='temporary')
        initial_weights = {k: v.clone() for k, v in source['model'].items()}

        def evaluation(model, batches, mode, manifest, device, chunk_size, step):
            row = dict(loss=1., precision=.5, recall=1., f1=2/3, roc_auc=.5,
                       circuit_macro_roc_auc=.5)
            return dict(step=step, fixed_balanced=row, original_full_pairs=dict(row))

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            args = SimpleNamespace(output=Path(directory), device='cpu', chunk_size=1000, log_interval=1)
            with patch.object(entry, 'evaluate', evaluation), patch.object(entry, 'CHECK_STEPS', (0, 1, 2)):
                for mode in ('D0', 'D1'):
                    result = entry.run_mode(args, mode, manifest, source, subset)
                    self.assertEqual(result['training_checks']['completed_steps'], 2)
                    self.assertTrue(result['training_checks']['gradient_finite'])
                    self.assertTrue(result['training_checks']['parameters_updated'])
                    run_dir = Path(directory) / mode
                    history = json.loads((run_dir / 'history.json').read_text())
                    self.assertEqual([r['step'] for r in history], [0, 1, 2])
                    self.assertEqual(len((run_dir / 'steps.jsonl').read_text().splitlines()), 2)
                    self.assertTrue((run_dir / 'last.pt').is_file())
                    with self.assertRaises(FileExistsError):
                        entry.run_mode(args, mode, manifest, source, subset)
            self.assertTrue(all(torch.equal(v, source['model'][k]) for k, v in initial_weights.items()))


if __name__ == '__main__':
    unittest.main()
