"""E keeps B's effective functional coefficient, samples and validation."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .test_pair_finetune import TinyDataset, tiny_item, tiny_items, tiny_source
from ..model import TriWaveGNN
from ..trainer import move
from ..utils.batch import collate_circuits
from ..utils.functional_only import functional_only_objective
from ..utils.losses import loss_terms
from ..utils.model_utils import model_outputs
from ..utils.pair_sampling import sample_functional_pairs


def branch(name):
    if name.startswith(('updates.s_', 'aggregators.s_', 'initialization.internal.s')):
        return 'structure'
    if name.startswith(('updates.f_', 'aggregators.f_', 'initialization.internal.f', 'initialization.pi_function')):
        return 'function'
    if name.startswith(('initialization.logic_embedding', 'initialization.role_embedding', 'initialization.pin_embedding')):
        return 'shared_embeddings'
    return 'unused_behavior_or_heads'


def real_weight_self_check(source, device):
    from examples import finetune_functional_pairs as base
    from examples import finetune_functional_only as entry
    items = tiny_items(source['role_vocab'])
    dataset = TinyDataset(items)
    names = [i.name for i in items]
    args = SimpleNamespace(batch_size=source['config']['batch_size'], seed=source['config']['seed'],
        num_workers=0, device=device, chunk_size=128, progress_interval=0, epochs=5,
        sampler_seed=source['config']['seed'] + 1_000_003, cap=1024)
    order = base.epoch_orders(names, args.seed, 5)[0]
    batch = collate_circuits([items[names.index(name)] for name in order])
    view, _, expected_rows = sample_functional_pairs(batch, 'circuit_cap',
        torch.Generator().manual_seed(args.sampler_seed), 1024)
    model = TriWaveGNN(len(source['logic_vocab']), len(source['role_vocab']),
                       len(source['pin_vocab']), source['config']['hidden']).to(device)
    model.load_state_dict(source['model'])
    optimizer = base.make_optimizer(model, source, 3e-5)
    assert not optimizer.state
    before = {key: p.detach().cpu().clone() for key, p in model.named_parameters()}
    source_snapshot = {key: p.clone() for key, p in source['model'].items()}
    model.train()
    full, selected = move(batch, device), move(view, device)
    output = model_outputs(model, full)
    output['hf'].retain_grad()
    loss, details = functional_only_objective(output, selected, 128)
    assert details['functional_coefficient_before'] == details['functional_coefficient_after'] == .25
    loss.backward()
    assert model.training and loss.requires_grad and bool(torch.isfinite(output['hf'].grad).all())
    grads, changed = {}, {}
    for name, p in model.named_parameters():
        group = branch(name)
        if p.grad is not None:
            assert bool(torch.isfinite(p.grad).all())
            grads[group] = grads.get(group, 0.) + float(p.grad.double().norm())
    assert grads['structure'] > 0 and grads['function'] > 0
    assert grads.get('unused_behavior_or_heads', 0.) == 0.
    optimizer.zero_grad(set_to_none=True)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        args.b_run = root / 'B'
        args.b_run.mkdir()
        destination = root / 'E'
        destination.mkdir()
        (args.b_run / 'sampling.jsonl').write_text(json.dumps({'epoch': 1, 'batch': 1,
            'circuits': expected_rows}) + '\n')
        trained = entry.train_epoch(model, optimizer, dataset, order,
            {name: i for i, name in enumerate(names)}, args, 1, destination)
        logs = [json.loads(x) for x in (destination / 'sampling.jsonl').read_text().splitlines()]
        assert logs[-1]['functional_coefficient_before'] == logs[-1]['functional_coefficient_after'] == .25
        assert trained['optimizer_steps'] == 1 and model.training
    for name, p in model.named_parameters():
        if not torch.equal(p.detach().cpu(), before[name]):
            group = branch(name)
            changed[group] = changed.get(group, 0) + 1
    assert changed.get('structure', 0) > 0 and changed.get('function', 0) > 0
    assert changed.get('unused_behavior_or_heads', 0) == 0
    with patch.object(entry, 'sample_functional_pairs', side_effect=AssertionError('Validation sampled pairs')), \
         patch.object(base, 'sample_functional_pairs', side_effect=AssertionError('Validation sampled pairs')):
        validation = entry.evaluate_validation(model, dataset, names,
            {name: i for i, name in enumerate(names)}, args, 0)
    assert validation['metric_counts']['tt_distance_mae'] == 2410
    assert 'mean_loss' not in validation
    assert 'validation_original_multitask_total_loss' in validation
    for metric in ('prob_mae', 'rc_accuracy', 'tt_distance_mae', 'hb_tp_mae', 'hb_first_mean_mae',
                   'hb_stable_mean_mae', 'hb_glitch_rate_mae'):
        assert metric in validation['metrics']
    assert all(torch.equal(v, source_snapshot[k]) for k, v in source['model'].items())
    return dict(status='passed', fixture='temporary graphs, original checkpoint weights',
        original_coefficient=.25, E_coefficient=.25, gradient_norm_by_branch=grads,
        changed_parameter_tensors_by_branch=changed, train_mode=True,
        optimizer_steps=1, optimizer_started_fresh=True, gradients_finite=True,
        all_hf_path_parameters_unfrozen=all(p.requires_grad for p in model.parameters()),
        inactive_heads_unchanged=True, source_weights_unchanged=True,
        validation_full_pair_count=2410, validation_training_sampler_calls=0,
        E_samples_equal_B_samples=True)


class FunctionalOnlyTest(unittest.TestCase):
    def test_exact_partial_gradient_and_active_denominator(self):
        for rc in (False, True):
            for behavior in (False, True):
                with self.subTest(rc=rc, behavior=behavior):
                    torch.manual_seed(41)
                    item = tiny_item('coefficient', [0., .1, .9, 0.], ['pi','cell_input','cell_output','po'])
                    item.hf_pairs = torch.tensor([[4, 4, 6, 2], [6, 7, 7, 6]])
                    item.hs_mask.fill_(rc)
                    for key in ('hb_tp_mask', 'hb_first_train_mask', 'hb_stable_train_mask', 'hb_glitch_rate_train_mask'):
                        item['pin'][key].fill_(behavior)
                    output = {'hf': torch.randn(8, 6, dtype=torch.float64, requires_grad=True),
                        'hf_prob_logits': torch.randn(8, dtype=torch.float64, requires_grad=True),
                        'hs_rc_logits': torch.randn(2, dtype=torch.float64, requires_grad=True),
                        'hb_raw': torch.randn(8, 4, dtype=torch.float64, requires_grad=True)}
                    original, terms = loss_terms(output, item, 3)
                    expected = 2 / (3 + int(rc) + 2 + 2 * int(behavior))
                    actual, details = functional_only_objective(output, item, chunk_size=2)
                    self.assertEqual(details['functional_coefficient_before'], expected)
                    self.assertEqual(details['functional_coefficient_after'], expected)
                    self.assertAlmostEqual(float(actual), expected * terms['func'], places=12)
                    g_original = torch.autograd.grad(original, output['hf'], retain_graph=True)[0]
                    g_e = torch.autograd.grad(actual, output['hf'], retain_graph=True)[0]
                    torch.testing.assert_close(g_original, g_e, atol=1e-12, rtol=1e-12)
                    actual.backward()
                    for key in ('hf_prob_logits', 'hs_rc_logits', 'hb_raw'):
                        self.assertIsNone(output[key].grad)

    def test_no_valid_functional_term_retains_zero_contribution(self):
        item = tiny_item('empty', [0.], ['pi','cell_input','cell_output','po'])
        output = {'hf': torch.randn(8, 4, requires_grad=True), 'hf_prob_logits': torch.zeros(8),
                  'hs_rc_logits': torch.zeros(2), 'hb_raw': torch.zeros(8, 4)}
        loss, details = functional_only_objective(output, item)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(details['functional_coefficient_after'], 0.)
        loss.backward()
        self.assertEqual(float(output['hf'].grad.norm()), 0.)

    def test_actual_model_optimizer_and_validation(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = real_weight_self_check(tiny_source(), 'cpu')
        self.assertTrue(result['inactive_heads_unchanged'])

    def test_sampling_mismatch_is_rejected(self):
        from examples.finetune_functional_only import compare_sampling
        expected = {'epoch': 1, 'batch': 2, 'circuits': [{'selected_columns_sha256': 'original'}]}
        compare_sampling(expected['circuits'], expected, 1, 2)
        with self.assertRaises(ValueError):
            compare_sampling([{'selected_columns_sha256': 'different'}], expected, 1, 2)

    def test_validation_is_original_B_implementation_with_loss_renamed(self):
        from examples import finetune_functional_only as entry
        result = {'mean_loss': .123, 'metrics': {'prob_mae': .04}, 'functional_equivalence': {'f1': .01}}
        with patch.object(entry.base, 'evaluate_validation', return_value=dict(result)) as call:
            supplied = ('model', 'dataset', ['val1'], {'val1': 0}, 'args', 1)
            actual = entry.evaluate_validation(*supplied)
        call.assert_called_once_with(*supplied)
        self.assertEqual(actual['validation_original_multitask_total_loss'], .123)
        self.assertNotIn('mean_loss', actual)
        self.assertEqual(actual['metrics'], result['metrics'])

    def test_disposable_end_to_end_run_keeps_loss_names_and_checkpoints(self):
        from examples import finetune_functional_pairs as base
        from examples import finetune_functional_only as entry
        from examples.evaluate import digest_file
        source = tiny_source()
        source.update(architecture_id=TriWaveGNN.architecture_id, epoch=40, stage=3,
            feature_normalization={}, source_sha256={'graphs.npz': 'fixture', 'labels.npz': 'fixture'})
        items = tiny_items(source['role_vocab']) + tiny_items(source['role_vocab'])
        for i, item in enumerate(items):
            item.name = ('train_' if i < 3 else 'val_') + item.name
        dataset = TinyDataset(items)
        dataset.ids = [item.name for item in items]
        dataset.source_sha256 = source['source_sha256']
        splits = {'train': dataset.ids[:3], 'val': dataset.ids[3:], 'test': ['never_loaded']}
        source['split_ids'] = splits
        orders = base.epoch_orders(splits['train'], 42, 1)

        def loader(checkpoint, supplied, device):
            model = TriWaveGNN(1, 4, 1, hidden=8).to(device)
            model.load_state_dict(source['model'])
            return model.eval(), source

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            b, e = root / 'B', root / 'E'
            b.mkdir(); e.mkdir()
            checkpoint = root / 'stage3_best.pt'
            torch.save(source, checkpoint)
            indices = {name: i for i, name in enumerate(dataset.ids)}
            batch = collate_circuits([items[indices[n]] for n in orders[0]])
            _, _, rows = sample_functional_pairs(batch, 'circuit_cap', torch.Generator().manual_seed(1000045), 1024)
            (b / 'sampling.jsonl').write_text(json.dumps({'epoch': 1, 'batch': 1, 'circuits': rows}) + '\n')
            for dest in (b, e):
                base.write_json(dest / 'split.json', splits)
                base.write_json(dest / 'epoch_orders.json', orders)
            reference = {'metrics': {}}
            base.write_json(b / 'val_epoch_1.json', reference)
            base.write_json(e / 'baseline_B_reference.json', reference)
            plan = dict(group='E', status='prepared', comparison_b_run=str(b),
                data=str(root), initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=digest_file(checkpoint),
                b_sampling_sha256=digest_file(b / 'sampling.jsonl'), code_sha256={},
                lr=3e-5, epochs=1, cap=1024, batch_size=32, seed=42, num_workers=0,
                sampler_seed=1000045, lr_step=20, evaluation_chunk_size=128)
            base.write_json(e / 'plan.json', plan)
            args = SimpleNamespace(output=e, chunk_size=128, progress_interval=0, device='cpu')
            with patch.object(entry, 'CircuitDataset', return_value=dataset), \
                 patch.object(entry, 'load_model', side_effect=loader):
                entry.run(args)
                with self.assertRaises(FileExistsError):
                    entry.run(args)
            history = json.loads((e / 'history.json').read_text())
            self.assertIn('training_functional_objective', history[0]['train'])
            self.assertNotIn('mean_loss', history[0]['train'])
            self.assertIn('validation_original_multitask_total_loss', history[0]['val'])
            self.assertNotIn('mean_loss', history[0]['val'])
            self.assertEqual(history[0]['val']['metric_counts']['tt_distance_mae'], 2410)
            self.assertEqual(history[0]['train']['functional_coefficients'], [.25])
            for name in ('best_loss.pt', 'best_equiv.pt', 'last.pt'):
                self.assertTrue((e / name).is_file())
            last = torch.load(e / 'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(last['epoch'], 1)
            self.assertTrue(all(int(s['step']) == 1 for s in last['optimizer']['state'].values()))
            self.assertEqual(json.loads((e / 'run.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
