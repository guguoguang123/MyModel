"""Prepare/self-check or manually run D0/D1 train-subset fitting diagnostics.

No default action starts training. --prepare --self-check prepares immutable
inputs and uses disposable models; --run runs D0 then D1 from the same source.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.compare_functional_stages import cosine_scores64, within_circuit_auc
from examples.evaluate import RUN_DIR, digest_file
from examples.finetune_functional_pairs import direct_log, make_optimizer, write_json
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.model import TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import collate_circuits
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint
from triwave_gnn.utils.config import DEFAULT_DATA_DIR
from triwave_gnn.utils.equivalence_audit import pair_metrics, QUANTILES
from triwave_gnn.utils.equivalence_fit import (NOTICE, balanced_columns, batch_schedule,
    diagnostic_loss, pair_owners, pair_view, select_circuits, training_threshold)
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

MODES = ('D0', 'D1')
CHECK_STEPS = (0, 100, 250, 500)


def new_model(source, device, seed):
    set_seed(seed)
    model = TriWaveGNN(len(source['logic_vocab']), len(source['role_vocab']),
                       len(source['pin_vocab']), source['config']['hidden']).to(device)
    model.load_state_dict(source['model'])
    # Hf depends on structural propagation: never freeze Hs or shared modules.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def prepare(args):
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Refusing to overwrite diagnostic: {root}')
    source = load_checkpoint(args.checkpoint, 'cpu')
    if source.get('stage') != 3 or source.get('architecture_id') != TriWaveGNN.architecture_id:
        raise ValueError('Require original stage3_best.pt with current architecture')
    train = set(source['split_ids']['train'])
    with np.load(args.data / 'labels.npz', allow_pickle=False) as labels:
        ids = list(map(str, labels['design_ids']))
        ptr = labels['hf_tt_pair_ptr']
        target, mask = labels['hf_tt_distance'], labels['hf_tt_pair_mask']
    with np.load(args.data / 'graphs.npz', allow_pickle=False) as graphs:
        if list(map(str, graphs['design_ids'])) != ids:
            raise ValueError('Dataset design order differs')
        nodes = np.diff(graphs['node_ptr'])
    rows = []
    for i, name in enumerate(ids):
        if name not in train:
            continue
        start, stop = map(int, ptr[i:i + 2])
        t, valid = target[start:stop], mask[start:stop] & np.isfinite(target[start:stop])
        pos = np.isclose(t, 0, atol=1e-8, rtol=0)
        near = ~pos & (t > 0) & (t <= .2)
        rows.append(dict(circuit=name, dataset_index=i, node_count=int(nodes[i]),
                         valid_pairs=int(valid.sum()), positive_count=int((valid & pos).sum()),
                         near_count=int((valid & near).sum())))
    selected, rule = select_circuits(rows, args.seed, max_nodes=args.max_nodes)
    # Use the normal loader and checkpoint normalization; do not refit scaling.
    info('Loading original dataset once to snapshot the 16 selected circuits')
    dataset = CircuitDataset(args.data, cache_mode='none')
    model, _ = load_model(args.checkpoint, dataset, device='cpu')
    del model
    generator = torch.Generator().manual_seed(args.seed + 1_000_003)
    items, selections = {}, {}
    role = dataset.role_vocab.index('cell_output')
    for row in selected:
        name = row['circuit']
        item = dataset[row['dataset_index']]
        columns = balanced_columns(item.hf_tt_target, item.hf_tt_mask, generator, args.pairs_per_class)
        view = pair_view(item, columns)
        pair_owners(view)
        if not bool((item['pin'].node_role[view.hf_pairs] == role).all()):
            raise ValueError(f'{name}: selected endpoint is not cell_output')
        if columns.unique().numel() != columns.numel():
            raise ValueError('Repeated source pair column')
        # Existing data have distinct pairs; fail rather than silently duplicate endpoints.
        unique_endpoints = torch.unique(view.hf_pairs.sort(dim=0).values.T, dim=0).shape[0]
        if unique_endpoints != columns.numel():
            raise ValueError(f'{name}: selected pairs repeat the same unordered endpoints')
        items[name], selections[name] = item, columns
        row.update(selected_per_class=columns.numel() // 2, selected_pairs=columns.numel(),
                   source_columns=columns.tolist(), local_node_pairs=view.hf_pairs.T.tolist(),
                   targets=view.hf_tt_target.tolist(), masks=view.hf_tt_mask.tolist(),
                   node_depth=int(item['pin'].depth.max()))
    names = [r['circuit'] for r in selected]
    schedule = batch_schedule(names, args.steps, args.batch_size)
    root.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(items=items, selections=selections), root / 'subset.pt')
    manifest = dict(status='prepared', notice=NOTICE, data=str(args.data.resolve()),
        checkpoint=str(args.checkpoint.resolve()), initial_checkpoint_sha256=digest_file(args.checkpoint),
        subset_sha256=digest_file(root / 'subset.pt'), source_sha256=dataset.source_sha256,
        seed=args.seed, pair_seed=args.seed + 1_000_003, selection_rule=rule, circuits=selected,
        circuit_order=names, optimizer_step_batches=schedule, steps=args.steps,
        batch_size=args.batch_size, lr=args.lr, margin=args.margin, pairs_per_class=args.pairs_per_class,
        check_steps=list(CHECK_STEPS), gradient_clip_norm=1.,
        diagnostic_fit_criterion=dict(f1_at_least=.95, pooled_auc_at_least=.99,
                                     note='diagnostic convention only, not a generalization claim'),
        training=dict(D0='original standardized-distance L1, pooled within the current mini-batch',
                      D1='mean over circuits of .5*mean(d_pos)+.5*mean(relu(margin-d_neg))',
                      other_task_losses=False, freeze_hf_path=False, restore_optimizer=False,
                      negative_training_definition='not positive and 0<TT<=0.2',
                      positive_definition='isclose(TT,0,atol=1e-8,rtol=0)',
                      model_precision='float32, no autocast',
                      optimizer='fresh Adam with source hyperparameters except lr=1e-4',
                      unused_modules='behavior and readout heads receive no loss; structural branch stays active'),
        evaluation=dict(model_mode='eval + no_grad, always restored to train before backward',
                        cosine='audited chunked cosine_scores64: cast endpoints before cosine',
                        threshold='max pooled F1 on fixed balanced training pairs; grouped ties; highest threshold on F1 ties',
                        full_pairs='all original valid pairs of same 16 TRAIN circuits; same threshold as balanced set',
                        loss='same objective/mini-batch grouping as training, averaged by circuit count; full-pair D1 uses all original nonzero negatives',
                        test_evaluated=False, validation_evaluated=False),
        inherited=['model weights', 'vocabularies', 'feature_normalization', 'original train membership'])
    write_json(root / 'manifest.json', manifest)
    info(NOTICE)
    info(f'Prepared 16 train circuits: {sum(r["selected_pairs"] for r in selected)} fixed pairs; '
         f'{sum(r["valid_pairs"] for r in selected)} original valid pairs')
    for row in selected:
        info(f'{row["circuit"]}: nodes={row["node_count"]} full_pairs={row["valid_pairs"]} '
             f'fixed={row["selected_per_class"]}+{row["selected_per_class"]}')
    info(f'Prepared only: {root}; no 500-step experiment started')


def load_prepared(args):
    manifest = json.loads((args.output / 'manifest.json').read_text())
    if manifest['steps'] != 500 or manifest['check_steps'] != list(CHECK_STEPS):
        raise ValueError('Diagnostic must run exactly 500 steps with specified reporting steps')
    if digest_file(Path(manifest['checkpoint'])) != manifest['initial_checkpoint_sha256']:
        raise ValueError('Original checkpoint changed since preparation')
    if digest_file(args.output / 'subset.pt') != manifest['subset_sha256']:
        raise ValueError('Prepared subset changed')
    source = load_checkpoint(manifest['checkpoint'], 'cpu')
    subset = load_checkpoint(args.output / 'subset.pt', 'cpu')
    return manifest, source, subset


def make_batches(subset, manifest):
    result = []
    names = manifest['circuit_order']
    for start in range(0, len(names), manifest['batch_size']):
        group = names[start:start + manifest['batch_size']]
        graphs = [subset['items'][name] for name in group]
        full = collate_circuits(graphs)
        offset, columns = 0, []
        for name, graph in zip(group, graphs):
            selected = subset['selections'][name]
            columns.append(selected + offset)
            offset += graph.hf_tt_target.numel()
        columns = torch.cat(columns)
        view = pair_view(full, columns)
        pair_owners(view)
        expected = torch.cat([subset['items'][name].hf_tt_target[subset['selections'][name]] for name in group])
        if not torch.equal(view.hf_tt_target, expected) or not bool(view.hf_tt_mask.all()):
            raise ValueError('Batched selection lost target/mask alignment')
        result.append(dict(names=group, full=full, columns=columns))
    expected_schedule = [row['names'] for row in result]
    if any(names != expected_schedule[i % len(result)] for i, names in enumerate(manifest['optimizer_step_batches'])):
        raise ValueError('Optimizer batch order differs from prepared schedule')
    return result


def parameter_group(name):
    if name.startswith(('aggregators.s_', 'updates.s_', 'initialization.internal.s')):
        return 'structure'
    if name.startswith(('aggregators.f_', 'updates.f_', 'initialization.internal.f', 'initialization.pi_function')):
        return 'function'
    if name.startswith(('initialization.logic_embedding', 'initialization.role_embedding', 'initialization.pin_embedding')):
        return 'shared_embeddings'
    return 'unused_behavior_or_heads'


def gradient_report(model, hf):
    groups = {name: dict(norm=0., tensors_with_grad=0, nonzero_tensors=0, finite=True)
              for name in ('structure', 'function', 'shared_embeddings', 'unused_behavior_or_heads')}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = groups[parameter_group(name)]
        group['tensors_with_grad'] += 1
        finite = bool(torch.isfinite(p.grad).all())
        group['finite'] &= finite
        if finite:
            group['norm'] = math.hypot(group['norm'], float(p.grad.double().norm()))
            group['nonzero_tensors'] += int(bool((p.grad != 0).any()))
    result = dict(groups=groups, hf_gradient_finite=hf.grad is not None and bool(torch.isfinite(hf.grad).all()),
                  hf_gradient_norm=float(hf.grad.double().norm()) if hf.grad is not None else None)
    result['finite'] = result['hf_gradient_finite'] and all(g['finite'] for g in groups.values())
    if not result['finite']:
        raise FloatingPointError('Nonfinite or missing Hf gradient')
    return result


def parameter_changes(model, initial):
    groups = {name: dict(changed_tensors=0, max_abs_change=0., l2_change=0.)
              for name in ('structure', 'function', 'shared_embeddings', 'unused_behavior_or_heads')}
    for name, p in model.named_parameters():
        delta = p.detach().cpu().double() - initial[name].double()
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError(f'Nonfinite updated parameter: {name}')
        g = groups[parameter_group(name)]
        g['changed_tensors'] += int(bool((delta != 0).any()))
        g['max_abs_change'] = max(g['max_abs_change'], float(delta.abs().max()))
        g['l2_change'] = math.hypot(g['l2_change'], float(delta.norm()))
    return groups


def distance_quantiles(scores, labels):
    result = {}
    for name, mask in (('positive', labels), ('negative', ~labels)):
        values = 1 - scores[mask]
        result[name] = dict(count=values.numel(), median=float(torch.quantile(values, .5)) if values.numel() else None,
            quantiles={str(q): float(torch.quantile(values, q)) for q in QUANTILES} if values.numel() else None)
    return result


def summarize(parts, labels, names, offsets, loss, threshold):
    scores, truth = torch.cat(parts), torch.cat(labels)
    rows = [dict(circuit=name, **pair_metrics(scores[a:b], truth[a:b], threshold))
            for name, a, b in zip(names, offsets, offsets[1:])]
    return dict(loss=loss, **pair_metrics(scores, truth, threshold), **within_circuit_auc(rows),
                distance_distribution=distance_quantiles(scores, truth), per_circuit=rows)


def evaluate(model, batches, mode, manifest, device, chunk_size, step):
    was_training = model.training
    model.eval()
    collections = {scope: dict(scores=[], labels=[], names=[], offsets=[0], loss=0.)
                   for scope in ('fixed_balanced', 'original_full_pairs')}
    try:
        with torch.no_grad():
            for batch in batches:
                item = move(batch['full'], device)
                fixed = pair_view(item, batch['columns'].to(device))
                _, hf, _ = model(item, include_behavior=False)
                for scope, view in (('fixed_balanced', fixed), ('original_full_pairs', item)):
                    c = collections[scope]
                    valid = view.hf_tt_mask.bool() & torch.isfinite(view.hf_tt_target)
                    scores = cosine_scores64(hf, view.hf_pairs, valid, chunk_size)
                    targets = view.hf_tt_target[valid].cpu()
                    labels = torch.isclose(targets, torch.zeros_like(targets), atol=1e-8, rtol=0)
                    owners = pair_owners(view)[valid].cpu()
                    for i, name in enumerate(batch['names']):
                        local = owners == i
                        c['scores'].append(scores[local]); c['labels'].append(labels[local])
                        c['names'].append(name); c['offsets'].append(c['offsets'][-1] + int(local.sum()))
                    value = float(diagnostic_loss(hf, view, mode, manifest['margin'], chunk_size))
                    if not math.isfinite(value):
                        raise FloatingPointError('Nonfinite evaluation loss')
                    c['loss'] += value * len(batch['names']) / len(manifest['circuit_order'])
        fixed = collections['fixed_balanced']
        threshold = training_threshold(torch.cat(fixed['scores']), torch.cat(fixed['labels']))
        result = dict(step=step, notice=NOTICE, threshold_source='fixed balanced TRAIN subset only')
        for scope, c in collections.items():
            result[scope] = summarize(c['scores'], c['labels'], c['names'], c['offsets'], c['loss'], threshold)
            row = result[scope]
            info(f'{mode} step={step} {scope} | ' + ' | '.join(f'{k}={row[k]:.10g}' for k in
                 ('loss', 'precision', 'recall', 'f1', 'roc_auc', 'circuit_macro_roc_auc', 'similarity_threshold')))
            info(f'{mode} step={step} {scope} distance_quantiles=' + json.dumps(row['distance_distribution']))
        return result
    finally:
        model.train(was_training)


def self_check(args):
    manifest, source, subset = load_prepared(args)
    batches = make_batches(subset, manifest)
    results = {}
    for mode in MODES:
        model = new_model(source, args.device, manifest['seed'])
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        optimizer = make_optimizer(model, source, manifest['lr'])
        assert not optimizer.state, 'Optimizer must start without old moments'
        batch = batches[0]
        item = move(batch['full'], args.device)
        view = pair_view(item, batch['columns'].to(args.device))
        model.train()
        with torch.enable_grad():
            _, hf, _ = model(item, include_behavior=False)
            hf.retain_grad()
            loss = diagnostic_loss(hf, view, mode, manifest['margin'], args.chunk_size)
            assert loss.requires_grad and model.training and not torch.is_inference_mode_enabled()
            loss.backward()
            gradients = gradient_report(model, hf)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        changes = parameter_changes(model, initial)
        assert all(gradients['groups'][g]['norm'] > 0 and changes[g]['changed_tensors'] > 0
                   for g in ('structure', 'function')), 'Hf structural/function path must update'
        assert changes['unused_behavior_or_heads']['changed_tensors'] == 0
        assert digest_file(Path(manifest['checkpoint'])) == manifest['initial_checkpoint_sha256']
        results[mode] = dict(loss=float(loss.detach()), gradient_norm=float(norm), gradients=gradients,
                             parameter_changes=changes, train_mode=model.training,
                             original_checkpoint_unchanged=True, independent_initialization=True)
        info(f'SELF-CHECK {mode} passed: finite backward + Hs/Hf parameters updated; disposable model only')
        del loss, hf, optimizer, model, item, view
    # One disposable step-0 evaluation verifies the real full-pair threshold contract.
    model = new_model(source, args.device, manifest['seed'])
    model.train()
    eval_result = evaluate(model, batches, 'D1', manifest, args.device, args.chunk_size, 0)
    assert model.training, 'Evaluation must restore train mode'
    fixed, full = eval_result['fixed_balanced'], eval_result['original_full_pairs']
    assert fixed['similarity_threshold'] == full['similarity_threshold']
    assert full['positive_count'] + full['negative_count'] == sum(r['valid_pairs'] for r in manifest['circuits'])
    assert fixed['positive_count'] == fixed['negative_count'] == sum(r['selected_per_class'] for r in manifest['circuits'])
    write_json(args.output / 'self_check.json', dict(status='passed', modes=results,
        step0=eval_result, full_pairs_share_fixed_threshold=True, evaluation_restores_train=True,
        initial_checkpoint_sha256=manifest['initial_checkpoint_sha256']))
    info('SELF-CHECK passed; no formal D0/D1 run started')


def save_checkpoint(path, model, optimizer, source, manifest, mode, step, result):
    state = {k: deepcopy(source[k]) for k in ('architecture_id', 'logic_vocab', 'role_vocab', 'pin_vocab',
             'feature_normalization', 'split_ids', 'source_sha256')}
    state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), stage=3, step=step,
                 config=dict(source['config'], lr=manifest['lr'], resume=None), diagnostic=manifest,
                 diagnostic_mode=mode, diagnostic_metrics=result, diagnostic_only=True)
    atomic_save(state, path)


def run_mode(args, mode, manifest, source, subset):
    run_dir = args.output / mode
    run_dir.mkdir(exist_ok=True)
    if (run_dir / 'run.json').exists():
        raise FileExistsError(f'Diagnostic already started, preserve results: {run_dir}')
    record = dict(status='running', mode=mode, notice=NOTICE, completed_steps=0,
                  initial_checkpoint_sha256=manifest['initial_checkpoint_sha256'],
                  subset_sha256=manifest['subset_sha256'], lr=manifest['lr'], margin=manifest['margin'])
    write_json(run_dir / 'run.json', record)
    with direct_log(run_dir / 'console.log'):
        info(NOTICE)
        info(f'{mode}: fresh weights + optimizer | lr={manifest["lr"]} | margin={manifest["margin"]} | {manifest["steps"]} steps')
        model = new_model(source, args.device, manifest['seed'])
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        optimizer = make_optimizer(model, source, manifest['lr'])
        assert not optimizer.state
        batches = make_batches(subset, manifest)
        history = [evaluate(model, batches, mode, manifest, args.device, args.chunk_size, 0)]
        write_json(run_dir / 'history.json', history)
        write_json(run_dir / 'metrics_step_0.json', history[0])
        first_changes, grad_min, grad_max, structural_steps, function_steps = None, math.inf, 0., 0, 0
        started = time.monotonic()
        with (run_dir / 'steps.jsonl').open('w', buffering=1) as stream:
            for step in range(1, manifest['steps'] + 1):
                model.train()
                batch = batches[(step - 1) % len(batches)]
                item = move(batch['full'], args.device)
                view = pair_view(item, batch['columns'].to(args.device))
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    _, hf, _ = model(item, include_behavior=False)
                    hf.retain_grad()
                    loss = diagnostic_loss(hf, view, mode, manifest['margin'], args.chunk_size)
                    if not model.training or not loss.requires_grad or not bool(torch.isfinite(loss)):
                        raise RuntimeError('Training must be train-mode, finite, differentiable')
                    loss.backward()
                    gradients = gradient_report(model, hf)
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                    optimizer.step()
                grad_min, grad_max = min(grad_min, norm), max(grad_max, norm)
                structural_steps += int(gradients['groups']['structure']['norm'] > 0)
                function_steps += int(gradients['groups']['function']['norm'] > 0)
                if step == 1:
                    first_changes = parameter_changes(model, initial)
                    if not all(first_changes[g]['changed_tensors'] for g in ('structure', 'function')):
                        raise RuntimeError('First optimizer step did not update Hs/Hf paths')
                stream.write(json.dumps(dict(step=step, circuits=batch['names'], loss=float(loss.detach()),
                    actual_lr=[g['lr'] for g in optimizer.param_groups], gradient_norm_before_clip=norm,
                    gradients=gradients), allow_nan=False) + '\n')
                if step == 1 or step % args.log_interval == 0:
                    info(f'{mode} step={step}/{manifest["steps"]} loss={float(loss.detach()):.9g} grad_norm={norm:.9g} '
                         f'lr={optimizer.param_groups[0]["lr"]} elapsed={time.monotonic()-started:.1f}s')
                del hf, loss, item, view
                if step in CHECK_STEPS:
                    result = evaluate(model, batches, mode, manifest, args.device, args.chunk_size, step)
                    history.append(result)
                    write_json(run_dir / f'metrics_step_{step}.json', result)
                    write_json(run_dir / 'history.json', history)
                    save_checkpoint(run_dir / 'last.pt', model, optimizer, source, manifest, mode, step, result)
                    record.update(completed_steps=step)
                    write_json(run_dir / 'run.json', record)
        changes = parameter_changes(model, initial)
        fit = history[-1]['fixed_balanced']
        success = fit['f1'] >= .95 and fit['roc_auc'] >= .99
        record.update(status='completed', completed_steps=manifest['steps'], gradient_finite=True,
            gradient_norm_range=[grad_min, grad_max], structure_nonzero_gradient_steps=structural_steps,
            function_nonzero_gradient_steps=function_steps, first_step_parameter_changes=first_changes,
            final_parameter_changes=changes, parameters_updated=any(g['changed_tensors'] for g in changes.values()),
            fit_criterion_met=success, conclusion='达到本次预设拟合标准' if success else '本次预算下未成功')
        write_json(run_dir / 'run.json', record)
        info(f'Completed {mode}: {record["conclusion"]}; all {manifest["steps"]} optimizer steps executed; no test/val evaluated')
        del model, optimizer
        return dict(mode=mode, initial=history[0], final=history[-1], training_checks=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--checkpoint', type=Path, default=RUN_DIR / 'stage3_best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--run', choices=('both', 'D0', 'D1'))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--pairs-per-class', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-nodes', type=int, default=20000)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not (args.prepare or args.self_check or args.run) or (args.run and (args.prepare or args.self_check)):
        parser.error('Use --prepare [--self-check], --self-check, or --run; no implicit training')
    if args.steps != 500 or args.lr <= 0 or args.margin <= 0 or not 1 <= args.pairs_per_class <= 128:
        parser.error('Require 500 steps, positive lr/margin, 1..128 pairs per class')
    if min(args.batch_size, args.max_nodes, args.chunk_size, args.log_interval) < 1:
        parser.error('Batch size, node budget, chunk size, log interval must be positive')
    if args.prepare:
        prepare(args)
    if args.self_check:
        self_check(args)
    if args.run:
        manifest, source, subset = load_prepared(args)
        modes = MODES if args.run == 'both' else (args.run,)
        # Check all output names before starting either model.
        for mode in modes:
            if (args.output / mode / 'run.json').exists():
                raise FileExistsError(f'{mode} already started; use a new prepared experiment directory')
        report = {mode: run_mode(args, mode, manifest, source, subset) for mode in modes}
        if digest_file(Path(manifest['checkpoint'])) != manifest['initial_checkpoint_sha256']:
            raise RuntimeError('Original checkpoint changed during run')
        write_json(args.output / ('comparison.json' if args.run == 'both' else f'{args.run}_summary.json'), report)
        info(NOTICE)
        for mode, result in report.items():
            for phase in ('initial', 'final'):
                for scope in ('fixed_balanced', 'original_full_pairs'):
                    row = result[phase][scope]
                    info(f'SUMMARY {mode} {phase} {scope} | ' + ' | '.join(f'{k}={row[k]:.9g}' for k in
                         ('loss', 'precision', 'recall', 'f1', 'roc_auc', 'circuit_macro_roc_auc')))
            info(f'SUMMARY {mode}: gradients_finite={result["training_checks"]["gradient_finite"]} '
                 f'parameters_updated={result["training_checks"]["parameters_updated"]} '
                 f'conclusion={result["training_checks"]["conclusion"]}')


if __name__ == '__main__':
    main()
