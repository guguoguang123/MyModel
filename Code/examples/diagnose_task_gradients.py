"""Fixed 32 train batches, two checkpoints, no optimizer or training updates."""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.evaluate import RUN_DIR, digest_file, validate_splits
from examples.finetune_functional_pairs import direct_log, epoch_orders, write_json
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS
from triwave_gnn.utils.checkpoint import load as load_checkpoint
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.pair_sampling import sample_functional_pairs
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.utils.task_gradient_diagnostic import (
    TASKS, SCOPES, COMPARISONS, module_parameter_groups, weighted_task_losses,
    task_gradients, gradient_measurements, aggregate, prioritized_findings)

ROOT = RUN_DIR.parent


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f'Preserving existing directory: {args.output}')
    b = json.loads((args.b_run / 'run.json').read_text())
    if b['status'] != 'completed' or b['cap'] != 1024 or b['mode'] != 'circuit_cap':
        raise ValueError('Require completed B circuit_cap=1024')
    source_path = Path(b['initial_checkpoint'])
    if digest_file(source_path) != b['initial_checkpoint_sha256']:
        raise ValueError('Original stage3 checkpoint differs from B')
    source = load_checkpoint(source_path, 'cpu')
    e = load_checkpoint(args.e_checkpoint, 'cpu')
    if e.get('finetune_mode') != 'E_circuit_cap_functional_only':
        raise ValueError('E checkpoint is not the E functional-only experiment')
    for key in ('split_ids', 'config', 'logic_vocab', 'role_vocab', 'pin_vocab'):
        if key == 'config':
            if e[key]['batch_size'] != b['batch_size'] or e[key]['hidden'] != source[key]['hidden']:
                raise ValueError('Model/batch configuration differs')
        elif e[key] != source[key]:
            raise ValueError(f'Checkpoint {key} differs')
    for relation in ('net', 'cell'):
        for stat in ('mean', 'std'):
            if not torch.equal(torch.as_tensor(e['feature_normalization'][relation][stat]),
                               torch.as_tensor(source['feature_normalization'][relation][stat])):
                raise ValueError('Feature normalization differs between checkpoints')
    splits = json.loads((args.b_run / 'split.json').read_text())
    validate_splits(splits, source)
    orders = json.loads((args.b_run / 'epoch_orders.json').read_text())
    if orders != epoch_orders(splits['train'], b['seed'], b['epochs']):
        raise ValueError('B saved epoch order differs from seed')
    train = set(splits['train'])
    references = []
    with (args.b_run / 'sampling.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['epoch'] != 1 or len(references) == 32:
                break
            expected = orders[0][len(references) * b['batch_size']:(len(references) + 1) * b['batch_size']]
            if [x['circuit'] for x in row['circuits']] != expected or any(x not in train for x in expected):
                raise ValueError('Selected batches differ from original train order')
            references.append({'batch': row['batch'], 'circuits': row['circuits']})
    if len(references) != 32:
        raise ValueError('Need 32 complete B epoch-1 batches')
    coverage = [x for r in references for x in r['circuits']]
    checks = {'stage3_original': source_path.resolve(), 'E_best_equiv': args.e_checkpoint.resolve()}
    code_dir = Path(__file__).parents[1]
    code_paths = [Path(__file__), Path(__file__).with_name('finetune_functional_pairs.py'),
        *[code_dir / p for p in ('triwave_gnn/utils/task_gradient_diagnostic.py', 'triwave_gnn/utils/losses.py',
            'triwave_gnn/utils/pair_sampling.py', 'triwave_gnn/model.py', 'triwave_gnn/arch/initialization.py',
            'triwave_gnn/arch/propagation.py', 'triwave_gnn/arch/message.py', 'triwave_gnn/utils/model_utils.py')]]
    plan = dict(status='prepared', seed=b['seed'], sampler_seed=b['sampler_seed'], cap=1024,
        batch_size=b['batch_size'], num_workers=b['num_workers'], data=b['data'],
        checkpoints={k: str(v) for k, v in checks.items()}, checkpoint_sha256={k: digest_file(v) for k, v in checks.items()},
        E_checkpoint_epoch=e['epoch'], batches=references,
        selected_batches=32, split='train only', selection_rule='B epoch-1 first 32 complete saved batches; no metric-based selection',
        circuit_visits=len(coverage), unique_circuits=len({r['circuit'] for r in coverage}),
        sampled_functional_positive_count=sum(r['sampled_strata']['zero'] for r in coverage),
        sampled_functional_negative_count=sum(r['sampled_pairs'] - r['sampled_strata']['zero'] for r in coverage),
        full_train_order=orders[0], source_sha256=source['source_sha256'], chunk_size=100000,
        code_sha256={str(p.resolve()): digest_file(p) for p in code_paths},
        dependency_review={
            'shared_initialization': 'actual logic/role/pin embedding module parameters shared by s/f/b initialization',
            'structure_path': 'internal[s], s_net/s_cell aggregators+GRU updates; Hs sources feed f and b propagation',
            'function_path': 'internal[f], pi_function, f_net/f_cell aggregators+GRU updates; Hf sources feed b',
            'behavior_path': 'internal[b], pi_behavior, b_net/b_cell aggregators+GRU updates; cannot feed Hs/Hf',
            'readouts': 'prob(Hf); RC(pair_features(Hs)); behavior(Hb)',
            'validation': 'autograd None vs connected zero recorded for every task/parameter on actual batches'},
        measurement={
            'model_mode': 'train with gradients enabled; reviewed model contains no dropout/BatchNorm',
            'forward': 'one forward per checkpoint/batch, four autograd.grad calls from same graph',
            'weights': 'original active stage3 denominator; usual prob=3/8 rc=1/8 func=2/8 behavior=2/8',
            'cosine_support': 'intersection of autograd-connected parameter tensors, all tensor elements including zero entries',
            'combined_other_support': 'func-connected tensors intersecting union of prob/rc/behavior dependencies',
            'total_support': 'all func-connected tensors; compare func with sum of all four weighted gradients',
            'norm_ratios': 'weighted other norm / weighted func norm on same common tensors; task full module norms also saved',
            'not_applicable': 'None cosine and ratio for no common dependency or either norm zero',
            'quantiles': [0, .05, .25, .5, .75, .95, 1],
            'aggregation': 'each applicable batch has equal weight, negative fraction denominator excludes N/A',
            'numerics': 'original float32 model/loss; gradient dot/norm reductions on CPU float64; no autocast or clipping',
            'interpretation': '优化器更新前的诊断，不等同于 Adam 实际更新方向或长期训练效果；梯度冲突不是最终因果结论',
            'optimizer_steps': 0, 'validation_or_test': False})
    args.output.mkdir(parents=True)
    write_json(args.output / 'plan.json', plan)
    info(f'Prepared: 32 train batches x {b["batch_size"]}; circuits={plan["unique_circuits"]} '
         f'func positive={plan["sampled_functional_positive_count"]} negative={plan["sampled_functional_negative_count"]}')
    info(f'Checkpoints: original stage3 and E best_equiv epoch={e["epoch"]}; optimizer steps=0')


def run_checkpoint(args, plan, label, dataset, indices):
    set_seed(plan['seed'])
    model, state = load_model(Path(plan['checkpoints'][label]), dataset, args.device)
    # No optimizer object is created anywhere in this diagnostic.
    model.train()
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for m in model.modules()):
        raise ValueError('Re-review required: mutable/stochastic train-mode layer introduced')
    names, groups = module_parameter_groups(model)
    parameters = tuple(dict(model.named_parameters())[name] for name in names)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    dependencies = {task: {name: {'group': groups[name], 'autograd_connected_batches': 0,
        'nonzero_gradient_batches': 0} for name in names} for task in TASKS}
    selected_names = plan['full_train_order'][:32 * plan['batch_size']]
    loader = DataLoader(Subset(dataset, [indices[n] for n in selected_names]), batch_size=plan['batch_size'],
        shuffle=False, num_workers=plan['num_workers'], exclude_keys=SCHEDULE_KEYS,
        generator=torch.Generator().manual_seed(plan['seed'] + 1))
    sampler = torch.Generator().manual_seed(plan['sampler_seed'])
    rows, fixed_columns = [], []
    started = time.monotonic()
    directory = args.output / label
    directory.mkdir(exist_ok=False)
    with (directory / 'batches.jsonl').open('w', buffering=1) as stream:
        for number, batch in enumerate(loader, 1):
            view, columns, circuits = sample_functional_pairs(batch, 'circuit_cap', sampler, 1024)
            if circuits != plan['batches'][number - 1]['circuits']:
                raise ValueError(f'{label} batch {number} differs from saved B graph/pair selection')
            fixed_columns.append({'batch': number, 'circuits': batch.name, 'source_batch_pair_columns': columns.tolist()})
            item = move(batch, args.device)
            loss_item = copy(item)
            for key in ('hf_pairs', 'hf_tt_target', 'hf_tt_mask'):
                loss_item[key] = view[key].to(args.device)
            with torch.enable_grad():
                output = model_outputs(model, item)
                weighted, details = weighted_task_losses(output, loss_item, plan['chunk_size'])
                grads, residual = task_gradients(weighted, parameters, verify_total=number == 1)
            for task in TASKS:
                for name, grad in zip(names, grads.get(task, [None] * len(names))):
                    dependencies[task][name]['autograd_connected_batches'] += int(grad is not None)
                    dependencies[task][name]['nonzero_gradient_batches'] += int(grad is not None and bool((grad != 0).any()))
            comparisons, norms = gradient_measurements(grads, names, groups)
            row = dict(batch=number, circuits=batch.name, loss_details=details, comparisons=comparisons,
                task_weighted_module_norms=norms, all_gradients_finite=True, B_pair_columns_match=True,
                weighted_sum_vs_direct_total_max_abs_residual=residual,
                positive_count=sum(r['sampled_strata']['zero'] for r in circuits),
                negative_count=sum(r['sampled_pairs'] - r['sampled_strata']['zero'] for r in circuits))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            # .grad is never populated by autograd.grad; no hidden optimizer step.
            if any(p.grad is not None for p in parameters):
                raise ValueError('Unexpected parameter .grad accumulation')
            if number == 1 or number % 4 == 0:
                c = comparisons['full_multitask_total']['all_common']
                info(f'{label} batch={number}/32 elapsed={time.monotonic()-started:.1f}s '
                     f'func_vs_total={c["cosine"]} finite=True no_update=True')
            del output, weighted, grads, item, loss_item, batch, view
    if len(rows) != 32:
        raise ValueError('Did not process exactly 32 batches')
    if not all(torch.equal(value.detach().cpu(), before[name]) for name, value in model.state_dict().items()):
        raise ValueError('Model parameters/buffers changed during read-only diagnostic')
    report = dict(checkpoint=label, checkpoint_epoch=state['epoch'], summary=aggregate(rows),
        parameter_groups=groups, parameter_dependencies=dependencies, model_state_unchanged=True,
        B_pair_columns_match=True, optimizer_steps=0, all_gradients_finite=True, batches=rows)
    write_json(directory / 'result.json', report)
    write_json(directory / 'selected_columns.json', fixed_columns)
    info(f'Completed {label}: 32/32 batches, model state unchanged, optimizer steps=0')
    del model, state
    return report, fixed_columns


def print_summary(report):
    info('优化器更新前诊断；不等同于 Adam 实际更新方向，不构成长期效果或最终因果结论')
    for label, checkpoint in report['checkpoints'].items():
        info(f'SUMMARY {label} (32 train batches)')
        for comparison in COMPARISONS:
            for scope in SCOPES:
                row = checkpoint['summary'][comparison][scope]
                if row['cosine_quantiles'] is None:
                    info(f'func vs {comparison} / {scope}: 不适用 ({row["not_applicable_reasons"]})')
                else:
                    info(f'func vs {comparison} / {scope}: n={row["applicable_batches"]}/32 '
                         f'cos_median={row["cosine_quantiles"]["0.5"]:.5f} '
                         f'cos_q05={row["cosine_quantiles"]["0.05"]:.5f} '
                         f'cos_q95={row["cosine_quantiles"]["0.95"]:.5f} '
                         f'negative_fraction={row["negative_cosine_batch_fraction"]:.3f} '
                         f'weighted_norm_ratio_median={row["weighted_norm_ratio_to_func_quantiles"]["0.5"]:.5g}')
    findings = report['findings']
    info('优先单独消融候选（两个检查点均满足描述性筛选标准）: ' + json.dumps(findings['priority_separate_ablations'], ensure_ascii=False))
    info('梯度量级差异: ' + json.dumps(findings['scale_differences'], ensure_ascii=False))
    info('不一致或不适用项: ' + json.dumps(findings['inconsistent_or_insufficient'], ensure_ascii=False))
    info(findings['causal_conclusion'])


def run(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    for name, expected in plan['code_sha256'].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f'Code changed since preparation: {name}')
    for label, path in plan['checkpoints'].items():
        if digest_file(Path(path)) != plan['checkpoint_sha256'][label]:
            raise ValueError('Checkpoint changed since preparation')
    if (args.output / 'run.json').exists():
        raise FileExistsError('Preserving previously started run')
    with direct_log(args.output / 'console.log'):
        write_json(args.output / 'run.json', {'status': 'running', 'optimizer_steps': 0})
        info(f'Gradient diagnostic only: 32 train batches per checkpoint, batch_size={plan["batch_size"]}, '
             f'circuits={plan["unique_circuits"]}, positive={plan["sampled_functional_positive_count"]}, '
             f'negative={plan["sampled_functional_negative_count"]}; no optimizer/val/test')
        set_seed(plan['seed'])
        dataset = CircuitDataset(plan['data'], cache_mode='none')
        if dataset.source_sha256 != plan['source_sha256']:
            raise ValueError('Data changed since original experiment')
        indices = {name: i for i, name in enumerate(dataset.ids)}
        checkpoints, first = {}, None
        for label in plan['checkpoints']:
            result, columns = run_checkpoint(args, plan, label, dataset, indices)
            if first is not None and first != columns:
                raise ValueError('Checkpoints used different pair columns')
            first = columns
            checkpoints[label] = result
        unchanged = all(digest_file(Path(p)) == plan['checkpoint_sha256'][label] for label, p in plan['checkpoints'].items())
        if not unchanged:
            raise ValueError('Checkpoint file changed during diagnostic')
        report = dict(metadata=plan, checkpoints=checkpoints,
            findings=prioritized_findings({k: v['summary'] for k, v in checkpoints.items()}),
            original_checkpoint_files_unchanged=True, both_checkpoints_same_graphs_and_pairs=True)
        write_json(args.output / 'comparison.json', report)
        print_summary(report)
        write_json(args.output / 'run.json', {'status': 'completed', 'optimizer_steps': 0,
            'batches_per_checkpoint': 32, 'original_checkpoint_files_unchanged': True,
            'both_checkpoints_same_graphs_and_pairs': True, 'val_test_evaluated': False})
        info(f'Completed task gradient diagnostic: {args.output / "comparison.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--b-run', type=Path, default=ROOT / 'functional_pair_sampling_stage3/circuit_cap')
    parser.add_argument('--e-checkpoint', type=Path, default=ROOT / 'functional_pair_sampling_stage3_E_functional_only_20261109/best_equiv.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not (args.prepare or args.run):
        parser.error('Explicit --prepare and/or --run required')
    if args.prepare:
        prepare(args)
    if args.run:
        run(args)


if __name__ == '__main__':
    main()
