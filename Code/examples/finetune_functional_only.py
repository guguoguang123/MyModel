"""E: replay B's circuit_cap sampling, backpropagate only its functional term.

--prepare [--self-check] never starts a formal experiment. --run is manual.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
import json
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import finetune_functional_pairs as base
from examples.evaluate import RUN_DIR, digest_file, validate_splits
from triwave_gnn.dataset import CircuitDataset
from triwave_gnn.model import TriWaveGNN
from triwave_gnn.trainer import move
from triwave_gnn.utils.batch import SCHEDULE_KEYS, collate_circuits
from triwave_gnn.utils.checkpoint import atomic_save, load as load_checkpoint, random_state
from triwave_gnn.utils.functional_only import functional_only_objective
from triwave_gnn.utils.logger import AverageMeter, BatchProgress, info
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.pair_sampling import sample_functional_pairs, sampling_summary
from triwave_gnn.utils.random_seed import set_seed
from triwave_gnn.utils.inference import load_model

DEFAULT_B = RUN_DIR.parent / 'functional_pair_sampling_stage3' / 'circuit_cap'
DIAGNOSTIC_NOTICE = '训练电路诊断，不能当作验证或测试成绩'


def summarize_existing_diagnostics(directory, output):
    """Read stored scalar metrics; never run a model to recover absent scores."""
    report = {'notice': DIAGNOSTIC_NOTICE, 'models': {}}
    for mode in ('D0', 'D1'):
        root = directory / mode
        history = json.loads((root / 'history.json').read_text())
        rows = []
        for result in history:
            full = result['original_full_pairs']
            rows.append({'step': result['step'], 'balanced_subset_threshold_result': {
                key: full[key] for key in ('similarity_threshold', 'precision', 'recall', 'f1',
                                          'positive_count', 'negative_count', 'tp', 'fp')}})
        # The prior script stores summaries and quantiles, not its score arrays.
        # last.pt contains weights + diagnostic_metrics, never model predictions.
        report['models'][mode] = {
            'stored_original_pair_metrics': rows,
            'score_status': 'missing: no per-pair scores/labels were persisted by diagnose_equivalence_fit.py',
            'full_pair_max_f1_threshold': {'status': 'unavailable_missing_scores',
                'reason': 'cannot reconstruct ranking from summary metrics or quantiles; no inference performed'},
            'max_recall_at_precision': {str(p): {'status': 'unavailable_missing_scores', 'recall': None}
                                       for p in (.1, .5, .9)},
            'no_retraining_or_inference': True,
        }
        final = rows[-1]['balanced_subset_threshold_result']
        info(f'{DIAGNOSTIC_NOTICE} | {mode} final original pairs at balanced threshold: '
             + ' | '.join(f'{k}={v}' for k, v in final.items()))
    info('D0/D1 逐对分数未保存：原始配对重新选最大 F1 阈值、Precision>=10%/50%/90% 的最大 Recall 无法计算；未重跑推理')
    base.write_json(output, report)


def prepare(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite E output: {args.output}')
    bplan = json.loads((args.b_run / 'plan.json').read_text())
    brun = json.loads((args.b_run / 'run.json').read_text())
    if bplan['mode'] != 'circuit_cap' or brun['status'] != 'completed':
        raise ValueError('E requires a completed original B circuit_cap experiment')
    if (bplan['lr'], bplan['epochs'], bplan['cap']) != (3e-5, 5, 1024):
        raise ValueError('B settings must be lr=3e-5, epochs=5, cap=1024')
    source_path = Path(bplan['initial_checkpoint'])
    if digest_file(source_path) != bplan['initial_checkpoint_sha256']:
        raise ValueError('Original stage3 checkpoint differs from B')
    source = load_checkpoint(source_path, 'cpu')
    if source.get('stage') != 3:
        raise ValueError('Require original stage3_best.pt')
    splits = json.loads((args.b_run / 'split.json').read_text())
    validate_splits(splits, source)
    orders = json.loads((args.b_run / 'epoch_orders.json').read_text())
    if orders != base.epoch_orders(splits['train'], bplan['seed'], 5):
        raise ValueError('B orders differ from its source seed/splits')
    code_paths = [Path(base.__file__), Path(__file__),
        Path(__file__).parents[1] / 'triwave_gnn/utils/functional_only.py',
        Path(__file__).parents[1] / 'triwave_gnn/utils/losses.py',
        Path(__file__).parents[1] / 'triwave_gnn/utils/pair_sampling.py',
        Path(__file__).parents[1] / 'triwave_gnn/model.py',
        Path(__file__).parents[1] / 'triwave_gnn/arch/initialization.py',
        Path(__file__).parents[1] / 'triwave_gnn/arch/propagation.py',
        Path(__file__).with_name('compare_functional_stages.py')]
    plan = dict(bplan, group='E', status='prepared', comparison_b_run=str(args.b_run.resolve()),
        loss='functional_only_objective: original normalized-distance L1 * ORIGINAL active functional coefficient',
        training_objective_name='training_functional_objective',
        validation_total_loss_name='validation_original_multitask_total_loss',
        backward_tasks=['func'], freeze_hf_path=False,
        coefficient={'usual_before': 2/8, 'usual_after': 2/8,
            'per_batch': '2/sum(original active stage3 task weights); 0 if original func term absent',
            'renormalize_after_disabling_tasks': False},
        code_sha256={str(p.resolve()): digest_file(p) for p in code_paths},
        b_run_json_sha256=digest_file(args.b_run / 'run.json'),
        b_sampling_sha256=digest_file(args.b_run / 'sampling.jsonl'),
        evaluation_chunk_size=brun.get('evaluation_chunk_size', 100000),
        compare_b_sampling_each_batch=True,
        baseline='evaluate original complete validation once before E; saved B baseline retained separately')
    args.output.mkdir(parents=True)
    for name in ('split.json', 'epoch_orders.json'):
        shutil.copyfile(args.b_run / name, args.output / name)
    shutil.copyfile(args.b_run / 'baseline.json', args.output / 'baseline_B_reference.json')
    base.write_json(args.output / 'plan.json', plan)
    summarize_existing_diagnostics(args.diagnostics, args.output / 'D0_D1_existing_diagnostic.json')
    info(f'Prepared E: {args.output}; functional coefficient before=0.25 after=0.25 (usual all-task batch)')
    info('Prepared only: no formal training started')


def expected_b_batches(path, epoch):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['epoch'] == epoch:
                yield row


def compare_sampling(rows, expected, epoch, batch):
    if expected is None or (expected['epoch'], expected['batch']) != (epoch, batch):
        raise ValueError('B sampling batch order differs')
    # Whole dictionaries include circuit name/order, class counts and exact
    # selected-column hashes. Equality is stronger than equal pair counts.
    if rows != expected['circuits']:
        raise ValueError(f'E sampled different original pair columns than B at epoch={epoch}, batch={batch}')


def train_epoch(model, optimizer, dataset, order, indices, args, epoch, run_dir):
    model.train()
    generator = torch.Generator().manual_seed(args.sampler_seed + epoch - 1)
    loader = DataLoader(Subset(dataset, [indices[name] for name in order]), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, exclude_keys=SCHEDULE_KEYS,
        generator=torch.Generator().manual_seed(args.seed + epoch))
    expected = iter(expected_b_batches(args.b_run / 'sampling.jsonl', epoch))
    all_rows, objective_meter, raw_meter = [], AverageMeter(), AverageMeter()
    coefficients, functional_batches = set(), 0
    with (run_dir / 'sampling.jsonl').open('a', buffering=1) as stream, \
         BatchProgress('train', 3, epoch, len(loader), args.progress_interval, epoch_total=args.epochs) as progress:
        for number, batch in enumerate(loader, 1):
            view, _, rows = sample_functional_pairs(batch, 'circuit_cap', generator, args.cap)
            compare_sampling(rows, next(expected, None), epoch, number)
            item = move(batch, args.device)
            loss_item = copy(item)
            for key in ('hf_pairs', 'hf_tt_target', 'hf_tt_mask'):
                loss_item[key] = view[key].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model_outputs(model, item)
            total, details = functional_only_objective(output, loss_item, args.chunk_size)
            if not model.training or not total.requires_grad or not bool(torch.isfinite(total)):
                raise RuntimeError('E must run finite differentiable loss in train mode')
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            count = len(batch.name)
            objective_meter.update(details['training_functional_objective'], count)
            raw_meter.update(details['training_raw_functional_loss'], count)
            coefficients.add(details['functional_coefficient_after'])
            functional_batches += int(details['functional_term_active'])
            record = dict(epoch=epoch, batch=number, lr=[g['lr'] for g in optimizer.param_groups],
                **details, gradient_norm=float(grad_norm), gradients_finite=True,
                B_pair_columns_match=True, sampling=sampling_summary(rows), circuits=rows)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            all_rows.extend(rows)
            if number == 1 or (args.progress_interval and number % args.progress_interval == 0):
                info(f'E epoch={epoch} batch={number}: functional coefficient before='
                     f'{details["functional_coefficient_before"]} after={details["functional_coefficient_after"]} '
                     f'| training_functional_objective={details["training_functional_objective"]:.9g}')
            progress.update(number, objective_meter.avg, {'func_raw': raw_meter.avg}, {})
            del total, output, item, loss_item, batch, view
    if next(expected, None) is not None:
        raise ValueError('E has fewer batches than B')
    return dict(training_functional_objective=objective_meter.avg,
        training_raw_functional_loss=raw_meter.avg, functional_coefficients=sorted(coefficients),
        functional_active_batches=functional_batches, sampling=sampling_summary(all_rows),
        optimizer_steps=len(loader), actual_lr=[g['lr'] for g in optimizer.param_groups],
        all_gradients_finite=True, B_pair_columns_match=True, sampler_state=generator.get_state())


def evaluate_validation(model, dataset, names, indices, args, epoch):
    # Exactly B's implementation, including its original multitask loss.
    result = base.evaluate_validation(model, dataset, names, indices, args, epoch)
    result['validation_original_multitask_total_loss'] = result.pop('mean_loss')
    return result


def print_validation(result, label, b_reference=None):
    # Existing formatter receives only a temporary compatibility copy.
    base.print_validation(dict(result, mean_loss=result['validation_original_multitask_total_loss']), label)
    info(f'{label} validation_original_multitask_total_loss={result["validation_original_multitask_total_loss"]}')
    if b_reference:
        delta = {key: result['metrics'][key] - value for key, value in b_reference['metrics'].items()
                 if value is not None and result['metrics'].get(key) is not None}
        info(f'{label} metric_delta_vs_B_same_epoch=' + json.dumps(delta))
        degraded = {key: (value < 0 if key == 'rc_accuracy' else value > 0) for key, value in delta.items()}
        info(f'{label} metric_degraded_vs_B_same_epoch=' + json.dumps(degraded))


def save_checkpoint(path, model, optimizer, source, args, epoch, validation, sampler_state=None):
    # Reuse B's checkpoint layout, adding an explicit objective label.
    state = {key: deepcopy(source[key]) for key in ('architecture_id', 'logic_vocab', 'role_vocab',
        'pin_vocab', 'source_sha256', 'split_ids', 'feature_normalization')}
    state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), stage=3, epoch=epoch,
        stage_complete=False, finetune_mode='E_circuit_cap_functional_only', finetune_epochs=args.epochs,
        initial_checkpoint=str(args.checkpoint), sampler_state=sampler_state, random_state=random_state(),
        config=dict(source['config'], lr=args.lr, output=str(args.output), resume=None),
        training_objective='functional contribution only, original active denominator retained',
        validation_original_multitask_total_loss=validation['validation_original_multitask_total_loss'],
        validation_f1=validation['functional_equivalence']['f1'],
        validation_threshold=validation['functional_equivalence']['similarity_threshold'])
    atomic_save(state, path)


def load_plan(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    if plan['group'] != 'E' or plan['status'] != 'prepared':
        raise ValueError('Not a prepared E experiment')
    for name, expected in plan['code_sha256'].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f'Code changed after E preparation: {name}; prepare a new directory')
    args.b_run = Path(plan['comparison_b_run'])
    args.data, args.checkpoint = Path(plan['data']), Path(plan['initial_checkpoint'])
    for key in ('lr', 'epochs', 'cap', 'batch_size', 'seed', 'num_workers', 'sampler_seed'):
        setattr(args, key, plan[key])
    if args.chunk_size != plan['evaluation_chunk_size']:
        raise ValueError('Use the same functional/evaluation chunk size as B')
    if digest_file(args.checkpoint) != plan['initial_checkpoint_sha256']:
        raise ValueError('Original checkpoint changed after preparation')
    if digest_file(args.b_run / 'sampling.jsonl') != plan['b_sampling_sha256']:
        raise ValueError('B sampling reference changed')
    splits = json.loads((args.output / 'split.json').read_text())
    orders = json.loads((args.output / 'epoch_orders.json').read_text())
    if splits != json.loads((args.b_run / 'split.json').read_text()):
        raise ValueError('E splits differ from B')
    if orders != json.loads((args.b_run / 'epoch_orders.json').read_text()):
        raise ValueError('E graph batch orders differ from B')
    return plan, splits, orders


def run(args):
    plan, splits, orders = load_plan(args)
    if (args.output / 'run.json').exists():
        raise FileExistsError(f'E already started, preserving results: {args.output}')
    with base.direct_log(args.output / 'console.log'):
        record = dict(plan, status='running', completed_epochs=0, device=args.device)
        base.write_json(args.output / 'run.json', record)
        info(f'E: circuit_cap=1024 | fresh original stage3 weights + optimizer | lr={args.lr} '
             f'| batch_size={args.batch_size} | seed={args.seed} | epochs=5')
        info('Original functional coefficient=2/(3+1+2+2)=0.25; E=0.25; per-batch original active denominator retained')
        set_seed(args.seed)
        dataset = CircuitDataset(args.data, cache_mode='small')
        model, source = load_model(args.checkpoint, dataset, args.device)
        if source['source_sha256'] != dataset.source_sha256:
            raise ValueError('Dataset files differ from B source')
        val_key = validate_splits(splits, source)
        indices = {name: i for i, name in enumerate(dataset.ids)}
        optimizer = base.make_optimizer(model, source, args.lr)
        if optimizer.state:
            raise RuntimeError('Optimizer must start without resumed state')
        baseline = evaluate_validation(model, dataset, splits[val_key], indices, args, 0)
        base.write_json(args.output / 'baseline.json', baseline)
        print_validation(baseline, 'E BASELINE', json.loads((args.output / 'baseline_B_reference.json').read_text()))
        best_loss = baseline['validation_original_multitask_total_loss']
        best_f1 = baseline['functional_equivalence']['f1']
        for name in ('best_loss.pt', 'best_equiv.pt'):
            save_checkpoint(args.output / name, model, optimizer, source, args, 0, baseline)
        history = []
        for epoch, order in enumerate(orders, 1):
            info(f'E epoch={epoch}/5 actual_lr={[g["lr"] for g in optimizer.param_groups]}')
            train = train_epoch(model, optimizer, dataset, order, indices, args, epoch, args.output)
            validation = evaluate_validation(model, dataset, splits[val_key], indices, args, epoch)
            base.write_json(args.output / f'val_epoch_{epoch}.json', validation)
            sampler_state = train.pop('sampler_state')
            val_loss = validation['validation_original_multitask_total_loss']
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(args.output / 'best_loss.pt', model, optimizer, source, args, epoch, validation, sampler_state)
            f1 = validation['functional_equivalence']['f1']
            if f1 is not None and (best_f1 is None or f1 > best_f1):
                best_f1 = f1
                save_checkpoint(args.output / 'best_equiv.pt', model, optimizer, source, args, epoch, validation, sampler_state)
            save_checkpoint(args.output / 'last.pt', model, optimizer, source, args, epoch, validation, sampler_state)
            history.append({'epoch': epoch, 'train': train,
                'val': {k: v for k, v in validation.items() if k != 'per_circuit'}})
            base.write_json(args.output / 'history.json', history)
            reference = json.loads((args.b_run / f'val_epoch_{epoch}.json').read_text())
            print_validation(validation, f'E VAL epoch={epoch}', reference)
            info(f'E epoch={epoch} training_functional_objective={train["training_functional_objective"]} '
                 f'| sampling={train["sampling"]}')
            if plan['lr_step'] and epoch % plan['lr_step'] == 0:
                for group in optimizer.param_groups:
                    group['lr'] *= .1
            record.update(completed_epochs=epoch)
            base.write_json(args.output / 'run.json', record)
        if digest_file(args.checkpoint) != plan['initial_checkpoint_sha256']:
            raise RuntimeError('Original checkpoint changed')
        record.update(status='completed', best_validation_original_multitask_total_loss=best_loss,
            best_f1=best_f1, B_pair_columns_match=True, test_evaluated=False)
        base.write_json(args.output / 'run.json', record)
        info('Completed E; best_loss.pt / best_equiv.pt / last.pt / history.json | test not evaluated')


def self_check(args):
    from triwave_gnn.tests.test_functional_only import real_weight_self_check
    plan, splits, orders = load_plan(args)
    source = load_checkpoint(args.checkpoint, 'cpu')
    report = real_weight_self_check(source, args.device)
    # One actual B batch, one disposable update. Never save its model/optimizer.
    info('E SELF-CHECK: checking B epoch-1 first real batch; disposable model only')
    set_seed(args.seed)
    dataset = CircuitDataset(args.data, cache_mode='none')
    model, _ = load_model(args.checkpoint, dataset, args.device)
    if dataset.source_sha256 != source['source_sha256']:
        raise ValueError('Data differ from original B')
    indices = {name: i for i, name in enumerate(dataset.ids)}
    group = orders[0][:args.batch_size]
    batch = collate_circuits([dataset[indices[name]] for name in group])
    view, _, rows = sample_functional_pairs(batch, 'circuit_cap',
        torch.Generator().manual_seed(args.sampler_seed), args.cap)
    compare_sampling(rows, next(expected_b_batches(args.b_run / 'sampling.jsonl', 1)), 1, 1)
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    optimizer = base.make_optimizer(model, source, args.lr)
    assert not optimizer.state
    item, selected = move(batch, args.device), move(view, args.device)
    model.train()
    output = model_outputs(model, item)
    output['hf'].retain_grad()
    objective, details = functional_only_objective(output, selected, args.chunk_size)
    objective.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    hf_norm = float(output['hf'].grad.norm())
    if not bool(torch.isfinite(output['hf'].grad).all()):
        raise ValueError('Nonfinite actual-batch Hf gradient')
    gradients = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
    optimizer.step()
    from triwave_gnn.tests.test_functional_only import branch
    changes = {}
    for name, p in model.named_parameters():
        if not torch.equal(p.detach().cpu(), before[name]):
            category = branch(name)
            changes[category] = changes.get(category, 0) + 1
    if not changes.get('structure') or not changes.get('function') or changes.get('unused_behavior_or_heads', 0):
        raise ValueError('Actual-batch update did not isolate the full Hf path')
    report['actual_B_first_batch'] = dict(circuits=group, B_pair_columns_match=True,
        sampling=sampling_summary(rows), objective_details=details,
        gradient_norm_before_clip=norm, hf_gradient_norm=hf_norm, gradients_finite=True,
        parameter_gradient_norms=gradients, changed_parameter_tensors_by_branch=changes,
        optimizer_steps=1, disposable_model_only=True)
    report['initial_checkpoint_unchanged'] = digest_file(args.checkpoint) == plan['initial_checkpoint_sha256']
    if not report['initial_checkpoint_unchanged']:
        raise RuntimeError('Original checkpoint changed during self-check')
    base.write_json(args.output / 'self_check.json', report)
    info('E SELF-CHECK passed: disposable optimizer update; Hs/Hf gradients finite; other heads untouched; full original validation reused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--b-run', type=Path, default=DEFAULT_B)
    parser.add_argument('--diagnostics', type=Path, default=RUN_DIR.parent / 'equivalence_fit_diagnostic_20261109')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--progress-interval', type=int, default=100)
    args = parser.parse_args()
    if not (args.prepare or args.self_check or args.run) or (args.run and (args.prepare or args.self_check)):
        parser.error('Use --prepare [--self-check], --self-check, or --run; no implicit training')
    if args.chunk_size < 1 or args.progress_interval < 0:
        parser.error('Require positive chunk size and nonnegative progress interval')
    if args.prepare:
        prepare(args)
    if args.self_check:
        self_check(args)
    if args.run:
        run(args)


if __name__ == '__main__':
    main()
