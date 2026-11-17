"""Manual cache-only continuation to a bounded circuit-exposure budget."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import expand_training_frozen_head as coverage
from examples import compare_circuit_balanced_head as control
from examples import continue_circuit_balanced_head as continuation
from examples import expanded_frozen_head_validation as extended
from examples import finetune_hard_negative_head as hard
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash, within_circuit_auc
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save, random_state
from triwave_gnn.utils.equivalence_audit import pair_metrics
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

SOURCE = original.EXP / 'frozen_head_train512_hard_negative_20261113'
MAX_STEP, WALL_SECONDS, EVAL_RESERVE, MONITOR_INTERVAL = 35000, 300, 20, 2000
MONITOR_SEED = 20261113
RESTORE_NOTICE = ('head and Adam restored; dedicated negative RNG restored; original seed42 circuit schedule '
                  'reconstructed and checked through step5000 using saved sampler position; source did NOT save '
                  'global Python/NumPy/torch CPU/CUDA RNG states, so this is NOT a full global-RNG exact resume. '
                  'MLP has no stochastic layers; mining RNG and reconstructed circuit RNG remain independent.')
NOTICE = coverage.NOTICE + '；本次仅增加固定规则训练预算；达到步数/时间上限不等于收敛'


def read(path):
    return json.loads(Path(path).read_text())


def schedule_after_source(samples, source, source_step):
    schedule = control.circuit_schedule(len(samples), 4, MAX_STEP, original.SEED)
    names = [[samples[i]['circuit'] for i in batch.tolist()] for batch in schedule]
    saved = read(source / 'circuit_schedule.json')
    updates = read(source / 'update_history.json')
    if (saved['start_total_step'] != 1 or saved['batches'] != names[:source_step]
            or [row['step'] for row in updates] != list(range(1, source_step+1))):
        raise ValueError('Saved schedule/completed-update prefix does not match reconstructed progress')
    visits = sum(map(len, saved['batches']))
    target = math.ceil(len(samples) * 312.5)
    return schedule, names, visits, target


def check_saved_samples(cache, sample):
    columns = sample['cache_pair_columns']
    labels = cache['labels'][columns]
    if (not torch.equal(cache['original_pair_columns'][columns], sample['original_pair_columns'])
            or int(labels.sum()) != sample['positive_count']
            or int((~labels).sum()) != sample['negative_count']
            or max(sample['positive_count'], sample['negative_count']) > control.CAP):
        raise ValueError(f'Existing training pair identity/budget changed: {sample["circuit"]}')


def build_monitor(samples, lookup, plan, output):
    """Independent fixed class samples; never fed into training or mining."""
    rng = torch.Generator().manual_seed(MONITOR_SEED)
    monitor, records = [], []
    for sample in samples:
        cache = coverage.load_train_cache(lookup[sample['circuit']], plan)
        check_saved_samples(cache, sample)
        row = control.sample_circuit(cache, rng)
        monitor.append({**row, 'features': row['features'].to(plan['device']),
                        'labels': row['labels'].to(plan['device'])})
        records.append({key: value for key, value in row.items() if key not in ('features', 'labels')})
    atomic_save({'seed': MONITOR_SEED, 'cap_per_class': control.CAP, 'circuits': records,
                 'objective': control.OBJECTIVE, 'used_for_training_sampling': False,
                 'monitor_rng_state': rng.get_state()}, output / 'monitor_pairs.pt')
    return monitor


@torch.no_grad()
def monitor_fit(head, rows, step, visits):
    head.eval()
    losses, scores, labels, aucs = [], [], [], []
    for row in rows:
        logits = head(row['features']).squeeze(-1)
        continuation.finite(logits, '固定监测 logits')
        loss = control.circuit_loss(logits, row['labels'])
        continuation.finite(loss, '固定完整训练目标')
        losses.append(float(loss))
        scores.append(logits.cpu())
        labels.append(row['labels'].cpu())
        aucs.append({'circuit': row['circuit'], **pair_metrics(scores[-1], labels[-1], None)})
    pooled = pair_metrics(torch.cat(scores), torch.cat(labels), None)
    result = {'notice': '固定训练监测配对上的拟合指标，不是泛化成绩；不用于选择检查点或改变训练',
              'total_step': step, 'cumulative_circuit_visits': visits,
              'circuit_balanced_objective': math.fsum(losses) / len(rows),
              'pooled_auc': pooled['roc_auc'], **within_circuit_auc(aucs),
              'active_circuits': len(rows), 'positive_count': pooled['positive_count'],
              'negative_count': pooled['negative_count'], 'single_class_in_objective': True}
    info(f"FIXED TRAIN MONITOR step={step} visits={visits}: objective={result['circuit_balanced_objective']:.8f} | "
         f"pooled/macro AUC={result['pooled_auc']}/{result['circuit_macro_roc_auc']} | "
         f"macro circuits={result['macro_participating_circuits']}; observation only")
    return result


def checkpoint(head, optimizer, step, visits, rng, sampling_file, record):
    state = continuation.snapshot(head, optimizer, step, record)
    state.update(cumulative_circuit_visits=visits, negative_rng_state=rng.get_state().clone(),
                 sampling_file=sampling_file, fixed_samples_file=record['fixed_samples_file'],
                 global_random_state=random_state(), circuit_schedule_seed=42,
                 circuit_schedule_rule='original uniform permutation sweeps, batches4 with remainder2',
                 restore_limitations=RESTORE_NOTICE)
    return state


def budget_fields(step, visits, target, active, reason):
    return {'completed_total_step': step, 'actual_cumulative_circuit_visits': visits,
            'average_circuit_visits': visits / active, 'target_circuit_visits': target,
            'target_average_circuit_visits': 312.5, 'exposure_target_reached': visits >= target,
            'stop_reason': reason, 'convergence_established': False}


@torch.inference_mode()
def evaluate_final(args, plan, head, record):
    """Final finite weights only; calibrate dev32 once, then fixed eval128 once."""
    head.eval()
    dev_rows = [r for r in read(extended.DEV / 'cache_manifest.json') if r['split'] == 'val']
    saved = torch.load(extended.DEV / 'validation_scores.pt', map_location='cpu', weights_only=False)
    if [r['circuit'] for r in dev_rows] != saved['circuits']:
        raise ValueError('Original32 validation circuit identity/order changed')
    record['dev_calibration_passes_started'] = 1
    write_json(args.output / 'run.json', record)
    pieces = []
    for i, row in enumerate(dev_rows):
        cache = torch.load(row['cache_file'], map_location='cpu', weights_only=False)
        extended.check_cache(cache, row['circuit'], plan['cache_contract'], False)
        a, b = saved['offsets'][i:i+2]
        if not torch.equal(cache['labels'], saved['labels'][a:b]):
            raise ValueError('Dev32 cached labels/order changed')
        pieces.append(coverage.score_one_cache(head, cache, args.device))
    scores = torch.cat(pieces)
    calibration = control.score_summary(scores, saved['labels'], saved['offsets'], saved['circuits'], 'logit')
    thresholds = {'main_threshold': calibration['maximum_f1']['threshold'],
                  'old_low_fp_threshold': calibration['fp_le_69']['threshold'], 'score_kind': 'logit',
                  'source': 'final finite continuation checkpoint, original32 development only',
                  'low_fp_rule': 'dev32 FP<=69 maxTP, then minFP, then highest threshold; whole equal-score groups'}
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'development32.json', {'notice': NOTICE, 'calibration': calibration})
    atomic_save({'head_logits': scores, 'labels': saved['labels'], 'offsets': saved['offsets'],
                 'circuits': saved['circuits']}, args.output / 'development32_scores.pt')
    record.update(dev_calibration_passes_completed=1, expanded_validation_passes_started=1)
    write_json(args.output / 'run.json', record)
    ext_plan = read(coverage.EXTENDED / 'plan.json')
    baseline = read(args.source / 'comparison.json')
    if (ext_plan['cache_contract'] != plan['cache_contract']
            or not baseline['complete_128_circuit_result']
            or baseline['selected_ids'] != ext_plan['selected_ids']):
        raise ValueError('Require same complete128 baseline and compatible caches; no encoder fallback')
    lookup = {r['circuit']: r for r in read(coverage.EXTENDED / 'cache_manifest.json')}
    score_dir = args.output / 'expanded_scores'
    score_dir.mkdir()
    pieces, labels, offsets = [], [], [0]
    for name in ext_plan['selected_ids']:
        cache = torch.load(lookup[name]['cache_file'], map_location='cpu', weights_only=False)
        extended.check_cache(cache, name, plan['cache_contract'], True)
        scores = coverage.score_one_cache(head, cache, args.device)
        atomic_save({'circuit': name, 'head_logits': scores, 'labels': cache['labels'],
                     'original_pair_columns': cache['original_pair_columns']},
                    score_dir / f'{extended.cache_key(name)}.pt')
        pieces.append(scores)
        labels.append(cache['labels'])
        offsets.append(offsets[-1] + scores.numel())
    methods = {key: baseline['methods'][key] for key in
               ('hard_negative_head_step5000', 'hard_negative_head_train512_step5000')}
    key = f'hard_negative_head_train512_final_step{record["completed_total_step"]}'
    methods[key] = extended.summarize(torch.cat(pieces), torch.cat(labels), offsets,
                                     ext_plan['selected_ids'], thresholds)
    report = {'notice': NOTICE, 'complete_128_circuit_result': True, 'test_evaluated': False,
              'expanded128_used_for_development_decisions': True,
              'threshold_selection': 'final head: original32 calibration, fixed on128; baselines reuse existing results',
              'restore_limitations': RESTORE_NOTICE, 'methods': methods,
              'low_fp_note': 'FP<=69 only on original32 calibration; actual FP/FPR/Recall reported on128',
              **{k: record[k] for k in budget_fields(record['completed_total_step'],
                 record['actual_cumulative_circuit_visits'], record['target_circuit_visits'], 426,
                 record['stop_reason'])}}
    write_json(args.output / 'comparison.json', report)
    for name, result in methods.items():
        extended.print_result(name, result)
        for tag, row in [('pooled', result), *result['groups_by_original_valid_pairs'].items()]:
            low = row['at_old_low_fp_threshold']
            info(f"{name} / {tag}: fixed dev32 low-FP threshold={low['threshold']} | "
                 f"P/R/F1={low['precision']}/{low['recall']}/{low['f1']} | "
                 f"TP/FP/FN={low['tp']}/{low['fp']}/{low['fn']}")
    record.update(status='completed', expanded_validation_passes_completed=1,
                  complete_128_circuit_result=True)
    write_json(args.output / 'run.json', record)


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {args.output}')
    source = args.source.resolve()
    args.source = source
    prior = read(source / 'run.json')
    args.cache_output = Path(prior['source_run'])
    plan = coverage.verify_plan(args)
    state = torch.load(source / 'last.pt', map_location='cpu', weights_only=False)
    source_step = state['optimizer_steps']
    if (prior['status'] != 'completed' or source_step != 5000 or state['sampler_position'] != source_step
            or state['objective'] != control.OBJECTIVE or prior['lr'] != 1e-3
            or prior['circuits_per_step'] != 4 or prior['active_train_circuits'] != 426):
        raise ValueError('Require unchanged completed512 step5000 source and training rules')
    if not state.get('optimizer') or state.get('negative_rng_state') is None:
        raise ValueError('Missing Adam/mining RNG; stopping instead of substituting an alternative')
    sampling = torch.load(state['sampling_file'], map_location='cpu', weights_only=False)
    if sampling['refresh_total_step'] != 4500 or not torch.equal(sampling['negative_rng_after'], state['negative_rng_state']):
        raise ValueError('Source sampling progress and RNG differ')
    data = torch.load(state['fixed_samples_file'], map_location='cpu', weights_only=False)
    samples = data['circuits']
    manifest = read(args.cache_output / 'cache_manifest.json')
    lookup = {r['circuit']: r for r in manifest}
    expected = [r['circuit'] for r in manifest if r['original_valid_pairs']]
    if len(samples) != 426 or [r['circuit'] for r in samples] != expected:
        raise ValueError('Original active426 training sample order changed')
    if [r['circuit'] for r in sampling['circuits']] != expected:
        raise ValueError('Source hard-negative sample order changed')
    schedule, names, visits, target = schedule_after_source(samples, source, source_step)
    if visits != 19908:
        raise ValueError('Original cumulative visits differ from19908')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sampling').mkdir()
    group_counts = coverage.sample_counts(samples)
    classes = Counter(coverage.class_kind(r['positive_count'], r['negative_count']) for r in samples)
    pos_coefficient = (.5*classes['both_class'] + classes['positive_only']) / len(samples)
    record = {'status': 'preparing', 'notice': NOTICE, 'hidden': state['hidden'],
              'source_run': str(source), 'fixed_samples_file': state['fixed_samples_file'],
              'optimizer_restore': RESTORE_NOTICE, 'restore_limitations': RESTORE_NOTICE,
              'source_checkpoint_sha256': file_hash(source / 'last.pt'), 'entry_sha256': file_hash(Path(__file__)),
              'global_rng_from_source_available': False, 'dedicated_negative_rng_restored': True,
              'circuit_progress_reconstructed_and_prefix_checked': True,
              'encoder_loaded': False, 'encoder_forward_count': 0, 'test_evaluated': False,
              'lr': 1e-3, 'objective': control.OBJECTIVE, 'group_sample_counts': group_counts,
              'active_train_circuits': len(samples), 'source_total_step': source_step,
              'source_cumulative_circuit_visits': visits,
              'population_class_coefficients': {'positive': pos_coefficient, 'negative': 1-pos_coefficient},
              'class_coefficients_notice': 'loss coefficients, not gradient contribution; no global50:50 reweighting',
              'monitor_seed': MONITOR_SEED, 'monitor_interval_additional_steps': MONITOR_INTERVAL,
              'monitor_used_for_selection': False, 'hard_negative_refresh_interval': hard.REFRESH_INTERVAL,
              'maximum_total_step': MAX_STEP, 'wall_seconds': WALL_SECONDS,
              'final_evaluation_reserve_seconds': EVAL_RESERVE,
              'time_rule': 'single300s hard timer covers restore, training, monitoring and final evaluation; stop updates at280s to reserve20s for saving/evaluation; hard timeout marks evaluation incomplete',
              'complete_128_circuit_result': False,
              **budget_fields(source_step, visits, target, len(samples), None)}
    history, updates = [], []
    safe = None
    head = None
    write_json(args.output / 'run.json', record)
    atomic_save(state, args.output / 'source_step5000.pt')
    atomic_save(data, args.output / 'training_samples.pt')
    write_json(args.output / 'circuit_schedule.json', {'start_total_step': source_step+1,
               'batches': names[source_step:], 'source_cumulative_circuit_visits': visits})
    with direct_log(args.output / 'console.log'):
        info(RESTORE_NOTICE)
        info(f'Fixed coefficients positive/negative={pos_coefficient}/{1-pos_coefficient}; target visits={target}; source visits={visits}')
        try:
            with continuation.wall_limit(started + WALL_SECONDS):
                set_seed(original.SEED)  # original global states absent; these global draws do not drive mining/schedule
                head = original.make_head(state['hidden']).to(args.device)
                head.load_state_dict(state['head'], strict=True)
                optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
                optimizer.load_state_dict(state['optimizer'])
                if any(g['lr'] != 1e-3 for g in optimizer.param_groups):
                    raise ValueError('Saved Adam learning rate differs')
                rng = torch.Generator().set_state(state['negative_rng_state'])
                sampling_file = state['sampling_file']
                safe = checkpoint(head, optimizer, source_step, visits, rng, sampling_file, record)
                atomic_save(safe, args.output / 'last.pt')
                monitor = build_monitor(samples, lookup, plan, args.output)

                def inspect(step):
                    row = monitor_fit(head, monitor, step, visits)
                    history.append(row)
                    write_json(args.output / 'training_monitor.json', history)

                inspect(source_step)
                training_deadline = started + WALL_SECONDS - EVAL_RESERVE
                reason = 'step_limit'
                try:
                    for step in range(source_step+1, MAX_STEP+1):
                        hard.training_time_check(training_deadline)
                        if (step-1) % hard.REFRESH_INTERVAL == 0:
                            active, sampling_file = coverage.refresh(head, samples, lookup, plan, rng,
                                                                     step-1, training_deadline, args.output)
                            info(f'Refreshed train-only hard negatives at total_step={step-1}')
                        head.train()
                        optimizer.zero_grad(set_to_none=True)
                        local = []
                        batch = schedule[step-1].tolist()
                        for index in batch:
                            row = active[index]
                            logits = head(row['features']).squeeze(-1)
                            continuation.finite(logits, '训练 logits')
                            local.append(control.circuit_loss(logits, row['labels']))
                        loss = torch.stack(local).mean()
                        continuation.finite(loss, '训练损失')
                        loss.backward()
                        for p in head.parameters():
                            if p.grad is not None:
                                continuation.finite(p.grad, '分类头梯度')
                        optimizer.step()
                        for p in head.parameters():
                            continuation.finite(p, '分类头参数')
                        candidate = checkpoint(head, optimizer, step, visits+len(batch), rng, sampling_file, record)
                        safe = candidate
                        visits = candidate['cumulative_circuit_visits']
                        updates.append({'total_step': step, 'circuits': names[step-1],
                                        'cumulative_circuit_visits': visits,
                                        'changing_sample_minibatch_bce': float(loss.detach())})
                        if step % 500 == 0:
                            atomic_save(safe, args.output / 'last.pt')
                            info(f'Head step={step}/{MAX_STEP}; visits={visits}/{target}; average={visits/426:.4f}')
                        if (step-source_step) % MONITOR_INTERVAL == 0:
                            inspect(step)
                        if visits >= target:
                            reason = 'exposure_target'
                            break
                except hard.TrainingTimeLimit:
                    reason = 'time_limit'
                # Always use the final complete finite snapshot, never best-by-monitor or validation.
                head.load_state_dict(safe['head'], strict=True)
                visits = safe['cumulative_circuit_visits']
                record.update(**budget_fields(safe['optimizer_steps'], visits, target, len(samples), reason))
                atomic_save(safe, args.output / 'last.pt')
                atomic_save(safe, args.output / 'final_head.pt')
                write_json(args.output / 'update_history.json', updates)
                record['training_elapsed_wall_seconds'] = time.monotonic()-started
                if (not history or history[-1]['total_step'] != safe['optimizer_steps']) and time.monotonic() < training_deadline:
                    inspect(safe['optimizer_steps'])
                record['status'] = 'final_evaluation'
                write_json(args.output / 'run.json', record)
                evaluate_final(args, plan, head, record)
        except (continuation.WallLimit, continuation.NonFinite) as error:
            record.update(status='incomplete_timeout' if isinstance(error, continuation.WallLimit) else 'stopped_nonfinite',
                          stop_detail=str(error), complete_128_circuit_result=False)
            if isinstance(error, continuation.NonFinite):
                record['stop_reason'] = 'nonfinite'
            elif record.get('stop_reason') is None:
                record['stop_reason'] = 'time_limit'
            info('停止并保存最后完整有限更新；未完成评价不能作为完整128电路结果。')
        finally:
            if safe is not None:
                atomic_save(safe, args.output / 'last.pt')
                atomic_save(safe, args.output / 'final_head.pt')
                record.update(**budget_fields(safe['optimizer_steps'], safe['cumulative_circuit_visits'],
                                               target, len(samples), record.get('stop_reason')))
                record.update(additional_optimizer_steps=safe['optimizer_steps']-source_step,
                              additional_circuit_visits=safe['cumulative_circuit_visits']-
                              record['source_cumulative_circuit_visits'])
            write_json(args.output / 'update_history.json', updates)
            write_json(args.output / 'training_monitor.json', history)
            record['elapsed_wall_seconds'] = time.monotonic()-started
            write_json(args.output / 'run.json', record)
            info(f"status={record['status']} | stop={record['stop_reason']} | step={record['completed_total_step']} | "
                 f"visits={record['actual_cumulative_circuit_visits']}/{target} | target_reached={record['exposure_target_reached']} | "
                 f"elapsed={record['elapsed_wall_seconds']:.1f}s")
            info(NOTICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
