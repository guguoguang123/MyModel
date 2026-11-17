"""Manual G0/G1 train-circuit sampling contrast; cache-only, full-val only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import compare_circuit_balanced_head as control
from examples import continue_circuit_balanced_head as continuation
from examples import continue_train512_budget as budget
from examples import expand_training_frozen_head as coverage
from examples import expanded_frozen_head_validation as reuse
from examples import final_frozen_head_evaluation as final
from examples import finetune_hard_negative_head as hard
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save, random_state, restore_random_state
from triwave_gnn.utils.functional_equivalence import grouped_curve, select_threshold
from triwave_gnn.utils.equivalence_audit import pair_count_group_metrics
from triwave_gnn.utils.logger import info

SOURCE = original.EXP / 'frozen_head_train512_budget_continue_20261113'
FINAL_VAL = original.EXP / 'final_B_head33438_20261113'
START, ADDITIONAL, WALL, VAL_RESERVE = 33438, 10000, 300, 30
NOTICE = ('G0/G1仅改变训练电路抽取比例；密集指原始有效功能配对数>=10000，不称大规模电路；'
          '完整val开发验证，不是新测试成绩；不读取test结果，不运行图编码器、不训练原任务头；'
          '不自动扩大训练或替换正式版本')


def read(path):
    return final.read(path)


def load(path):
    return final.load(path)


def grouped_train(samples, manifest):
    lookup = {r['circuit']: r for r in manifest}
    groups = {'dense_pairs': [], 'other_pairs': []}
    counts = {g: {'circuits': 0, 'both_class': 0, 'positive_only': 0, 'negative_only': 0, 'single_class': 0}
              for g in groups}
    for i, sample in enumerate(samples):
        meta = lookup[sample['circuit']]
        if meta['split'] != 'train' or not meta['original_valid_pairs']:
            raise ValueError('Training sample is outside active original train circuits')
        group = 'dense_pairs' if meta['original_valid_pairs'] >= 10000 else 'other_pairs'
        kind = coverage.class_kind(meta['positive_count'], meta['negative_count'])
        groups[group].append(i)
        counts[group]['circuits'] += 1
        counts[group][kind] += 1
        counts[group]['single_class'] += kind != 'both_class'
    if len(groups['dense_pairs']) < 1 or len(groups['other_pairs']) < 3 or len(samples) < 4:
        raise ValueError(f'Cannot construct1 dense+3 other distinct circuits; no ratio changes: {counts}')
    return {g: torch.tensor(indices, dtype=torch.long) for g, indices in groups.items()}, counts


def draw_batch(branch, count, groups, generator):
    if branch == 'G0':
        return torch.randperm(count, generator=generator)[:4]
    dense, other = groups['dense_pairs'], groups['other_pairs']
    return torch.cat((dense[torch.randperm(len(dense), generator=generator)[:1]],
                      other[torch.randperm(len(other), generator=generator)[:3]]))


def minimum_fp_at_recall(scores, labels, required_tp, offsets=None):
    """Whole equal-score groups; min FP, then highest threshold; no split ties."""
    thresholds, tp, fp = grouped_curve(scores, labels)
    allowed = torch.nonzero(tp >= required_tp).flatten()
    if not allowed.numel():
        raise ValueError('Reference recall unattainable; no altered target or tie splitting')
    allowed = allowed[fp[allowed] == fp[allowed].min()]
    threshold = float(thresholds[allowed[0]])
    result = reuse.fixed_metrics(scores, labels, threshold)
    if offsets is not None:
        result['groups_by_original_valid_pairs'] = pair_count_group_metrics(scores, labels, offsets, threshold)
    return result


def preflight(args):
    state = load(SOURCE / 'final_head.pt')
    config = read(FINAL_VAL / 'frozen_config.json')  # saved configuration only, not test results
    calibration = read(FINAL_VAL / 'calibration_val1663.json')
    if (state['optimizer_steps'] != START or not state.get('optimizer')
            or state.get('negative_rng_state') is None or state.get('global_random_state') is None
            or state['objective'] != control.OBJECTIVE
            or file_hash(SOURCE / 'final_head.pt') != config['head_sha256']
            or config['head_actual_step'] != START or config['functional_net_identity'] is not False
            or args.device != config['device'] or not calibration['complete_val_calibration']):
        raise ValueError('Require exact33438 frozen base, Adam/RNG states, and complete full-val baseline')
    if file_hash(Path(config['encoder_checkpoint'])) != config['encoder_sha256']:
        raise ValueError('Frozen B checkpoint changed')
    if calibration['frozen_config_sha256'] != file_hash(FINAL_VAL / 'frozen_config.json'):
        raise ValueError('Baseline full-val config changed')
    train_plan = read(original.EXP / 'frozen_head_train512_cache_20261113/plan.json')
    manifest = read(original.EXP / 'frozen_head_train512_cache_20261113/cache_manifest.json')
    samples_file = load(state['fixed_samples_file'])
    samples = samples_file['circuits']
    splits = read(original.B_RUN / 'split.json')
    active_names = [r['circuit'] for r in manifest if r['original_valid_pairs']]
    if (len(manifest) != 512 or len(samples) != 426 or [r['circuit'] for r in samples] != active_names
            or not {r['circuit'] for r in manifest} <= set(splits['train'])
            or set(active_names) & set(config['val_ids'])
            or train_plan['cache_contract'] != config['cache_contract']):
        raise ValueError('Original512/active426 training identity or encoder contract changed')
    groups, counts = grouped_train(samples, manifest)
    # Check only necessary saved cache paths/provenance. No raw dataset scan or encoder fallback.
    val_plan = {'selected_ids': config['val_ids'], 'cache_contract': config['cache_contract'],
                'external_cache_manifests': [str(reuse.DEV / 'cache_manifest.json'),
                   str(coverage.EXTENDED / 'cache_manifest.json'), str(final.REMAINING / 'cache_manifest.json')]}
    val_cache = reuse.external_cache_index(val_plan)
    missing_val = sorted(set(config['val_ids']) - set(val_cache))
    missing_train = [r['cache_file'] for r in manifest if r['original_valid_pairs'] and not Path(r['cache_file']).is_file()]
    if missing_val or missing_train:
        raise FileNotFoundError(f'必要缓存缺失；停止，不补跑图编码器。val IDs={missing_val}; train cache files={missing_train}')
    sampling = load(state['sampling_file'])
    if (sampling['refresh_total_step'] != 33000
            or not torch.equal(sampling['negative_rng_after'], state['negative_rng_state'])
            or [r['circuit'] for r in sampling['circuits']] != active_names):
        raise ValueError('Source hard-negative sampling progress/RNG differs')
    baseline = load(FINAL_VAL / 'val_scores.pt')
    if (baseline['circuits'] != config['val_ids'] or len(set(baseline['circuits'])) != 1663
            or file_hash(FINAL_VAL / 'val_scores.pt') != calibration['val_scores_sha256']):
        raise ValueError('Full-val baseline scores identity/hash changed')
    ref = calibration['methods'][final.HEAD]['at_fixed_val_threshold']
    if ref['positive_count'] != int(baseline['labels'].sum()):
        raise ValueError('Baseline complete-val class count differs')
    reference = {'recall': ref['recall'], 'required_tp': ref['tp'], 'positive_count': ref['positive_count'],
                 'baseline_max_f1_threshold': ref['threshold'],
                 'rule': 'minimum FP with TP>=baseline full-val maxF1 TP; whole equal-score groups; ties choose higher threshold'}
    recipe = {'source_checkpoint': str((SOURCE / 'final_head.pt').resolve()),
              'source_checkpoint_sha256': file_hash(SOURCE / 'final_head.pt'),
              'encoder_checkpoint': config['encoder_checkpoint'], 'encoder_sha256': config['encoder_sha256'],
              'source_step': START, 'planned_additional_steps': ADDITIONAL, 'lr': .001,
              'objective': control.OBJECTIVE, 'circuits_per_step': 4, 'cap_per_class': 128,
              'hard_refresh_interval': 500, 'refresh_phase': 'original global steps; next33500',
              'mining_rng': 'independent CPU Generator per process; explicit seed-derived state; no sharing with circuit RNG',
              'circuit_rng_seed': args.seed,
              'negative_rng_seed': hard.MINING_SEED + args.seed * 1_000_003,
              'circuit_rng': 'independent CPU Generator per process; explicit seed; no sharing with mining RNG',
              'negative_rule': 'unchanged ceil(half) top logits + uniform without replacement from remaining known negatives',
              'wall_seconds': WALL, 'validation_reserve_seconds': VAL_RESERVE,
              'source_training_samples_sha256': file_hash(Path(state['fixed_samples_file'])),
              'baseline_calibration_sha256': file_hash(FINAL_VAL / 'calibration_val1663.json')}
    return state, config, calibration, train_plan, manifest, samples_file, sampling, groups, counts, val_cache, baseline, reference, recipe


def restore_active(samples, sampling, lookup, plan):
    active = []
    for sample, saved in zip(samples, sampling['circuits']):
        cache = coverage.load_train_cache(lookup[sample['circuit']], plan)
        budget.check_saved_samples(cache, sample)
        columns = sample['cache_pair_columns']
        positives = columns[cache['labels'][columns]]
        negatives = saved['negative_cache_columns']
        if (not torch.equal(saved['positive_cache_columns'], positives)
                or negatives.numel() != sample['negative_count']
                or negatives.unique().numel() != negatives.numel()
                or bool(cache['labels'][negatives].any())
                or not torch.equal(cache['original_pair_columns'][negatives], saved['negative_original_columns'])):
            raise ValueError('Source active class samples changed')
        chosen = torch.cat((positives, negatives))
        active.append({'circuit': sample['circuit'], 'features': hard.features_at(cache, chosen).to(plan['device']),
                       'labels': cache['labels'][chosen].to(plan['device'])})
    return active


def snapshot(head, optimizer, step, circuit_rng, negative_rng, sampling_file, record):
    state = continuation.snapshot(head, optimizer, step, record)
    state.update(global_random_state=random_state(), negative_rng_state=negative_rng.get_state().clone(),
                 circuit_rng_state=circuit_rng.get_state().clone(), sampling_file=sampling_file,
                 fixed_samples_file=record['fixed_samples_file'], additional_optimizer_steps=step-START,
                 branch=record['branch'], branch_circuit_visits=(step-START)*4)
    return state


def print_result(name, result, constraint):
    final.print_equivalence('VAL development only', name, result)
    m = constraint
    info(f"{name}: baseline-recall constraint minimum FP={m['fp']} | TP/FN={m['tp']}/{m['fn']} | "
         f"actual Recall={m['recall']} | threshold={m['threshold']} | P/F1={m['precision']}/{m['f1']}")
    for group, row in m.get('groups_by_original_valid_pairs', {}).items():
        info(f"{name} at recall-constraint threshold / {group}: P/R/F1={row['precision']}/{row['recall']}/{row['f1']} | "
             f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}")


@torch.inference_mode()
def evaluate(args, head, config, baseline, calibration, val_cache, reference, record):
    head.eval()
    score_dir = args.output / 'val_score_cache'
    score_dir.mkdir()
    pieces = []
    for name, a, b in zip(baseline['circuits'], baseline['offsets'], baseline['offsets'][1:]):
        cache = load(val_cache[name])
        reuse.check_cache(cache, name, config['cache_contract'], 'cache_contract' in cache)
        if not torch.equal(cache['labels'], baseline['labels'][a:b]):
            raise ValueError(f'Full-val cache label order differs: {name}; no encoder fallback')
        scores = coverage.score_one_cache(head, cache, args.device)
        atomic_save({'circuit': name, 'logits': scores, 'labels': cache['labels'],
                     'original_pair_columns': cache['original_pair_columns']},
                    score_dir / f'{reuse.cache_key(name)}.pt')
        pieces.append(scores)
    scores = torch.cat(pieces)
    threshold = select_threshold(scores, baseline['labels'])
    result = final.equivalence_summary(scores, baseline['labels'], baseline['offsets'], baseline['circuits'],
                                       threshold, 'val', 'logit')
    constraint = minimum_fp_at_recall(scores, baseline['labels'], reference['required_tp'], baseline['offsets'])
    baseline_result = calibration['methods'][final.HEAD]
    baseline_constraint = minimum_fp_at_recall(baseline['scores'][final.HEAD], baseline['labels'], reference['required_tp'], baseline['offsets'])
    methods = {'original_step33438': {'maximum_f1': baseline_result, 'minimum_fp_at_reference_recall': baseline_constraint},
               args.branch: {'maximum_f1': result, 'minimum_fp_at_reference_recall': constraint}}
    thresholds = {name: {'max_f1_threshold': row['maximum_f1']['at_fixed_val_threshold']['threshold'],
                         'recall_constraint_threshold': row['minimum_fp_at_reference_recall']['threshold']}
                  for name, row in methods.items()}
    write_json(args.output / 'thresholds.json', thresholds)
    report = {'notice': NOTICE, 'complete_val_evaluation': True, 'test_evaluated': False,
              'branch': args.branch, 'completed_additional_steps': record['completed_additional_steps'],
              'completed_total_step': record['completed_total_step'], 'stop_reason': record['stop_reason'],
              'common_recipe_sha256': record['common_recipe_sha256'], 'reference_recall': reference,
              'methods': methods, 'selection': 'final finite checkpoint only; all thresholds selected on full val only'}
    for name, row in methods.items():
        info(f'VAL MODEL: {name}')
        print_result(final.HEAD, row['maximum_f1'], row['minimum_fp_at_reference_recall'])
    write_json(args.output / 'comparison.json', report)
    record.update(status='completed', complete_val_evaluation=True)
    write_json(args.output / 'run.json', record)
    return report


def combine_peer(args, current):
    if args.peer_output is None or not (args.peer_output / 'comparison.json').is_file():
        info('另一分支完整val报告尚未保存；当前不作G0/G1对照。')
        return
    other = read(args.peer_output / 'comparison.json')
    if (not other['complete_val_evaluation'] or other['branch'] == args.branch
            or other['common_recipe_sha256'] != current['common_recipe_sha256']
            or other['reference_recall'] != current['reference_recall']):
        raise ValueError('Peer branch does not share the frozen source/common recipe/reference recall')
    pair = {r['branch']: r for r in (current, other)}
    if set(pair) != {'G0', 'G1'}:
        raise ValueError('Expected separate G0 and G1 branches')
    equal_steps = pair['G0']['completed_additional_steps'] == pair['G1']['completed_additional_steps']
    full_budget = all(r['completed_additional_steps'] == ADDITIONAL for r in pair.values())
    methods = {'original_step33438': current['methods']['original_step33438'],
               **{g: pair[g]['methods'][g] for g in ('G0', 'G1')}}
    report = {'notice': NOTICE, 'test_evaluated': False, 'methods': methods,
              'same_completed_optimizer_steps': equal_steps, 'both_completed_planned10000': full_budget,
              'equal_planned_optimizer_budget_comparison': equal_steps and full_budget,
              'branch_steps': {g: pair[g]['completed_additional_steps'] for g in pair},
              'reference_recall': current['reference_recall'], 'common_recipe_sha256': current['common_recipe_sha256']}
    write_json(args.output / 'comparison_G0_G1.json', report)
    info(f'G0/G1 actual steps={report["branch_steps"]}; equal planned optimizer budget={equal_steps and full_budget}')
    if not equal_steps:
        info('完成步数不同，不能宣称等预算对照。')
    for name, row in methods.items():
        info(f'VAL COMPARISON MODEL: {name}')
        print_result(final.HEAD, row['maximum_f1'], row['minimum_fp_at_reference_recall'])
    g0, g1 = methods['G0'], methods['G1']
    a, b = g0['minimum_fp_at_reference_recall'], g1['minimum_fp_at_reference_recall']
    info(f'G1 minus G0 at reference Recall: FP delta={b["fp"]-a["fp"]}; actual Recall G0/G1={a["recall"]}/{b["recall"]}')
    for group in original.GROUPS:
        a = g0['maximum_f1']['groups_by_original_valid_pairs'][group]['at_fixed_val_threshold']
        b = g1['maximum_f1']['groups_by_original_valid_pairs'][group]['at_fixed_val_threshold']
        info(f'G1 versus G0 group={group}: P/R/F1={b["precision"]}/{b["recall"]}/{b["f1"]} vs '
             f'{a["precision"]}/{a["recall"]}/{a["f1"]}; TP/FP/FN={b["tp"]}/{b["fp"]}/{b["fn"]} vs '
             f'{a["tp"]}/{a["fp"]}/{a["fn"]}')


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite branch output: {args.output}')
    (state, config, calibration, train_plan, manifest, samples_file, sampling, groups,
     group_counts, val_cache, baseline, reference, recipe) = preflight(args)
    samples = samples_file['circuits']
    lookup = {r['circuit']: r for r in manifest}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sampling').mkdir()
    write_json(args.output / 'common_recipe.json', recipe)
    atomic_save(state, args.output / 'source_step33438.pt')
    atomic_save(samples_file, args.output / 'training_samples.pt')
    record = {'status': 'preparing', 'notice': NOTICE, 'branch': args.branch, 'source_run': str(SOURCE),
              'hidden': state['hidden'], 'fixed_samples_file': state['fixed_samples_file'],
              'optimizer_restore': 'same source33438 Adam, global RNG and mining RNG restored independently in each branch',
              'common_recipe_sha256': reuse.digest_json(recipe), 'group_counts': group_counts,
              'empty_train_circuits': sum(not r['original_valid_pairs'] for r in manifest),
              'source_checkpoint_sha256': recipe['source_checkpoint_sha256'], 'entry_sha256': file_hash(Path(__file__)),
              'planned_additional_steps': ADDITIONAL, 'completed_additional_steps': 0, 'completed_total_step': START,
              'wall_seconds': WALL, 'validation_reserve_seconds': VAL_RESERVE,
              'circuit_sampler': 'uniform4 distinct per step' if args.branch == 'G0' else 'uniform1 dense+3 other distinct per step',
              'circuit_rng_seed': args.seed,
              'negative_rng_seed': recipe['negative_rng_seed'], 'reference_recall': reference, 'stop_reason': None,
              'encoder_loaded': False, 'encoder_forward_count': 0, 'test_evaluated': False,
              'test_results_read': False, 'complete_val_evaluation': False}
    safe, updates, report = None, [], None
    with direct_log(args.output / 'console.log'):
        write_json(args.output / 'run.json', record)
        for group, counts in group_counts.items():
            info(f'Train pair-count group={group}: {counts}')
        info(f'Excluded empty train circuits={record["empty_train_circuits"]}; reference full-val Recall={reference["recall"]}, required TP={reference["required_tp"]}')
        try:
            with continuation.wall_limit(started + WALL):
                head = original.make_head(state['hidden']).to(args.device)
                head.load_state_dict(state['head'], strict=True)
                optimizer = torch.optim.Adam(head.parameters(), lr=.001)
                optimizer.load_state_dict(state['optimizer'])
                if any(g['lr'] != .001 for g in optimizer.param_groups):
                    raise ValueError('Source Adam learning rate changed')
                restore_random_state(state['global_random_state'])
                circuit_rng = torch.Generator().manual_seed(args.seed)
                negative_rng = torch.Generator().manual_seed(recipe['negative_rng_seed'])
                sampling_file = state['sampling_file']
                safe = snapshot(head, optimizer, START, circuit_rng, negative_rng, sampling_file, record)
                atomic_save(safe, args.output / 'last.pt')
                active = restore_active(samples, sampling, lookup, train_plan)
                deadline = started + WALL - VAL_RESERVE
                stop_reason = 'step_limit'
                try:
                    for step in range(START+1, START+ADDITIONAL+1):
                        hard.training_time_check(deadline)
                        if (step-1) % hard.REFRESH_INTERVAL == 0:
                            active, sampling_file = coverage.refresh(head, samples, lookup, train_plan, negative_rng,
                                                                     step-1, deadline, args.output)
                            info(f'{args.branch} train-only negative refresh at total_step={step-1}')
                        batch = draw_batch(args.branch, len(samples), groups, circuit_rng)
                        head.train()
                        optimizer.zero_grad(set_to_none=True)
                        losses = []
                        for i in batch.tolist():
                            row = active[i]
                            logits = head(row['features']).squeeze(-1)
                            continuation.finite(logits, '训练 logits')
                            losses.append(control.circuit_loss(logits, row['labels']))
                        loss = torch.stack(losses).mean()
                        continuation.finite(loss, '训练损失')
                        loss.backward()
                        for p in head.parameters():
                            if p.grad is not None:
                                continuation.finite(p.grad, '分类头梯度')
                        optimizer.step()
                        for p in head.parameters():
                            continuation.finite(p, '分类头参数')
                        safe = snapshot(head, optimizer, step, circuit_rng, negative_rng, sampling_file, record)
                        updates.append({'total_step': step, 'additional_step': step-START,
                                        'circuits': [samples[i]['circuit'] for i in batch.tolist()],
                                        'changing_sample_minibatch_bce': float(loss.detach())})
                        if (step-START) % 500 == 0:
                            atomic_save(safe, args.output / 'last.pt')
                            info(f'{args.branch}: additional step={step-START}/{ADDITIONAL}; changing-minibatch BCE={float(loss.detach()):.6f}; not fixed-objective convergence evidence')
                except hard.TrainingTimeLimit:
                    stop_reason = 'time_limit'
                head.load_state_dict(safe['head'], strict=True)
                atomic_save(safe, args.output / 'last.pt')
                atomic_save(safe, args.output / 'final_head.pt')
                record.update(status='evaluating_val', completed_total_step=safe['optimizer_steps'],
                              completed_additional_steps=safe['optimizer_steps']-START, stop_reason=stop_reason,
                              training_elapsed_wall_seconds=time.monotonic()-started)
                write_json(args.output / 'run.json', record)
                write_json(args.output / 'update_history.json', updates)
                report = evaluate(args, head, config, baseline, calibration, val_cache, reference, record)
                combine_peer(args, report)
        except (continuation.WallLimit, continuation.NonFinite) as error:
            complete = (args.output / 'comparison.json').is_file()
            record.update(status='completed' if complete else 'incomplete_timeout' if isinstance(error, continuation.WallLimit) else 'stopped_nonfinite',
                          complete_val_evaluation=complete, stop_detail=str(error))
            if record['stop_reason'] is None:
                record['stop_reason'] = 'time_limit' if isinstance(error, continuation.WallLimit) else 'nonfinite'
            info('保存最后完整有限更新；未完成val不能作为完整验证结果。')
        finally:
            if safe is not None:
                atomic_save(safe, args.output / 'last.pt')
                atomic_save(safe, args.output / 'final_head.pt')
                record.update(completed_total_step=safe['optimizer_steps'],
                              completed_additional_steps=safe['optimizer_steps']-START,
                              branch_circuit_visits=(safe['optimizer_steps']-START)*4)
            write_json(args.output / 'update_history.json', updates)
            record['elapsed_wall_seconds'] = time.monotonic()-started
            write_json(args.output / 'run.json', record)
            info(f'{args.branch}: status={record["status"]}; completed additional steps={record["completed_additional_steps"]}/{ADDITIONAL}; stop={record["stop_reason"]}; elapsed={record["elapsed_wall_seconds"]:.1f}s')
            info(NOTICE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--branch', choices=('G0', 'G1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--peer-output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, required=True,
                        help='independent unused sampling seed, e.g. 43 or 44')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
