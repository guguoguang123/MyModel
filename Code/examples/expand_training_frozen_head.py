"""Manual 512-train-circuit cache stage and bounded cache-only head training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import compare_circuit_balanced_head as control
from examples import continue_circuit_balanced_head as continuation
from examples import expanded_frozen_head_validation as extended
from examples import finetune_hard_negative_head as hard
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.random_seed import set_seed

EXTENDED = original.EXP / 'frozen_heads_extended_val128_20261113'
FIXED = original.EXP / 'frozen_equivalence_head_circuit_balanced_20261113'
NOTICE = ('512训练电路覆盖对照；32电路仅用于校准新头阈值；128电路已用于开发决策，'
          '不是最终独立测试；不运行test，不根据128电路结果调阈值或追加训练')
SEED, TOTAL, ADDED, STEPS = 20261113, 512, 448, 5000


def read(path):
    return json.loads(Path(path).read_text())


def choose_train_ids(splits, retained):
    if len(retained) != 64 or len(set(retained)) != 64 or not set(retained) <= set(splits['train']):
        raise ValueError('Require the original 64 train IDs')
    eligible = sorted(set(splits['train']) - set(retained))
    if len(eligible) < ADDED:
        raise ValueError('Insufficient remaining train IDs; no replacement')
    added = random.Random(SEED).sample(eligible, ADDED)
    names = retained + added
    if set(names) & (set(splits['val']) | set(splits['test'])):
        raise ValueError('Train/val/test overlap; stopping')
    return names, added


def prepare(args):
    """Only identities, saved run metadata, and small checkpoint hashes; no graphs."""
    output = args.cache_output
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    base = read(EXTENDED / 'plan.json')
    if not read(EXTENDED / 'comparison.json').get('complete_128_circuit_result'):
        raise ValueError('Require completed existing expanded cache/result')
    # Verify saved encoder provenance, original head implementation/initialization recipe.
    dev_run = read(extended.DEV / 'run.json')
    prior = read(FIXED / 'run.json')
    for path, expected in ((Path(base['encoder_checkpoint']), base['cache_contract']['encoder_checkpoint_sha256']),
                           (Path(original.__file__), dev_run['source_sha256']),
                           (Path(control.__file__), prior['entry_sha256'])):
        if file_hash(path) != expected:
            raise ValueError(f'Existing encoder/head recipe changed: {path}')
    contract = base['cache_contract']
    if (contract['encoder_epoch'] != 5 or contract['functional_net_identity'] is not False
            or contract['device'] != args.device or extended.input_stats(Path(base['data'])) != contract['input_files']):
        raise ValueError('Require unchanged frozen B epoch5, original propagation, device and input metadata')
    selection = read(extended.DEV / 'selection.json')
    retained = [row['circuit'] for row in selection['circuits'] if row['split'] == 'train']
    # Uniform selection before reading ANY pair counts, labels or predictions.
    names, added = choose_train_ids(read(original.B_RUN / 'split.json'), retained)
    plan = {'notice': NOTICE, 'seed': SEED, 'selected_ids': names, 'retained_ids': retained,
            'added_ids': added, 'requested_circuits': TOTAL,
            'selection_rule': 'retain original64; random.Random(20261113).sample(sorted(original train minus retained64),448); no filtering or replacement',
            'cache_contract': contract, 'encoder_checkpoint': base['encoder_checkpoint'],
            'data': base['data'], 'device': args.device,
            'external_cache_manifests': [str(extended.DEV / 'cache_manifest.json'),
                                         str(EXTENDED / 'cache_manifest.json'),
                                         *[str(p.resolve()) for p in args.cache_manifest]],
            'locked_files': {}, 'cache_wall_seconds': 1800, 'test_evaluated': False}
    locked = [Path(__file__), Path(original.__file__), Path(control.__file__), Path(hard.__file__),
              Path(continuation.__file__), Path(extended.__file__), Path(base['encoder_checkpoint']),
              original.B_RUN / 'split.json', FIXED / 'training_samples.pt', FIXED / 'initial_head.pt',
              extended.DEV / 'selection.json', EXTENDED / 'plan.json', EXTENDED / 'comparison.json']
    plan['locked_files'] = {str(p.resolve()): file_hash(p) for p in locked}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'selection.json', {k: plan[k] for k in
               ('seed', 'selection_rule', 'selected_ids', 'retained_ids', 'added_ids')})
    write_json(output / 'plan.json', plan)
    write_json(output / 'run.json', {'status': 'prepared', 'complete_512_cache': False, 'notice': NOTICE})
    info('Prepared original64 + uniform448 train IDs; no inference, no label filtering')
    return plan


def verify_plan(args):
    plan = read(args.cache_output / 'plan.json')
    if args.device != plan['device']:
        raise ValueError('Keep prepared device')
    names, added = choose_train_ids(read(original.B_RUN / 'split.json'), plan['retained_ids'])
    if names != plan['selected_ids'] or added != plan['added_ids']:
        raise ValueError('Saved train selection changed')
    for path, expected in plan['locked_files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Prepared input changed: {path}')
    if extended.input_stats(Path(plan['data'])) != plan['cache_contract']['input_files']:
        raise ValueError('Input file size/mtime changed; no full audit attempted')
    return plan


def class_kind(pos, neg):
    return 'both_class' if pos and neg else 'positive_only' if pos else 'negative_only' if neg else 'empty'


def cache_counts(rows):
    counts = {g: {kind: 0 for kind in ('both_class', 'positive_only', 'negative_only', 'empty')}
              | {'positive_pairs': 0, 'negative_pairs': 0} for g in (*original.GROUPS, 'zero_valid_pairs')}
    for row in rows:
        pos, neg = row['positive_count'], row['negative_count']
        group = original.group_name(pos + neg) if pos + neg else 'zero_valid_pairs'
        counts[group][class_kind(pos, neg)] += 1
        counts[group]['positive_pairs'] += pos
        counts[group]['negative_pairs'] += neg
    return counts


def cache_stage(args):
    started = time.monotonic()
    if not (args.cache_output / 'plan.json').is_file():
        prepare(args)
    plan = verify_plan(args)
    cache_dir = args.cache_output / 'hf_cache'
    cache_dir.mkdir(exist_ok=True)
    manifest_path = args.cache_output / 'cache_manifest.json'
    rows = {r['circuit']: r for r in read(manifest_path)} if manifest_path.is_file() else {}
    record = {'status': 'caching', 'notice': NOTICE, 'complete_512_cache': False,
              'requested_circuits': TOTAL, 'encoder_forwards_this_attempt': 0,
              'reused_circuits_this_attempt': 0, 'wall_seconds': 1800, 'test_evaluated': False}
    with direct_log(args.cache_output / 'console.log'), ExitStack() as stack:
        try:
            with continuation.wall_limit(started + 1800):
                # Compatible manifests are indexed by identity/provenance, never label balance.
                external = extended.external_cache_index(plan)
                graph = None
                missing = [n for n in plan['selected_ids'] if n not in rows and n not in external
                           and not (cache_dir / f'{extended.cache_key(n)}.pt').is_file()]
                if missing:
                    graph = original.NpzSlices(Path(plan['data']) / 'graphs.npz')
                    stack.callback(graph.close)
                    archive = {str(n): i for i, n in enumerate(graph.read('design_ids'))}
                    order = sorted(plan['selected_ids'], key=archive.__getitem__)
                else:
                    order = plan['selected_ids']
                reader, encoder = None, None
                set_seed(original.SEED)
                for index, name in enumerate(order, 1):
                    own = cache_dir / f'{extended.cache_key(name)}.pt'
                    source = Path(rows[name]['cache_file']) if name in rows else own if own.is_file() else external.get(name)
                    if source is not None:
                        if not source.is_file():
                            raise FileNotFoundError(f'Saved cache missing; refusing to repeat encoder: {source}')
                        cache = torch.load(source, map_location='cpu', weights_only=False)
                        extended.check_cache(cache, name, plan['cache_contract'], source == own)
                        record['reused_circuits_this_attempt'] += 1
                    else:
                        if reader is None:
                            label = original.NpzSlices(Path(plan['data']) / 'labels.npz')
                            stack.callback(label.close)
                            state = torch.load(plan['encoder_checkpoint'], map_location='cpu', weights_only=False)
                            reader = original.SelectedReader(graph, label, state)
                            encoder, _ = load_model(Path(plan['encoder_checkpoint']), reader, 'cpu')
                            original.freeze_encoder(encoder)
                            encoder.to(args.device)
                        cache = extended.extract_cache(reader, encoder, name, plan['cache_contract'], args.device)
                        cache['metadata']['split'] = 'train'
                        atomic_save(cache, own)
                        source = own
                        record['encoder_forwards_this_attempt'] += 1
                    pos = int(cache['labels'].sum())
                    rows[name] = {'circuit': name, 'split': 'train', 'cache_file': str(source.resolve()),
                                  'original_valid_pairs': cache['labels'].numel(), 'positive_count': pos,
                                  'negative_count': cache['labels'].numel() - pos,
                                  'retained_original64': name in plan['retained_ids']}
                    del cache
                    write_json(manifest_path, [rows[n] for n in plan['selected_ids'] if n in rows])
                    record['completed_cache_circuits'] = len(rows)
                    write_json(args.cache_output / 'run.json', record)
                    info(f'Train cache {index}/{TOTAL}: {name} | pos/neg={rows[name]["positive_count"]}/{rows[name]["negative_count"]}')
                record.update(status='completed', complete_512_cache=len(rows) == TOTAL,
                              group_counts=cache_counts(rows.values()))
                if len(rows) != TOTAL:
                    raise ValueError('Incomplete selected cache; no complete result')
                write_json(args.cache_output / 'counts.json', record['group_counts'])
                for group, counts in record['group_counts'].items():
                    info(f'Train cache group={group}: {counts}')
        except continuation.WallLimit:
            record.update(status='incomplete_timeout', stop_reason='30 minute cache limit', complete_512_cache=False)
            info('缓存未完成；已完成电路保留。用同一缓存命令复用，不能作为完整512电路结果。')
        record.update(completed_cache_circuits=len(rows), elapsed_wall_seconds=time.monotonic() - started)
        write_json(args.cache_output / 'run.json', record)


def load_train_cache(row, plan):
    cache = torch.load(row['cache_file'], map_location='cpu', weights_only=False)
    extended.check_cache(cache, row['circuit'], plan['cache_contract'],
                         'cache_contract' in cache)
    if cache['labels'].numel() != row['original_valid_pairs'] or int(cache['labels'].sum()) != row['positive_count']:
        raise ValueError(f'Train cache labels/count changed: {row["circuit"]}')
    return cache


def fixed_samples(manifest, plan, output):
    old = torch.load(FIXED / 'training_samples.pt', map_location='cpu', weights_only=False)
    old = {r['circuit']: r for r in old['circuits']}
    generator = torch.Generator().manual_seed(original.SEED)
    samples, active = [], []
    for meta in manifest:
        cache = load_train_cache(meta, plan)
        if not cache['labels'].numel():
            continue  # selection/count retained, no loss and no replacement
        if meta['circuit'] in plan['retained_ids']:
            row = old[meta['circuit']]
            columns = row['cache_pair_columns']
            if not torch.equal(cache['original_pair_columns'][columns], row['original_pair_columns']):
                raise ValueError('Original64 fixed pair mapping changed')
            labels = cache['labels'][columns]
            if int(labels.sum()) != row['positive_count'] or int((~labels).sum()) != row['negative_count']:
                raise ValueError('Original64 fixed pair labels changed')
            feature = hard.features_at(cache, columns)
            sampled = dict(row)
        else:
            sampled = control.sample_circuit(cache, generator)
            feature, labels = sampled.pop('features'), sampled.pop('labels')
        samples.append(sampled)
        active.append({'circuit': meta['circuit'], 'features': feature.to(plan['device']),
                       'labels': labels.to(plan['device'])})
    if not samples:
        raise ValueError('No active train circuits')
    atomic_save({'objective': control.OBJECTIVE, 'circuits': samples,
                 'new_sample_rng_state': generator.get_state()}, output / 'training_samples.pt')
    return samples, active


@torch.no_grad()
def refresh(head, samples, lookup, plan, generator, step, deadline, output):
    """One full circuit cache at a time; bounded candidate feature chunks."""
    records, active = [], []
    before = generator.get_state().clone()
    for sample in samples:
        hard.training_time_check(deadline)
        cache = load_train_cache(lookup[sample['circuit']], plan)
        columns = sample['cache_pair_columns']
        positives = columns[cache['labels'][columns]]
        candidates = torch.nonzero(~cache['labels']).flatten()
        negatives, mining = hard.choose_negatives(head, cache, candidates, sample['negative_count'],
                                                  generator, plan['device'], training_deadline=deadline)
        chosen = torch.cat((positives, negatives))
        active.append({'circuit': sample['circuit'], 'features': hard.features_at(cache, chosen).to(plan['device']),
                       'labels': cache['labels'][chosen].to(plan['device'])})
        records.append({'circuit': sample['circuit'], 'positive_cache_columns': positives,
                        'negative_cache_columns': negatives,
                        'negative_original_columns': cache['original_pair_columns'][negatives], **mining})
        del cache
    path = output / 'sampling' / f'refresh_step_{step}.pt'
    atomic_save({'refresh_total_step': step, 'negative_rng_before': before,
                 'negative_rng_after': generator.get_state(), 'circuits': records}, path)
    return active, str(path.resolve())


def sample_counts(samples):
    counts = {g: {'both_class': 0, 'positive_only': 0, 'negative_only': 0,
                  'positive_samples': 0, 'negative_samples': 0} for g in original.GROUPS}
    for row in samples:
        entry = counts[row['pair_count_group']]
        pos, neg = row['positive_count'], row['negative_count']
        entry[class_kind(pos, neg)] += 1
        entry['positive_samples'] += pos
        entry['negative_samples'] += neg
    return counts


@torch.inference_mode()
def score_one_cache(head, cache, device):
    parts = []
    for pairs in cache['pairs'].split(hard.CHUNK_SIZE, dim=1):
        left, right = pairs
        scores = head(original.pair_features(cache['hf_l2'][left], cache['hf_l2'][right]).to(device)).squeeze(-1)
        continuation.finite(scores, '评价 logits')
        parts.append(scores.cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def evaluate(args, plan, head, record):
    """Only completed final weights. Calibrate on dev32, then fixed eval128 once."""
    head.eval().requires_grad_(False)
    dev_manifest = [r for r in read(extended.DEV / 'cache_manifest.json') if r['split'] == 'val']
    saved = torch.load(extended.DEV / 'validation_scores.pt', map_location='cpu', weights_only=False)
    if [r['circuit'] for r in dev_manifest] != saved['circuits']:
        raise ValueError('Original32 cache circuit order differs')
    parts = []
    for i, row in enumerate(dev_manifest):
        cache = torch.load(row['cache_file'], map_location='cpu', weights_only=False)
        extended.check_cache(cache, row['circuit'], plan['cache_contract'], False)
        a, b = saved['offsets'][i:i+2]
        if not torch.equal(cache['labels'], saved['labels'][a:b]):
            raise ValueError('Development cache label order differs')
        parts.append(score_one_cache(head, cache, args.device))
    scores = torch.cat(parts)
    # Existing complete equal-score group rules: max F1; FP<=69 max TP, min FP, highest threshold.
    calibration = control.score_summary(scores, saved['labels'], saved['offsets'], saved['circuits'], 'logit')
    thresholds = {'main_threshold': calibration['maximum_f1']['threshold'],
                  'old_low_fp_threshold': calibration['fp_le_69']['threshold'], 'score_kind': 'logit',
                  'source': 'new head on original32 development circuits only',
                  'low_threshold_note': 'calibrated to FP<=69 on original32, not constrained to69 on expanded128'}
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'development32.json', {'notice': NOTICE, 'calibration': calibration})
    atomic_save({'head_logits': scores, 'labels': saved['labels'], 'offsets': saved['offsets'],
                 'circuits': saved['circuits']}, args.output / 'development32_scores.pt')
    record.update(status='evaluating_expanded128', dev_calibration_completed=True, expanded_evaluation_started=True)
    write_json(args.output / 'run.json', record)
    ext_plan = read(EXTENDED / 'plan.json')
    ext_report = read(EXTENDED / 'comparison.json')
    if ext_plan['cache_contract'] != plan['cache_contract'] or not ext_report['complete_128_circuit_result']:
        raise ValueError('Missing compatible complete128 cache; no inference allowed')
    lookup = {r['circuit']: r for r in read(EXTENDED / 'cache_manifest.json')}
    parts, labels, offsets = [], [], [0]
    score_dir = args.output / 'expanded_scores'
    score_dir.mkdir(exist_ok=True)
    for name in ext_plan['selected_ids']:
        cache = torch.load(lookup[name]['cache_file'], map_location='cpu', weights_only=False)
        extended.check_cache(cache, name, plan['cache_contract'], True)
        logits = score_one_cache(head, cache, args.device)
        atomic_save({'circuit': name, 'head_logits': logits, 'labels': cache['labels'],
                     'original_pair_columns': cache['original_pair_columns']}, score_dir / f'{extended.cache_key(name)}.pt')
        parts.append(logits)
        labels.append(cache['labels'])
        offsets.append(offsets[-1] + logits.numel())
        del cache
    new = extended.summarize(torch.cat(parts), torch.cat(labels), offsets, ext_plan['selected_ids'], thresholds)
    methods = {name: ext_report['methods'][name] for name in ('B_cosine', 'hard_negative_head_step5000')}
    methods['hard_negative_head_train512_step5000'] = new
    report = {'notice': NOTICE, 'complete_512_training': True, 'optimizer_steps': STEPS,
              'complete_128_circuit_result': True, 'selected_ids': ext_plan['selected_ids'],
              'zero_valid_pair_circuits': ext_report['zero_valid_pair_circuits'],
              'threshold_selection': 'new head calibrated on original32 only, fixed on expanded128; old methods reuse saved results and thresholds',
              'expanded128_used_for_development_decisions': True, 'test_evaluated': False,
              'low_fp_note': thresholds['low_threshold_note'], 'methods': methods}
    write_json(args.output / 'comparison.json', report)
    for name, result in methods.items():
        extended.print_result(name, result)
    record.update(status='completed', complete_result=True, expanded_evaluation_completed=True)
    write_json(args.output / 'run.json', record)
    info(NOTICE)
    info(f'Completed512 training and fixed-threshold128 evaluation: {args.output / "comparison.json"}')


def train_stage(args):
    started = time.monotonic()
    plan = verify_plan(args)
    if not read(args.cache_output / 'run.json').get('complete_512_cache'):
        raise ValueError('512 train cache incomplete; no implicit encoder inference')
    manifest = read(args.cache_output / 'cache_manifest.json')
    if [r['circuit'] for r in manifest] != plan['selected_ids']:
        raise ValueError('Require all exact512 selected train caches in saved order')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite training output: {args.output}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'sampling').mkdir()
    record = {'status': 'training', 'notice': NOTICE, 'hidden': plan['cache_contract']['hidden'],
              'source_run': str(args.cache_output.resolve()), 'optimizer_restore': 'new Adam, lr=1e-3; same seed42 initial head',
              'objective': control.OBJECTIVE, 'seed': 42, 'lr': 1e-3, 'circuits_per_step': 4,
              'planned_optimizer_steps': STEPS, 'training_wall_seconds': 300,
              'encoder_loaded': False, 'encoder_forward_count': 0, 'test_evaluated': False,
              'complete_result': False, 'complete_512_training': False,
              'sampling': 'original64 fixed pairs retained; added circuits independent cap128/class no replacement; positives fixed; random negatives steps1..500; top-half+uniform refresh at500,1000,..4500',
              'selection': 'final complete5000 only, no validation-based checkpoint selection',
              'expanded128_used_for_development_decisions': True}
    safe = None
    updates = []
    head = None
    with direct_log(args.output / 'console.log'):
        write_json(args.output / 'run.json', record)
        try:
            with continuation.wall_limit(started + 300):
                set_seed(original.SEED)
                head = original.make_head(record['hidden']).to(args.device)
                digest = original.state_digest(head)
                old_init = torch.load(FIXED / 'initial_head.pt', map_location='cpu', weights_only=False)
                if digest != old_init['state_sha256']:
                    raise ValueError('Original head random initialization differs; no alternative')
                optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
                safe = continuation.snapshot(head, optimizer, 0, record)
                atomic_save(safe, args.output / 'initial_head.pt')
                atomic_save(safe, args.output / 'last.pt')
                samples, active = fixed_samples(manifest, plan, args.output)
                record.update(initial_state_sha256=digest, active_train_circuits=len(samples),
                              group_sample_counts=sample_counts(samples), cache_group_counts=cache_counts(manifest))
                for group, counts in record['group_sample_counts'].items():
                    info(f'Train sample group={group}: {counts}')
                info('按电路及类别取均值；各组总体标签比例不保证1:1；无有效配对电路保留但不参与损失')
                write_json(args.output / 'run.json', record)
                schedule = control.circuit_schedule(len(samples), 4, STEPS, original.SEED)
                write_json(args.output / 'circuit_schedule.json', {
                    'start_total_step': 1, 'batches': [[samples[i]['circuit'] for i in batch.tolist()] for batch in schedule]})
                rng = torch.Generator().manual_seed(hard.MINING_SEED)
                lookup = {r['circuit']: r for r in manifest}
                refresh_file = None
                for step, batch in enumerate(schedule, 1):
                    hard.training_time_check(started + 300)
                    if step > 500 and (step - 501) % hard.REFRESH_INTERVAL == 0:
                        active, refresh_file = refresh(head, samples, lookup, plan, rng, step-1, started+300, args.output)
                        info(f'Hard negatives refreshed at total_step={step-1}')
                    head.train()
                    optimizer.zero_grad(set_to_none=True)
                    local = []
                    for i in batch.tolist():
                        row = active[i]
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
                    candidate = continuation.snapshot(head, optimizer, step, record)
                    candidate.update(negative_rng_state=rng.get_state(), sampling_file=refresh_file,
                                     fixed_samples_file=str((args.output / 'training_samples.pt').resolve()))
                    safe = candidate
                    updates.append({'step': step, 'changing_sample_minibatch_bce': float(loss.detach()),
                                    'refresh_step': (step-1)//500*500 if step > 500 else None})
                    if step % 500 == 0:
                        atomic_save(safe, args.output / 'last.pt')
                        atomic_save(safe, args.output / f'head_step{step}.pt')
                        info(f'Head step={step}/{STEPS}; changing-sample loss={float(loss.detach()):.6f} (not convergence evidence)')
                # The timer must include the last safe snapshot; no partial train result is evaluated.
                hard.training_time_check(started + 300)
                record.update(status='trained', complete_512_training=True, completed_optimizer_steps=STEPS,
                              stop_reason='step_limit')
        except (continuation.WallLimit, hard.TrainingTimeLimit, continuation.NonFinite) as error:
            record.update(status='incomplete_nonfinite' if isinstance(error, continuation.NonFinite) else 'incomplete_timeout',
                          stop_reason='nonfinite' if isinstance(error, continuation.NonFinite) else '5 minute training limit',
                          stop_detail=str(error), complete_512_training=False,
                          completed_optimizer_steps=safe['optimizer_steps'] if safe else 0)
            info('未完成：保留最后完整有限更新，不校准阈值、不评价128电路。')
        finally:
            if safe is not None:
                atomic_save(safe, args.output / 'last.pt')
            write_json(args.output / 'update_history.json', updates)
            record['training_elapsed_wall_seconds'] = time.monotonic() - started
            write_json(args.output / 'run.json', record)
        if not record['complete_512_training']:
            return
        atomic_save(safe, args.output / 'final_head.pt')
        head.load_state_dict(safe['head'], strict=True)
        evaluate(args, plan, head, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--cache', action='store_true')
    action.add_argument('--train', action='store_true')
    parser.add_argument('--cache-output', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.train and args.output is None:
        parser.error('--train requires a fresh --output directory')
    if args.cache:
        cache_stage(args)
    else:
        train_stage(args)


if __name__ == '__main__':
    main()
