"""Manual full-val calibration, followed by one resumable frozen-candidate test."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import evaluate as legacy
from examples import expanded_frozen_head_validation as reuse
from examples import frozen_equivalence_head as original
from examples import remaining_frozen_head_validation as remaining
from examples.compare_functional_stages import file_hash, cosine_scores64
from examples.continue_circuit_balanced_head import WallLimit, wall_limit
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.trainer import move
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.functional_equivalence import select_threshold
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.model_utils import model_outputs
from triwave_gnn.utils.net_identity import checkpoint_net_identity
from triwave_gnn.utils.random_seed import set_seed

REMAINING = original.EXP / 'frozen_heads_remaining_val1503_20261113'
COS, HEAD = 'B_original_cosine_equivalence', 'B_plus_512_head_equivalence'
METHODS = (COS, HEAD)
WALL_SECONDS = 1800
NOTICE = ('固定B编码器epoch5、原传播、独立512电路分类头step33438；'
          '仅完整val选择最大F1阈值，test固定使用；原余弦与新增分类头分别命名；'
          '概率/RC/TT距离/行为指标来自冻结B及其原任务头；不训练、不根据test追加实验')


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def score_sources(plan):
    sources = {}
    for name in plan['excluded_development32_ids']:
        sources[name] = {'kind': 'development32', 'cosine': str(reuse.DEV / 'validation_scores.pt'),
                         'head': str(remaining.CANDIDATE / 'development32_scores.pt')}
    for name in plan['excluded_expanded128_ids']:
        key = reuse.cache_key(name)
        sources[name] = {'kind': 'expanded128', 'cosine': str(remaining.EXTENDED / 'score_cache' / f'{key}.pt'),
                         'head': str(remaining.CANDIDATE / 'expanded_scores' / f'{key}.pt')}
    for name in plan['selected_ids']:
        path = str(REMAINING / 'score_cache' / f'{reuse.cache_key(name)}.pt')
        if name in sources:
            raise ValueError(f'Overlapping validation source ID: {name}')
        sources[name] = {'kind': 'remaining1503', 'cosine': path, 'head': path}
    return sources


def frozen_config(args):
    plan = read(REMAINING / 'plan.json')
    status = read(REMAINING / 'run.json')
    if not status.get('complete_remaining_validation') or status['completed_score_circuits'] != 1503:
        raise ValueError('Remaining1503 scores incomplete; no validation inference fallback')
    encoder_path = Path(plan['encoder_checkpoint'])
    head_path = Path(plan['head_checkpoints'][remaining.NEW])
    encoder, head = load(encoder_path), load(head_path)
    if (encoder['epoch'] != 5 or checkpoint_net_identity(encoder)
            or file_hash(encoder_path) != plan['encoder_sha256']
            or head['optimizer_steps'] != 33438
            or file_hash(head_path) != plan['head_config'][remaining.NEW]['sha256']
            or head_path != remaining.CANDIDATE / 'final_head.pt'):
        raise ValueError('Exact frozen encoder/head checkpoint mismatch; no best-file fallback')
    if args.device != plan['device']:
        raise ValueError('Keep original score/inference device')
    splits = read(original.B_RUN / 'split.json')
    val_key = legacy.validate_splits(splits, encoder)
    rest, counts = remaining.remaining_ids(splits, plan['excluded_development32_ids'], plan['excluded_expanded128_ids'])
    if rest != plan['selected_ids']:
        raise ValueError('Saved160/1503 partition differs from original split')
    sources = score_sources(plan)
    if set(sources) != set(splits[val_key]) or len(sources) != 1663:
        raise ValueError('Full val score IDs have omissions/extras')
    missing = sorted({path for row in sources.values() for key, path in row.items()
                      if key in ('cosine', 'head') and not Path(path).is_file()})
    if missing:
        raise FileNotFoundError(f'缺少已保存val分数文件，共{len(missing)}个；停止，不补跑验证推理：\n' + '\n'.join(missing))
    source_config_paths = [original.B_RUN / 'split.json', REMAINING / 'plan.json',
                           remaining.CANDIDATE / 'run.json', remaining.TRAIN_CACHE / 'plan.json']
    if (original.B_RUN / 'config.json').is_file():
        source_config_paths.append(original.B_RUN / 'config.json')
    code_paths = [Path(__file__), Path(original.__file__), Path(reuse.__file__), Path(legacy.__file__),
                  Path(sys.modules[load_model.__module__].__file__), Path(sys.modules[model_outputs.__module__].__file__),
                  Path(sys.modules[select_threshold.__module__].__file__),
                  Path(__file__).resolve().parents[1] / 'triwave_gnn/model.py',
                  Path(__file__).resolve().parents[1] / 'triwave_gnn/arch/propagation.py']
    locked = [encoder_path, head_path, *source_config_paths, *code_paths]
    return {'notice': NOTICE, 'frozen_config_path': str((args.output / 'frozen_config.json').resolve()),
            'encoder_checkpoint': str(encoder_path.resolve()),
            'encoder_sha256': file_hash(encoder_path), 'encoder_epoch': 5,
            'head_checkpoint': str(head_path.resolve()), 'head_sha256': file_hash(head_path),
            'head_actual_step': 33438, 'encoder_config': encoder['config'],
            'head_config': plan['head_config'][remaining.NEW],
            'source_configs': {str(p.resolve()): file_hash(p) for p in source_config_paths},
            'functional_net_identity': False, 'model_mode': 'eval, all original parameters frozen, no_grad',
            'feature': 'concat(abs(L2(hf_u)-L2(hf_v)),L2(hf_u)*L2(hf_v)); eps1e-12; hf only',
            'cosine_policy': plan['cache_contract']['cosine_policy'],
            'original_task_metric_implementation': str(Path(legacy.__file__).resolve()),
            'data': plan['data'], 'input_files': plan['cache_contract']['input_files'], 'device': args.device,
            'cache_contract': plan['cache_contract'], 'val_ids': splits[val_key], 'test_ids': splits['test'],
            'val_partition_counts': counts, 'val_score_sources': sources,
            'locked_files': {str(p.resolve()): file_hash(p) for p in locked},
            'wall_seconds_per_command': WALL_SECONDS, 'training': False,
            'threshold_rule': 'maximum pooled full-val F1; whole equal-score groups; equal F1 chooses higher threshold',
            'test_threshold_search_allowed': False, 'test_checkpoint_selection_allowed': False}


def verify_config(config, args):
    if args.device != config['device']:
        raise ValueError('Keep frozen device')
    for path, expected in config['locked_files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Frozen checkpoint/config/code changed: {path}')
    if reuse.input_stats(Path(config['data'])) != config['input_files']:
        raise ValueError('Data file size/mtime changed; no full raw-data audit attempted')


def valid_vectors(scores, labels, name):
    if (labels.dtype != torch.bool or labels.ndim != 1
            or any(s.shape != labels.shape or not bool(torch.isfinite(s).all()) for s in scores)):
        raise ValueError(f'Saved validation score/label vectors missing or inconsistent: {name}; no inference fallback')


def collect_val_scores(config):
    plan = read(REMAINING / 'plan.json')
    rem_hash = reuse.digest_json(plan)
    ext_hash = reuse.digest_json(read(remaining.EXTENDED / 'plan.json'))
    base = load(reuse.DEV / 'validation_scores.pt')
    new = load(remaining.CANDIDATE / 'development32_scores.pt')
    if (base['circuits'] != new['circuits'] or base['offsets'] != new['offsets']
            or not torch.equal(base['labels'], new['labels'])
            or set(base['circuits']) != set(plan['excluded_development32_ids'])):
        raise ValueError('Saved original32 vectors do not align')
    dev_index = {n: (a, b) for n, a, b in zip(base['circuits'], base['offsets'], base['offsets'][1:])}
    pieces, label_parts, offsets = {COS: [], HEAD: []}, [], [0]
    for name in config['val_ids']:
        src = config['val_score_sources'][name]
        if src['kind'] == 'development32':
            a, b = dev_index[name]
            cosine, logits, labels = base['cosine_scores64'][a:b], new['head_logits'][a:b], base['labels'][a:b]
        elif src['kind'] == 'expanded128':
            saved, current = load(src['cosine']), load(src['head'])
            if (saved['plan_sha256'] != ext_hash or saved['metadata']['circuit'] != name
                    or current['circuit'] != name or not torch.equal(saved['labels'], current['labels'])
                    or not torch.equal(saved['original_pair_columns'], current['original_pair_columns'])):
                raise ValueError(f'Saved expanded128 score mapping/provenance changed: {name}')
            cosine, logits, labels = saved['scores']['B_cosine'], current['head_logits'], saved['labels']
        else:
            saved = load(src['cosine'])
            if saved['plan_sha256'] != rem_hash or saved['metadata']['circuit'] != name:
                raise ValueError(f'Saved remaining1503 score provenance changed: {name}')
            cosine, logits, labels = saved['scores']['B_cosine'], saved['scores'][remaining.NEW], saved['labels']
        valid_vectors((cosine, logits), labels, name)
        pieces[COS].append(cosine)
        pieces[HEAD].append(logits)
        label_parts.append(labels)
        offsets.append(offsets[-1] + labels.numel())
    return {'scores': {k: torch.cat(v) for k, v in pieces.items()}, 'labels': torch.cat(label_parts),
            'offsets': offsets, 'circuits': config['val_ids']}


def equivalence_summary(scores, labels, offsets, names, threshold, split, score_kind):
    # Reuse the existing fixed-threshold pooled/group/per-circuit AUC implementation.
    result = reuse.summarize(scores, labels, offsets, names,
                             {'main_threshold': threshold, 'old_low_fp_threshold': threshold, 'score_kind': score_kind})
    result['at_fixed_val_threshold'] = result.pop('at_old_dev_max_f1_threshold')
    result.pop('at_old_low_fp_threshold')
    result.pop('thresholds_reselected')
    result.update(threshold_source='full original1663 val circuits only', threshold_selected_on_this_split=split == 'val')
    for row in result['per_circuit']:
        row.pop('at_old_low_fp_threshold')
    for row in result['groups_by_original_valid_pairs'].values():
        row['at_fixed_val_threshold'] = row.pop('at_old_dev_max_f1_threshold')
        row.pop('at_old_low_fp_threshold')
    return result


def print_equivalence(split, method, result):
    title = 'B原余弦等价识别' if method == COS else 'B＋512分类头等价识别（独立新增分类头）'
    for group, row in [('pooled', result), *result['groups_by_original_valid_pairs'].items()]:
        m = row['at_fixed_val_threshold']
        info(f"{split} {title} / {group}: threshold={m['threshold']} | "
             f"P/R/F1={m['precision']}/{m['recall']}/{m['f1']} | "
             f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} | "
             f"positive/negative={m['positive_count']}/{m['negative_count']} | "
             f"pooled/macro/within AUC={m['roc_auc']}/{row['circuit_macro_roc_auc']}/{row['within_circuit_weighted_auc']} | "
             f"macro circuits={row['macro_participating_circuits']} | FPR={m['false_positive_rate']}")


def calibrate(args):
    started = time.monotonic()
    if args.output.exists():
        config = read(args.output / 'frozen_config.json')
        verify_config(config, args)
        if (args.output / 'calibration_val1663.json').is_file():
            completed = read(args.output / 'calibration_val1663.json')
            if (not completed.get('complete_val_calibration')
                    or completed['frozen_config_sha256'] != file_hash(args.output / 'frozen_config.json')
                    or completed['final_thresholds_sha256'] != file_hash(args.output / 'final_val_thresholds.json')):
                raise ValueError('Saved complete calibration/config/threshold artifact changed')
            info('Full-val calibration already saved; reuse without choosing thresholds again')
            return
        if (args.output / 'test_plan.json').exists():
            raise ValueError('Test plan already frozen; cannot recalibrate')
    else:
        config = frozen_config(args)  # coverage checked before creating output; never reads test labels
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'frozen_config.json', config)
    record = {'status': 'calibrating_val', 'complete_val_calibration': False, 'test_labels_read': False,
              'test_inference_started': False, 'encoder_forward_count': 0, 'training': False}
    with direct_log(args.output / 'calibration_console.log'):
        try:
            with wall_limit(started + WALL_SECONDS):
                verify_config(config, args)
                cache_meta = args.output / 'val_score_cache.json'
                cache_path = args.output / 'val_scores.pt'
                if cache_meta.is_file():
                    provenance = read(cache_meta)
                    if (provenance['frozen_config_sha256'] != file_hash(args.output / 'frozen_config.json')
                            or provenance['val_scores_sha256'] != file_hash(cache_path)):
                        raise ValueError('Merged val score cache provenance changed')
                    saved = load(cache_path)
                    if saved['circuits'] != config['val_ids']:
                        raise ValueError('Merged val score cache identity changed')
                else:
                    saved = collect_val_scores(config)
                    atomic_save(saved, cache_path)
                    write_json(cache_meta, {'frozen_config_sha256': file_hash(args.output / 'frozen_config.json'),
                                           'val_scores_sha256': file_hash(cache_path)})
                thresholds = {}
                for method in METHODS:
                    value = select_threshold(saved['scores'][method], saved['labels'])
                    if value is None:
                        raise ValueError('Full val lacks both classes; cannot establish final threshold; no test fallback')
                    thresholds[method] = {'threshold': value,
                                          'score_kind': 'original float64 cosine' if method == COS else 'logit',
                                          'selected_on': 'full val1663 only', 'rule': config['threshold_rule']}
                methods = {k: equivalence_summary(saved['scores'][k], saved['labels'], saved['offsets'],
                           saved['circuits'], thresholds[k]['threshold'], 'val', thresholds[k]['score_kind']) for k in METHODS}
                write_json(args.output / 'final_val_thresholds.json', thresholds)
                report = {'notice': NOTICE, 'complete_val_calibration': True, 'circuit_count': 1663,
                          'val_partition_counts': config['val_partition_counts'], 'thresholds': thresholds,
                          'thresholds_selected_from_merged_raw_scores': True, 'test_labels_read': False,
                          'encoder_forward_count': 0, 'methods': methods,
                          'frozen_config_path': config['frozen_config_path'],
                          'frozen_config_sha256': file_hash(args.output / 'frozen_config.json'),
                          'final_thresholds_sha256': file_hash(args.output / 'final_val_thresholds.json'),
                          'val_scores_sha256': file_hash(args.output / 'val_scores.pt')}
                for k, result in methods.items():
                    print_equivalence('VAL calibration', k, result)
                write_json(args.output / 'calibration_val1663.json', report)
                record.update(status='completed', complete_val_calibration=True)
        except WallLimit:
            final_calibration = args.output / 'calibration_val1663.json'
            complete = final_calibration.is_file() and read(final_calibration).get('complete_val_calibration') is True
            record.update(status='completed' if complete else 'incomplete_timeout', complete_val_calibration=complete)
            info('完整校准已保存；test尚未运行。' if complete else
                 '校准未完成；保留合并分数缓存，用同一校准命令继续；不运行test，不把部分val标为完整校准')
        record['elapsed_wall_seconds'] = time.monotonic()-started
        write_json(args.output / 'calibration_run.json', record)
        info(f"Calibration status={record['status']}; test not started")


@torch.inference_mode()
def extract_test_cache(reader, encoder, name, contract, device):
    item = reader.item(name)
    valid = item['hf_tt_mask'].bool() & torch.isfinite(item['hf_tt_target'])
    pairs = item['hf_pairs'][:, valid].cpu().clone()
    target = item['hf_tt_target'][valid]
    labels = torch.isclose(target, torch.zeros_like(target), atol=1e-8, rtol=0).cpu()
    columns = torch.nonzero(valid).flatten().cpu()
    item = move(item, device)
    output = model_outputs(encoder, item, include_behavior=True)  # exactly one original model forward
    sums, counts, _, legacy_labels = legacy.collect_legacy(output, item)
    if not torch.equal(labels, legacy_labels):
        raise ValueError(f'Legacy/equivalence valid pair label mapping differs: {name}')
    hf = output['hf'].detach().cpu()
    if not bool(torch.isfinite(hf).all()):
        raise ValueError(f'Nonfinite frozen B hf: {name}')
    return {'hf_raw': hf, 'hf_l2': F.normalize(hf, p=2, dim=-1, eps=1e-12),
            'pairs': pairs, 'labels': labels, 'original_pair_columns': columns,
            'cosine_scores64': cosine_scores64(hf, pairs, torch.ones(labels.numel(), dtype=torch.bool), reuse.CHUNK),
            'legacy_sums': sums, 'legacy_counts': counts, 'test_cache_contract': contract,
            'metadata': {'circuit': name, 'split': 'test', 'node_count': hf.shape[0],
                         'original_valid_pairs': labels.numel(), 'encoder_forward_count': 1,
                         'include_behavior': True, 'original_tasks_source': 'frozen B and its unchanged original task heads'}}


@torch.inference_mode()
def score_test_cache(cache, head, device, plan_hash):
    scored = reuse.score_cache(cache, {HEAD: head}, device, plan_hash)
    scored['scores'][COS] = scored['scores'].pop('B_cosine')
    scored.update(legacy_sums=cache['legacy_sums'], legacy_counts=cache['legacy_counts'])
    return scored


def test_plan(args, config):
    path = args.output / 'test_plan.json'
    threshold_path = args.output / 'final_val_thresholds.json'
    calibration_path = args.output / 'calibration_val1663.json'
    calibration = read(calibration_path)
    if (not calibration['complete_val_calibration'] or calibration['circuit_count'] != 1663
            or calibration['frozen_config_sha256'] != file_hash(args.output / 'frozen_config.json')
            or calibration['final_thresholds_sha256'] != file_hash(threshold_path)):
        raise ValueError('Complete frozen full-val calibration required before any test labels/inference')
    plan = {'frozen_config_sha256': file_hash(args.output / 'frozen_config.json'),
            'final_thresholds_sha256': file_hash(threshold_path), 'calibration_sha256': file_hash(calibration_path),
            'thresholds': read(threshold_path), 'test_ids': config['test_ids'],
            'test_cache_contract': {'encoder_sha256': config['encoder_sha256'],
                'functional_net_identity': False, 'input_files': config['input_files'], 'device': args.device,
                'include_behavior': True, 'legacy_metric_implementation_sha256': file_hash(Path(legacy.__file__)),
                'feature': config['feature']}, 'wall_seconds': WALL_SECONDS,
            'test_threshold_search': False, 'training': False}
    if path.is_file():
        if read(path) != plan:
            raise ValueError('Frozen test plan or final thresholds changed; no test restart with new configuration')
    else:
        write_json(path, plan)
    return plan


def run_test(args):
    started = time.monotonic()
    config = read(args.output / 'frozen_config.json')
    verify_config(config, args)
    plan = test_plan(args, config)  # this checks val calibration, before any test data reader is opened
    plan_hash = reuse.digest_json(plan)
    final = args.output / 'test_complete.json'
    if final.is_file():
        if read(final).get('plan_sha256') != plan_hash or not read(final).get('complete_test'):
            raise ValueError('Existing test report differs')
        info('Final test already complete; reuse saved result, no additional test pass')
        return
    hf_dir, score_dir = args.output / 'test_hf_cache', args.output / 'test_score_cache'
    hf_dir.mkdir(exist_ok=True)
    score_dir.mkdir(exist_ok=True)
    names = config['test_ids']
    record = {'status': 'running_test', 'notice': NOTICE, 'plan_sha256': plan_hash,
              'requested_circuits': len(names), 'completed_score_circuits': 0, 'complete_test': False,
              'encoder_forwards_this_attempt': 0, 'reused_hf_this_attempt': 0, 'reused_scores_this_attempt': 0,
              'test_thresholds_fixed_from_val': True, 'test_threshold_search': False, 'training': False,
              'original_tasks_source': 'same frozen B checkpoint and its original task heads', 'wall_seconds': WALL_SECONDS}
    manifest_path = args.output / 'test_cache_manifest.json'
    manifest = {r['circuit']: r for r in read(manifest_path)} if manifest_path.is_file() else {}
    by_name = {}
    with direct_log(args.output / 'test_console.log'), ExitStack() as stack:
        write_json(args.output / 'test_run.json', record)
        try:
            with wall_limit(started + WALL_SECONDS):
                head = original.make_head(config['head_config']['hidden'])
                head.load_state_dict(load(config['head_checkpoint'])['head'], strict=True)
                head.to(args.device).eval().requires_grad_(False)
                graph = original.NpzSlices(Path(config['data']) / 'graphs.npz')
                stack.callback(graph.close)
                archive = {str(n): i for i, n in enumerate(graph.read('design_ids'))}
                reader, encoder = None, None
                set_seed(original.SEED)
                for index, name in enumerate(sorted(names, key=archive.__getitem__), 1):
                    key = reuse.cache_key(name)
                    hf_path, score_path = hf_dir / f'{key}.pt', score_dir / f'{key}.pt'
                    if score_path.is_file():
                        scored = load(score_path)
                        if scored['plan_sha256'] != plan_hash or scored['metadata']['circuit'] != name:
                            raise ValueError(f'Saved test score provenance differs: {name}')
                        record['reused_scores_this_attempt'] += 1
                    else:
                        if hf_path.is_file():
                            cache = load(hf_path)
                            if cache.get('test_cache_contract') != plan['test_cache_contract']:
                                raise ValueError(f'Saved test hf/task cache contract differs: {name}')
                            reuse.check_cache(cache, name, config['cache_contract'], False)
                            if any(k not in cache for k in ('legacy_sums', 'legacy_counts')):
                                raise ValueError('Missing cached original-task statistics; no duplicate encoder inference')
                            record['reused_hf_this_attempt'] += 1
                        else:
                            if reader is None:
                                label = original.NpzSlices(Path(config['data']) / 'labels.npz')
                                stack.callback(label.close)
                                state = load(config['encoder_checkpoint'])
                                reader = original.SelectedReader(graph, label, state)
                                encoder, _ = load_model(Path(config['encoder_checkpoint']), reader, 'cpu')
                                original.freeze_encoder(encoder)
                                encoder.to(args.device)
                            cache = extract_test_cache(reader, encoder, name, plan['test_cache_contract'], args.device)
                            atomic_save(cache, hf_path)
                            record['encoder_forwards_this_attempt'] += 1
                        manifest[name] = {'circuit': name, 'cache_file': str(hf_path.resolve()),
                                          'original_valid_pairs': cache['labels'].numel()}
                        write_json(manifest_path, list(manifest.values()))
                        scored = score_test_cache(cache, head, args.device, plan_hash)
                        atomic_save(scored, score_path)
                        del cache
                    if set(scored['scores']) != set(METHODS) or any(k not in scored for k in ('legacy_sums', 'legacy_counts')):
                        raise ValueError(f'Incomplete test score/task cache: {name}')
                    valid_vectors(tuple(scored['scores'].values()), scored['labels'], name)
                    by_name[name] = scored
                    record['completed_score_circuits'] = len(by_name)
                    write_json(args.output / 'test_run.json', record)
                    info(f'Test cached {index}/{len(names)}: {name} | valid pairs={scored["labels"].numel()}')
                offsets = [0]
                sums, counts, task_rows = {}, {}, []
                for name in names:
                    row = by_name[name]
                    offsets.append(offsets[-1] + row['labels'].numel())
                    for k, count in row['legacy_counts'].items():
                        counts[k] = counts.get(k, 0) + count
                        sums[k] = sums.get(k, 0.) + row['legacy_sums'].get(k, 0.)
                    task_rows.append({'circuit': name, 'node_count': row['metadata']['node_count'],
                                      **legacy.legacy_results(row['legacy_sums'], row['legacy_counts'])})
                y = torch.cat([by_name[n]['labels'] for n in names])
                methods = {}
                for k in METHODS:
                    scores = torch.cat([by_name[n]['scores'][k] for n in names])
                    methods[k] = equivalence_summary(scores, y, offsets, names, plan['thresholds'][k]['threshold'],
                                                      'test', plan['thresholds'][k]['score_kind'])
                tasks = legacy.legacy_results(sums, counts)
                info('以下原任务指标均来自冻结B编码器及原任务头；原TEST equivalence行仅指B原余弦识别。')
                legacy.print_summary({**tasks, 'functional_equivalence': methods[COS]['at_fixed_val_threshold']})
                for k, result in methods.items():
                    print_equivalence('TEST fixed-val threshold', k, result)
                report = {'notice': NOTICE, 'complete_test': True, 'plan_sha256': plan_hash,
                          'requested_circuits': len(names), 'completed_circuits': len(by_name), 'test_ids': names,
                          'zero_valid_pair_circuits': sum(a == b for a, b in zip(offsets, offsets[1:])),
                          'thresholds': plan['thresholds'], 'threshold_source': 'full val1663; no test threshold search',
                          'test_threshold_search': False, 'test_checkpoint_selection': False, 'training': False,
                          'equivalence_methods': methods, 'original_B_tasks': tasks, 'original_B_tasks_per_circuit': task_rows,
                          'frozen_config_path': config['frozen_config_path'],
                          'frozen_config_sha256': plan['frozen_config_sha256'],
                          'calibration_sha256': plan['calibration_sha256'],
                          'final_thresholds_sha256': plan['final_thresholds_sha256']}
                verify_config(config, args)
                write_json(final, report)  # only complete original test split, never partial metrics
                record.update(status='completed', complete_test=True)
                info(f'Complete fixed-candidate test saved: {final}; 本轮结束，不追加训练或调参。')
        except WallLimit:
            complete = final.is_file() and read(final).get('complete_test') is True
            record.update(status='completed' if complete else 'incomplete_timeout', complete_test=complete,
                          stop_reason='30 minute hard wall limit')
            info('test已完整保存。' if complete else 'test未完成；缓存及进度保留，用同一test命令继续；不能作为完整test结果。')
        record.update(elapsed_wall_seconds=time.monotonic()-started,
                      completed_score_cache_files=sum((score_dir/f'{reuse.cache_key(n)}.pt').is_file() for n in names))
        write_json(args.output / 'test_run.json', record)
        info(f"status={record['status']} | complete_test={record['complete_test']} | cached={record['completed_score_cache_files']}/{len(names)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--calibrate', action='store_true')
    stage.add_argument('--test', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args)
    else:
        run_test(args)


if __name__ == '__main__':
    main()
