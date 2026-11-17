"""Manual fixed-candidate validation on original val minus the used160 circuits."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import expanded_frozen_head_validation as reuse
from examples import frozen_equivalence_head as original
from examples.compare_functional_stages import file_hash
from examples.continue_circuit_balanced_head import WallLimit, wall_limit
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.inference import load_model
from triwave_gnn.utils.logger import info
from triwave_gnn.utils.net_identity import checkpoint_net_identity
from triwave_gnn.utils.random_seed import set_seed

EXP = original.EXP
EXTENDED = EXP / 'frozen_heads_extended_val128_20261113'
CANDIDATE = EXP / 'frozen_head_train512_budget_continue_20261113'
TRAIN_CACHE = EXP / 'frozen_head_train512_cache_20261113'
COS = 'B_cosine'
OLD = 'independent_head_train64_hard_negative_step5000'
NEW = 'independent_head_train512_hard_negative_step33438'
METHODS = (COS, OLD, NEW)
WALL_SECONDS = 1800
NOTICE = ('原val排除已用于分类头开发的32+128电路后的剩余1503电路；'
          '尚未参与分类头开发，不是最终test，也不称整个模型完全未见过的数据；'
          '余弦识别和独立分类头分别报告，不代表原余弦或概率预测指标得到改善')


def read(path):
    return json.loads(Path(path).read_text())


def remaining_ids(splits, dev, expanded):
    val = splits['val']
    overlap = sorted(set(dev) & set(expanded))
    outside = sorted((set(dev) | set(expanded)) - set(val))
    remaining = sorted(set(val) - set(dev) - set(expanded))
    counts = {'original_val': len(val), 'unique_val': len(set(val)),
              'development32': len(dev), 'unique_development32': len(set(dev)),
              'expanded128': len(expanded), 'unique_expanded128': len(set(expanded)),
              'used_overlap_ids': overlap, 'used_ids_outside_val': outside,
              'remaining': len(remaining), 'expected_val': 1663, 'expected_remaining': 1503}
    if (len(val) != 1663 or len(set(val)) != 1663 or len(dev) != 32 or len(set(dev)) != 32
            or len(expanded) != 128 or len(set(expanded)) != 128 or overlap or outside
            or len(remaining) != 1503):
        raise ValueError('验证范围不符；停止，不静默改变范围：' + json.dumps(counts, ensure_ascii=False))
    if set(remaining) & (set(splits['train']) | set(splits['test'])):
        raise ValueError('Remaining val IDs overlap train/test; stopping')
    return remaining, counts


def make_plan(args):
    prior = read(EXTENDED / 'plan.json')
    dev_selection = read(reuse.DEV / 'selection.json')
    dev = [r['circuit'] for r in dev_selection['circuits'] if r['split'] == 'val']
    expanded = prior['selected_ids']
    names, counts = remaining_ids(read(original.B_RUN / 'split.json'), dev, expanded)
    thresholds_new = read(CANDIDATE / 'thresholds.json')
    calibration = read(CANDIDATE / 'development32.json')['calibration']
    candidate_run = read(CANDIDATE / 'run.json')
    candidate_result = read(CANDIDATE / 'comparison.json')
    candidate_method = candidate_result['methods']['hard_negative_head_train512_final_step33438']
    if (thresholds_new['main_threshold'] != 0.5088928937911987
            or thresholds_new['old_low_fp_threshold'] != 1.4913545846939087
            or calibration['maximum_f1']['threshold'] != thresholds_new['main_threshold']
            or calibration['fp_le_69']['threshold'] != thresholds_new['old_low_fp_threshold']
            or candidate_method['at_old_dev_max_f1_threshold']['threshold'] != thresholds_new['main_threshold']
            or candidate_method['at_old_low_fp_threshold']['threshold'] != thresholds_new['old_low_fp_threshold']):
        raise ValueError('保存阈值与指定数值不一致；停止，不重新选择阈值')
    if (candidate_run['status'] != 'completed' or candidate_run['completed_total_step'] != 33438
            or candidate_result['completed_total_step'] != 33438
            or not candidate_result['complete_128_circuit_result']):
        raise ValueError('Require completed final candidate step33438')
    source512 = Path(candidate_run['source_run'])
    source512_run = read(source512 / 'run.json')
    if (Path(source512_run['source_run']).resolve() != TRAIN_CACHE.resolve()
            or candidate_run['source_checkpoint_sha256'] != file_hash(source512 / 'last.pt')):
        raise ValueError('Candidate does not match the saved512 training/cache lineage')
    cache_plan = read(TRAIN_CACHE / 'plan.json')
    contract = prior['cache_contract']
    if cache_plan['cache_contract'] != contract or args.device != contract['device']:
        raise ValueError('Candidate B encoder/cache/device contract differs')
    if reuse.input_stats(Path(prior['data'])) != contract['input_files']:
        raise ValueError('Input size/mtime changed; no full bundle audit attempted')
    checkpoint = Path(prior['encoder_checkpoint'])
    b_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (file_hash(checkpoint) != contract['encoder_checkpoint_sha256'] or b_state['epoch'] != 5
            or checkpoint_net_identity(b_state) or contract['functional_net_identity'] is not False):
        raise ValueError('Require exactly frozen B epoch5 and original propagation')
    # Verify the existing provenance without reading any raw graph/label arrays.
    for path, expected in cache_plan['locked_files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Encoder/head/cache provenance changed: {path}')
    old_path = Path(prior['head_checkpoints']['hard_negative_head_step5000'])
    if file_hash(old_path) != prior['locked_files'][str(old_path)]:
        raise ValueError('Original64 head checkpoint changed')
    heads = {OLD: old_path, NEW: CANDIDATE / 'final_head.pt'}
    head_config = {}
    for name, path in heads.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        step = 5000 if name == OLD else 33438
        if state['optimizer_steps'] != step or state['hidden'] != contract['hidden']:
            raise ValueError(f'Unexpected exact checkpoint step/width: {path}')
        if name == NEW:
            last = torch.load(CANDIDATE / 'last.pt', map_location='cpu', weights_only=False)
            if (last['optimizer_steps'] != 33438 or state['objective'] != candidate_run['objective']
                    or Path(state['source_run']).resolve() != source512.resolve()
                    or state['head'].keys() != last['head'].keys()
                    or any(not torch.equal(value, last['head'][key]) for key, value in state['head'].items())):
                raise ValueError('Explicit final33438 checkpoint differs from the completed final update')
        head_config[name] = {'checkpoint': str(path.resolve()), 'sha256': file_hash(path),
                             'actual_step': step, 'architecture': '2*hidden ->64 ReLU ->1 logits',
                             'feature': 'concat(abs(L2(hf_u)-L2(hf_v)),L2(hf_u)*L2(hf_v)); hf only',
                             'hidden': state['hidden']}
    # Thresholds for each method were already determined on the original32 only.
    thresholds = {COS: prior['thresholds'][COS], OLD: prior['thresholds']['hard_negative_head_step5000'],
                  NEW: {**thresholds_new, 'main_source': str(CANDIDATE / 'thresholds.json'),
                        'low_fp_source': str(CANDIDATE / 'thresholds.json')}}
    old_calibration = read(reuse.HARD5000 / 'comparison.json')['methods']['hard_negative_final']
    cos_calibration = read(reuse.DEV / 'comparison.json')['cosine_baseline']
    cos_low = read(reuse.STEP500 / 'comparison.json')['methods']['cosine_baseline']['fp_le_69']['threshold']
    if (thresholds[COS]['main_threshold'] != cos_calibration['threshold']
            or thresholds[COS]['old_low_fp_threshold'] != cos_low
            or thresholds[OLD]['main_threshold'] != old_calibration['maximum_f1']['threshold']
            or thresholds[OLD]['old_low_fp_threshold'] != old_calibration['fp_le_69']['threshold']):
        raise ValueError('Existing baseline threshold provenance changed')
    if any(not isinstance(r[k], (float, int)) or not math.isfinite(r[k])
           for r in thresholds.values() for k in ('main_threshold', 'old_low_fp_threshold')):
        raise ValueError('Missing/nonfinite fixed threshold; no calibration allowed')
    locked = [Path(__file__), Path(reuse.__file__), Path(original.__file__), checkpoint, *heads.values(),
              original.B_RUN / 'split.json', reuse.DEV / 'selection.json', EXTENDED / 'plan.json',
              EXTENDED / 'comparison.json', CANDIDATE / 'run.json', CANDIDATE / 'comparison.json',
              CANDIDATE / 'thresholds.json', CANDIDATE / 'development32.json',
              CANDIDATE / 'last.pt', source512 / 'run.json', source512 / 'last.pt',
              reuse.HARD5000 / 'comparison.json', reuse.DEV / 'comparison.json', reuse.STEP500 / 'comparison.json',
              reuse.DEV / 'validation_scores.pt', reuse.HARD5000 / 'validation_scores.pt',
              CANDIDATE / 'development32_scores.pt']
    return {'notice': NOTICE, 'requested_circuits': len(names), 'selected_ids': names,
            'excluded_development32_ids': dev, 'excluded_expanded128_ids': expanded, 'scope_counts': counts,
            'selection_rule': 'all original val IDs minus exactly the used32 and128 IDs; no filtering/replacements',
            'data': prior['data'], 'device': args.device, 'wall_seconds': WALL_SECONDS,
            'encoder_checkpoint': str(checkpoint), 'encoder_config': b_state['config'],
            'encoder_sha256': file_hash(checkpoint), 'encoder_epoch': 5, 'functional_net_identity': False,
            'head_checkpoints': {k: str(v.resolve()) for k, v in heads.items()},
            'head_config': head_config, 'candidate_training_config': candidate_run,
            'cache_contract': contract, 'thresholds': thresholds,
            'locked_files': {str(p.resolve()): file_hash(p) for p in locked},
            'external_cache_manifests': [str(p.resolve()) for p in
                [reuse.DEV / 'cache_manifest.json', EXTENDED / 'cache_manifest.json',
                 TRAIN_CACHE / 'cache_manifest.json', *args.cache_manifest]],
            'training': False, 'test_evaluated': False, 'thresholds_reselected': False,
            'full_data_audit': False, 'probability_prediction_evaluated': False}


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    plan = make_plan(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'selection.json', {k: plan[k] for k in
               ('selected_ids', 'excluded_development32_ids', 'excluded_expanded128_ids', 'scope_counts', 'selection_rule')})
    write_json(args.output / 'thresholds.json', plan['thresholds'])
    write_json(args.output / 'run.json', {'status': 'prepared', 'complete_remaining_validation': False,
                                        'requested_circuits': len(plan['selected_ids'])})
    info(f"Prepared exact remaining scope: {plan['scope_counts']}; no inference")
    return plan


def print_methods(scope, methods):
    info(f'SEPARATE SCOPE: {scope}; fixed original32 thresholds, no reselection')
    for method, result in methods.items():
        reuse.print_result(method, result)
        for group, row in [('pooled', result), *result['groups_by_original_valid_pairs'].items()]:
            main, low = row['at_old_dev_max_f1_threshold'], row['at_old_low_fp_threshold']
            info(f"{method} / {group}: main actual FPR={main['false_positive_rate']}; "
                 f"low threshold={low['threshold']} P/R/F1={low['precision']}/{low['recall']}/{low['f1']} | "
                 f"TP/FP/FN={low['tp']}/{low['fp']}/{low['fn']} | actual FPR={low['false_positive_rate']}")


def used160_report(plan, output, plan_hash):
    path = output / 'used_development160.json'
    if path.is_file():
        report = read(path)
        if report.get('plan_sha256') != plan_hash:
            raise ValueError('Existing used160 report differs')
        return report
    base = torch.load(reuse.DEV / 'validation_scores.pt', map_location='cpu', weights_only=False)
    old = torch.load(reuse.HARD5000 / 'validation_scores.pt', map_location='cpu', weights_only=False)
    new = torch.load(CANDIDATE / 'development32_scores.pt', map_location='cpu', weights_only=False)
    for saved in (old, new):
        if (saved['circuits'] != base['circuits'] or saved['offsets'] != base['offsets']
                or not torch.equal(saved['labels'], base['labels'])):
            raise ValueError('Saved used32 score mapping differs; no inference fallback')
    if set(base['circuits']) != set(plan['excluded_development32_ids']) or old['final_total_step'] != 5000:
        raise ValueError('Used32 saved identity/old head step differs')
    names = base['circuits'] + plan['excluded_expanded128_ids']
    offsets = list(base['offsets'])
    labels = [base['labels']]
    parts = {COS: [base['cosine_scores64']], OLD: [old['hard_negative_final_logits']], NEW: [new['head_logits']]}
    ext_hash = reuse.digest_json(read(EXTENDED / 'plan.json'))
    for name in plan['excluded_expanded128_ids']:
        key = reuse.cache_key(name)
        saved = torch.load(EXTENDED / 'score_cache' / f'{key}.pt', map_location='cpu', weights_only=False)
        current = torch.load(CANDIDATE / 'expanded_scores' / f'{key}.pt', map_location='cpu', weights_only=False)
        if (saved['metadata']['circuit'] != name or current['circuit'] != name
                or saved['plan_sha256'] != ext_hash or not torch.equal(saved['labels'], current['labels'])
                or not torch.equal(saved['original_pair_columns'], current['original_pair_columns'])):
            raise ValueError('Saved used128 pair/label/score provenance differs; no inference fallback')
        labels.append(saved['labels'])
        offsets.append(offsets[-1] + saved['labels'].numel())
        parts[COS].append(saved['scores'][COS])
        parts[OLD].append(saved['scores']['hard_negative_head_step5000'])
        parts[NEW].append(current['head_logits'])
    y = torch.cat(labels)
    score_vectors = {k: torch.cat(v) for k, v in parts.items()}
    methods = {k: reuse.summarize(s, y, offsets, names, plan['thresholds'][k]) for k, s in score_vectors.items()}
    subsets = {}
    split_pair = offsets[32]
    for tag, a, b, pair_a, pair_b in [('calibration32', 0, 32, 0, split_pair),
                                      ('used_expanded128', 32, 160, split_pair, y.numel())]:
        off = [v-pair_a for v in offsets[a:b+1]]
        subsets[tag] = {k: reuse.summarize(s[pair_a:pair_b], y[pair_a:pair_b], off,
                                          names[a:b], plan['thresholds'][k]) for k, s in score_vectors.items()}
    report = {'notice': '已用于分类头开发的160电路；仅读取保存的逐对分数，不重复模型推理；与剩余1503分开报告',
              'plan_sha256': plan_hash, 'circuit_count': len(names), 'selected_ids': names,
              'methods': methods, 'subsets': subsets, 'thresholds_reselected': False, 'test_evaluated': False}
    write_json(path, report)
    return report


def run(args):
    started = time.monotonic()
    plan_path = args.output / 'plan.json'
    plan = read(plan_path) if plan_path.is_file() else prepare(args)
    names = plan['selected_ids']
    current, _ = remaining_ids(read(original.B_RUN / 'split.json'),
                               plan['excluded_development32_ids'], plan['excluded_expanded128_ids'])
    if current != names or args.device != plan['device']:
        raise ValueError('Prepared remaining scope/device changed')
    for path, expected in plan['locked_files'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError(f'Prepared checkpoint/config/calibration/code changed: {path}')
    if reuse.input_stats(Path(plan['data'])) != plan['cache_contract']['input_files']:
        raise ValueError('Prepared input metadata changed; no full-data audit')
    plan_hash = reuse.digest_json(plan)
    final = args.output / 'comparison_remaining1503.json'
    if final.is_file():
        completed = read(final)
        if completed.get('plan_sha256') != plan_hash or not completed.get('complete_remaining_validation'):
            raise ValueError('Existing remaining report differs or incomplete')
        info(f'Already completed all1503; reusing without inference: {final}')
        return
    hf_dir, score_dir = args.output / 'hf_cache', args.output / 'score_cache'
    hf_dir.mkdir(exist_ok=True)
    score_dir.mkdir(exist_ok=True)
    manifest_path = args.output / 'cache_manifest.json'
    manifest = {r['circuit']: r for r in read(manifest_path)} if manifest_path.is_file() else {}
    record = {'status': 'running', 'notice': NOTICE, 'plan_sha256': plan_hash, 'requested_circuits': len(names),
              'completed_score_circuits': 0, 'complete_remaining_validation': False,
              'encoder_forwards_this_attempt': 0, 'reused_hf_this_attempt': 0, 'reused_scores_this_attempt': 0,
              'wall_seconds': WALL_SECONDS, 'thresholds_reselected': False, 'training': False,
              'test_evaluated': False, 'probability_prediction_evaluated': False}
    scored_by_name = {}
    with direct_log(args.output / 'console.log'), ExitStack() as stack:
        info(NOTICE)
        write_json(args.output / 'run.json', record)
        try:
            with wall_limit(started + WALL_SECONDS):
                used = used160_report(plan, args.output, plan_hash)
                print_methods('already-used development160; NOT remaining validation', used['methods'])
                external = reuse.external_cache_index(plan)
                heads = {}
                for method, path in plan['head_checkpoints'].items():
                    state = torch.load(path, map_location='cpu', weights_only=False)
                    head = original.make_head(plan['cache_contract']['hidden'])
                    head.load_state_dict(state['head'], strict=True)
                    heads[method] = head.to(args.device).eval().requires_grad_(False)
                graph = original.NpzSlices(Path(plan['data']) / 'graphs.npz')
                stack.callback(graph.close)
                archive_order = {str(n): i for i, n in enumerate(graph.read('design_ids'))}
                reader, encoder = None, None
                set_seed(original.SEED)
                # Archive ordering only optimizes selected-slice I/O; it never changes membership.
                for index, name in enumerate(sorted(names, key=archive_order.__getitem__), 1):
                    key = reuse.cache_key(name)
                    hf_path, score_path = hf_dir / f'{key}.pt', score_dir / f'{key}.pt'
                    if score_path.is_file():
                        scored = torch.load(score_path, map_location='cpu', weights_only=False)
                        if scored.get('plan_sha256') != plan_hash or scored['metadata']['circuit'] != name:
                            raise ValueError(f'Saved score plan/identity differs: {name}')
                        record['reused_scores_this_attempt'] += 1
                    else:
                        source = hf_path if hf_path.is_file() else external.get(name)
                        if source is not None:
                            cache = torch.load(source, map_location='cpu', weights_only=False)
                            reuse.check_cache(cache, name, plan['cache_contract'], source == hf_path)
                            record['reused_hf_this_attempt'] += 1
                        else:
                            if reader is None:
                                label = original.NpzSlices(Path(plan['data']) / 'labels.npz')
                                stack.callback(label.close)
                                state = torch.load(plan['encoder_checkpoint'], map_location='cpu', weights_only=False)
                                reader = original.SelectedReader(graph, label, state)
                                encoder, _ = load_model(Path(plan['encoder_checkpoint']), reader, 'cpu')
                                original.freeze_encoder(encoder)
                                encoder.to(args.device)
                            cache = reuse.extract_cache(reader, encoder, name, plan['cache_contract'], args.device)
                            cache['metadata']['split'] = 'val_remaining_after_development160'
                            atomic_save(cache, hf_path)  # persist before head scoring, for interruption reuse
                            source = hf_path
                            record['encoder_forwards_this_attempt'] += 1
                        manifest[name] = {'circuit': name, 'cache_file': str(source.resolve()),
                                          'original_valid_pairs': cache['labels'].numel()}
                        write_json(manifest_path, list(manifest.values()))
                        scored = reuse.score_cache(cache, heads, args.device, plan_hash)
                        atomic_save(scored, score_path)
                        del cache
                    if set(scored['scores']) != set(METHODS) or any(v.shape != scored['labels'].shape for v in scored['scores'].values()):
                        raise ValueError(f'Saved method/label score dimensions differ: {name}')
                    scored_by_name[name] = scored
                    record['completed_score_circuits'] = len(scored_by_name)
                    write_json(args.output / 'run.json', record)
                    info(f'Remaining val {index}/{len(names)}: {name} | original valid pairs={scored["labels"].numel()}')
                offsets = [0]
                for name in names:
                    offsets.append(offsets[-1] + scored_by_name[name]['labels'].numel())
                y = torch.cat([scored_by_name[n]['labels'] for n in names])
                methods = {}
                for method in METHODS:
                    scores = torch.cat([scored_by_name[n]['scores'][method] for n in names])
                    methods[method] = reuse.summarize(scores, y, offsets, names, plan['thresholds'][method])
                report = {'notice': NOTICE, 'plan_sha256': plan_hash, 'complete_remaining_validation': True,
                          'requested_circuits': len(names), 'completed_circuits': len(scored_by_name),
                          'selected_ids': names, 'scope_counts': plan['scope_counts'],
                          'zero_valid_pair_circuits': sum(a == b for a, b in zip(offsets, offsets[1:])),
                          'methods': methods, 'thresholds': plan['thresholds'], 'thresholds_reselected': False,
                          'test_evaluated': False, 'training': False, 'probability_prediction_evaluated': False,
                          'used160_report': str((args.output / 'used_development160.json').resolve()),
                          'mixed_used160_and_remaining_result_generated': False,
                          'low_fp_note': 'Fixed original32 low-FP thresholds; actual remaining-subset FP need not be<=69'}
                print_methods('remaining1503 only; excluded all used160', methods)
                write_json(final, report)  # only after all1503 and every method/group metric completes
                record.update(status='completed', complete_remaining_validation=True,
                              zero_valid_pair_circuits=report['zero_valid_pair_circuits'])
                info(f'Completed all1503 remaining validation circuits: {final}')
        except WallLimit:
            # A report is committed only when every requested circuit and metric is complete.
            complete = final.is_file() and read(final).get('complete_remaining_validation') is True
            record.update(status='completed' if complete else 'incomplete_timeout',
                          complete_remaining_validation=complete, stop_reason='30 minute hard wall limit')
            info('30分钟上限：已完成结果可复用。' if complete else
                 '未完成：已保存电路缓存和进度，不能作为完整1503电路结果；使用同一命令复用缓存继续。')
        record.update(elapsed_wall_seconds=time.monotonic()-started,
                      completed_score_cache_files=sum((score_dir/f'{reuse.cache_key(n)}.pt').is_file() for n in names))
        write_json(args.output / 'run.json', record)
        info(f"status={record['status']} | complete_remaining={record['complete_remaining_validation']} | "
             f"cached={record['completed_score_cache_files']}/{len(names)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
