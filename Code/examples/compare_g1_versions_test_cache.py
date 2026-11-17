"""Cache-only test comparison for fixed step33438, G0 and G1 candidates."""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import expanded_frozen_head_validation as extended
from examples import final_frozen_head_evaluation as final
from examples import frozen_equivalence_head as original
from examples.continue_circuit_balanced_head import WallLimit, wall_limit
from examples.compare_functional_stages import file_hash
from examples.finetune_functional_pairs import direct_log, write_json
from triwave_gnn.utils.checkpoint import atomic_save
from triwave_gnn.utils.logger import info

EXP = original.EXP
TEST_RUN = EXP / 'final_B_head33438_20261113'
BASELINE_TEST = TEST_RUN / 'test_complete.json'
G0_RUN = EXP / 'frozen_head_dense_sampling_G0_20261113'
G1_RUN = EXP / 'frozen_head_dense_sampling_G1_20261113'
NOTICE = ('既有test划分上的固定版本对照；不是新收集、从未查看过的独立测试集；'
          'step33438、G0、G1均使用各自完整val最大F1阈值；不在test重新选阈值，不训练，不运行编码器')
METHODS = ('step33438', 'G0', 'G1')
WALL = 1800


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def check_run(path, branch, expected_step):
    run = read(path / 'run.json')
    comp = read(path / 'comparison.json')
    if run.get('status') != 'completed' or run.get('completed_additional_steps') != 10000:
        raise ValueError(f'{branch}:续训未完整完成10000步')
    if comp.get('complete_val_evaluation') is not True or comp.get('test_evaluated'):
        raise ValueError(f'{branch}:缺少完整val结果或误读test')
    method = comp['methods'][branch]
    state = load(path / 'final_head.pt')
    if state['optimizer_steps'] != expected_step:
        raise ValueError(f'{branch}:检查点步数不符')
    threshold = method['maximum_f1']['at_fixed_val_threshold']['threshold']
    return {'run': run, 'comparison': comp, 'state': state,
            'checkpoint': str((path / 'final_head.pt').resolve()),
            'checkpoint_sha256': file_hash(path / 'final_head.pt'),
            'threshold': threshold, 'threshold_source': str((path / 'comparison.json').resolve())}


def check_test_cache(config, manifest):
    rows = read(manifest)
    if len(rows) != len(config['test_ids']) or {r['circuit'] for r in rows} != set(config['test_ids']):
        raise ValueError('Existing test cache manifest does not cover the exact test split')
    by_name = {r['circuit']: r for r in rows}
    rows = [by_name[name] for name in config['test_ids']]
    first = load(rows[0]['cache_file'])
    contract = first['test_cache_contract']
    if (contract['encoder_sha256'] != config['encoder_sha256']
            or contract['functional_net_identity'] is not False
            or contract['input_files'] != config['input_files']
            or contract['device'] != config['device']):
        raise ValueError('Existing test feature cache is not compatible with frozen B')
    for row in rows:
        path = Path(row['cache_file'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing existing test hf cache: {path}')
    return rows, contract


def score_head(cache, head, device):
    chunks = []
    with torch.inference_mode():
        head.eval()
        for pairs in cache['pairs'].split(4096, dim=1):
            left, right = pairs
            features = original.pair_features(cache['hf_l2'][left], cache['hf_l2'][right])
            values = head(features.to(device)).squeeze(-1).cpu()
            if not bool(torch.isfinite(values).all()):
                raise ValueError('Nonfinite cached-head logits')
            chunks.append(values)
    return torch.cat(chunks) if chunks else torch.empty(0)


def run(args):
    started = time.monotonic()
    config = read(TEST_RUN / 'frozen_config.json')
    if args.device != config['device']:
        raise ValueError('Use frozen test cache device')
    baseline = read(BASELINE_TEST)
    if not baseline.get('complete_test') or baseline.get('test_threshold_search'):
        raise ValueError('Existing step33438 test result is incomplete or was selected on test')
    # Existing full-val calibration remains the source of the step33438 threshold.
    val_thresholds = read(TEST_RUN / 'final_val_thresholds.json')
    step_threshold = val_thresholds['B_plus_512_head_equivalence']['threshold']
    old = baseline['equivalence_methods']['B_plus_512_head_equivalence']
    if old['at_fixed_val_threshold']['threshold'] != step_threshold:
        raise ValueError('Existing step33438 test result/val threshold mismatch')
    g0 = check_run(G0_RUN, 'G0', 43438)
    g1 = check_run(G1_RUN, 'G1', 43438)
    rows, cache_contract = check_test_cache(config, TEST_RUN / 'test_cache_manifest.json')
    if (set(config['test_ids']) != set(baseline['test_ids']) or len(config['test_ids']) != 1663
            or baseline['threshold_source'] != 'full val1663; no test threshold search'):
        raise ValueError('Existing test split/baseline provenance differs')
    plan = {'notice': NOTICE, 'device': args.device, 'test_ids': config['test_ids'],
            'test_cache_manifest': str((TEST_RUN / 'test_cache_manifest.json').resolve()),
            'encoder_checkpoint': config['encoder_checkpoint'], 'encoder_sha256': config['encoder_sha256'],
            'functional_net_identity': False, 'cache_contract': cache_contract,
            'methods': {
                'step33438': {'checkpoint': config['head_checkpoint'], 'checkpoint_sha256': config['head_sha256'],
                              'actual_step': 33438, 'threshold': step_threshold,
                              'threshold_source': str((TEST_RUN / 'final_val_thresholds.json').resolve())},
                'G0': {'checkpoint': g0['checkpoint'], 'checkpoint_sha256': g0['checkpoint_sha256'],
                       'actual_step': 43438, 'threshold': g0['threshold'], 'threshold_source': g0['threshold_source']},
                'G1': {'checkpoint': g1['checkpoint'], 'checkpoint_sha256': g1['checkpoint_sha256'],
                       'actual_step': 43438, 'threshold': g1['threshold'], 'threshold_source': g1['threshold_source']}},
            'threshold_rule': 'each method fixed at its saved complete-val maximum-F1 threshold; whole equal-score groups',
            'training': False, 'encoder_forwards': 0, 'test_threshold_search': False}
    if args.output.exists():
        existing_plan = args.output / 'plan.json'
        if not existing_plan.is_file() or read(existing_plan) != plan:
            raise ValueError('Existing output belongs to another frozen comparison; preserve it and choose a new directory')
        if (args.output / 'comparison_test_existing.json').is_file():
            info('Existing cache-only comparison already complete; reusing without scoring')
            return
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'plan.json', plan)
    heads = {}
    for method, info_row in plan['methods'].items():
        if method == 'step33438':
            # The old complete test report is reused directly; no head forward is needed.
            continue
        state = load(info_row['checkpoint'])
        head = original.make_head(config['head_config']['hidden'])
        head.load_state_dict(state['head'], strict=True)
        heads[method] = head.to(args.device).eval().requires_grad_(False)
    manifest_path = args.output / 'score_cache_manifest.json'
    score_dir = args.output / 'score_cache'; score_dir.mkdir(exist_ok=True)
    saved_rows = {r['circuit']: r for r in read(manifest_path)} if manifest_path.is_file() else {}
    parts = {'G0': [], 'G1': []}; labels, offsets, names = [], [0], []
    record = {'status': 'running', 'notice': NOTICE, 'requested_circuits': 1663,
              'completed_score_circuits': 0, 'complete_test_comparison': False,
              'encoder_forwards': 0, 'hf_cache_reused': 1663, 'test_threshold_search': False,
              'training': False, 'wall_seconds': WALL}
    with direct_log(args.output / 'console.log'), ExitStack() as stack:
        write_json(args.output / 'run.json', record)
        try:
            with wall_limit(started + WALL):
                for index, row in enumerate(rows, 1):
                    name = row['circuit']; cache = load(row['cache_file'])
                    extended.check_cache(cache, name, config['cache_contract'], False)
                    labels.append(cache['labels']); names.append(name)
                    for method, head in heads.items():
                        path = score_dir / f'{extended.cache_key(name)}_{method}.pt'
                        if path.is_file():
                            scores = load(path)['scores']
                        else:
                            scores = score_head(cache, head, args.device)
                            atomic_save({'circuit': name, 'scores': scores, 'labels': cache['labels'],
                                         'original_pair_columns': cache['original_pair_columns'],
                                         'checkpoint_sha256': plan['methods'][method]['checkpoint_sha256']}, path)
                        if scores.shape != cache['labels'].shape:
                            raise ValueError(f'Score/label shape mismatch: {name}/{method}')
                        parts[method].append(scores)
                    offsets.append(offsets[-1] + cache['labels'].numel())
                    saved_rows[name] = {'circuit': name, 'score_files': {m: str((score_dir / f'{extended.cache_key(name)}_{m}.pt').resolve()) for m in heads}}
                    record['completed_score_circuits'] = index
                    write_json(manifest_path, list(saved_rows.values()))
                    write_json(args.output / 'run.json', record)
                    if index % 100 == 0: info(f'Cached head scores {index}/1663: {name}')
                y = torch.cat(labels)
                results = {'step33438': old}
                for method in ('G0', 'G1'):
                    scores = torch.cat(parts[method])
                    results[method] = final.equivalence_summary(scores, y, offsets, names,
                                                               plan['methods'][method]['threshold'], 'test', 'logit')
                report = {'notice': NOTICE, 'complete_test_comparison': True,
                          'test_split': 'existing final test split; not newly collected and not unseen',
                          'test_evaluated': True, 'encoder_forwards': 0, 'hf_cache_reused': 1663,
                          'threshold_source': 'each method complete-val maximum-F1 threshold; no test search',
                          'methods': results, 'plan_sha256': extended.digest_json(plan),
                          'old_step33438_report_reused': str(BASELINE_TEST.resolve()),
                          'original_formal_test_result_preserved': True}
                for method, result in results.items():
                    final.print_equivalence('TEST existing split version comparison', method if method != 'step33438' else final.HEAD, result)
                write_json(args.output / 'comparison_test_existing.json', report)
                record.update(status='completed', complete_test_comparison=True)
                info(f'Completed cache-only version comparison: {args.output / "comparison_test_existing.json"}')
        except WallLimit:
            record.update(status='incomplete_timeout', complete_test_comparison=False,
                          stop_reason='30 minute hard wall limit')
            info('比较未完成；保留已有hf及分类头分数缓存，不能作为完整版本对照。')
        record.update(elapsed_wall_seconds=time.monotonic()-started,
                      completed_score_circuits=len(saved_rows))
        write_json(args.output / 'run.json', record)
        info(f"status={record['status']} | complete={record['complete_test_comparison']} | cached={record['completed_score_circuits']}/1663")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); run(args)


if __name__ == '__main__':
    main()
