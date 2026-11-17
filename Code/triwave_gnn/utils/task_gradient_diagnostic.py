"""Read-only, pre-optimizer task-gradient measurements for the stage-3 model."""
import math

import torch
from torch.nn import functional as F

from .losses import BEHAVIOR, STAGE_WEIGHTS, functional_distance, loss_terms, zero_normalize

TASKS = ('prob', 'rc', 'func', 'behavior')
SCOPES = ('all_common', 'shared_initialization', 'structure_path', 'function_path')
COMPARISONS = ('prob', 'rc', 'behavior', 'other_weighted_sum', 'full_multitask_total')


def module_parameter_groups(model):
    """Map module object parameter identities to their real named_parameters.

    Module membership reflects reviewed forward equations; graph dependencies
    are measured separately, never inferred from prefixes of parameter names.
    """
    init = model.initialization
    objects = {
        'shared_initialization': [init.logic_embedding, init.role_embedding, init.pin_embedding],
        'structure_path': [init.internal['s']],
        'function_path': [init.internal['f'], init.pi_function],
        'behavior_path': [init.internal['b'], init.pi_behavior],
        'task_heads': [model.readout_prob, model.readout_rc, model.readout_behavior],
    }
    for branch, scope in (('s', 'structure_path'), ('f', 'function_path'), ('b', 'behavior_path')):
        for relation in ('net', 'cell'):
            objects[scope].extend([model.aggregators[branch + '_' + relation], model.updates[branch + '_' + relation]])
    assignment = {}
    for scope, modules in objects.items():
        for module in modules:
            parameters = [module] if isinstance(module, torch.nn.Parameter) else module.parameters()
            for parameter in parameters:
                if id(parameter) in assignment:
                    raise ValueError('Parameter appears in multiple module groups')
                assignment[id(parameter)] = scope
    names, groups = [], {}
    for name, parameter in model.named_parameters():
        if id(parameter) not in assignment:
            raise ValueError(f'Unreviewed parameter: {name}')
        names.append(name)
        groups[name] = assignment[id(parameter)]
    return names, groups


def weighted_task_losses(output, item, chunk_size):
    """Mirror the unchanged four loss formulas, preserving its active divisor.

    loss_terms returns detached terms, so this diagnostic-only helper exposes
    differentiable terms. Value and gradient equivalence are tested separately.
    """
    pins = item['pin']
    raw = {'prob': F.l1_loss(output['hf_prob_logits'].sigmoid(), pins.hf_prob)}
    if bool(item['hs_mask'].any()):
        mask = item['hs_mask']
        raw['rc'] = F.binary_cross_entropy_with_logits(output['hs_rc_logits'][mask], item['hs_target'][mask])
    valid = item['hf_tt_mask'] & torch.isfinite(item['hf_tt_target'])
    if int(valid.sum()) >= 2:
        distance = functional_distance(output['hf'], item['hf_pairs'], chunk_size)[valid]
        raw['func'] = F.l1_loss(zero_normalize(distance), zero_normalize(item['hf_tt_target'][valid]))
    behavior = []
    for index, (name, mask_name, kind) in enumerate(BEHAVIOR):
        target = pins[name]
        mask = pins[mask_name] & torch.isfinite(target)
        if not bool(mask.any()):
            continue
        prediction = output['hb_raw'][:, index][mask]
        if kind == 'probability':
            value = F.binary_cross_entropy_with_logits(prediction, target[mask])
        else:
            value = F.smooth_l1_loss(prediction, torch.log1p(target[mask].clamp_min(0)))
        behavior.append(value)
    if behavior:
        raw['behavior'] = torch.stack(behavior).mean()
    weights = STAGE_WEIGHTS[3]
    divisor = sum(weights[key] for key in raw)
    weighted = {key: value * (weights[key] / divisor) for key, value in raw.items()}
    # A same-forward detached reference proves formulas on every actual batch.
    with torch.no_grad():
        original, reference = loss_terms(output, item, 3, functional_chunk_size=chunk_size)
    for key, value in raw.items():
        if not torch.isclose(value.detach(), value.new_tensor(reference[key]), rtol=1e-6, atol=1e-7):
            raise ValueError(f'Diagnostic loss differs from original: {key}')
    if not torch.isclose(sum(weighted.values()).detach(), original, rtol=1e-6, atol=1e-7):
        raise ValueError('Weighted task sum differs from original total')
    details = dict(raw={key: float(value.detach()) for key, value in raw.items()},
                   weighted={key: float(value.detach()) for key, value in weighted.items()},
                   effective_weights={key: weights[key] / divisor for key in raw},
                   active_divisor=divisor, original_total=float(original), formula_check=True)
    return weighted, details


def task_gradients(weighted, parameters, verify_total=False):
    """One forward, four VJPs; None remains distinct from a connected zero."""
    gradients = {}
    items = list(weighted.items())
    for index, (task, value) in enumerate(items):
        grads = torch.autograd.grad(value, parameters, allow_unused=True,
                                    retain_graph=verify_total or index < len(items) - 1)
        cpu = [None if g is None else g.detach().cpu().double() for g in grads]
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in cpu):
            raise FloatingPointError(f'Nonfinite gradient: {task}')
        gradients[task] = cpu
        del grads
    residual = None
    if verify_total:
        direct = torch.autograd.grad(sum(weighted.values()), parameters, allow_unused=True)
        residual = 0.
        for i, grad in enumerate(direct):
            parts = [g[i] for g in gradients.values() if g[i] is not None]
            summed = sum(parts) if parts else None
            if grad is None:
                if summed is not None:
                    raise ValueError('Task sum and original total disagree on parameter dependency')
            else:
                actual = grad.detach().cpu().double()
                torch.testing.assert_close(actual, summed, atol=3e-6, rtol=3e-4)
                residual = max(residual, float((actual - summed).abs().max()))
    return gradients, residual


def sum_gradients(gradients, keys, count):
    result = []
    for i in range(count):
        parts = [gradients[key][i] for key in keys if key in gradients and gradients[key][i] is not None]
        result.append(sum(parts) if parts else None)
    return result


def compare_gradients(left, right, indices):
    """Cosine on the intersection of autograd-connected parameter tensors.

    All elements of shared tensors are retained, including zero elements.
    Sparse overlap of active entries is not silently selected away.
    """
    common = [i for i in indices if left[i] is not None and right[i] is not None]
    a2 = math.fsum(float(left[i].square().sum()) for i in common)
    b2 = math.fsum(float(right[i].square().sum()) for i in common)
    a, b = math.sqrt(a2), math.sqrt(b2)
    result = dict(common_parameter_tensors=len(common),
        common_parameter_elements=sum(left[i].numel() for i in common),
        func_weighted_norm=a, compared_weighted_norm=b, norm_ratio_to_func=None,
        cosine=None, status='不适用', reason=None)
    if not common:
        result['reason'] = 'no common autograd dependency'
    elif a == 0 or b == 0:
        result['reason'] = 'zero gradient norm on common parameters'
    else:
        dot = math.fsum(float((left[i] * right[i]).sum()) for i in common)
        result.update(status='ok', cosine=max(-1., min(1., dot / (a * b))), norm_ratio_to_func=b / a)
    return result


def gradient_measurements(gradients, names, groups):
    func = gradients.get('func', [None] * len(names))
    compared = {key: gradients.get(key, [None] * len(names)) for key in ('prob', 'rc', 'behavior')}
    compared['other_weighted_sum'] = sum_gradients(gradients, ('prob', 'rc', 'behavior'), len(names))
    compared['full_multitask_total'] = sum_gradients(gradients, TASKS, len(names))
    # Combined measurements use the same common-dependency rule. For full total,
    # this is exactly all function-connected parameters, including pure-func ones.
    result = {}
    for key, right in compared.items():
        result[key] = {}
        for scope in SCOPES:
            indices = [i for i, name in enumerate(names) if scope == 'all_common' or groups[name] == scope]
            result[key][scope] = compare_gradients(func, right, indices)
    norms = {}
    for task, grads in gradients.items():
        norms[task] = {}
        for scope in (*SCOPES, 'behavior_path', 'task_heads'):
            indices = [i for i, name in enumerate(names) if scope == 'all_common' or groups[name] == scope]
            present = [grads[i] for i in indices if grads[i] is not None]
            norms[task][scope] = dict(weighted_norm=math.sqrt(math.fsum(float(g.square().sum()) for g in present)),
                dependent_parameter_tensors=len(present), nonzero_parameter_tensors=sum(bool((g != 0).any()) for g in present))
    return result, norms


def quantiles(values):
    if not values:
        return None
    x = torch.tensor(values, dtype=torch.float64)
    return {str(q): float(torch.quantile(x, q)) for q in (0., .05, .25, .5, .75, .95, 1.)}


def aggregate(rows):
    result = {}
    for comparison in COMPARISONS:
        result[comparison] = {}
        for scope in SCOPES:
            values = [r['comparisons'][comparison][scope] for r in rows]
            valid = [v for v in values if v['status'] == 'ok']
            reasons = {}
            for value in values:
                if value['status'] != 'ok':
                    reasons[value['reason']] = reasons.get(value['reason'], 0) + 1
            cos = [v['cosine'] for v in valid]
            result[comparison][scope] = dict(
                total_batches=len(rows), applicable_batches=len(valid), not_applicable_reasons=reasons,
                negative_cosine_batch_fraction=sum(v < 0 for v in cos) / len(cos) if cos else None,
                negative_fraction_denominator='applicable batches only; no synthetic zero cosines',
                cosine_quantiles=quantiles(cos),
                weighted_func_norm_quantiles=quantiles([v['func_weighted_norm'] for v in valid]),
                weighted_other_norm_quantiles=quantiles([v['compared_weighted_norm'] for v in valid]),
                weighted_norm_ratio_to_func_quantiles=quantiles([v['norm_ratio_to_func'] for v in valid]))
    return result


def prioritized_findings(summaries):
    """Transparent descriptive screening, not a causal or significance test."""
    result = {'rule': 'prioritize only when BOTH checkpoints show negative fraction >= .75 and median cosine < -.1 in same module; scale differences median ratio >10 or <0.1 reported separately',
              'priority_separate_ablations': [], 'inconsistent_or_insufficient': [],
              'scale_differences': [], 'causal_conclusion': '不足以定位最终因果；仅用于确定下一步单独消融优先级'}
    for task in ('prob', 'rc', 'behavior'):
        for scope in SCOPES[1:]:
            entries = [s[task][scope] for s in summaries.values()]
            ok = [s for s in entries if s['cosine_quantiles'] is not None]
            if len(ok) == len(entries) and len(entries) == 2 and all(
                s['negative_cosine_batch_fraction'] >= .75 and s['cosine_quantiles']['0.5'] < -.1 for s in ok):
                result['priority_separate_ablations'].append({'task': task, 'module': scope})
            else:
                result['inconsistent_or_insufficient'].append({'task': task, 'module': scope,
                    'conclusion': '不适用' if not ok else '不足以定位'})
            for checkpoint, summary in summaries.items():
                q = summary[task][scope]['weighted_norm_ratio_to_func_quantiles']
                if q is not None and (q['0.5'] > 10 or q['0.5'] < .1):
                    result['scale_differences'].append(dict(checkpoint=checkpoint, task=task,
                        module=scope, median_ratio=q['0.5']))
    return result
