"""Per-circuit and pooled audits of sampled functional response agreement."""

import math

import torch

from .functional_equivalence import grouped_curve

METRICS = ("precision", "recall", "f1", "accuracy", "roc_auc")
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)


def pair_metrics(scores, labels, threshold):
    """Undefined ratios are null; zero F1 remains valid when its denominator > 0."""
    scores, labels = scores.detach().cpu(), labels.detach().cpu().bool()
    positive = int(labels.sum())
    negative = labels.numel() - positive
    result = dict(threshold=threshold, similarity_threshold=threshold,
                  positive_count=positive, negative_count=negative,
                  positive_ratio=positive / labels.numel() if labels.numel() else None,
                  tp=None, fp=None, tn=None, fn=None, **dict.fromkeys(METRICS))
    if not scores.numel():
        result["status"] = "no_valid_pairs"
        result["undefined_reasons"] = {key: "no valid functional pairs" for key in METRICS}
        return result
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite audit similarities")
    if positive and negative:
        _, tps, fps = grouped_curve(scores, labels)
        tpr = torch.cat((torch.zeros(1), tps.double() / positive))
        fpr = torch.cat((torch.zeros(1), fps.double() / negative))
        result["roc_auc"] = float(torch.trapz(tpr, fpr))
    if threshold is None:
        result["status"] = "no_validation_threshold"
        result["undefined_reasons"] = {key: "validation threshold unavailable"
                                       for key in METRICS if key != "roc_auc"}
        if result["roc_auc"] is None:
            result["undefined_reasons"]["roc_auc"] = "requires positive and negative pairs"
        return result
    predicted = scores >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    tn, fn = negative - fp, positive - tp
    result.update(tp=tp, fp=fp, tn=tn, fn=fn,
                  precision=tp / (tp + fp) if tp + fp else None,
                  recall=tp / positive if positive else None,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  accuracy=(tp + tn) / labels.numel(),
                  status="ok" if positive and negative else "single_class")
    reasons = {"precision": "no predicted positive pairs", "recall": "no ground-truth positive pairs",
               "f1": "no ground-truth or predicted positive pairs", "roc_auc": "requires positive and negative pairs"}
    result["undefined_reasons"] = {key: reason for key, reason in reasons.items() if result[key] is None}
    return result


def circuit_macro(rows):
    """Arithmetic mean for each metric over circuits where it is defined."""
    result = {"averaging": "unweighted arithmetic mean over circuits with a defined metric; null excluded separately per metric",
              "circuit_count": len(rows),
              "circuits_with_pairs": sum(row["positive_count"] + row["negative_count"] > 0 for row in rows),
              "both_class_circuit_count": sum(row["positive_count"] > 0 and row["negative_count"] > 0 for row in rows),
              "contributing_circuits": {}}
    for key in METRICS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = math.fsum(values) / len(values) if values else None
        result["contributing_circuits"][key] = len(values)
    return result


def similarity_distribution(scores, labels):
    """Do not clamp scores: expose roundoff beyond one as well as saturation."""
    scores, labels = scores.detach().cpu().double(), labels.detach().cpu().bool()
    result = {}
    for name, mask in (("all", torch.ones_like(labels)), ("positive", labels), ("negative", ~labels)):
        values = scores[mask]
        count = values.numel()
        entry = {"count": count, "quantiles": None, "equal_one_fraction": None,
                 "greater_than_one_fraction": None, "near_one_fractions": {}}
        if count:
            # torch.quantile rejects very large arrays; rank interpolation on sorted CPU scores
            # has the same linear definition and no 2**24 sample-size limitation.
            ordered = values.sort().values
            quantiles = {}
            for q in QUANTILES:
                position = (count - 1) * q
                low, high = math.floor(position), math.ceil(position)
                quantiles[str(q)] = float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
            entry.update(quantiles=quantiles,
                         equal_one_fraction=float((values == 1).double().mean()),
                         greater_than_one_fraction=float((values > 1).double().mean()))
            for tolerance in (1e-8, 1e-7, 1e-6, 1e-4):
                entry["near_one_fractions"][str(tolerance)] = float(((values - 1).abs() <= tolerance).double().mean())
        result[name] = entry
    result["near_one_definition"] = "abs(score - 1) <= atol, rtol=0; includes exactly-one scores"
    return result


def metric_differences(original, audited):
    result = {}
    for key in (*METRICS, "threshold", "positive_count", "negative_count", "tp", "fp", "tn", "fn"):
        before, after = original.get(key), audited.get(key)
        result[key] = {"original": before, "audited": after,
                       "delta": after - before if before is not None and after is not None else None}
    return result
