"""Validation-calibrated equivalence metrics, with grouped ties and CPU storage."""

import warnings

import torch

from .logger import info


def equivalence_samples(distance, target, mask):
    valid = mask.bool() & torch.isfinite(target)
    targets = target[valid].detach().cpu()
    if bool((targets < 0).any()):
        raise ValueError("Functional truth-table distances must be nonnegative")
    scores = (1 - distance[valid]).detach().cpu()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Nonfinite functional similarity scores on valid pairs")
    labels = torch.isclose(targets, torch.zeros_like(targets), atol=1e-8, rtol=0.0)
    return scores, labels


def check_samples(scores, labels, split):
    if scores.ndim != 1 or labels.shape != scores.shape or labels.dtype != torch.bool:
        raise ValueError("Expected equally sized score and boolean label vectors")
    if not scores.numel():
        raise ValueError(f"{split}: no valid functional pairs (mask=true and finite target)")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError(f"{split}: nonfinite similarity scores")
    positive = int(labels.sum())
    negative = labels.numel() - positive
    info(f"{split}: positive_count={positive}, negative_count={negative}")
    if not positive or not negative:
        warnings.warn(f"{split}: positive_count={positive}, negative_count={negative}; "
                      "both classes are required for a meaningful equivalence comparison. "
                      "Regenerate evaluation pairs; model retraining is not required.", stacklevel=2)
    elif positive < 100:
        warnings.warn(f"{split}: only {positive} positive pairs (<100); equivalence metrics "
                      "may be unstable. Regenerate more evaluation pairs; model retraining "
                      "is not required.", stacklevel=2)
    return positive, negative


def grouped_curve(scores, labels):
    """Counts after each entire equal-score group, in descending score order."""
    scores = scores.detach().cpu().double()
    labels = labels.detach().cpu()
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    ends = torch.cat((torch.nonzero(sorted_scores[:-1] != sorted_scores[1:]).flatten(),
                      torch.tensor([scores.numel() - 1])))
    tp = labels[order].to(torch.int64).cumsum(0)[ends]
    fp = ends + 1 - tp
    return sorted_scores[ends], tp, fp


def select_threshold(scores, labels):
    """Maximize pooled validation F1; equal F1 chooses the higher threshold."""
    positive, negative = check_samples(scores, labels, "validation")
    if not positive or not negative:
        return None
    thresholds, tp, fp = grouped_curve(scores, labels)
    f1 = 2 * tp.double() / (positive + tp + fp)
    # With positives present, predicting none has F1=0 and cannot be optimal.
    return float(thresholds[int(torch.argmax(f1))])


def equivalence_metrics(scores, labels, threshold, split="test"):
    positive, negative = check_samples(scores, labels, split)
    result = dict(threshold=threshold, similarity_threshold=threshold,
                  precision=None, recall=None, f1=None, accuracy=None, roc_auc=None,
                  positive_count=positive, negative_count=negative,
                  tp=None, fp=None, tn=None, fn=None)
    if positive and negative:
        _, tp_curve, fp_curve = grouped_curve(scores, labels)
        tpr = torch.cat((torch.zeros(1), tp_curve.double() / positive))
        fpr = torch.cat((torch.zeros(1), fp_curve.double() / negative))
        result["roc_auc"] = float(torch.trapz(tpr, fpr))
    if threshold is None:
        result["status"] = "unavailable: validation must contain both classes"
        return result
    predicted = scores >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    tn, fn = negative - fp, positive - tp
    result.update(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp + tn) / labels.numel())
    if not positive or not negative:
        result["status"] = "unavailable: evaluation split must contain both classes"
        return result
    result.update(precision=tp / (tp + fp) if tp + fp else 0.0,
                  recall=tp / positive, f1=2 * tp / (2 * tp + fp + fn), status="ok")
    return result
