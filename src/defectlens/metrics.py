"""Evaluation metrics: per-class / macro top-k accuracy, confusion matrix.

Inputs are assumed pre-validated upstream (taxonomy.load_mapping guarantees
unified labels come from UNIFIED_CLASSES). Labels outside `classes` are
ignored by per_class_topk_accuracy and raise KeyError in confusion_matrix —
if taxonomy validation is ever loosened, revisit these functions.
"""
from __future__ import annotations

import math
from collections import defaultdict


def per_class_topk_accuracy(
    y_true: list[str], ranked_preds: list[list[str]], classes: list[str], k: int
) -> dict[str, float]:
    """Accuracy per class; NaN for classes absent from y_true."""
    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for true, ranked in zip(y_true, ranked_preds, strict=True):
        totals[true] += 1
        if true in ranked[:k]:
            hits[true] += 1
    return {
        c: (hits[c] / totals[c]) if totals[c] else float("nan") for c in classes
    }


def macro_topk_accuracy(
    y_true: list[str], ranked_preds: list[list[str]], classes: list[str], k: int
) -> float:
    """Mean of per-class accuracies over classes that appear in y_true.

    Returns NaN when no class in `classes` has any samples.
    """
    per = per_class_topk_accuracy(y_true, ranked_preds, classes, k)
    vals = [v for v in per.values() if not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def confusion_matrix(
    y_true: list[str], top1_preds: list[str], classes: list[str]
) -> list[list[int]]:
    """rows = true class, cols = predicted class, in `classes` order."""
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, top1_preds, strict=True):
        m[idx[t]][idx[p]] += 1
    return m
