import math

from defectlens.metrics import (
    confusion_matrix,
    macro_topk_accuracy,
    per_class_topk_accuracy,
)

CLASSES = ["a", "b", "c"]


def test_per_class_top1():
    y_true = ["a", "a", "b"]
    ranked = [["a", "b", "c"], ["b", "a", "c"], ["b", "c", "a"]]
    per = per_class_topk_accuracy(y_true, ranked, CLASSES, k=1)
    assert per["a"] == 0.5
    assert per["b"] == 1.0
    assert math.isnan(per["c"])  # no samples of class c


def test_top3_hits_anywhere_in_top3():
    y_true = ["a"]
    ranked = [["c", "b", "a"]]
    per = per_class_topk_accuracy(y_true, ranked, CLASSES, k=3)
    assert per["a"] == 1.0


def test_macro_ignores_absent_classes():
    y_true = ["a", "a", "b"]
    ranked = [["a", "x", "x"], ["b", "x", "x"], ["b", "x", "x"]]
    # a: 1/2, b: 1/1, c: absent -> macro = (0.5 + 1.0) / 2
    assert macro_topk_accuracy(y_true, ranked, CLASSES, k=1) == 0.75


def test_confusion_matrix():
    y_true = ["a", "a", "b"]
    top1 = ["a", "b", "b"]
    m = confusion_matrix(y_true, top1, CLASSES)
    assert m[0][0] == 1  # a predicted a
    assert m[0][1] == 1  # a predicted b
    assert m[1][1] == 1  # b predicted b
    assert sum(sum(row) for row in m) == 3
