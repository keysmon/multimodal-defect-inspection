"""Regression locks for scripts/crop_utils.py (loaded by path like the other
outside-package script tests — see tests/test_package_sagemaker.py).

The overlap rules encode two non-obvious decisions that were each the fix for
a real mislabeling mode observed while preparing the v2 datasets:
- overlap coefficient (intersection / min area), NOT IoU: nested
  different-class boxes have low IoU but pollute the containing crop;
- `benign` object classes: a defect box inside an insulator OBJECT box must
  keep its crop, while the object box containing it must lose its `normal`
  crop (a symmetric rule kept only 7 of ~3,500 insulator defect crops).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "crop_utils", REPO_ROOT / "scripts" / "crop_utils.py"
)
assert spec and spec.loader
crop_utils = importlib.util.module_from_spec(spec)
# @dataclass resolves cls.__module__ through sys.modules — register before exec.
sys.modules[spec.name] = crop_utils
spec.loader.exec_module(crop_utils)

Box = crop_utils.Box


def test_overlap_coefficient_containment_is_full():
    outer = Box("a", 0, 0, 100, 100)
    inner = Box("b", 10, 10, 30, 30)
    # IoU here would be 400/10000 = 0.04; the coefficient sees full containment.
    assert crop_utils.overlap_coefficient(outer, inner) == pytest.approx(1.0)


def test_overlap_coefficient_disjoint_and_degenerate():
    a = Box("a", 0, 0, 10, 10)
    assert crop_utils.overlap_coefficient(a, Box("b", 20, 20, 30, 30)) == 0.0
    assert crop_utils.overlap_coefficient(a, Box("b", 5, 5, 5, 40)) == 0.0  # zero width


def test_conflicting_same_class_never_conflicts():
    a = Box("crack", 0, 0, 50, 50)
    b = Box("crack", 10, 10, 60, 60)
    assert not crop_utils.conflicting(a, [a, b])


def test_conflicting_nested_different_class():
    outer = Box("abscission", 0, 0, 100, 100)
    inner = Box("crack", 10, 10, 30, 30)
    boxes = [outer, inner]
    assert crop_utils.conflicting(outer, boxes)
    assert crop_utils.conflicting(inner, boxes)


def test_benign_object_class_is_one_way():
    """Defect inside an object box: defect crop survives, object crop dies."""
    insulator = Box("normal", 0, 0, 100, 100)
    defect = Box("broken", 10, 10, 30, 30)
    boxes = [insulator, defect]
    benign = frozenset({"normal"})
    assert not crop_utils.conflicting(defect, boxes, benign=benign)
    assert crop_utils.conflicting(insulator, boxes, benign=benign)


def test_defect_vs_defect_stays_symmetric_with_benign():
    flashover = Box("pollution_flashover", 0, 0, 40, 40)
    broken = Box("broken", 10, 10, 30, 30)
    boxes = [flashover, broken]
    benign = frozenset({"normal"})
    assert crop_utils.conflicting(flashover, boxes, benign=benign)
    assert crop_utils.conflicting(broken, boxes, benign=benign)


def test_expanded_pixel_box_margin_and_clamp():
    box = Box("crack", 100, 100, 300, 200)  # 200x100, 15% margin = 30x15
    assert crop_utils.expanded_pixel_box(box, 1280, 720) == (70, 85, 330, 215)
    # Clamped at the image edge:
    edge = Box("crack", 0, 0, 200, 150)
    assert crop_utils.expanded_pixel_box(edge, 210, 160) == (0, 0, 210, 160)


def test_expanded_pixel_box_min_side_floor():
    small = Box("crack", 0, 0, 50, 500)  # 50px + margin < 96 on the x side
    assert crop_utils.expanded_pixel_box(small, 1280, 720) is None


def test_yolo_to_box_roundtrip_and_validation():
    names = ["pollution_flashover", "broken", "normal"]
    box = crop_utils.yolo_to_box("2 0.5 0.5 0.25 0.5", names, 400, 200)
    assert box.label == "normal"
    assert (box.x1, box.y1, box.x2, box.y2) == (150.0, 50.0, 250.0, 150.0)
    with pytest.raises(ValueError):
        crop_utils.yolo_to_box("3 0.5 0.5 0.1 0.1", names, 400, 200)  # bad id
    with pytest.raises(ValueError):
        crop_utils.yolo_to_box("1 0.5 0.5 0.1", names, 400, 200)  # malformed
