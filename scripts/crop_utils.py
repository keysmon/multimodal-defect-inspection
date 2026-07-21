"""Shared bbox->classification-crop rules for detection-derived datasets.

Used by scripts/prepare_mbdd.py and scripts/prepare_insulator.py (imported as a
sibling module: `python scripts/prepare_X.py` puts scripts/ on sys.path).

Crop contract (plan Task A1, applied to both datasets):
- crop = bbox expanded by MARGIN_FRAC on every side, clamped to the image;
- crops with min side < MIN_SIDE px are skipped (too small to classify);
- a box overlapping a DIFFERENT-class box is skipped so crops stay single-label.
  Overlap uses the overlap coefficient (intersection / min box area), not plain
  IoU: nested boxes (a defect region inside a larger object box) have low IoU
  but the containing crop would include the other class's pixels — the exact
  mislabeling the rule exists to prevent.
- exact-duplicate crops are dropped by content hash (sha1 of encoded bytes).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

MARGIN_FRAC = 0.15
MIN_SIDE = 96
OVERLAP_SKIP = 0.3


@dataclass(frozen=True)
class Box:
    """Pixel-space box, half-open [x1, x2) x [y1, y2)."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def overlap_coefficient(a: Box, b: Box) -> float:
    """Intersection area / min(area a, area b); 0.0 when either is degenerate."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    denom = min(a.area, b.area)
    return (ix * iy) / denom if denom > 0 else 0.0


def conflicting(
    box: Box,
    others: list[Box],
    threshold: float = OVERLAP_SKIP,
    benign: frozenset[str] = frozenset(),
) -> bool:
    """True when any DIFFERENT-class box overlaps enough to pollute the crop.

    `benign` labels never pollute OTHER crops (but can themselves be polluted):
    an OBJECT-class box (e.g. the insulator dataset's whole-insulator boxes)
    is label-compatible with a defect crop nested inside it, while a defect box
    inside an object box makes that object crop unusable as a clean example.
    Defect-vs-defect overlap stays symmetric.
    """
    return any(
        other.label != box.label
        and other.label not in benign
        and overlap_coefficient(box, other) > threshold
        for other in others
    )


def expanded_pixel_box(
    box: Box, width: int, height: int, margin: float = MARGIN_FRAC
) -> tuple[int, int, int, int] | None:
    """Margin-expanded, image-clamped int crop box; None if below MIN_SIDE."""
    dx = (box.x2 - box.x1) * margin
    dy = (box.y2 - box.y1) * margin
    x1 = max(0, int(round(box.x1 - dx)))
    y1 = max(0, int(round(box.y1 - dy)))
    x2 = min(width, int(round(box.x2 + dx)))
    y2 = min(height, int(round(box.y2 + dy)))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return x1, y1, x2, y2


def content_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def yolo_to_box(
    line: str, names: list[str], width: int, height: int
) -> Box:
    """Parse one YOLO txt line (class cx cy w h, normalized) into a pixel Box."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"malformed YOLO line: {line!r}")
    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
    if not 0 <= cls < len(names):
        raise ValueError(f"class id {cls} outside names list ({len(names)} entries)")
    return Box(
        label=names[cls],
        x1=(cx - w / 2) * width,
        y1=(cy - h / 2) * height,
        x2=(cx + w / 2) * width,
        y2=(cy + h / 2) * height,
    )
