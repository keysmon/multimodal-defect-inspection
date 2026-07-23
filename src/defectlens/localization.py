"""Qwen2.5-VL grounding helpers: bbox JSON parsing + resized->original mapping.

Qwen2.5-VL emits bounding boxes as absolute pixel coordinates RELATIVE TO ITS
SMART-RESIZED input, not the original image. The processor reports the resized
extent via image_grid_thw (in 14-px vision patches), so the mapping back to
original pixels is a pure scale + clamp. Used by the localization spike and,
if the spike passes, vendored into deploy/sagemaker/inference.py.
"""
from __future__ import annotations

import json
import math

from defectlens.llm_json import balanced_array_candidates

PATCH = 14  # Qwen2.5-VL ViT patch edge: image_grid_thw counts 14-px patches

GROUNDING_PROMPT = (
    "Locate every visible {name} in this image. Output ONLY a JSON array of "
    'objects like [{{"bbox_2d": [x1, y1, x2, y2], "label": "{name}"}}] using '
    "absolute pixel coordinates. If none are visible, output []."
)


def input_size_from_grid(grid_thw) -> tuple[int, int]:
    """(input_h, input_w) pixels from one image_grid_thw row [t, h, w]."""
    _t, h, w = (int(v) for v in grid_thw)
    return h * PATCH, w * PATCH


def _valid_entry(entry) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
        return False
    box = entry.get("bbox_2d")
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
        return False
    return x1 < x2 and y1 < y2


def parse_boxes(text: str) -> list[dict]:
    """Extract [{"bbox_2d": [x1,y1,x2,y2], "label": str}, ...] from model text.

    Balanced-scan over the raw output (Qwen wraps JSON in prose/fences);
    malformed entries are dropped, never raised - grounding must degrade to
    "no boxes", not crash a report path.
    """
    for candidate in reversed(balanced_array_candidates(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            kept = [
                {"bbox_2d": [int(float(v)) for v in e["bbox_2d"]], "label": e["label"]}
                for e in data
                if _valid_entry(e)
            ]
            if kept or data == []:
                return kept
    return []


def rescale_box(
    box: list[int], input_size: tuple[int, int], orig_size: tuple[int, int]
) -> list[int]:
    """Map one [x1, y1, x2, y2] from resized-input coords to original pixels.

    Sizes are (height, width). Clamps into the original bounds - the model
    occasionally overshoots edges by a few pixels.
    """
    ih, iw = input_size
    oh, ow = orig_size
    x1, y1, x2, y2 = box
    sx, sy = ow / iw, oh / ih
    return [
        max(0, min(ow, round(x1 * sx))),
        max(0, min(oh, round(y1 * sy))),
        max(0, min(ow, round(x2 * sx))),
        max(0, min(oh, round(y2 * sy))),
    ]
