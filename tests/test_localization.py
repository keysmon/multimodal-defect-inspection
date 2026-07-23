"""Unit tests for Qwen2.5-VL grounding helpers (spike + gated overlays)."""
from defectlens.localization import (
    GROUNDING_PROMPT,
    input_size_from_grid,
    parse_boxes,
    rescale_box,
)


def test_input_size_from_grid_is_patches_times_14():
    # image_grid_thw row [t, h, w] counts 14-px patches
    assert input_size_from_grid([1, 54, 76]) == (756, 1064)


def test_parse_boxes_extracts_bbox_entries_from_prose():
    text = (
        'Here are the cracks: [{"bbox_2d": [10, 20, 110, 220], "label": "crack"},'
        ' {"bbox_2d": [5, 5, 50, 50], "label": "crack"}] as requested.'
    )
    boxes = parse_boxes(text)
    assert boxes == [
        {"bbox_2d": [10, 20, 110, 220], "label": "crack"},
        {"bbox_2d": [5, 5, 50, 50], "label": "crack"},
    ]


def test_parse_boxes_drops_malformed_entries_never_raises():
    text = '[{"bbox_2d": [1, 2, 3], "label": "crack"}, {"bbox_2d": [30, 10, 20, 40]}, "junk"]'
    assert parse_boxes(text) == []  # wrong arity, x1>x2 after norm-check, non-dict
    assert parse_boxes("no json at all") == []
    assert parse_boxes('[{"bbox_2d": [20, 10, 5, 40], "label": "c"}]') == []  # x1 > x2


def test_rescale_box_maps_resized_coords_to_original_and_clamps():
    # input 756x1064 (h, w) -> original 1512x2128: exact 2x scale
    assert rescale_box([10, 20, 110, 220], (756, 1064), (1512, 2128)) == [20, 40, 220, 440]
    # out-of-range coords clamp to the original bounds
    assert rescale_box([-5, 0, 2000, 900], (756, 1064), (756, 1064)) == [0, 0, 1064, 756]


def test_grounding_prompt_names_the_class():
    p = GROUNDING_PROMPT.format(name="corrosion stain")
    assert "corrosion stain" in p and "bbox_2d" in p
