"""Variant wiring for the BFDD SegFormer comparison. No real training here.

Exception: test_build_model_backward_runs_on_mps does one real backward pass at
the production feature size to lock the MPS BatchNorm2d contiguity workaround;
it is skipped where MPS is unavailable (e.g. CI).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from defectlens.thermal.bfdd import CLASS_IDS, CLASS_NAMES
from defectlens.thermal.train_seg import (
    VARIANT_CHANNELS,
    build_metrics,
    build_model,
    compose_input,
    iou_from_confusion,
    per_class_iou_json,
)


def test_variant_channels():
    assert VARIANT_CHANNELS == {"rgb": 3, "ir": 3, "rgbir": 6, "rgbir_hybrid": 6}


def _stem(model):
    """The SegFormer stem patch-embed conv (verified path for transformers 5.x)."""
    return model.segformer.stages[0].patch_embeddings.proj


@pytest.fixture(scope="module")
def pretrained_stem():
    """The pretrained 3-channel mit-b0 stem weight/bias, loaded once."""
    from transformers import SegformerForSemanticSegmentation

    ref = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=6, ignore_mismatched_sizes=True
    )
    proj = _stem(ref)
    return proj.weight.detach().clone(), proj.bias.detach().clone()


def test_hybrid_stem_rgb_half_pretrained_ir_half_zero(pretrained_stem):
    ref_w, ref_b = pretrained_stem
    proj = _stem(build_model("rgbir_hybrid", num_labels=6))
    assert proj.weight.shape[1] == 6
    assert torch.allclose(proj.weight[:, :3], ref_w)  # RGB half starts pretrained
    assert torch.count_nonzero(proj.weight[:, 3:]) == 0  # IR half starts at zero
    assert torch.allclose(proj.bias, ref_b)


def test_compose_input_parity_rgbir_and_hybrid():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    ir = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)
    assert torch.allclose(
        compose_input(rgb, ir, "rgbir"), compose_input(rgb, ir, "rgbir_hybrid")
    )


def test_rgb_stem_stays_pretrained_no_surgery_leak(pretrained_stem):
    """The hybrid surgery must not leak into other variants: a plain rgb build
    keeps the pretrained 3-channel stem untouched."""
    ref_w, _ = pretrained_stem
    proj = _stem(build_model("rgb", num_labels=6))
    assert proj.weight.shape[1] == 3
    assert torch.allclose(proj.weight, ref_w)


def test_compose_input_shapes_and_variant_selection():
    rgb = np.zeros((512, 640, 3), dtype=np.uint8)
    ir = np.full((512, 640, 3), 255, dtype=np.uint8)
    x_rgb = compose_input(rgb, ir, "rgb")
    x_ir = compose_input(rgb, ir, "ir")
    x_fused = compose_input(rgb, ir, "rgbir")
    assert x_rgb.shape == (3, 512, 640) and x_fused.shape == (6, 512, 640)
    # normalized IR (all-255) has strictly larger mean than all-0 rgb
    assert x_ir.mean() > x_rgb.mean()
    # fusion stacks rgb first, ir second
    assert torch.allclose(x_fused[:3], x_rgb) and torch.allclose(x_fused[3:], x_ir)


def test_build_model_in_channels_and_labels():
    m3 = build_model("ir", num_labels=6)
    m6 = build_model("rgbir", num_labels=6)
    assert m3.config.num_channels == 3 and m6.config.num_channels == 6
    assert m3.config.num_labels == 6
    x = torch.zeros(1, 6, 128, 160)
    out = m6(pixel_values=x)
    assert out.logits.shape[1] == 6  # (B, num_labels, H/4, W/4)


def test_iou_from_confusion_known_values():
    conf = np.array([[2, 1], [1, 2]], dtype=np.int64)
    ious = iou_from_confusion(conf)
    assert np.allclose(ious, [2 / 4, 2 / 4])


def test_build_metrics_schema_records_run_config():
    ious = np.full(len(CLASS_IDS), 0.5)
    m = build_metrics(
        "rgb", ious, epochs=25, batch_size=4, lr=6e-5, seed=42,
        steps=3675, train_pairs=586, test_pairs=126, final_train_loss=0.1228,
    )
    expected = {
        "variant", "epochs", "batch_size", "lr", "seed", "steps",
        "train_pairs", "test_pairs", "final_train_loss", "per_class_iou",
        "mean_defect_iou",
    }
    assert expected <= set(m)
    assert (m["seed"], m["lr"], m["batch_size"], m["epochs"]) == (42, 6e-5, 4, 25)
    assert m["final_train_loss"] == 0.1228
    assert set(m["per_class_iou"]) == set(CLASS_NAMES.values())
    # loss is optional (no training steps) -> null, not a crash.
    assert build_metrics(
        "rgb", ious, epochs=0, batch_size=4, lr=6e-5, seed=42,
        steps=0, train_pairs=0, test_pairs=126,
    )["final_train_loss"] is None


def test_per_class_iou_json_absent_class_serializes_as_null():
    import json

    # Zero a class's row and column -> undefined union -> NaN IoU for that class.
    n = len(CLASS_IDS)
    conf = np.eye(n, dtype=np.int64) * 5
    conf[3, :] = 0
    conf[:, 3] = 0
    ious = iou_from_confusion(conf)
    d = per_class_iou_json(ious)
    assert d[CLASS_NAMES[3]] is None
    # The real bug was invalid JSON: bare NaN. Prove the dump is clean.
    assert "NaN" not in json.dumps(d)
    assert all(v is None or isinstance(v, float) for v in d.values())


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS-specific backward regression"
)
def test_build_model_backward_runs_on_mps():
    """Regression lock for the decode-head BatchNorm2d contiguity workaround:
    without it, backward at the real 512x640 feature size crashes on MPS with a
    view/stride error. One tiny (batch-2) step proves the loop trains on MPS."""
    model = build_model("rgbir", num_labels=6).to("mps")
    x = torch.randn(2, 6, 512, 640, device="mps")
    y = torch.randint(0, 6, (2, 512, 640), device="mps")
    logits = model(pixel_values=x).logits
    logits = F.interpolate(logits, size=(512, 640), mode="bilinear", align_corners=False)
    loss = F.cross_entropy(logits, y.to("mps"))
    loss.backward()  # must not raise
    assert torch.isfinite(loss).item()
