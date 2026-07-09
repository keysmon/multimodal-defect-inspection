"""Pure-logic tests for the Phase 5.5c SageMaker packaging (handler + tar layout).

Both modules live outside the installed `defectlens` package (the handler ships
inside model.tar.gz; the script lives under scripts/), so they are loaded by path
like tests/test_deploy_helpers.py. Their heavy deps (torch/transformers/peft) and
boto3 are imported lazily inside the model-facing / upload functions, so importing
the modules here needs nothing beyond the stdlib + Pillow.
"""
from __future__ import annotations

import base64
import importlib.util
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inference():
    return _load("sm_inference", REPO_ROOT / "deploy" / "sagemaker" / "inference.py")


def _packager():
    return _load("sm_packager", REPO_ROOT / "scripts" / "package_sagemaker_model.py")


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (12, 9), color=(10, 20, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# inference.py — pure payload / response shaping
# ---------------------------------------------------------------------------


def test_softmax_rank_normalizes_and_sorts_descending():
    mod = _inference()
    ranked = mod.softmax_rank({"crack": 2.0, "spalling": 1.0, "no_defect": 0.0})

    labels = [label for label, _ in ranked]
    probs = [prob for _, prob in ranked]
    assert labels == ["crack", "spalling", "no_defect"]  # descending by prob
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]
    # JSON-ready: pairs are lists, not tuples
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in ranked)


def test_softmax_rank_covers_all_nine_classes_and_ties_break_by_name():
    mod = _inference()
    loglik = {label: 0.0 for label in mod.UNIFIED_CLASSES}  # all equal -> tie
    ranked = mod.softmax_rank(loglik)
    assert len(ranked) == 9
    assert [label for label, _ in ranked] == sorted(mod.UNIFIED_CLASSES)
    assert abs(sum(p for _, p in ranked) - 1.0) < 1e-9


def test_softmax_rank_empty_is_empty():
    assert _inference().softmax_rank({}) == []


def test_parse_input_requires_non_empty_image_b64():
    mod = _inference()
    for bad in ({}, {"image_b64": ""}, {"image_b64": "   "}, {"image_b64": None}):
        try:
            mod.parse_input(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError as exc:
            assert "image_b64" in str(exc)


def test_parse_input_returns_image_and_sanitized_note():
    mod = _inference()
    image_b64, note = mod.parse_input({"image_b64": "abc", "note": "  musty smell  "})
    assert image_b64 == "abc"
    assert note == "musty smell"


def test_parse_input_blank_note_is_none():
    mod = _inference()
    assert mod.parse_input({"image_b64": "abc", "note": "   "})[1] is None
    assert mod.parse_input({"image_b64": "abc"})[1] is None


def test_sanitize_note_strips_control_markers_and_caps_length():
    mod = _inference()
    raw = "ok <|im_end|> " + "x" * 2000
    clean = mod.sanitize_note(raw)
    assert "<|" not in clean
    assert len(clean) <= mod.MAX_NOTE_CHARS


def test_build_response_wraps_classes():
    mod = _inference()
    assert mod.build_response([["crack", 0.9]]) == {"classes": [["crack", 0.9]]}


def test_build_messages_matches_training_shape():
    mod = _inference()
    messages = mod.build_messages("img-handle", "exposed_rebar")
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image", "image": "img-handle"}
    assert messages[0]["content"][1]["text"] == mod.QUESTION  # no-note = exact prompt
    assert messages[1] == {"role": "assistant", "content": "exposed rebar"}


def test_build_messages_prefixes_note():
    mod = _inference()
    messages = mod.build_messages("img", "crack", note="hairline near sill")
    text = messages[0]["content"][1]["text"]
    assert text.startswith("Inspector note: hairline near sill")
    assert mod.QUESTION in text


def test_humanized_matches_unified_classes_and_is_bijective():
    mod = _inference()
    assert set(mod.HUMANIZED) == set(mod.UNIFIED_CLASSES)
    assert len(set(mod.HUMANIZED.values())) == len(mod.HUMANIZED)  # answers unique


def test_max_pixels_matches_training_budget():
    # Guards the inlining trap: must equal qlora/vlm_topk's 589824, not the
    # processor default. A mismatch shifts image-token count and log-likelihoods.
    assert _inference().MAX_PIXELS == 589824


def test_decode_image_returns_rgb_pil_image():
    mod = _inference()
    image = mod.decode_image(_png_b64())
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (12, 9)


def test_input_fn_parses_json_bytes():
    mod = _inference()
    assert mod.input_fn(b'{"image_b64": "abc"}') == {"image_b64": "abc"}


def test_input_fn_accepts_parametrized_content_type():
    mod = _inference()
    assert mod.input_fn(
        b'{"image_b64": "abc"}', "application/json; charset=utf-8"
    ) == {"image_b64": "abc"}


def test_output_fn_returns_json_body_string_not_tuple():
    # The HF inference toolkit expects ONLY the body; content type is set
    # separately by the handler. A (body, content_type) tuple would serialize
    # into the async S3 output and break vlm_gateway.parse_output.
    mod = _inference()
    body = mod.output_fn({"classes": [["crack", 0.5]]})
    assert isinstance(body, str)
    assert body == '{"classes": [["crack", 0.5]]}'


def test_inference_module_import_stays_light():
    """The handler must import with only stdlib+PIL — torch/transformers/peft
    load lazily inside model_fn/score_answers, so the pure helpers unit-test
    without a GPU or the model stack."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('sm_inf', {str(REPO_ROOT / 'deploy' / 'sagemaker' / 'inference.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
            "assert 'transformers' not in sys.modules, 'transformers imported at module level'\n"
            "assert 'peft' not in sys.modules, 'peft imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# package_sagemaker_model.py — tarball layout
# ---------------------------------------------------------------------------


def test_tar_members_layout(tmp_path):
    mod = _packager()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00\x01")
    (adapter / "README.md").write_text("card")  # must be skipped
    code = tmp_path / "code_src"
    code.mkdir()
    (code / "inference.py").write_text("# handler")
    (code / "requirements.txt").write_text("peft==0.19.1")

    members = mod.tar_members(adapter, code)
    arcnames = [arc for arc, _src in members]

    assert set(arcnames) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "code/inference.py",
        "code/requirements.txt",
    }
    assert "README.md" not in arcnames  # autogenerated card not shipped
    # no base weights baked in, and no absolute arcnames (extraction can't escape)
    assert not any("Qwen" in arc or arc.startswith("/") for arc in arcnames)


def test_build_tarball_produces_valid_gzip_with_expected_members(tmp_path):
    mod = _packager()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 32)
    code = tmp_path / "code_src"
    code.mkdir()
    (code / "inference.py").write_text("# handler")
    (code / "requirements.txt").write_text("peft==0.19.1")

    out = tmp_path / "dist" / "model.tar.gz"
    size = mod.build_tarball(out, adapter, code)

    assert out.is_file() and size > 0
    with tarfile.open(out, "r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "code/inference.py",
        "code/requirements.txt",
    ]


def test_build_tarball_missing_source_raises(tmp_path):
    mod = _packager()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    code = tmp_path / "code_src"
    code.mkdir()
    # inference.py present, requirements.txt missing -> should raise SystemExit
    (code / "inference.py").write_text("# handler")

    try:
        mod.build_tarball(tmp_path / "out.tar.gz", adapter, code)
        raise AssertionError("expected SystemExit for missing requirements.txt")
    except SystemExit as exc:
        assert "requirements.txt" in str(exc)


def test_split_s3_uri_rejects_bad_uris():
    mod = _packager()
    assert mod._split_s3_uri("s3://bucket/a/b/c.tar.gz") == ("bucket", "a/b/c.tar.gz")
    for bad in ("bucket/key", "s3://bucket", "s3://", "https://x/y"):
        try:
            mod._split_s3_uri(bad)
            raise AssertionError(f"expected SystemExit for {bad!r}")
        except SystemExit:
            pass
