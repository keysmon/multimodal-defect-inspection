"""BedrockDescriber: cloud-path description via Claude Haiku on Bedrock.

The boto3 client is stubbed (request captured, canned response returned) — NO
real Bedrock calls here. The one real smoke call lives outside the test suite.
"""
from __future__ import annotations

import subprocess
import sys
from io import BytesIO

from PIL import Image

from defectlens.serve.bedrock_describer import (
    BedrockDescriber,
    describer_is_bedrock,
)


def _png_image():
    return Image.new("RGB", (8, 8))


class FakeBedrockClient:
    """Captures the converse request; returns a canned assistant message."""

    def __init__(self, text="Visible longitudinal cracking with rust staining.", raises=None):
        self.text = text
        self.raises = raises
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return {"output": {"message": {"content": [{"text": self.text}]}}}


# ---------------------------------------------------------------------------
# describer_is_bedrock env gate
# ---------------------------------------------------------------------------


def test_describer_is_bedrock_true_only_for_bedrock(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_DESCRIBER", "bedrock")
    assert describer_is_bedrock() is True
    monkeypatch.setenv("DEFECTLENS_DESCRIBER", "BEDROCK")  # case-insensitive
    assert describer_is_bedrock() is True


def test_describer_is_bedrock_false_when_unset_or_local(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_DESCRIBER", raising=False)
    assert describer_is_bedrock() is False
    monkeypatch.setenv("DEFECTLENS_DESCRIBER", "local")
    assert describer_is_bedrock() is False


# ---------------------------------------------------------------------------
# describe() — request wiring + response parsing (stubbed client)
# ---------------------------------------------------------------------------


def test_describe_sends_prompt_and_jpeg_and_returns_text():
    d = BedrockDescriber(model_id="test-model")
    fake = FakeBedrockClient(text="  Diagonal crack, moderate width.  ")
    d._client = fake

    out = d.describe(_png_image(), ["crack", "spalling"])
    assert out == "Diagonal crack, moderate width."  # stripped

    assert len(fake.calls) == 1
    req = fake.calls[0]
    assert req["modelId"] == "test-model"
    content = req["messages"][0]["content"]
    image_block = next(b for b in content if "image" in b)
    text_block = next(b for b in content if "text" in b)
    # JPEG bytes, not base64/PIL — Converse takes raw bytes.
    assert image_block["image"]["format"] == "jpeg"
    assert isinstance(image_block["image"]["source"]["bytes"], (bytes, bytearray))
    assert image_block["image"]["source"]["bytes"][:2] == b"\xff\xd8"  # JPEG SOI
    # the reused describer.build_prompt text names the top classes
    assert "crack" in text_block["text"]
    assert "spalling" in text_block["text"]


def test_describe_forwards_audio_band_into_prompt():
    d = BedrockDescriber()
    fake = FakeBedrockClient()
    d._client = fake

    d.describe(_png_image(), ["crack"], audio_band="anomalous")
    text = next(b for b in fake.calls[0]["messages"][0]["content"] if "text" in b)["text"]
    assert "anomalous" in text


def test_describe_swallows_bedrock_errors_and_returns_empty():
    d = BedrockDescriber()
    d._client = FakeBedrockClient(raises=RuntimeError("AccessDeniedException"))
    # describe() must degrade to "" — description is optional.
    assert d.describe(_png_image(), ["crack"]) == ""


def test_converse_surfaces_errors_for_the_smoke_test():
    """The non-swallowing path used by the smoke test must raise, so a wrong
    model id / access issue is visible instead of masked as ''."""
    import pytest

    d = BedrockDescriber()
    d._client = FakeBedrockClient(raises=RuntimeError("ValidationException"))
    with pytest.raises(RuntimeError, match="ValidationException"):
        d.converse(_png_image(), ["crack"])


# ---------------------------------------------------------------------------
# health-shape attributes — no local VLM, no adapter, no rank_classes
# ---------------------------------------------------------------------------


def test_bedrock_describer_has_no_local_vlm_or_adapter_or_ranker():
    d = BedrockDescriber()
    d.load()  # no-op, must not build a client or import boto3
    assert d.model is None  # /health vlm_loaded stays False
    assert d.adapter_loaded is False  # /health classifier stays clip-fused
    assert not hasattr(d, "rank_classes")  # api.py getattr default -> [] -> CLIP-fused


# ---------------------------------------------------------------------------
# Client config — fail fast, never boto3's default 5-attempt backoff
# ---------------------------------------------------------------------------


def test_bedrock_client_fails_fast_no_retries():
    """Regression lock: with zero applied Bedrock quota every Converse call
    throttles, and boto3's default retries (initial + 4 backoff attempts) were
    adding ~10s to every /analyze. describe()'s ""-fallback is the retry
    strategy — the client itself must make exactly one attempt."""
    d = BedrockDescriber()
    client = d._bedrock_client()  # builds the client; no network call
    cfg = client.meta.config
    # total_max_attempts counts the initial call — 1 = zero retries.
    # (botocore normalizes max_attempts=N to total_max_attempts=N+1, so
    # asserting the normalized key locks the actual behavior.)
    assert cfg.retries == {"total_max_attempts": 1, "mode": "standard"}
    assert cfg.connect_timeout == 3
    assert cfg.read_timeout == 15


# ---------------------------------------------------------------------------
# Import sanity — module must not pull boto3 at import time (cheap import)
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_boto3_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.serve.bedrock_describer\n"
            "assert 'boto3' not in sys.modules, 'boto3 imported at module level'\n"
            "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
