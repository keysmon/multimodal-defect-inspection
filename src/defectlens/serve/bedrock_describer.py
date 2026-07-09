"""Bedrock-backed condition description (Phase 5.5 cloud path).

Same ``describe(image, top_classes, audio_band=None) -> str`` contract as
serve.describer.Describer, but the narration comes from Claude Haiku on Amazon
Bedrock rather than a local Qwen VLM. The cloud Lambda runs DEFECTLENS_NO_VLM=1
(no torch model in the image) and delegates description to Bedrock; env gate
DEFECTLENS_DESCRIBER=bedrock selects this implementation in serve.api's lifespan
(default stays the local Qwen Describer).

Reuses describer.build_prompt so the instruction text is identical across
backends. boto3 is imported lazily so the module stays cheap to import (like
Describer) and unit tests need no AWS. Any Bedrock failure from the public
``describe`` returns "" — description is optional; the CLIP-fused classification
and RAG cards carry the report without it.

Honest note (README): the fine-tuned adapter is never in play here — Bedrock
sees the base Haiku model, so classification stays CLIP-fused (this class has no
rank_classes, so serve.api falls back to the fused ranking).

Model id: ca-central-1 offers Haiku 4.5 through cross-region **inference
profiles** only (the bare foundation-model id is INFERENCE_PROFILE-only, not
on-demand), so the default is the global profile. Override with
DEFECTLENS_BEDROCK_MODEL / DEFECTLENS_BEDROCK_REGION.
"""
from __future__ import annotations

import logging
import os
from io import BytesIO

from defectlens.serve.describer import build_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "ca-central-1"
MAX_TOKENS = 300


def describer_is_bedrock() -> bool:
    """True when the env gate selects the Bedrock backend."""
    return os.environ.get("DEFECTLENS_DESCRIBER", "").strip().lower() == "bedrock"


def _image_to_jpeg_bytes(image, quality: int = 90) -> bytes:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class BedrockDescriber:
    """Describer.describe contract, served by Claude Haiku on Bedrock Converse."""

    def __init__(self, model_id: str | None = None, region: str | None = None) -> None:
        self.model_id = model_id or os.environ.get(
            "DEFECTLENS_BEDROCK_MODEL", DEFAULT_MODEL_ID
        )
        self.region = region or os.environ.get(
            "DEFECTLENS_BEDROCK_REGION", DEFAULT_REGION
        )
        # No local VLM: /health reads model (vlm_loaded) and adapter_loaded.
        self.model = None
        self.adapter_loaded = False
        self._client = None

    def load(self) -> None:
        # No-op: the boto3 client is built lazily on first describe() so import
        # and construction stay dependency-free (mirrors Describer.load()'s
        # "cheap unless needed" shape). Kept for a uniform load() contract.
        pass

    def _bedrock_client(self):
        if self._client is None:
            import boto3  # lazy: keeps the module import AWS-free

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def converse(self, image, top_classes, audio_band=None) -> str:
        """Raw Bedrock Converse call — RAISES on failure.

        The public describe() swallows errors; this path does not, so the smoke
        test sees the real ValidationException / AccessDeniedException instead of
        a misleading empty string.
        """
        prompt = build_prompt(top_classes, audio_band)
        jpeg = _image_to_jpeg_bytes(image)
        resp = self._bedrock_client().converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
                        {"text": prompt},
                    ],
                }
            ],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.0},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()

    def describe(self, image, top_classes, audio_band=None) -> str:
        try:
            return self.converse(image, top_classes, audio_band)
        except Exception:
            # Description is optional — a Bedrock outage/throttle must not fail
            # the analysis; classification + RAG cards still return. Log at
            # warning so a persistent misconfig (bad model id, missing IAM/model
            # access) is visible in CloudWatch instead of silently empty.
            logger.warning("Bedrock describe failed", exc_info=True)
            return ""
