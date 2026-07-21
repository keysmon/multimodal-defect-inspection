"""LLM provider abstraction for the inspection workflow.

One interface, three implementations:
- MockProvider: deterministic, for tests and dry runs.
- LocalQwenProvider: wraps the loaded Describer (adapter OFF), $0, day-one.
- BedrockHaikuProvider: personal-AWS path; account Bedrock quota is 0 as of
  2026-07-10, so it stays live-untested until activation (fail-fast client
  config per serve.bedrock_describer).
Token counts are estimates (len/4) for local/mock; Bedrock reports real usage.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class ProviderCall:
    prompt: str
    had_image: bool
    n_images: int = 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _as_image_list(image, images) -> list:
    """Normalize the (image, images) pair to a list; both set is a caller bug."""
    if images is not None and image is not None:
        raise ValueError("pass either image or images, not both")
    if images is not None:
        return list(images)
    return [image] if image is not None else []


def _image_to_png_bytes(image) -> bytes:
    """Encode a PIL image as lossless PNG bytes; PNG carries RGBA natively."""
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class LLMProvider(Protocol):
    name: str

    def complete(
        self, prompt: str, image=None, max_tokens: int = 1024, images: list | None = None
    ) -> str: ...

    def usage(self) -> Usage: ...


@dataclass
class MockProvider:
    responses: list[str]
    name: str = "mock"
    calls: list[ProviderCall] = field(default_factory=list)
    _usage: Usage = field(default_factory=Usage)

    def complete(
        self, prompt: str, image=None, max_tokens: int = 1024, images: list | None = None
    ) -> str:
        imgs = _as_image_list(image, images)
        response = self.responses[len(self.calls)]
        self.calls.append(
            ProviderCall(prompt=prompt, had_image=bool(imgs), n_images=len(imgs))
        )
        self._usage.calls += 1
        self._usage.input_tokens += _estimate_tokens(prompt)
        self._usage.output_tokens += _estimate_tokens(response)
        return response

    def usage(self) -> Usage:
        return dataclasses.replace(self._usage)


class LocalQwenProvider:
    """Adapter-OFF base Qwen2.5-VL via the already-loaded Describer."""

    name = "local-qwen2.5-vl-3b"

    def __init__(self, describer) -> None:
        self._describer = describer
        self._usage = Usage()

    def complete(
        self, prompt: str, image=None, max_tokens: int = 1024, images: list | None = None
    ) -> str:
        response = self._describer.chat(
            prompt, image=image, images=images, max_new_tokens=max_tokens
        )
        self._usage.calls += 1
        self._usage.input_tokens += _estimate_tokens(prompt)
        self._usage.output_tokens += _estimate_tokens(response)
        return response  # cost stays 0.0: local compute

    def usage(self) -> Usage:
        return dataclasses.replace(self._usage)


# Haiku 4.5 on-demand pricing (USD per million tokens), for cost-per-report.
_HAIKU_IN_PER_MTOK = 1.00
_HAIKU_OUT_PER_MTOK = 5.00


class BedrockHaikuProvider:
    """Bedrock converse path (personal AWS). Live-untested until quota > 0."""

    name = "bedrock-haiku-4.5"

    def __init__(self, model_id: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                 region: str = "ca-central-1") -> None:
        self._model_id = model_id
        self._region = region
        self._client = None
        self._usage = Usage()

    def _ensure_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=Config(retries={"total_max_attempts": 1, "mode": "standard"},
                              connect_timeout=3, read_timeout=60),
            )
        return self._client

    def complete(
        self, prompt: str, image=None, max_tokens: int = 1024, images: list | None = None
    ) -> str:
        content: list[dict] = [
            {"image": {"format": "png", "source": {"bytes": _image_to_png_bytes(img)}}}
            for img in _as_image_list(image, images)
        ]
        content.append({"text": prompt})
        resp = self._ensure_client().converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": content}],
            # temperature 0 (matches bedrock_describer): the walkthrough eval
            # regression-gates on a frozen golden set, so sampling noise would
            # false-positive the 0.02 tolerance.
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        u = resp.get("usage", {})
        self._usage.calls += 1
        self._usage.input_tokens += u.get("inputTokens", 0)
        self._usage.output_tokens += u.get("outputTokens", 0)
        self._usage.cost_usd += (
            u.get("inputTokens", 0) * _HAIKU_IN_PER_MTOK
            + u.get("outputTokens", 0) * _HAIKU_OUT_PER_MTOK
        ) / 1_000_000
        return resp["output"]["message"]["content"][0]["text"]

    def usage(self) -> Usage:
        return dataclasses.replace(self._usage)
