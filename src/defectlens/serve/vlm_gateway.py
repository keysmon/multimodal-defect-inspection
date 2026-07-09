"""SageMaker async VLM gateway — the serve API's bridge to the GPU endpoint (5.5c).

The fine-tuned Qwen2.5-VL classifier runs on a scale-to-zero SageMaker ASYNC
endpoint (infra/stacks/gpu_stack.py). Async rather than real-time precisely
because the endpoint sleeps at 0 instances: the client drops the request payload
in S3, the endpoint wakes (~5-8 min on a cold start), runs, and writes the result
to another S3 key the client polls. This gateway wraps that two-step dance behind
submit()/status() so serve.api's /analyze-vlm and /vlm-status stay thin.

Enabled only when SAGEMAKER_ENDPOINT is set (build_gateway_from_env returns None
otherwise), so the CPU-only deploy — which has no GPU endpoint — makes serve.api
return a graceful 503 instead of invoking a missing endpoint.

boto3 is imported lazily (like BedrockDescriber), so the module stays cheap to
import and the pure payload/response helpers unit-test with no AWS. Tests may also
inject fake s3/runtime clients to exercise submit()/status() offline.
"""
from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any
from urllib.parse import urlparse

# botocore ClientError codes that mean "the async result isn't written yet".
_MISSING_CODES = {"NoSuchKey", "NotFound", "404"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no AWS involved)
# ---------------------------------------------------------------------------


def build_payload(image_bytes: bytes, note: str | None) -> bytes:
    """The endpoint's request body: {"image_b64": ..., "note"?: ...}. Pure."""
    payload: dict[str, Any] = {"image_b64": base64.b64encode(image_bytes).decode("ascii")}
    if note:
        payload["note"] = note
    return json.dumps(payload).encode("utf-8")


def parse_output(raw: bytes) -> list[dict]:
    """Reshape the endpoint's {"classes": [[label, prob], ...]} into the
    [{"label", "score"}, ...] shape /analyze and the frontend already use. Pure.
    """
    data = json.loads(raw)
    return [{"label": label, "score": score} for label, score in data["classes"]]


def split_s3_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/key -> (bucket, key). Raises ValueError on anything else."""
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    return parsed.netloc, key


def _is_missing(exc: Exception) -> bool:
    """True when a boto3 S3 error means 'object not there yet' (still pending)."""
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in _MISSING_CODES


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class SageMakerAsyncGateway:
    """submit() an image to the async endpoint; status() polls its S3 output."""

    def __init__(
        self,
        endpoint_name: str,
        input_s3_uri: str,
        region: str | None = None,
        s3_client: Any = None,
        runtime_client: Any = None,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.input_s3_uri = input_s3_uri.rstrip("/")  # s3://bucket/prefix
        self.region = region
        self._s3 = s3_client
        self._runtime = runtime_client

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint_name and self.input_s3_uri)

    def _s3_client(self):
        if self._s3 is None:
            import boto3  # lazy: keeps the module import AWS-free

            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def _runtime_client(self):
        if self._runtime is None:
            import boto3

            self._runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        return self._runtime

    def submit(self, image_bytes: bytes, note: str | None) -> dict:
        """Upload the payload to the async-in prefix and fire the async invoke.

        Returns the job id + the S3 locations the caller polls. invoke_endpoint_async
        returns immediately (HTTP 202) — the model runs later, after the endpoint
        wakes.
        """
        bucket, prefix = split_s3_uri(self.input_s3_uri)
        key = f"{prefix.rstrip('/')}/{uuid.uuid4().hex}.json"
        self._s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=build_payload(image_bytes, note),
            ContentType="application/json",
        )
        resp = self._runtime_client().invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{bucket}/{key}",
            ContentType="application/json",
        )
        return {
            "job_id": resp["InferenceId"],
            "output_location": resp["OutputLocation"],
            "failure_location": resp.get("FailureLocation"),
        }

    def status(
        self, output_location: str, failure_location: str | None = None
    ) -> tuple[str, list[dict] | None]:
        """Poll the async result. ("ready", classes) once the output is written,
        ("failed", None) if a failure object appeared, else ("pending", None).
        """
        out_bucket, out_key = split_s3_uri(output_location)
        raw = self._get_object(out_bucket, out_key)
        if raw is not None:
            return "ready", parse_output(raw)
        if failure_location:
            fail_bucket, fail_key = split_s3_uri(failure_location)
            if self._get_object(fail_bucket, fail_key) is not None:
                return "failed", None
        return "pending", None

    def _get_object(self, bucket: str, key: str) -> bytes | None:
        """Read an S3 object, or None if it isn't written yet (still pending)."""
        try:
            return self._s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise


def build_gateway_from_env() -> SageMakerAsyncGateway | None:
    """Construct the gateway from env, or None when the GPU path isn't deployed.

    SAGEMAKER_ENDPOINT is the sole on-switch (serve.api returns 503 when it's
    unset). ASYNC_INPUT_S3 is the s3://bucket/prefix the request payloads go to;
    without it the gateway can't submit, so treat that as not-deployed too.
    """
    endpoint = os.environ.get("SAGEMAKER_ENDPOINT", "").strip()
    input_s3 = os.environ.get("ASYNC_INPUT_S3", "").strip()
    if not endpoint or not input_s3:
        return None
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("DEFECTLENS_BEDROCK_REGION")
        or None
    )
    return SageMakerAsyncGateway(endpoint, input_s3, region=region)
