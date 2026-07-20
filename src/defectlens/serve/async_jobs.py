"""CPU async-job path - the /analyze cold-start fix (Phase 1 of the async design).

Cold CPU-Lambda `/analyze` blows the 29s API-gateway integration cap (model load
alone is ~24-29s), so the first click on a cold/recycled env 503s. This module
moves the heavy work OFF the request: a model-free submit route drops the job in
S3 and async self-invokes the SAME Lambda (InvocationType=Event, which has no
gateway cap), a poll route reads the S3 result, and the worker - dispatched by
lambda_handler on the worker event - loads models and runs the shared
``serve.api.run_analysis`` pipeline. No SQS/DynamoDB, no new function: one
container reprocesses its own queue.

Mirrors ``vlm_gateway`` (the SageMaker async path): pure helpers unit-test with
no AWS, boto3 is imported lazily, and callers may inject fake s3/lambda clients.
``split_s3_uri`` / ``_is_missing`` are reused from vlm_gateway (identical S3
semantics), so this module stays cheap to import - the serve.api import-sanity
test (no torch/transformers) still holds when api imports this lazily.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from io import BytesIO
from typing import Any

from PIL import Image

from defectlens.serve.vlm_gateway import _is_missing, split_s3_uri

logger = logging.getLogger(__name__)

# The worker has no gateway cap, so it uses a generous describe budget (the point
# of the async path: description is always included), still bounded so a stalled
# Bedrock call can't burn the whole function timeout.
_DEFAULT_WORKER_DESCRIBE_BUDGET_S = 60.0


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no AWS involved)
# ---------------------------------------------------------------------------


def build_job_payload(image_bytes: bytes, note: str | None, audio_bytes: bytes | None) -> bytes:
    """Encode a job's inputs as the S3 payload: image (+ optional note/audio). Pure."""
    payload: dict[str, Any] = {"image_b64": base64.b64encode(image_bytes).decode("ascii")}
    if note:
        payload["note"] = note
    if audio_bytes:
        payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
    return json.dumps(payload).encode("utf-8")


def parse_job_payload(raw: bytes) -> dict:
    """Decode a job payload into {image_bytes, note, audio_bytes}. Pure."""
    data = json.loads(raw)
    audio_b64 = data.get("audio_b64")
    return {
        "image_bytes": base64.b64decode(data["image_b64"]),
        "note": data.get("note"),
        "audio_bytes": base64.b64decode(audio_b64) if audio_b64 else None,
    }


def is_worker_event(event: Any) -> bool:
    """True when a Lambda event is a worker self-invocation (vs an HTTP event)."""
    return isinstance(event, dict) and "defectlens_job" in event


def worker_job_id(event: dict) -> str:
    return event["defectlens_job"]["job_id"]


def _worker_describe_budget() -> float:
    raw = os.environ.get("DEFECTLENS_WORKER_DESCRIBE_TIMEOUT_S", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("bad DEFECTLENS_WORKER_DESCRIBE_TIMEOUT_S=%r; using default", raw)
    return _DEFAULT_WORKER_DESCRIBE_BUDGET_S


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class CpuJobStore:
    """S3 job I/O + Lambda self-invoke for the CPU async path.

    submit() writes the payload under ``<prefix>/in/<job_id>.json`` and fires an
    Event invoke of ``function_name`` (itself); the worker reads that input, runs
    the analysis, and writes ``<prefix>/out/<job_id>.json`` (or ``.../err/...``);
    status() polls those.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        function_name: str,
        region: str | None = None,
        s3_client: Any = None,
        lambda_client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.function_name = function_name
        self.region = region
        self._s3 = s3_client
        self._lambda = lambda_client

    @property
    def enabled(self) -> bool:
        return bool(self.bucket and self.function_name)

    def _key(self, kind: str, job_id: str) -> str:
        return f"{self.prefix}/{kind}/{job_id}.json"

    def _s3_client(self):
        if self._s3 is None:
            import boto3  # lazy: keeps the module import AWS-free

            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def _lambda_client(self):
        if self._lambda is None:
            import boto3

            self._lambda = boto3.client("lambda", region_name=self.region)
        return self._lambda

    def submit(self, image_bytes: bytes, note: str | None, audio_bytes: bytes | None) -> dict:
        """Write the job input to S3 and async self-invoke the worker. Fast (no
        models): returns the job id the caller polls."""
        job_id = uuid.uuid4().hex
        self._s3_client().put_object(
            Bucket=self.bucket,
            Key=self._key("in", job_id),
            Body=build_job_payload(image_bytes, note, audio_bytes),
            ContentType="application/json",
        )
        self._lambda_client().invoke(
            FunctionName=self.function_name,
            InvocationType="Event",  # async: no gateway cap, internal retries
            Payload=json.dumps({"defectlens_job": {"job_id": job_id}}).encode("utf-8"),
        )
        return {"job_id": job_id}

    def status(self, job_id: str) -> tuple[str, dict | None]:
        """("ready", result) once the output is written, ("failed", err) if the
        worker wrote an error, else ("pending", None)."""
        out = self._get(self._key("out", job_id))
        if out is not None:
            return "ready", json.loads(out)
        err = self._get(self._key("err", job_id))
        if err is not None:
            return "failed", json.loads(err)
        return "pending", None

    def get_input(self, job_id: str) -> bytes:
        """Read the job's input payload (worker side). Raises if it isn't there."""
        return self._s3_client().get_object(
            Bucket=self.bucket, Key=self._key("in", job_id)
        )["Body"].read()

    def put_output(self, job_id: str, obj: dict) -> None:
        self._put(self._key("out", job_id), obj)

    def put_error(self, job_id: str, obj: dict) -> None:
        self._put(self._key("err", job_id), obj)

    def _put(self, key: str, obj: dict) -> None:
        self._s3_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(obj).encode("utf-8"),
            ContentType="application/json",
        )

    def _get(self, key: str) -> bytes | None:
        """Read an S3 object, or None if it isn't written yet (still pending)."""
        try:
            return self._s3_client().get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as exc:
            if _is_missing(exc):
                return None
            raise


def build_cpu_job_store_from_env() -> CpuJobStore | None:
    """Construct the store from env, or None when the async path isn't wired.

    CPU_JOBS_S3 (s3://bucket/prefix) is where jobs live; AWS_LAMBDA_FUNCTION_NAME
    (set automatically inside Lambda) is the self-invoke target. Without both -
    e.g. local dev - return None so the submit/status routes answer 503 and the
    frontend falls back to sync /analyze.
    """
    cpu_jobs_s3 = os.environ.get("CPU_JOBS_S3", "").strip()
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "").strip()
    if not cpu_jobs_s3 or not function_name:
        return None
    bucket, prefix = split_s3_uri(cpu_jobs_s3)
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("DEFECTLENS_BEDROCK_REGION")
        or None
    )
    return CpuJobStore(bucket=bucket, prefix=prefix, function_name=function_name, region=region)


def run_worker(app: Any, event: dict, store: CpuJobStore | None = None) -> dict:
    """Process one worker self-invocation: read the S3 input, run the shared
    analysis pipeline, write the S3 output (or an error object).

    ``ensure_loaded``/``run_analysis`` are imported lazily to avoid a
    serve.api <-> async_jobs import cycle and keep this module cheap to import.
    Any failure is captured to the err/ prefix so the poll route can surface it
    instead of the client hanging.
    """
    if store is None:
        store = build_cpu_job_store_from_env()
    if store is None:
        raise RuntimeError(
            "CPU async job store not configured (need CPU_JOBS_S3 + AWS_LAMBDA_FUNCTION_NAME)"
        )

    job_id = worker_job_id(event)
    try:
        payload = parse_job_payload(store.get_input(job_id))
        from defectlens.serve.api import ensure_loaded, run_analysis

        ensure_loaded(app)
        image_bytes = payload["image_bytes"]
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        result = run_analysis(
            app,
            image_bytes,
            img,
            payload["note"],
            payload["audio_bytes"],
            describe_budget=_worker_describe_budget(),
        )
        store.put_output(job_id, result)
    except Exception as exc:  # worker is fire-and-forget; surface via err/
        logger.exception("cpu async worker job %s failed", job_id)
        store.put_error(job_id, {"error": str(exc)})
    return {"job_id": job_id}
