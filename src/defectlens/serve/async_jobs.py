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
import threading
import uuid
from io import BytesIO
from typing import Any

from PIL import Image

from defectlens.serve.vlm_gateway import _is_missing, split_s3_uri

logger = logging.getLogger(__name__)

# The worker has no gateway cap, so it uses a generous describe budget (the point
# of the async path: description is always included), but bounded so the
# worst-case cold worker - model load (~24-29s) + classify/RAG + this budget -
# stays under the 120s Lambda timeout with slack to write its result before the
# process is killed. (A stalled Bedrock call at a larger budget could otherwise
# leave no err/ object and the client polling forever.)
_DEFAULT_WORKER_DESCRIBE_BUDGET_S = 30.0


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no AWS involved)
# ---------------------------------------------------------------------------


def build_job_payload(image_bytes: bytes, note: str | None, audio_bytes: bytes | None) -> bytes:
    """Encode a single-photo analyze job as the S3 payload. Pure."""
    payload: dict[str, Any] = {"image_b64": base64.b64encode(image_bytes).decode("ascii")}
    if note:
        payload["note"] = note
    if audio_bytes:
        payload["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
    return json.dumps(payload).encode("utf-8")


def build_walkthrough_job_payload(photos: list[dict], visit_note: str | None) -> bytes:
    """Encode a walkthrough job (N photos + visit note) as the S3 payload. Pure.

    Each photo: {"photo_id", "image_bytes", "note"}. The "kind" discriminator
    routes the worker; analyze payloads carry no kind (back-compat with jobs
    already in flight when this shipped).
    """
    payload = {
        "kind": "walkthrough",
        "visit_note": visit_note,
        "photos": [
            {
                "photo_id": p["photo_id"],
                "image_b64": base64.b64encode(p["image_bytes"]).decode("ascii"),
                "note": p.get("note"),
            }
            for p in photos
        ],
    }
    return json.dumps(payload).encode("utf-8")


def parse_job_payload(raw: bytes) -> dict:
    """Decode a job payload; the returned dict always carries "kind". Pure.

    analyze (default, incl. legacy kind-less payloads):
      {"kind": "analyze", "image_bytes", "note", "audio_bytes"}
    walkthrough:
      {"kind": "walkthrough", "visit_note", "photos": [{photo_id, image_bytes, note}]}
    """
    data = json.loads(raw)
    if data.get("kind") == "walkthrough":
        return {
            "kind": "walkthrough",
            "visit_note": data.get("visit_note"),
            "photos": [
                {
                    "photo_id": p["photo_id"],
                    "image_bytes": base64.b64decode(p["image_b64"]),
                    "note": p.get("note"),
                }
                for p in data.get("photos", [])
            ],
        }
    audio_b64 = data.get("audio_b64")
    return {
        "kind": "analyze",
        "image_bytes": base64.b64decode(data["image_b64"]),
        "note": data.get("note"),
        "audio_bytes": base64.b64decode(audio_b64) if audio_b64 else None,
    }


def is_worker_event(event: Any) -> bool:
    """True when a Lambda event is a worker self-invocation (vs an HTTP event)."""
    return isinstance(event, dict) and "defectlens_job" in event


def worker_job_id(event: dict) -> str:
    return event["defectlens_job"]["job_id"]


def is_warmup_event(event: Any) -> bool:
    """True when a Lambda event is a keep-warm warmup (load models, no job).
    A distinct key from the worker event so neither is mistaken for the other."""
    return isinstance(event, dict) and event.get("defectlens_warmup") is True


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
        """Submit a single-photo analyze job (delegates to submit_payload)."""
        return self.submit_payload(build_job_payload(image_bytes, note, audio_bytes))

    def submit_payload(self, payload: bytes) -> dict:
        """Write a pre-built job payload to S3 and async self-invoke the worker.
        Fast (no models): returns the job id the caller polls."""
        job_id = uuid.uuid4().hex
        self._s3_client().put_object(
            Bucket=self.bucket,
            Key=self._key("in", job_id),
            Body=payload,
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

    def put_enrichment(self, job_id: str, obj: dict) -> None:
        """Persist the walkthrough job's GPU-enrichment fan-out mapping
        ({photo_id: {output_location, failure_location}}) under enr/."""
        self._put(self._key("enr", job_id), obj)

    def get_enrichment(self, job_id: str) -> dict | None:
        raw = self._get(self._key("enr", job_id))
        return json.loads(raw) if raw is not None else None

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


class LocalCpuJobStore:
    """In-process async job store for local dev (DEFECTLENS_LOCAL_JOBS=1).

    The walkthrough has NO sync route, so local development needs an async
    path without S3/Lambda: submit stores the payload in memory and runs the
    worker in a daemon thread; status() polls the same dicts. Mirrors
    CpuJobStore's public surface so the serve routes stay store-agnostic.
    bind(app) must be called before submits (the routes do this lazily -
    the worker needs the app to reach the loaded components).
    """

    def __init__(self, worker=None) -> None:
        self._worker = worker if worker is not None else run_worker
        self._app: Any = None
        self._inputs: dict[str, bytes] = {}
        self._outputs: dict[str, dict] = {}
        self._errors: dict[str, dict] = {}
        self._enrichments: dict[str, dict] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return True

    def bind(self, app: Any) -> None:
        self._app = app

    def submit(self, image_bytes: bytes, note: str | None, audio_bytes: bytes | None) -> dict:
        return self.submit_payload(build_job_payload(image_bytes, note, audio_bytes))

    def submit_payload(self, payload: bytes) -> dict:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._inputs[job_id] = payload
        event = {"defectlens_job": {"job_id": job_id}}
        threading.Thread(
            target=self._worker, args=(self._app, event, self), daemon=True
        ).start()
        return {"job_id": job_id}

    def status(self, job_id: str) -> tuple[str, dict | None]:
        with self._lock:
            if job_id in self._outputs:
                return "ready", self._outputs[job_id]
            if job_id in self._errors:
                return "failed", self._errors[job_id]
        return "pending", None

    def get_input(self, job_id: str) -> bytes:
        with self._lock:
            return self._inputs[job_id]

    def put_output(self, job_id: str, obj: dict) -> None:
        with self._lock:
            self._outputs[job_id] = obj

    def put_error(self, job_id: str, obj: dict) -> None:
        with self._lock:
            self._errors[job_id] = obj

    def put_enrichment(self, job_id: str, obj: dict) -> None:
        with self._lock:
            self._enrichments[job_id] = obj

    def get_enrichment(self, job_id: str) -> dict | None:
        with self._lock:
            return self._enrichments.get(job_id)


def build_cpu_job_store_from_env() -> CpuJobStore | LocalCpuJobStore | None:
    """Construct the store from env, or None when the async path isn't wired.

    DEFECTLENS_LOCAL_JOBS=1 selects the in-process LocalCpuJobStore (local
    dev: no S3/Lambda; the walkthrough has no sync fallback, so this is how
    it runs locally). Otherwise CPU_JOBS_S3 (s3://bucket/prefix) is where
    jobs live and AWS_LAMBDA_FUNCTION_NAME (set automatically inside Lambda)
    is the self-invoke target. Without either wiring - return None so the
    submit/status routes answer 503 and the frontend falls back to sync
    /analyze.
    """
    if os.environ.get("DEFECTLENS_LOCAL_JOBS", "").strip() == "1":
        return LocalCpuJobStore()
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


def run_worker(app: Any, event: dict, store: CpuJobStore | LocalCpuJobStore | None = None) -> dict:
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
        if payload["kind"] == "walkthrough":
            # Lazy module-style import: keeps this module cheap AND lets tests
            # monkeypatch serve.walkthrough.run_walkthrough_job.
            from defectlens.serve import walkthrough

            result = walkthrough.run_walkthrough_job(app, payload)
        else:
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
