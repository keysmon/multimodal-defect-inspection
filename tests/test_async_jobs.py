"""Unit tests for the CPU async-job path (serve/async_jobs.py).

Mirrors the vlm_gateway test style: pure helpers unit-tested with no AWS, and
the store exercised with injected fake s3/lambda clients so submit/status/worker
run fully offline.
"""
from io import BytesIO

import pytest
from PIL import Image

from defectlens.corpus import Card
from defectlens.rag.retrieve import Hit
from defectlens.serve.api import create_app
from defectlens.serve.async_jobs import (
    CpuJobStore,
    build_cpu_job_store_from_env,
    build_job_payload,
    is_worker_event,
    parse_job_payload,
    run_worker,
    worker_job_id,
)
from defectlens.serve.recognizer import RecognitionResult


def make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pure helpers (no AWS)
# ---------------------------------------------------------------------------


def test_job_payload_roundtrips_image_note_and_audio():
    png = make_png_bytes()
    raw = build_job_payload(png, "musty smell", b"\x00\x01wav")
    parsed = parse_job_payload(raw)
    assert parsed["image_bytes"] == png
    assert parsed["note"] == "musty smell"
    assert parsed["audio_bytes"] == b"\x00\x01wav"


def test_job_payload_omits_absent_note_and_audio():
    png = make_png_bytes()
    parsed = parse_job_payload(build_job_payload(png, None, None))
    assert parsed["image_bytes"] == png
    assert parsed["note"] is None
    assert parsed["audio_bytes"] is None


def test_is_worker_event_true_only_for_worker_shape():
    assert is_worker_event({"defectlens_job": {"job_id": "j1"}}) is True
    # An HTTP-API (Mangum) event has no defectlens_job key.
    assert is_worker_event({"version": "2.0", "routeKey": "POST /analyze"}) is False
    assert is_worker_event("not-a-dict") is False
    assert is_worker_event(None) is False


def test_worker_job_id_extracts_id():
    assert worker_job_id({"defectlens_job": {"job_id": "abc123"}}) == "abc123"


# ---------------------------------------------------------------------------
# Fake AWS clients
# ---------------------------------------------------------------------------


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Missing(Exception):
    """Mimics a botocore NoSuchKey ClientError shape."""

    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.puts: list[tuple[str, str, str | None]] = []

    def put_object(self, Bucket, Key, Body, ContentType=None):
        data = Body if isinstance(Body, bytes) else Body.encode()
        self.objects[(Bucket, Key)] = data
        self.puts.append((Bucket, Key, ContentType))

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _Missing()
        return {"Body": _Body(self.objects[(Bucket, Key)])}


class FakeLambda:
    def __init__(self):
        self.invocations: list[dict] = []

    def invoke(self, FunctionName, InvocationType, Payload):
        self.invocations.append(
            {"FunctionName": FunctionName, "InvocationType": InvocationType, "Payload": Payload}
        )
        return {"StatusCode": 202}


def make_store(s3=None, lam=None) -> CpuJobStore:
    return CpuJobStore(
        bucket="b",
        prefix="jobs",
        function_name="fn",
        s3_client=s3 or FakeS3(),
        lambda_client=lam or FakeLambda(),
    )


# ---------------------------------------------------------------------------
# CpuJobStore
# ---------------------------------------------------------------------------


def test_enabled_requires_bucket_and_function_name():
    assert make_store().enabled is True
    assert CpuJobStore(bucket="", prefix="jobs", function_name="fn").enabled is False
    assert CpuJobStore(bucket="b", prefix="jobs", function_name="").enabled is False


def test_submit_writes_input_and_async_self_invokes():
    import json

    s3, lam = FakeS3(), FakeLambda()
    store = make_store(s3, lam)
    png = make_png_bytes()

    out = store.submit(png, "a note", None)
    job_id = out["job_id"]
    assert job_id

    # Input payload written under jobs/in/<job_id>.json ...
    in_key = f"jobs/in/{job_id}.json"
    assert ("b", in_key) in s3.objects
    assert parse_job_payload(s3.objects[("b", in_key)])["image_bytes"] == png

    # ... and the SAME function was async-invoked (Event) with the job id.
    assert len(lam.invocations) == 1
    inv = lam.invocations[0]
    assert inv["FunctionName"] == "fn"
    assert inv["InvocationType"] == "Event"
    assert json.loads(inv["Payload"]) == {"defectlens_job": {"job_id": job_id}}


def test_status_ready_returns_written_output():
    store = make_store()
    store.put_output("j1", {"classes": [{"label": "crack", "score": 0.9}]})
    state, obj = store.status("j1")
    assert state == "ready"
    assert obj == {"classes": [{"label": "crack", "score": 0.9}]}


def test_status_failed_when_error_written():
    store = make_store()
    store.put_error("j1", {"error": "boom"})
    state, obj = store.status("j1")
    assert state == "failed"
    assert obj == {"error": "boom"}


def test_status_pending_when_neither_written():
    store = make_store()
    state, obj = store.status("nope")
    assert state == "pending"
    assert obj is None


def test_get_input_reads_back_submitted_payload():
    s3 = FakeS3()
    store = make_store(s3)
    out = store.submit(make_png_bytes(), None, None)
    raw = store.get_input(out["job_id"])
    assert parse_job_payload(raw)["note"] is None


# ---------------------------------------------------------------------------
# build_cpu_job_store_from_env
# ---------------------------------------------------------------------------


def test_build_from_env_returns_none_without_config(monkeypatch):
    monkeypatch.delenv("CPU_JOBS_S3", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    assert build_cpu_job_store_from_env() is None


def test_build_from_env_returns_none_without_function_name(monkeypatch):
    monkeypatch.setenv("CPU_JOBS_S3", "s3://b/phase5/cpu-jobs/")
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    assert build_cpu_job_store_from_env() is None


def test_build_from_env_constructs_store(monkeypatch):
    monkeypatch.setenv("CPU_JOBS_S3", "s3://mybucket/phase5/cpu-jobs/")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "serve-fn")
    store = build_cpu_job_store_from_env()
    assert store is not None
    assert store.bucket == "mybucket"
    assert store.function_name == "serve-fn"
    # key still lands under the configured prefix (trailing slash normalized)
    assert store._key("in", "j1") == "phase5/cpu-jobs/in/j1.json"


# ---------------------------------------------------------------------------
# run_worker (fake store + injected app components)
# ---------------------------------------------------------------------------


def _make_card(cid, tags, severity="urgent"):
    return Card(
        id=cid, title=f"title-{cid}", class_tags=tags, severity=severity,
        index_sentence=f"idx-{cid}", passage=f"passage-{cid}", citation=f"cite-{cid}",
        source_name=f"src-{cid}", source_url=f"https://example.com/{cid}",
        source_license="CC-BY-4.0",
    )


def _result():
    hits = [Hit(card=_make_card("c1", ["crack"]), distance=0.1)]
    return RecognitionResult(classes=[("crack", 0.9), ("spalling", 0.4)], severity="urgent", hits=hits)


class StubRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def analyze_image_bytes(self, data, k=5, note=None):
        self.calls.append((data, note))
        return self.result


class StubDescriber:
    def __init__(self, text="a description"):
        self.text = text

    def describe(self, image, top_labels, audio_band=None):
        return self.text


class StubAudioAnalyzer:
    enabled = False

    def analyze(self, wav_bytes):  # pragma: no cover - disabled in these tests
        raise AssertionError("disabled analyzer must not run")


class FakeStore:
    def __init__(self, input_payload: bytes):
        self._input = input_payload
        self.outputs: dict[str, dict] = {}
        self.errors: dict[str, dict] = {}

    def get_input(self, job_id):
        return self._input

    def put_output(self, job_id, obj):
        self.outputs[job_id] = obj

    def put_error(self, job_id, obj):
        self.errors[job_id] = obj


def _worker_app():
    return create_app(
        recognizer=StubRecognizer(_result()),
        describer=StubDescriber("a diagonal crack"),
        audio_analyzer=StubAudioAnalyzer(),
    )


def test_run_worker_writes_full_analysis_output():
    store = FakeStore(build_job_payload(make_png_bytes(), "note here", None))
    run_worker(_worker_app(), {"defectlens_job": {"job_id": "job-1"}}, store=store)

    assert "job-1" not in store.errors
    result = store.outputs["job-1"]
    assert result["classes"][0] == {"label": "crack", "score": 0.9}
    assert result["cards"][0]["id"] == "c1"
    assert result["note"] == "note here"
    # The worker has no gateway cap, so the description is always included.
    assert result["description"] == "a diagonal crack"


def test_run_worker_writes_error_on_failure():
    class BoomRecognizer:
        def analyze_image_bytes(self, *a, **k):
            raise RuntimeError("model exploded")

    app = create_app(
        recognizer=BoomRecognizer(),
        describer=StubDescriber(),
        audio_analyzer=StubAudioAnalyzer(),
    )
    store = FakeStore(build_job_payload(make_png_bytes(), None, None))
    run_worker(app, {"defectlens_job": {"job_id": "job-2"}}, store=store)

    assert "job-2" not in store.outputs
    assert "model exploded" in store.errors["job-2"]["error"]


def test_run_worker_uses_provided_store_or_env(monkeypatch):
    # With no env and no injected store, run_worker can't resolve a store.
    monkeypatch.delenv("CPU_JOBS_S3", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    with pytest.raises(RuntimeError):
        run_worker(_worker_app(), {"defectlens_job": {"job_id": "j"}})
