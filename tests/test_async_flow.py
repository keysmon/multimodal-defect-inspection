"""End-to-end wiring test for the CPU async path.

Drives the WHOLE flow through the real CpuJobStore, the real /analyze-jobs
routes, run_worker, and run_analysis - only the boto3 clients (in-memory S3) and
the models (stubs) are faked. The fake Lambda client runs the worker
synchronously on the Event self-invoke, so submit -> S3 payload -> worker ->
run_analysis -> S3 result -> poll is exercised as one connected path. This
catches payload-shape / S3-key / response-dict mismatches that the isolated unit
tests (which stub the store) cannot.
"""
import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defectlens.corpus import Card
from defectlens.rag.retrieve import Hit
from defectlens.serve.api import create_app
from defectlens.serve.async_jobs import CpuJobStore, run_worker
from defectlens.serve.recognizer import RecognitionResult


def make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


# --- fakes -----------------------------------------------------------------


class _Body:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class _Missing(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}  # post-ListBucket shape (404)


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _Missing()
        return {"Body": _Body(self.objects[(Bucket, Key)])}


class SyncInvokeLambda:
    """Simulates the async self-invoke by running the worker synchronously
    against the SAME store, so the result is written before submit returns."""

    def __init__(self):
        self.app = None
        self.store = None
        self.invocations = []

    def bind(self, app, store):
        self.app = app
        self.store = store

    def invoke(self, FunctionName, InvocationType, Payload):
        self.invocations.append((FunctionName, InvocationType))
        run_worker(self.app, json.loads(Payload), self.store)
        return {"StatusCode": 202}


# --- stubs (no models) -----------------------------------------------------


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

    def analyze_image_bytes(self, data, k=5, note=None):
        return self.result


class StubDescriber:
    def __init__(self, text="a description"):
        self.text = text

    def describe(self, image, top_labels, audio_band=None):
        return self.text


class StubAudioAnalyzer:
    enabled = False


def _build_flow(recognizer=None, describer=None):
    s3 = FakeS3()
    lam = SyncInvokeLambda()
    store = CpuJobStore(
        bucket="b", prefix="jobs", function_name="fn", s3_client=s3, lambda_client=lam
    )
    app = create_app(
        cpu_job_store=store,
        recognizer=recognizer or StubRecognizer(_result()),
        describer=describer or StubDescriber("a diagonal crack"),
        audio_analyzer=StubAudioAnalyzer(),
    )
    lam.bind(app, store)
    return app, store, s3, lam


def test_submit_worker_poll_end_to_end():
    app, store, s3, lam = _build_flow()
    client = TestClient(app)

    submit = client.post(
        "/analyze-jobs",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "crack near sill"},
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # The worker ran on the self-invoke (Event) and wrote its result to S3 in/out.
    assert lam.invocations == [("fn", "Event")]
    assert ("b", f"jobs/in/{job_id}.json") in s3.objects
    assert ("b", f"jobs/out/{job_id}.json") in s3.objects

    poll = client.get(f"/analyze-jobs/{job_id}")
    assert poll.status_code == 200
    body = poll.json()
    assert body["classes"][0] == {"label": "crack", "score": 0.9}
    assert body["cards"][0]["id"] == "c1"
    assert body["note"] == "crack near sill"
    assert body["description"] == "a diagonal crack"  # always included (no cap)


def test_end_to_end_worker_failure_surfaces_as_500_generic():
    class BoomRecognizer:
        def analyze_image_bytes(self, *a, **k):
            raise RuntimeError("s3://defectlens-phase3-ca-002559670021 AccessDenied arn:aws:iam::002559670021:role/x")

    app, store, s3, lam = _build_flow(recognizer=BoomRecognizer())
    client = TestClient(app)

    submit = client.post(
        "/analyze-jobs", files={"file": ("t.png", make_png_bytes(), "image/png")}
    )
    job_id = submit.json()["job_id"]
    # The worker wrote an err/ object (not out/).
    assert ("b", f"jobs/err/{job_id}.json") in s3.objects
    assert ("b", f"jobs/out/{job_id}.json") not in s3.objects

    poll = client.get(f"/analyze-jobs/{job_id}")
    assert poll.status_code == 500
    detail = poll.json()["detail"]
    assert detail == "analysis failed"  # generic
    assert "AccessDenied" not in detail and "arn:aws" not in detail  # no recon leak


def test_end_to_end_pending_before_worker_writes_result():
    # A no-op invoke (worker deferred) leaves out/ and err/ unwritten -> the poll
    # reads them as missing -> 202 pending, through the real store's _get path.
    app, store, s3, lam = _build_flow()
    # defer: accept the self-invoke but don't run the worker (result unwritten)
    lam.invoke = lambda FunctionName, InvocationType, Payload: {"StatusCode": 202}

    client = TestClient(app)
    submit = client.post(
        "/analyze-jobs", files={"file": ("t.png", make_png_bytes(), "image/png")}
    )
    poll = client.get(f"/analyze-jobs/{submit.json()['job_id']}")
    assert poll.status_code == 202
    assert poll.json()["status"] == "pending"
