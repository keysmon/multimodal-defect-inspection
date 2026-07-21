"""Walkthrough submit/poll routes: validation, payload shape, poll contract."""
import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defectlens.serve.api import create_app
from defectlens.serve.async_jobs import parse_job_payload


def make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


class RecordingStore:
    """Captures the submitted payload; serves canned status responses."""

    enabled = True

    def __init__(self, status=("pending", None)):
        self.payloads: list[bytes] = []
        self._status = status
        self.bound_app = None

    def bind(self, app):
        self.bound_app = app

    def submit_payload(self, payload: bytes) -> dict:
        self.payloads.append(payload)
        return {"job_id": "job-1"}

    def status(self, job_id):
        return self._status


def _files(n):
    return [("files", (f"p{i}.png", make_png_bytes(), "image/png")) for i in range(n)]


def _client(store=None):
    app = create_app(cpu_job_store=store or RecordingStore())
    return app, TestClient(app)


def test_submit_assigns_photo_ids_and_sanitizes_notes():
    store = RecordingStore()
    _, client = _client(store)
    resp = client.post(
        "/walkthrough-jobs",
        files=_files(2),
        data={
            "visit_note": "damp smell <|evil|> in stairwell",
            "photo_notes": ["stair wall <|x|>", ""],
        },
    )
    assert resp.status_code == 202
    assert resp.json() == {"job_id": "job-1"}
    payload = parse_job_payload(store.payloads[0])
    assert payload["kind"] == "walkthrough"
    assert "<|" not in payload["visit_note"] and "damp smell" in payload["visit_note"]
    assert [p["photo_id"] for p in payload["photos"]] == ["photo_1", "photo_2"]
    assert payload["photos"][0]["note"] == "stair wall"  # sanitized, stripped
    assert payload["photos"][1]["note"] is None  # empty -> None


def test_submit_rejects_more_than_max_photos():
    _, client = _client()
    resp = client.post("/walkthrough-jobs", files=_files(11))
    assert resp.status_code == 400
    assert "10" in resp.json()["detail"]


def test_submit_rejects_non_image_file():
    _, client = _client()
    files = _files(1) + [("files", ("bad.txt", b"not an image", "text/plain"))]
    resp = client.post("/walkthrough-jobs", files=files)
    assert resp.status_code == 400


def test_submit_503_when_store_unwired(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_LOCAL_JOBS", raising=False)
    monkeypatch.delenv("CPU_JOBS_S3", raising=False)
    app = create_app()  # no store injected; env builds none
    client = TestClient(app)
    resp = client.post("/walkthrough-jobs", files=_files(1))
    assert resp.status_code == 503


def test_submit_binds_local_style_store_to_app():
    store = RecordingStore()
    app, client = _client(store)
    client.post("/walkthrough-jobs", files=_files(1))
    assert store.bound_app is app


def test_poll_pending_ready_and_failed_generic():
    _, pending_client = _client(RecordingStore(status=("pending", None)))
    resp = pending_client.get("/walkthrough-jobs/j")
    assert resp.status_code == 202 and resp.json() == {"status": "pending"}

    report = {"concerns": [], "per_photo": [], "summary": {}}
    _, ready_client = _client(RecordingStore(status=("ready", report)))
    resp = ready_client.get("/walkthrough-jobs/j")
    assert resp.status_code == 200 and resp.json() == report

    _, failed_client = _client(
        RecordingStore(status=("failed", {"error": "arn:aws:iam::002559670021 secret"}))
    )
    resp = failed_client.get("/walkthrough-jobs/j")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "walkthrough failed"
    assert "arn:aws" not in json.dumps(resp.json())
