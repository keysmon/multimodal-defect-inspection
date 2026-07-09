"""Tests for the GPU async endpoints /analyze-vlm and /vlm-status (Phase 5.5c).

Drives the real FastAPI routes with a stub gateway injected into create_app —
same stub-spy pattern as test_api.py. No AWS, no model.
"""
from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defectlens.serve.api import create_app


def make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


class StubGateway:
    """Fixed-response stand-in for SageMakerAsyncGateway."""

    def __init__(self, *, enabled=True, submit_result=None, status_result=("pending", None)):
        self.enabled = enabled
        self._submit_result = submit_result or {
            "job_id": "job-1",
            "output_location": "s3://b/async-out/job-1.out",
            "failure_location": "s3://b/async-fail/job-1.out",
        }
        self._status_result = status_result
        self.submit_calls = []
        self.status_calls = []

    def submit(self, image_bytes, note):
        self.submit_calls.append((image_bytes, note))
        return self._submit_result

    def status(self, output_location, failure_location=None):
        self.status_calls.append((output_location, failure_location))
        return self._status_result


# ---------------------------------------------------------------------------
# POST /analyze-vlm
# ---------------------------------------------------------------------------


def test_analyze_vlm_submits_and_returns_job_locations():
    gateway = StubGateway()
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)

    resp = client.post(
        "/analyze-vlm",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "hairline crack near sill"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "job_id": "job-1",
        "output_location": "s3://b/async-out/job-1.out",
        "failure_location": "s3://b/async-fail/job-1.out",
    }
    assert len(gateway.submit_calls) == 1
    image_bytes, note = gateway.submit_calls[0]
    assert isinstance(image_bytes, bytes)
    assert note == "hairline crack near sill"


def test_analyze_vlm_503_when_gateway_not_wired():
    # default create_app leaves vlm_gateway None -> GPU path not deployed
    app = create_app()
    client = TestClient(app)
    resp = client.post("/analyze-vlm", files={"file": ("t.png", make_png_bytes(), "image/png")})
    assert resp.status_code == 503
    assert "not deployed" in resp.json()["detail"].lower()


def test_analyze_vlm_503_when_gateway_disabled():
    app = create_app(vlm_gateway=StubGateway(enabled=False))
    client = TestClient(app)
    resp = client.post("/analyze-vlm", files={"file": ("t.png", make_png_bytes(), "image/png")})
    assert resp.status_code == 503


def test_analyze_vlm_400_on_non_image():
    gateway = StubGateway()
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)
    resp = client.post(
        "/analyze-vlm", files={"file": ("bad.txt", b"not an image", "text/plain")}
    )
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()
    assert gateway.submit_calls == []  # never submitted a bad payload


def test_analyze_vlm_blank_note_passes_none():
    gateway = StubGateway()
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)
    resp = client.post(
        "/analyze-vlm",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "   "},
    )
    assert resp.status_code == 200
    assert gateway.submit_calls[0][1] is None


# ---------------------------------------------------------------------------
# GET /vlm-status
# ---------------------------------------------------------------------------


def test_vlm_status_ready_returns_200_with_classes():
    classes = [{"label": "crack", "score": 0.9}, {"label": "spalling", "score": 0.1}]
    gateway = StubGateway(status_result=("ready", classes))
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)

    resp = client.get("/vlm-status", params={"output_location": "s3://b/async-out/job-1.out"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "classes": classes}
    assert gateway.status_calls == [("s3://b/async-out/job-1.out", None)]


def test_vlm_status_pending_returns_202():
    gateway = StubGateway(status_result=("pending", None))
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)

    resp = client.get("/vlm-status", params={"output_location": "s3://b/async-out/job-1.out"})
    assert resp.status_code == 202
    assert resp.json() == {"status": "pending"}


def test_vlm_status_failed_returns_200_with_failed_status():
    gateway = StubGateway(status_result=("failed", None))
    app = create_app(vlm_gateway=gateway)
    client = TestClient(app)

    resp = client.get(
        "/vlm-status",
        params={
            "output_location": "s3://b/async-out/job-1.out",
            "failure_location": "s3://b/async-fail/job-1.out",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "failed", "classes": None}
    assert gateway.status_calls == [
        ("s3://b/async-out/job-1.out", "s3://b/async-fail/job-1.out")
    ]


def test_vlm_status_503_when_gateway_not_wired():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/vlm-status", params={"output_location": "s3://b/async-out/job-1.out"})
    assert resp.status_code == 503
