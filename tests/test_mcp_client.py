"""SiteCheckClient against an httpx MockTransport - no network."""
import json

import pytest

httpx = pytest.importorskip("httpx")

from defectlens.mcp_server.client import DEFAULT_API_URL, SiteCheckClient  # noqa: E402


def _client(handler, **kwargs):
    return SiteCheckClient(
        base_url="http://testserver",
        poll_interval_s=0.0,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_analyze_photo_submits_multipart_then_polls_to_ready(tmp_path):
    img = tmp_path / "wall.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")
    polls = {"n": 0}

    def handler(request):
        if request.url.path == "/analyze-jobs" and request.method == "POST":
            assert b"fakejpeg" in request.read()
            return httpx.Response(202, json={"job_id": "j1"})
        assert request.url.path == "/analyze-jobs/j1"
        polls["n"] += 1
        if polls["n"] < 2:
            return httpx.Response(202, json={"status": "pending"})
        return httpx.Response(200, json={"classes": [{"label": "crack", "score": 0.9}]})

    result = _client(handler).analyze_photo(str(img), note="north wall")
    assert result["classes"][0]["label"] == "crack"
    assert polls["n"] == 2


def test_poll_failure_raises_with_api_detail(tmp_path):
    img = tmp_path / "wall.jpg"
    img.write_bytes(b"x")

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "j1"})
        return httpx.Response(500, json={"detail": "analysis failed"})

    with pytest.raises(RuntimeError, match="analysis failed"):
        _client(handler).analyze_photo(str(img))


def test_poll_deadline_raises_timeout(tmp_path):
    img = tmp_path / "wall.jpg"
    img.write_bytes(b"x")

    def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json={"job_id": "j1"})
        return httpx.Response(202, json={"status": "pending"})

    with pytest.raises(TimeoutError):
        _client(handler, timeout_s=0.0).analyze_photo(str(img))


def test_search_standards_posts_query():
    def handler(request):
        assert json.loads(request.read()) == {"query": "efflorescence basement"}
        return httpx.Response(200, json={"cards": [{"id": "epa-001"}]})

    assert _client(handler).search_standards("efflorescence basement")["cards"]


def test_run_walkthrough_sends_all_photos_and_notes(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"p{i}.jpg"
        p.write_bytes(b"img%d" % i)
        paths.append(str(p))

    def handler(request):
        if request.url.path == "/walkthrough-jobs" and request.method == "POST":
            body = request.read()
            assert b"img0" in body and b"img1" in body and b"damp smell" in body
            return httpx.Response(202, json={"job_id": "w1"})
        return httpx.Response(200, json={"report": {"per_photo": []}})

    result = _client(handler).run_walkthrough(
        paths, visit_note="damp smell", photo_notes=["", "stain here"]
    )
    assert "report" in result


def test_default_api_url_is_cloudfront_api():
    assert DEFAULT_API_URL == "https://d2wxjiu5re5mow.cloudfront.net/api"
