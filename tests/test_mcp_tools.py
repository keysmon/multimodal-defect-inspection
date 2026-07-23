"""MCP tool layer: registration + delegation to SiteCheckClient.

The `mcp` SDK is an optional dependency (like boto3): CI does not install it,
so importorskip - the CI-red lesson from the Bedrock config test.
"""
import pytest

pytest.importorskip("mcp")

from defectlens.mcp_server import server  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def analyze_photo(self, path, note=""):
        self.calls.append(("analyze", path, note))
        return {"classes": [{"label": "crack", "score": 0.9}]}

    def search_standards(self, query):
        self.calls.append(("search", query))
        return {"cards": []}

    def run_walkthrough(self, photo_paths, visit_note="", photo_notes=None):
        self.calls.append(("walkthrough", tuple(photo_paths), visit_note))
        return {"report": {"per_photo": []}}


@pytest.fixture()
def fake_client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(server, "_client", lambda: fake)
    return fake


def test_three_tools_registered():
    import anyio

    tools = anyio.run(server.mcp.list_tools)
    assert {t.name for t in tools} == {
        "analyze_photo",
        "search_standards",
        "run_walkthrough",
    }


def test_analyze_tool_delegates(fake_client):
    result = server.analyze_photo(path="/tmp/x.jpg", note="hall")
    assert result["classes"][0]["label"] == "crack"
    assert fake_client.calls == [("analyze", "/tmp/x.jpg", "hall")]


def test_walkthrough_tool_delegates(fake_client):
    server.run_walkthrough(photo_paths=["/a.jpg", "/b.jpg"], visit_note="damp")
    assert fake_client.calls == [("walkthrough", ("/a.jpg", "/b.jpg"), "damp")]
