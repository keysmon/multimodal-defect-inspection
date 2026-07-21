"""serve.walkthrough: provider selection precedence + worker glue."""
import io
import json
from dataclasses import dataclass

import pytest
from PIL import Image

from defectlens.agent.providers import MockProvider
from defectlens.serve.api import create_app
from defectlens.serve.walkthrough import build_report_provider, run_walkthrough_job


@dataclass(frozen=True)
class FakeCard:
    id: str
    title: str = "t"
    class_tags: tuple = ("crack",)
    passage: str = "p"
    severity: str = "monitor"
    citation: str = "c"
    source_name: str = "s"
    source_url: str = "https://example.com"


@dataclass(frozen=True)
class FakeHit:
    card: FakeCard


class StubRecognizer:
    def analyze_image_bytes(self, data, k=5, note=None):
        class R:
            hits = [FakeHit(FakeCard("crack-01"))]

        return R()

    def search_text(self, query, k=5):
        return []


class ChatDescriber:
    def chat(self, prompt, image=None, max_new_tokens=400, images=None):
        return ""


class NoChatDescriber:
    """Mirrors BedrockDescriber: describes, but cannot chat."""


def _app(**kwargs):
    defaults = dict(
        recognizer=StubRecognizer(),
        describer=NoChatDescriber(),
        text_searcher=object(),
        audio_analyzer=type("A", (), {"enabled": False})(),
        vlm_gateway=type("G", (), {"enabled": False})(),
    )
    defaults.update(kwargs)
    return create_app(**defaults)


def test_injected_provider_wins(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_DESCRIBER", "bedrock")
    app = _app()
    injected = MockProvider(responses=[])
    app.state.report_provider = injected
    assert build_report_provider(app) is injected


def test_bedrock_env_selects_bedrock_provider(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_DESCRIBER", "bedrock")
    monkeypatch.setenv("DEFECTLENS_BEDROCK_MODEL", "my-model")
    monkeypatch.setenv("DEFECTLENS_BEDROCK_REGION", "us-east-1")
    provider = build_report_provider(_app())
    assert provider.name == "bedrock-haiku-4.5"
    assert provider._model_id == "my-model"
    assert provider._region == "us-east-1"


def test_local_chat_describer_selected_when_not_bedrock(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_DESCRIBER", raising=False)
    d = ChatDescriber()
    provider = build_report_provider(_app(describer=d))
    assert provider.name == "local-qwen2.5-vl-3b"
    assert provider._describer is d


def test_no_provider_available_returns_none(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_DESCRIBER", raising=False)
    assert build_report_provider(_app()) is None


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def test_run_walkthrough_job_returns_report_dict(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_DESCRIBER", raising=False)
    app = _app()
    app.state.report_provider = MockProvider(
        responses=[
            '["is it bad?"]',
            json.dumps(
                {
                    "per_photo": [
                        {"photo_id": "photo_1", "observation": "crack", "cited": ["crack-01"]}
                    ],
                    "summary": {
                        "overall_assessment": {"text": "One crack.", "citations": ["crack-01"]},
                        "action_items": [],
                        "answers": [
                            {"concern": "is it bad?", "answer": "monitor", "citations": ["crack-01"]}
                        ],
                    },
                }
            ),
        ]
    )
    payload = {
        "kind": "walkthrough",
        "visit_note": "worried about the crack",
        "photos": [{"photo_id": "photo_1", "image_bytes": _png(), "note": None}],
    }
    result = run_walkthrough_job(app, payload)
    assert result["per_photo"][0]["cited"] == ["crack-01"]
    assert result["summary"]["answers"][0]["concern"] == "is it bad?"
    assert result["cards"]["crack-01"]["title"] == "t"


def test_run_walkthrough_job_without_provider_raises(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_DESCRIBER", raising=False)
    with pytest.raises(RuntimeError, match="reasoning provider"):
        run_walkthrough_job(
            _app(),
            {"kind": "walkthrough", "visit_note": None,
             "photos": [{"photo_id": "photo_1", "image_bytes": _png(), "note": None}]},
        )
