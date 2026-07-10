import json

import pytest

from defectlens.agent.providers import MockProvider
from defectlens.agent.tools import (
    Trace,
    classify_image,
    observe_image,
    retrieve_guidance,
    score_audio,
)


class FakeDescriber:
    def rank_classes(self, image, note=None):
        return [("crack", 0.91), ("spalling", 0.05), ("no_defect", 0.04)]


class FakeHit:
    def __init__(self, card_id, title, class_tags):
        self.card_id = card_id
        self.title = title
        self.class_tags = class_tags


class FakeRecognizer:
    def search_text(self, query, k=5):
        return [FakeHit("epa-001", "Crack repair", ["crack"])]


class FakeAudioFinding:
    band = "investigate"


class FakeAudioAnalyzer:
    def analyze(self, wav_bytes):
        return FakeAudioFinding()


def test_trace_records_spans(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    with trace.span("demo", {"arg": 1}) as span:
        span["result_digest"] = "ok"
    lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["step"] == "demo" and record["args"] == {"arg": 1}
    assert "elapsed_ms" in record


def test_span_records_error_and_propagates(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="bad input"):
        with trace.span("boom", {"arg": 1}):
            raise ValueError("bad input")
    line = (tmp_path / "trace.jsonl").read_text().strip()
    assert '"error":' in line
    record = json.loads(line)
    assert record["error"] == "ValueError: bad input"
    assert "elapsed_ms" in record


def test_classify_image_returns_ranking_and_traces(tmp_path):
    trace = Trace(tmp_path / "t.jsonl")
    ranking = classify_image(FakeDescriber(), image="IMG", trace=trace)
    assert ranking[0] == ("crack", 0.91)
    assert "classify_image" in (tmp_path / "t.jsonl").read_text()


def test_observe_image_parses_json_list(tmp_path):
    responses = ['```json\n[{"finding": "corroded valve", "severity": "moderate"}]\n```']
    provider = MockProvider(responses=responses)
    obs = observe_image(provider, image="IMG", trace=Trace(tmp_path / "t.jsonl"))
    assert obs == [{"finding": "corroded valve", "severity": "moderate"}]


def test_observe_image_returns_empty_on_unparseable(tmp_path):
    provider = MockProvider(responses=["I see nothing structured"])
    obs = observe_image(provider, image="IMG", trace=Trace(tmp_path / "t.jsonl"))
    assert obs == []


def test_observe_image_filters_non_dict_elements(tmp_path):
    responses = ['["garbage", {"finding": "x", "severity": "moderate"}]']
    provider = MockProvider(responses=responses)
    obs = observe_image(provider, image="IMG", trace=Trace(tmp_path / "t.jsonl"))
    assert obs == [{"finding": "x", "severity": "moderate"}]


def test_observe_image_recovers_unclosed_fence(tmp_path):
    # Fence never closed: the regex misses, the bracket-balanced scan recovers.
    responses = ['```json\n[{"finding": "hairline gap", "severity": "monitor"}]']
    provider = MockProvider(responses=responses)
    obs = observe_image(provider, image="IMG", trace=Trace(tmp_path / "t.jsonl"))
    assert obs == [{"finding": "hairline gap", "severity": "monitor"}]


def test_observe_image_balanced_scan_ignores_brackets_in_strings(tmp_path):
    # Prose brackets before the array and brackets inside JSON strings must not
    # confuse the depth counter; the real array is validated last-to-first.
    raw = 'Note [draft]: ```json\n[{"finding": "stain on [east] wall", "severity": "cosmetic"}]'
    provider = MockProvider(responses=[raw])
    obs = observe_image(provider, image="IMG", trace=Trace(tmp_path / "t.jsonl"))
    assert obs == [{"finding": "stain on [east] wall", "severity": "cosmetic"}]


def test_retrieve_guidance_maps_hits_to_citations(tmp_path):
    cites = retrieve_guidance(FakeRecognizer(), "crack remediation", trace=Trace(tmp_path / "t.jsonl"))
    assert cites[0]["card_id"] == "epa-001" and cites[0]["class_tags"] == ["crack"]


def test_score_audio_returns_finding_and_traces(tmp_path):
    trace = Trace(tmp_path / "t.jsonl")
    finding = score_audio(FakeAudioAnalyzer(), b"RIFFxxxx", trace=trace)
    assert finding.band == "investigate"
    assert "score_audio" in (tmp_path / "t.jsonl").read_text()
