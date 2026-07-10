from defectlens.agent.inspect import run_inspection
from defectlens.agent.providers import MockProvider, Usage


class FakeDescriber:
    def rank_classes(self, image, note=None):
        if image == "crack.jpg":
            return [("crack", 0.91), ("no_defect", 0.09)]
        return [("no_defect", 0.88), ("crack", 0.12)]


class FakeHit:
    card_id = "epa-001"
    title = "Crack repair"
    class_tags = ["crack"]


class FakeRecognizer:
    def search_text(self, query, k=3):
        return [FakeHit()]


class RecordingProvider:
    """Mock that records the max_tokens each call was given."""

    name = "recording"

    def __init__(self, responses):
        self.responses = list(responses)
        self.max_tokens_seen: list[int] = []
        self._calls = 0

    def complete(self, prompt, image=None, max_tokens=1024):
        response = self.responses[self._calls]
        self._calls += 1
        self.max_tokens_seen.append(max_tokens)
        return response

    def usage(self):
        return Usage(calls=self._calls)


def _run(tmp_path, responses, provider=None):
    provider = provider or MockProvider(responses=responses)
    report, usage, trace_path = run_inspection(
        property_id="p1",
        image_paths=["crack.jpg", "clean.jpg"],
        describer=FakeDescriber(),
        recognizer=FakeRecognizer(),
        provider=provider,
        audio_analyzer=None,
        audio_bytes=None,
        out_dir=tmp_path,
        load_image=lambda p: p,  # identity: fakes take the path itself
    )
    return report, usage, provider


def test_measured_finding_from_confident_classification(tmp_path):
    report, _, _ = _run(tmp_path, ["[]", "[]", "Summary prose."])
    measured = [f for f in report.findings if f.tier == "measured"]
    assert len(measured) == 1
    assert measured[0].defect_class == "crack"
    assert measured[0].citations[0].card_id == "epa-001"


def test_no_defect_photo_produces_no_measured_finding(tmp_path):
    report, _, _ = _run(tmp_path, ["[]", "[]", "Summary."])
    assert all(f.evidence_photo != "clean.jpg" or f.tier == "observation"
               for f in report.findings)


def test_observation_findings_flow_through(tmp_path):
    obs = '[{"finding": "corroded valve", "severity": "moderate"}]'
    report, _, _ = _run(tmp_path, [obs, "[]", "Summary."])
    observations = [f for f in report.findings if f.tier == "observation"]
    assert observations and observations[0].finding == "corroded valve"


def test_summary_fallback_keeps_report_valid(tmp_path):
    # synthesis returns empty twice -> deterministic fallback summary
    report, _, provider = _run(tmp_path, ["[]", "[]", "", ""])
    assert report.summary  # non-empty
    assert provider.usage().calls == 4


def test_trace_file_written(tmp_path):
    _run(tmp_path, ["[]", "[]", "Summary."])
    traces = list(tmp_path.glob("*.jsonl"))
    assert traces and "classify_image" in traces[0].read_text()


def test_token_budgets_observe_512_synthesis_1024(tmp_path):
    provider = RecordingProvider(["[]", "[]", "Summary."])
    _run(tmp_path, None, provider=provider)
    assert provider.max_tokens_seen == [512, 512, 1024]
