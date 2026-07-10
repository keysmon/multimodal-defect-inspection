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


def _run(tmp_path, responses, provider=None, describer=None, load_image=None):
    provider = provider or MockProvider(responses=responses)
    report, usage, trace_path = run_inspection(
        property_id="p1",
        image_paths=["crack.jpg", "clean.jpg"],
        describer=describer or FakeDescriber(),
        recognizer=FakeRecognizer(),
        provider=provider,
        audio_analyzer=None,
        audio_bytes=None,
        out_dir=tmp_path,
        load_image=load_image or (lambda p: p),  # identity: fakes take the path itself
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


class RaisingDescriber:
    def rank_classes(self, image, note=None):
        if image == "clean.jpg":
            raise RuntimeError("classifier crashed")
        return [("crack", 0.91), ("no_defect", 0.09)]


class SummaryRaisingProvider:
    """Returns [] for image observations, raises on every text-only summary call."""

    name = "summary-raising"

    def __init__(self):
        self._calls = 0

    def complete(self, prompt, image=None, max_tokens=1024):
        self._calls += 1
        if image is None:
            raise RuntimeError("provider down")
        return "[]"

    def usage(self):
        return Usage(calls=self._calls)


def test_image_load_failure_isolated(tmp_path):
    # One corrupt image must not sink the report; findings come from the good one.
    def flaky_load(path):
        if path == "clean.jpg":
            raise OSError("corrupt file")
        return path

    report, _, _ = _run(tmp_path, ["[]", "Summary."], load_image=flaky_load)
    assert report.summary
    assert report.findings and all(f.evidence_photo == "crack.jpg" for f in report.findings)
    trace_text = next(tmp_path.glob("*.jsonl")).read_text()
    assert "image_error" in trace_text and "OSError: corrupt file" in trace_text


def test_classifier_failure_isolated(tmp_path):
    # rank_classes raising for one path is caught per-image, not fatal.
    report, _, _ = _run(tmp_path, ["[]", "Summary."], describer=RaisingDescriber())
    assert report.summary
    assert report.findings and all(f.evidence_photo == "crack.jpg" for f in report.findings)
    trace_text = next(tmp_path.glob("*.jsonl")).read_text()
    assert "image_error" in trace_text and "RuntimeError: classifier crashed" in trace_text


def test_synthesis_provider_exception_falls_back(tmp_path):
    # A provider that raises at synthesis must not discard the computed report.
    provider = SummaryRaisingProvider()
    report, _, _ = _run(tmp_path, None, provider=provider)
    measured = [f for f in report.findings if f.tier == "measured"]
    assert len(measured) == 1 and measured[0].defect_class == "crack"
    assert report.summary == "1 finding(s): crack."
    assert provider.usage().calls == 4  # 2 observe + 2 failed summary attempts
    trace_text = next(tmp_path.glob("*.jsonl")).read_text()
    assert "RuntimeError: provider down" in trace_text
