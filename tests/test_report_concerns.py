"""Concern extraction from the technician's visit note."""
from defectlens.agent.providers import MockProvider
from defectlens.report.concerns import MAX_CONCERNS, extract_concerns


def test_empty_or_none_note_returns_no_concerns_without_llm_call():
    p = MockProvider(responses=[])
    assert extract_concerns(p, None) == []
    assert extract_concerns(p, "   ") == []
    assert p.calls == []


def test_extracts_string_array_text_only():
    p = MockProvider(responses=['["is the crack active?", "damp smell in stairwell"]'])
    concerns = extract_concerns(p, "crack near window; damp smell")
    assert concerns == ["is the crack active?", "damp smell in stairwell"]
    assert p.calls[0].had_image is False
    assert "crack near window; damp smell" in p.calls[0].prompt


def test_dedupes_and_caps():
    many = [f"concern {i}" for i in range(12)] + ["concern 0"]
    import json

    p = MockProvider(responses=[json.dumps(many)])
    concerns = extract_concerns(p, "long note")
    assert len(concerns) == MAX_CONCERNS
    assert concerns == [f"concern {i}" for i in range(MAX_CONCERNS)]


def test_unparseable_falls_back_to_whole_note_as_one_concern():
    p = MockProvider(responses=["I think the main issues are cracks."])
    assert extract_concerns(p, "  the crack worries me  ") == ["the crack worries me"]


def test_provider_exception_falls_back_to_whole_note():
    class Boom:
        name = "boom"

        def complete(self, prompt, image=None, max_tokens=1024, images=None):
            raise RuntimeError("throttled")

    assert extract_concerns(Boom(), "note text") == ["note text"]
