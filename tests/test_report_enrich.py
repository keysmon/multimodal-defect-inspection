"""Enrichment gate: fine-tuned labels merge ONLY when confident + consistent."""
from defectlens.report.enrich import (
    CONFIDENCE_THRESHOLD,
    is_consistent,
    merge_enrichment,
)
from defectlens.report.schema import WalkthroughReport


def _report_dict():
    return {
        "concerns": [],
        "per_photo": [
            {
                "photo_id": "photo_1",
                "observation": "Patch of missing concrete with chunks on the floor below.",
                "cited": ["hud-008"],
                "no_evidence": False,
                "enrichment": None,
            },
            {
                "photo_id": "photo_2",
                "observation": "A gas furnace with a corroded flue pipe.",
                "cited": ["hud-008"],
                "no_evidence": False,
                "enrichment": None,
            },
            {
                "photo_id": "photo_3",
                "observation": "Not observed - verify on-site.",
                "cited": [],
                "no_evidence": True,
                "enrichment": None,
            },
        ],
        "summary": {
            "overall_assessment": "x",
            "assessment_citations": ["hud-008"],
            "action_items": [],
            "answers": [],
        },
        "disclaimer": "Initial diagnostic - verify before acting.",
        "flagged_claims": [],
        "cards": {},
    }


def test_is_consistent_keyword_match():
    assert is_consistent("spalling", "patch of missing concrete, spalled edges")
    assert is_consistent("spalling", "chunks of Missing Concrete on the floor")
    assert not is_consistent("spalling", "a gas furnace with a corroded flue")
    assert is_consistent("exposed_rebar", "corroded reinforcement is visible")
    assert not is_consistent("crack", "white powdery deposits")


def test_confident_consistent_label_merges():
    report, gate = merge_enrichment(_report_dict(), {"photo_1": ("spalling", 0.82)})
    f = report["per_photo"][0]
    assert f["enrichment"] == {"label": "spalling", "confidence": 0.82, "consistent": True}
    assert gate["kept"] == 1 and gate["dropped"] == []
    WalkthroughReport.model_validate(report)  # still schema-valid


def test_inconsistent_label_dropped():
    """Qwen forces out-of-scope photos into its 9 classes; the gate drops it."""
    report, gate = merge_enrichment(_report_dict(), {"photo_2": ("spalling", 0.95)})
    assert report["per_photo"][1]["enrichment"] is None
    assert gate["kept"] == 0
    assert gate["dropped"] == [
        {"photo_id": "photo_2", "label": "spalling", "confidence": 0.95,
         "reason": "inconsistent_with_observation"}
    ]


def test_low_confidence_dropped():
    report, gate = merge_enrichment(_report_dict(), {"photo_1": ("spalling", 0.31)})
    assert report["per_photo"][0]["enrichment"] is None
    assert gate["dropped"][0]["reason"] == "low_confidence"
    assert CONFIDENCE_THRESHOLD == 0.5


def test_no_evidence_photo_never_enriched():
    report, gate = merge_enrichment(_report_dict(), {"photo_3": ("crack", 0.9)})
    assert report["per_photo"][2]["enrichment"] is None
    assert gate["dropped"][0]["reason"] == "no_evidence_photo"


def test_unknown_photo_id_ignored_and_mixed_batch():
    labels = {
        "photo_1": ("spalling", 0.82),
        "photo_2": ("spalling", 0.95),
        "photo_9": ("crack", 0.9),
    }
    report, gate = merge_enrichment(_report_dict(), labels)
    assert gate["kept"] == 1 and len(gate["dropped"]) == 2
    reasons = {d["reason"] for d in gate["dropped"]}
    assert reasons == {"inconsistent_with_observation", "unknown_photo_id"}


def test_input_report_not_mutated():
    original = _report_dict()
    merged, _ = merge_enrichment(original, {"photo_1": ("spalling", 0.82)})
    assert original["per_photo"][0]["enrichment"] is None
    assert merged is not original
