import inspect
import json
import subprocess
import sys

import pytest

from defectlens.eval import vlm_topk
from defectlens.eval.vlm_topk import ANSWER_TO_LABEL, rank_answers, results_payload
from defectlens.serve.describer import Describer
from defectlens.taxonomy import UNIFIED_CLASSES
from defectlens.train.qlora import HUMANIZED

# ---------------------------------------------------------------------------
# rank_answers
# ---------------------------------------------------------------------------


def test_rank_answers_orders_by_loglik_desc():
    loglik = {"crack": -0.5, "spalling": -2.0, "no_defect": -0.1}
    assert rank_answers(loglik) == ["no_defect", "crack", "spalling"]


def test_rank_answers_tie_break_by_label_name():
    loglik = {"spalling": -1.0, "crack": -1.0, "no_defect": -1.0}
    # tied scores -> alphabetical order of the label names
    assert rank_answers(loglik) == ["crack", "no_defect", "spalling"]


def test_rank_answers_single_entry():
    assert rank_answers({"crack": -3.2}) == ["crack"]


# ---------------------------------------------------------------------------
# ANSWER_TO_LABEL
# ---------------------------------------------------------------------------


def test_answer_to_label_is_bijective_with_humanized():
    assert set(ANSWER_TO_LABEL) == set(HUMANIZED.values())
    assert set(ANSWER_TO_LABEL.values()) == set(UNIFIED_CLASSES)
    assert len(ANSWER_TO_LABEL) == len(HUMANIZED)
    for label, answer in HUMANIZED.items():
        assert ANSWER_TO_LABEL[answer] == label


# ---------------------------------------------------------------------------
# results_payload
# ---------------------------------------------------------------------------


def _full_ranking(first_three: list[str]) -> list[str]:
    """A full 9-label ranking with `first_three` in front, remaining labels after."""
    rest = [c for c in UNIFIED_CLASSES if c not in first_three]
    return [*first_three, *rest]


def test_results_payload_shape_and_values():
    y_true = ["crack", "crack", "spalling", "no_defect"]
    ranked = [
        _full_ranking(["crack", "spalling", "no_defect"]),
        _full_ranking(["spalling", "crack", "no_defect"]),
        _full_ranking(["spalling", "crack", "no_defect"]),
        _full_ranking(["no_defect", "crack", "spalling"]),
    ]

    payload = results_payload(y_true, ranked, k_values=(1, 3))

    assert payload["model_kind"] == "vlm_loglik"
    assert payload["n_images"] == 4
    assert payload["classes"] == UNIFIED_CLASSES

    # macro top-1: crack 1/2, spalling 1/1, no_defect 1/1 -> mean = 0.8333...
    assert payload["macro_top1"] == pytest.approx(5 / 6)
    # macro top-3: all three classes hit within top-3 -> 1.0
    assert payload["macro_top3"] == pytest.approx(1.0)

    assert set(payload["per_class_top1"]) == set(UNIFIED_CLASSES)
    assert set(payload["per_class_top3"]) == set(UNIFIED_CLASSES)
    # classes absent from y_true -> NaN -> None
    assert payload["per_class_top1"]["exposed_rebar"] is None
    assert payload["per_class_top3"]["mold_algae"] is None

    cm = payload["confusion_matrix"]
    assert len(cm) == len(UNIFIED_CLASSES)
    assert all(len(row) == len(UNIFIED_CLASSES) for row in cm)
    assert sum(sum(row) for row in cm) == 4

    # NaN must not survive into JSON (RFC 8259 requires allow_nan=False to raise on NaN)
    json.dumps(payload, allow_nan=False)


def test_results_payload_default_k_values():
    y_true = ["crack"]
    ranked = [_full_ranking(["crack"])]
    payload = results_payload(y_true, ranked)
    assert "macro_top1" in payload
    assert "macro_top3" in payload


# ---------------------------------------------------------------------------
# Import sanity — module must stay cheap to import (no heavy ML deps at
# module level; they're lazy inside the model-facing functions/main()).
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_heavy_ml_deps():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.eval.vlm_topk\n"
            "for mod in ('torch', 'transformers', 'peft', 'bitsandbytes'):\n"
            "    assert mod not in sys.modules, f'{mod} imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# note-aware signatures
# ---------------------------------------------------------------------------


def test_score_answers_accepts_note_kwarg():
    assert "note" in inspect.signature(vlm_topk.score_answers).parameters


def test_rank_classes_accepts_note_kwarg():
    assert "note" in inspect.signature(Describer.rank_classes).parameters


def test_score_answers_forwards_note_to_build_messages(monkeypatch):
    seen = []

    def spy(image, label, note=None):
        seen.append(note)
        raise RuntimeError("stop before torch")

    monkeypatch.setattr(vlm_topk, "build_messages", spy)
    with pytest.raises(RuntimeError, match="stop before torch"):
        vlm_topk.score_answers(None, None, "img", "cpu", note="my note")
    assert seen == ["my note"]


def test_rank_classes_forwards_note_to_score_answers(monkeypatch):
    seen = {}

    def spy(model, processor, image, device, note=None):
        seen["note"] = note
        return {"crack": -0.1}

    monkeypatch.setattr(vlm_topk, "score_answers", spy)
    d = Describer()
    d.adapter_loaded = True
    d.model = d.processor = object()
    d.device = "cpu"
    result = d.rank_classes("img", note="musty smell")
    assert seen["note"] == "musty smell"
    assert result[0][0] == "crack"
