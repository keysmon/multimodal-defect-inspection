import subprocess
import sys
from collections import Counter

from defectlens.ingest import ManifestRow
from defectlens.train.qlora import (
    HUMANIZED,
    QUESTION,
    build_messages,
    class_weights,
    sample_weights,
    subset_rows,
)

ALL_CLASSES = [
    "crack",
    "spalling",
    "efflorescence",
    "exposed_rebar",
    "corrosion_stain",
    "mold_algae",
    "water_damage",
    "peeling_paint",
    "no_defect",
]


def _rows(label_counts: dict[str, int]) -> list[ManifestRow]:
    rows = []
    for label, count in label_counts.items():
        for i in range(count):
            rows.append(ManifestRow(f"data/raw/x/{label}/{i}.jpg", "x", label, label))
    return rows


# ---------------------------------------------------------------------------
# class_weights
# ---------------------------------------------------------------------------


def test_class_weights_inverse_frequency():
    weights = class_weights(["crack"] * 90 + ["exposed_rebar"] * 10)
    assert weights["crack"] == 1.0
    assert weights["exposed_rebar"] == 9.0


def test_class_weights_cap_enforced():
    weights = class_weights(["a"] * 1000 + ["b"] * 1, cap=20.0)
    assert weights["a"] == 1.0
    assert weights["b"] == 20.0


def test_class_weights_deterministic():
    labels = ["crack", "crack", "spalling"]
    assert class_weights(labels) == class_weights(labels)


def test_class_weights_empty():
    assert class_weights([]) == {}


# ---------------------------------------------------------------------------
# sample_weights
# ---------------------------------------------------------------------------


def test_sample_weights_maps_rows_to_class_weight():
    rows = _rows({"crack": 3, "exposed_rebar": 1})
    weights = sample_weights(rows)
    expected = class_weights([r.unified_label for r in rows])
    assert weights == [expected[r.unified_label] for r in rows]
    rebar_idx = next(i for i, r in enumerate(rows) if r.unified_label == "exposed_rebar")
    crack_idx = next(i for i, r in enumerate(rows) if r.unified_label == "crack")
    assert weights[rebar_idx] > weights[crack_idx]


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_structure():
    messages = build_messages("data/raw/x/crack/0.jpg", "crack")
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0] == {"type": "image", "image": "data/raw/x/crack/0.jpg"}
    assert content[1] == {"type": "text", "text": QUESTION}
    assert messages[1] == {"role": "assistant", "content": "crack"}


def test_build_messages_humanizes_label():
    messages = build_messages("img.jpg", "exposed_rebar")
    assert messages[1]["content"] == "exposed rebar"


def test_question_lists_all_nine_answer_options():
    for humanized in HUMANIZED.values():
        assert humanized in QUESTION


def test_humanized_covers_all_unified_classes():
    assert set(HUMANIZED) == set(ALL_CLASSES)


# ---------------------------------------------------------------------------
# subset_rows
# ---------------------------------------------------------------------------


def test_subset_rows_balances_across_classes():
    rows = _rows({label: 5 for label in ALL_CLASSES})
    subset = subset_rows(rows, 18, seed=42)
    assert len(subset) == 18
    counts = Counter(r.unified_label for r in subset)
    assert len(counts) == 9
    assert all(c == 2 for c in counts.values())


def test_subset_rows_deterministic():
    rows = _rows({"crack": 5, "spalling": 5})
    assert subset_rows(rows, 4, seed=42) == subset_rows(rows, 4, seed=42)


def test_subset_rows_stable_when_new_class_added():
    rows = _rows({label: 5 for label in ALL_CLASSES})
    before = subset_rows(rows, 9, seed=42)

    extra = [
        ManifestRow(f"data/raw/x/zz_extra/{i}.jpg", "x", "zz_extra", "zz_extra")
        for i in range(5)
    ]
    after = subset_rows(rows + extra, 9, seed=42)

    assert before == after


def test_subset_rows_handles_fewer_rows_than_requested():
    rows = _rows({"crack": 2})
    subset = subset_rows(rows, 10, seed=42)
    assert len(subset) == 2


# ---------------------------------------------------------------------------
# Import sanity — module must stay cheap to import (no heavy ML deps at
# module level; they're lazy inside the training-assembly functions/main()).
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_heavy_ml_deps():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.train.qlora\n"
            "for mod in ('torch', 'transformers', 'peft', 'bitsandbytes'):\n"
            "    assert mod not in sys.modules, f'{mod} imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
