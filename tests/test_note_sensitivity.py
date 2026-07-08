from defectlens.eval.note_sensitivity import build_conditions, select_ambiguous_rows


def _row(path, label):
    return {"image_path": path, "unified_label": label}


def test_select_ambiguous_rows_balanced_and_deterministic():
    rows = [_row(f"a/crack_{i}.jpg", "crack") for i in range(30)]
    rows += [_row(f"a/plain_{i}.jpg", "no_defect") for i in range(30)]
    rows += [_row("a/mold.jpg", "mold_algae")]
    picked = select_ambiguous_rows(rows, per_class=20)
    assert len(picked) == 40
    labels = [r["unified_label"] for r in picked]
    assert labels.count("crack") == 20 and labels.count("no_defect") == 20
    assert picked == select_ambiguous_rows(rows, per_class=20)  # deterministic


def test_build_conditions_shapes():
    notes = {"a/crack_0.jpg": "hairline line on garage slab"}
    conds = build_conditions(notes, misleading="kitchen, repainted")
    assert conds["empty"]("a/crack_0.jpg") is None
    assert conds["informative"]("a/crack_0.jpg") == "hairline line on garage slab"
    assert conds["informative"]("a/unknown.jpg") is None  # missing note -> skip as empty
    assert conds["misleading"]("a/crack_0.jpg") == "kitchen, repainted"
