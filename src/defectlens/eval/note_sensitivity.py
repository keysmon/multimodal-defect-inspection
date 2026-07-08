"""Note-sensitivity study (spec Phase 5, photo+note workstream).

Three conditions over an ambiguous crack/no_defect subset:
  empty        - note=None; MUST reproduce the no-note baseline exactly
                 (build_messages equality is unit-locked; scoring is
                 deterministic, so accuracy must match to the last image)
  informative  - hand-authored context notes (data/notes/*.json). CAVEAT
                 (reported in output): notes were authored by the project
                 authors while viewing the images; this measures prompt
                 sensitivity, not field accuracy gain.
  misleading   - one fixed off-topic note for every image (robustness:
                 accuracy should not collapse under irrelevant text)

Usage:
  python -m defectlens.eval.note_sensitivity --notes data/notes/ambiguous_subset_notes.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MISLEADING_NOTE = "kitchen area, recently repainted, no issues reported"


def select_ambiguous_rows(rows: list[dict], per_class: int = 20) -> list[dict]:
    """First `per_class` crack + no_defect rows by sorted image_path (deterministic)."""
    picked: list[dict] = []
    for cls in ("crack", "no_defect"):
        cls_rows = sorted(
            (r for r in rows if r["unified_label"] == cls),
            key=lambda r: r["image_path"],
        )
        picked.extend(cls_rows[:per_class])
    return picked


def build_conditions(notes: dict[str, str], misleading: str = MISLEADING_NOTE):
    """Map condition name -> (image_path -> note-or-None)."""
    return {
        "empty": lambda path: None,
        "informative": lambda path: notes.get(path),
        "misleading": lambda path: misleading,
    }


def main(argv: list[str] | None = None) -> None:
    from PIL import Image
    from tqdm import tqdm

    from defectlens.eval.vlm_topk import (
        _load_model_and_processor,
        pick_device,
        rank_answers,
        score_answers,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--notes", type=Path, default=Path("data/notes/ambiguous_subset_notes.json"))
    parser.add_argument("--adapter", type=Path, default=Path("models/qwen25vl-lora-v1"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--out", type=Path, default=Path("results/note_sensitivity.json"))
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.test_manifest)))
    subset = select_ambiguous_rows(rows, per_class=args.per_class)
    notes = json.loads(args.notes.read_text(encoding="utf-8"))
    conditions = build_conditions(notes)

    device = pick_device()
    model, processor = _load_model_and_processor(args.adapter, device, args.max_pixels)

    results: dict = {"n": len(subset), "authorship_caveat": (
        "informative notes were hand-authored by the project authors while "
        "viewing the images; this measures prompt sensitivity, not field accuracy gain"
    ), "conditions": {}}
    for name, note_fn in conditions.items():
        correct = 0
        preds = []
        for row in tqdm(subset, desc=name):
            img = Image.open(row["image_path"]).convert("RGB")
            loglik = score_answers(model, processor, img, device, note=note_fn(row["image_path"]))
            top = rank_answers(loglik)[0]
            preds.append({"path": row["image_path"], "true": row["unified_label"], "pred": top})
            correct += top == row["unified_label"]
        results["conditions"][name] = {"accuracy": correct / len(subset), "preds": preds}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, c in results["conditions"].items():
        print(f"{name:>12}: accuracy {c['accuracy']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
