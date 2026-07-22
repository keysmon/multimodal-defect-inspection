"""VT corrosion severity secondary metric (plan B3).

Input: the JSONL from `vlm_topk --dump-per-image` over the frozen v2 test
split. For the vt_corrosion rows (whole bridge photos whose source_label is
the per-image WORST AASHTO condition state), report per-state recognition of
`corrosion_stain` at top-1 and top-3, plus the state -> severity band the
rule layer applies when the class IS recognized (fair -> monitor,
poor -> urgent, severe -> structural, per the spec).

This is deliberately a RECOGNITION metric: the model predicts the class, the
severity band is rule-mapped from the recorded state. It answers "does the
severity signal survive the classifier" — not "can the model grade severity"
(it is never asked to).

Usage:
  python scripts/eval_corrosion_severity.py --dump results/vlm_v2_per_image.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STATE_BAND = {"fair": "monitor", "poor": "urgent", "severe": "structural"}
TARGET = "corrosion_stain"


def severity_report(records: list[dict]) -> dict:
    """Per-state corrosion_stain recognition rates (pure; unit-tested)."""
    by_state: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["source_dataset"] == "vt_corrosion":
            by_state[r["source_label"]].append(r)
    if not by_state:
        raise SystemExit("no vt_corrosion rows in the dump")

    states = {}
    for state in sorted(by_state, key=lambda s: list(STATE_BAND).index(s) if s in STATE_BAND else 99):
        rows = by_state[state]
        top1 = sum(1 for r in rows if r["ranked"][0] == TARGET)
        top3 = sum(1 for r in rows if TARGET in r["ranked"][:3])
        states[state] = {
            "n": len(rows),
            "top1_corrosion_rate": round(top1 / len(rows), 4),
            "top3_corrosion_rate": round(top3 / len(rows), 4),
            "band_when_recognized": STATE_BAND.get(state, "unknown"),
        }
    return {
        "metric": (
            "per-AASHTO-state recognition of corrosion_stain on vt_corrosion "
            "test rows; severity band is RULE-mapped from the recorded state "
            "when the class is recognized (recognition metric, not model "
            "severity grading)"
        ),
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/corrosion_severity.json"))
    args = parser.parse_args()

    records = [json.loads(line) for line in args.dump.read_text().splitlines() if line.strip()]
    payload = severity_report(records)
    payload["source_dump"] = str(args.dump)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
