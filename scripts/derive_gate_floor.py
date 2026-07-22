"""Derive the enrich-gate confidence floor from per-image eval dumps (plan B4).

Input: the JSONL written by `vlm_topk --dump-per-image` over the frozen v2
test split. For every candidate floor f, the gate would merge exactly the
predictions with top-1 confidence >= f, so:

  kept_correct_frac    = (# correct with conf >= f) / (# correct overall)
  merged_incorrect_frac = (# incorrect with conf >= f) / (# merged at f)

The chosen floor MAXIMIZES kept-correct subject to merged-incorrect <= 5%
(the plan's constraint). The FULL curve is published, not just the point, so
the tradeoff is auditable. Provenance (dump file, n, adapter) rides along.

Usage:
  python scripts/derive_gate_floor.py --dump results/vlm_v2_per_image.jsonl \
      [--max-merged-incorrect 0.05] [--out results/gate_floor_v2.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_dump(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{path}: empty dump")
    return records


def floor_curve(records: list[dict], floors: list[float]) -> list[dict]:
    """One curve row per candidate floor (pure; unit-tested)."""
    scored = [
        (r["probs"][r["ranked"][0]], r["ranked"][0] == r["true"]) for r in records
    ]
    n_correct = sum(1 for _conf, ok in scored if ok)
    curve = []
    for f in floors:
        kept = [(conf, ok) for conf, ok in scored if conf >= f]
        kept_correct = sum(1 for _conf, ok in kept if ok)
        kept_incorrect = len(kept) - kept_correct
        curve.append({
            "floor": round(f, 4),
            "merged": len(kept),
            "kept_correct_frac": round(kept_correct / n_correct, 4) if n_correct else 0.0,
            "merged_incorrect_frac": round(kept_incorrect / len(kept), 4) if kept else 0.0,
        })
    return curve


def choose_floor(curve: list[dict], max_merged_incorrect: float) -> dict | None:
    """Max kept-correct among floors meeting the merged-incorrect constraint.

    Ties break toward the LOWER floor (more merges at equal quality). Returns
    None when no floor satisfies the constraint (report honestly, don't force).
    """
    ok = [row for row in curve if row["merged_incorrect_frac"] <= max_merged_incorrect]
    if not ok:
        return None
    return max(ok, key=lambda row: (row["kept_correct_frac"], -row["floor"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--max-merged-incorrect", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/gate_floor_v2.json"))
    args = parser.parse_args()

    records = load_dump(args.dump)
    floors = [round(0.005 * i, 4) for i in range(0, 200)]  # 0.000 .. 0.995
    curve = floor_curve(records, floors)
    chosen = choose_floor(curve, args.max_merged_incorrect)

    payload = {
        "source_dump": str(args.dump),
        "n_records": len(records),
        "constraint_max_merged_incorrect": args.max_merged_incorrect,
        "chosen": chosen,
        "note": (
            "chosen = max kept_correct_frac s.t. merged_incorrect_frac <= "
            "constraint; ties -> lower floor. curve published for audit."
        ),
        "curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if chosen is None:
        print("NO floor satisfies the constraint — see the curve; not forcing one.")
    else:
        print(f"chosen floor: {chosen}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
