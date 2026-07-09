"""Machine-type audio retrieval eval (Phase 5.3 — spec metric adaptation).

MIMII labels clips normal/abnormal only; fault families are unlabeled, so the
spec's "correct fault-family retrieval" is unmeasurable as written. We measure
the machine-type proxy instead: a fan clip's top-5 CLAP-retrieved hvac-* cards
should carry fan-family tags, not pump-family. Accuracy = fraction of clips
where >= threshold of the top-k cards carry a tag from the clip's machine family.
Measure-and-report; no gate. Fault-family retrieval quality stays qualitative.

Usage: python -m defectlens.eval.audio_retrieval
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from defectlens.taxonomy import AUDIO_FAULT_TAGS

# Machine -> fault-family tag set. The overlap tags (bearing_wear,
# motor_electrical, mounting_vibration, normal_operation) intentionally appear
# in BOTH sets: those faults occur on fans and pumps alike, so such a card is a
# correct hit for either machine.
FAN_FAMILY = frozenset(
    {
        "fan_imbalance", "belt_drive", "airflow_restriction",
        "bearing_wear", "motor_electrical", "mounting_vibration", "normal_operation",
    }
)
PUMP_FAMILY = frozenset(
    {
        "pump_cavitation", "pump_seal_leak",
        "bearing_wear", "motor_electrical", "mounting_vibration", "normal_operation",
    }
)
FAMILY = {"fan": FAN_FAMILY, "pump": PUMP_FAMILY}

# Sanity-check the family sets against the canonical audio tag universe so a
# taxonomy rename surfaces here instead of silently zeroing the accuracy.
assert FAN_FAMILY <= set(AUDIO_FAULT_TAGS), FAN_FAMILY - set(AUDIO_FAULT_TAGS)
assert PUMP_FAMILY <= set(AUDIO_FAULT_TAGS), PUMP_FAMILY - set(AUDIO_FAULT_TAGS)


def card_matches_family(card_tags, family) -> bool:
    """True if the card carries any tag in the machine's family set."""
    return any(t in family for t in card_tags)


def clip_is_correct(hit_tag_lists, family, threshold: int = 3) -> bool:
    """True if >= threshold of the retrieved cards carry a family tag."""
    return sum(card_matches_family(tags, family) for tags in hit_tag_lists) >= threshold


def sample_test_clips(rows, n: int, seed: int) -> list:
    """Deterministic seeded sample of n TEST-split clips (fewer if not enough)."""
    test = [r for r in rows if r.split == "test"]
    rng = random.Random(seed)
    return rng.sample(test, min(n, len(test)))


def main(argv: list[str] | None = None) -> None:
    import numpy as np
    from tqdm import tqdm

    from defectlens.audio.dataset import scan_machine_dir
    from defectlens.audio.embed import embed_audio_files, load_clap
    from defectlens.eval.clip_zeroshot import pick_device
    from defectlens.rag import audio_db

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/audio"))
    parser.add_argument("--machines", nargs="+", default=["fan", "pump"])
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/audio_retrieval.json"))
    args = parser.parse_args(argv)

    try:
        conn = audio_db.connect()
    except Exception as e:
        raise SystemExit("pgvector DB unreachable — docker compose up -d db") from e
    n_cards = conn.execute("SELECT count(*) FROM audio_card_vectors").fetchone()[0]
    if n_cards == 0:
        raise SystemExit(
            "audio_card_vectors is empty — run `python -m defectlens.rag.audio_embed_cards`"
        )

    device = pick_device()
    model, processor = load_clap(device)

    per_machine: dict = {}
    total_correct = 0
    total_clips = 0
    for machine in args.machines:
        rows = scan_machine_dir(args.audio_root / machine, machine=machine)
        clips = sample_test_clips(rows, args.sample, args.seed)
        if not clips:
            raise SystemExit(f"no test clips for {machine} under {args.audio_root}")
        embs = embed_audio_files(
            model, processor, [Path(r.path) for r in clips], device, args.batch_size
        )
        family = FAMILY[machine]
        correct = 0
        for emb in tqdm(embs, desc=machine):
            rows_out = audio_db.top_k(conn, emb, args.k)
            tag_lists = [tags for _cid, tags, _dist in rows_out]
            correct += int(clip_is_correct(tag_lists, family, args.threshold))
        acc = correct / len(clips)
        per_machine[machine] = {"n": len(clips), "correct": correct, "accuracy": acc}
        total_correct += correct
        total_clips += len(clips)
        print(f"{machine}: {acc:.3f} ({correct}/{len(clips)})")

    overall = total_correct / total_clips if total_clips else 0.0
    print(f"overall: {overall:.3f} ({total_correct}/{total_clips})")

    payload = {
        "metric": (
            f"machine-type retrieval accuracy (>={args.threshold} of top-{args.k} "
            "cards carry a tag from the clip's machine family)"
        ),
        "protocol": (
            f"MIMII test clips, sample {args.sample}/machine seed {args.seed}; "
            f"CLAP audio embedding -> audio_card_vectors top-{args.k} (cosine)"
        ),
        "note": (
            "MIMII lacks fault-family labels; machine-type retrieval is the "
            "measurable proxy (spec metric adaptation). Fault-family retrieval "
            "quality is qualitative (demo evidence)."
        ),
        "cards_indexed": n_cards,
        "per_machine": per_machine,
        "overall_accuracy": overall,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
