"""Exemplar-retrieval class-consistency eval (plan C2, spec §5.5).

For every row of the FROZEN v2 test split: CLIP-embed the photo, retrieve the
top-1 exemplar by cosine, and score whether that exemplar's class_tags contain
the row's unified label. This is deliberately worded as a CLASS-CONSISTENCY
metric, not correctness: sharing a class is a weak proxy for "visually similar
documented case", so numbers here bound plausibility, they don't prove
usefulness — the hand-rated spot check (results/exemplar_spotcheck.md) is the
human-judgment side.

Exemplar vectors are built directly from the served derivatives via
scripts/export_vector_artifacts.build_exemplars (no DB needed).

Usage:
  python scripts/eval_exemplar_retrieval.py [--subset 100] [--seed 42]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_vector_artifacts", Path(__file__).parent / "export_vector_artifacts.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--subset", type=int, default=None, help="cap rows (smoke)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spotcheck-n", type=int, default=20)
    args = parser.parse_args()

    import numpy as np

    from defectlens.ingest import read_manifest

    rows = read_manifest(args.test_manifest)
    rng = random.Random(args.seed)
    if args.subset:
        rows = rng.sample(rows, min(args.subset, len(rows)))

    export_mod = _load_export_module()
    e_ids, e_meta_json, e_emb = export_mod.build_exemplars()
    e_meta = [json.loads(m) for m in e_meta_json]
    e_tags = [set(m["class_tags"]) for m in e_meta]

    from transformers import CLIPModel, CLIPProcessor

    from defectlens.eval.clip_zeroshot import pick_device
    from defectlens.rag.embed import CLIP_MODEL, embed_images

    device = pick_device()
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    q_emb = embed_images(model, processor, [r.image_path for r in rows], device)

    dists = 1.0 - q_emb @ e_emb.T  # [N_rows, N_exemplars]
    top1 = np.argmin(dists, axis=1)

    per_class_hits: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        hit = int(row.unified_label in e_tags[top1[i]])
        per_class_hits[row.unified_label].append(hit)

    report = {
        "metric": (
            "top-1 retrieved exemplar shares the query's unified class "
            "(class-consistency proxy, NOT a correctness/usefulness claim)"
        ),
        "test_manifest": str(args.test_manifest),
        "n_queries": len(rows),
        "n_exemplars": len(e_ids),
        "subset": args.subset,
        "seed": args.seed,
        "overall_hit_rate": round(
            sum(sum(v) for v in per_class_hits.values()) / len(rows), 4
        ),
        "per_class_hit_rate": {
            cls: {"hit_rate": round(sum(v) / len(v), 4), "n": len(v)}
            for cls, v in sorted(per_class_hits.items())
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "exemplar_retrieval.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    # Hand-rating spot check, mirroring the walkthrough spot-check pattern.
    picks = rng.sample(range(len(rows)), min(args.spotcheck_n, len(rows)))
    lines = [
        "# Exemplar retrieval spot check",
        "",
        f"{len(picks)} random test-split queries with their top-1 retrieved",
        "exemplar. Rate each pair by hand: does the exemplar read as a",
        '"similar documented case" for the query photo? (good / weak / wrong)',
        "",
        "| # | query image | query class | exemplar | exemplar caption | rating |",
        "|---|---|---|---|---|---|",
    ]
    for j, i in enumerate(picks, 1):
        row = rows[i]
        m = e_meta[top1[i]]
        lines.append(
            f"| {j} | {row.image_path} | {row.unified_label} "
            f"| {m['id']} | {m['caption']} | |"
        )
    out_md = args.out_dir / "exemplar_spotcheck.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"spot check -> {out_md}")


if __name__ == "__main__":
    main()
