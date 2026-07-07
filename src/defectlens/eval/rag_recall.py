"""recall@5 for cross-modal retrieval on the frozen test split (spec §6)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from defectlens.corpus import load_corpus_dir
from defectlens.eval.clip_zeroshot import _nan_to_none, pick_device
from defectlens.ingest import read_manifest
from defectlens.rag import db
from defectlens.rag.retrieve import card_lookup, query_by_embedding
from defectlens.taxonomy import UNIFIED_CLASSES


def is_relevant(hit_tags: list[str], true_class: str) -> bool:
    return true_class in hit_tags


def recall_at_k(results: list[tuple[str, list[list[str]]]], k: int = 5) -> float:
    """Fraction of queries where any of the first k hits is relevant.

    results = [(true_class, [tags_of_hit1, tags_of_hit2, ...]), ...]
    """
    if not results:
        return float("nan")
    hits = 0
    for true_class, hit_tags_list in results:
        if any(is_relevant(tags, true_class) for tags in hit_tags_list[:k]):
            hits += 1
    return hits / len(results)


def per_class_recall(
    results: list[tuple[str, list[list[str]]]], classes: list[str], k: int = 5
) -> dict[str, float]:
    """recall@k per class; NaN for classes absent from results (mirrors metrics.py)."""
    grouped: dict[str, list[tuple[str, list[list[str]]]]] = {c: [] for c in classes}
    for true_class, hit_tags_list in results:
        if true_class in grouped:
            grouped[true_class].append((true_class, hit_tags_list))
    return {c: recall_at_k(grouped[c], k=k) for c in classes}


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: scale-free fusion of independent rankings.

    Standard fusion for rankings whose scores live on different scales
    (here: image↔image centroid distance vs. image↔text zero-shot similarity).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-manifest", type=Path, default=Path("data/manifests/test.csv")
    )
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--text-templates", type=Path, default=Path("configs/clip_prompts.yaml")
    )
    parser.add_argument(
        "--image-mode", choices=("centroid", "fused"), default="centroid",
        help="fused = RRF of centroid ranking + zero-shot class-prompt ranking",
    )
    parser.add_argument("--out-name", default="rag_recall.json")
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    from defectlens.rag.embed import embed_images, embed_texts

    cards = load_corpus_dir(args.corpus_dir)
    if not cards:
        raise SystemExit(f"no cards found in {args.corpus_dir}")
    lookup = card_lookup(cards)

    try:
        conn = db.connect()
    except Exception:
        raise SystemExit("pgvector DB unreachable — docker compose up -d db")

    cfg = yaml.safe_load(args.text_templates.read_text(encoding="utf-8"))
    class_phrases = cfg["class_phrases"]
    templates = cfg["templates"]

    device = pick_device()
    from defectlens.rag.embed import CLIP_MODEL

    model_name = cfg.get("model", CLIP_MODEL)
    print(f"Device: {device}; model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # IMAGE QUERIES
    rows = read_manifest(args.test_manifest)
    image_paths = [r.image_path for r in rows]
    image_embs = embed_images(model, processor, image_paths, device, batch_size=args.batch_size)

    if args.image_mode == "fused":
        # Zero-shot class-prompt features (prompt ensemble per class, normalized mean).
        import numpy as np

        from defectlens.rag.embed import normalize

        prompt_feats = []
        for cls in UNIFIED_CLASSES:
            prompts = [t.format(class_phrases[cls]) for t in templates]
            embs = embed_texts(model, processor, prompts, device)
            prompt_feats.append(normalize(embs.mean(axis=0)))
        prompt_feats = np.stack(prompt_feats)  # [9, 768]
        card_ids = [c.id for c in cards]
        card_best_class_idx = [
            [UNIFIED_CLASSES.index(t) for t in c.class_tags] for c in cards
        ]

    image_results: list[tuple[str, list[list[str]]]] = []
    for row, emb in zip(rows, image_embs, strict=True):
        if args.image_mode == "fused":
            centroid_rows = db.top_k(conn, emb, len(cards), ("image_centroid",))
            ranking_centroid = [cid for cid, _tags, _dist in centroid_rows]
            class_sims = emb @ prompt_feats.T  # [9]
            card_scores = [
                max(class_sims[i] for i in idxs) for idxs in card_best_class_idx
            ]
            ranking_prompt = [
                cid for _s, cid in sorted(zip(card_scores, card_ids), reverse=True)
            ]
            fused = rrf_fuse([ranking_centroid, ranking_prompt])[: args.k]
            tags_list = [lookup[cid].class_tags for cid in fused]
            image_results.append((row.unified_label, tags_list))
        else:
            hits = query_by_embedding(conn, lookup, emb, args.k, ("image_centroid",))
            image_results.append((row.unified_label, [h.card.class_tags for h in hits]))

    # TEXT QUERIES
    text_results: list[tuple[str, list[list[str]]]] = []
    for cls in UNIFIED_CLASSES:
        phrase = class_phrases[cls]
        texts = [t.format(phrase) for t in templates]
        text_embs = embed_texts(model, processor, texts, device)
        for emb in text_embs:
            hits = query_by_embedding(conn, lookup, emb, args.k, ("text",))
            text_results.append((cls, [h.card.class_tags for h in hits]))

    image_recall = recall_at_k(image_results, k=args.k)
    text_recall = recall_at_k(text_results, k=args.k)
    image_per_class = per_class_recall(image_results, UNIFIED_CLASSES, k=args.k)
    text_per_class = per_class_recall(text_results, UNIFIED_CLASSES, k=args.k)

    results = {
        "k": args.k,
        "image_mode": args.image_mode,
        "test_manifest": str(args.test_manifest),
        "n_image_queries": len(image_results),
        "n_text_queries": len(text_results),
        "image_recall_at_k": _nan_to_none(image_recall),
        "text_recall_at_k": _nan_to_none(text_recall),
        "image_per_class_recall_at_k": {c: _nan_to_none(v) for c, v in image_per_class.items()},
        "text_per_class_recall_at_k": {c: _nan_to_none(v) for c, v in text_per_class.items()},
        "classes": UNIFIED_CLASSES,
    }
    args.out_dir.mkdir(exist_ok=True)
    out_json = args.out_dir / args.out_name
    out_json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(f"image recall@{args.k}: {image_recall:.3f}  text recall@{args.k}: {text_recall:.3f}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
