"""Query the card-vector store by defect photo or text (spec §6)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from defectlens.corpus import Card, load_corpus_dir
from defectlens.rag import db


@dataclass(frozen=True)
class Hit:
    card: Card
    distance: float


def card_lookup(cards: list[Card]) -> dict[str, Card]:
    return {c.id: c for c in cards}


def hits_from_rows(
    rows: list[tuple[str, list[str], float]], lookup: dict[str, Card]
) -> list[Hit]:
    """Join db.top_k rows to Card metadata; unknown ids raise (index/corpus drift)."""
    # NOTE: guards missing ids only; edited class_tags without reindexing are
    # not detected — re-run `python -m defectlens.rag.embed` after corpus edits.
    out = []
    for card_id, _tags, dist in rows:
        if card_id not in lookup:
            raise KeyError(
                f"card {card_id!r} in index but not in corpus/ — re-run indexing"
            )
        out.append(Hit(card=lookup[card_id], distance=dist))
    return out


def query_by_embedding(conn, lookup, embedding, k: int, kinds: tuple[str, ...]) -> list[Hit]:
    return hits_from_rows(db.top_k(conn, embedding, k, kinds), lookup)


def query_by_text(conn, lookup, model, processor, device, text: str, k: int = 5) -> list[Hit]:
    from defectlens.rag.embed import embed_texts

    embs = embed_texts(model, processor, [text], device)
    return query_by_embedding(conn, lookup, embs[0], k, ("text",))


def query_by_image(
    conn, lookup, model, processor, device, image_path: str, k: int = 5
) -> list[Hit]:
    from defectlens.rag.embed import embed_images

    embs = embed_images(model, processor, [image_path], device)
    return query_by_embedding(conn, lookup, embs[0], k, ("image_centroid",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="free-text defect description")
    group.add_argument("--image", type=str, help="path to a defect photo")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    from defectlens.eval.clip_zeroshot import pick_device
    from transformers import CLIPModel, CLIPProcessor

    cards = load_corpus_dir(args.corpus_dir)
    if not cards:
        raise SystemExit(f"no cards found in {args.corpus_dir}")
    lookup = card_lookup(cards)

    try:
        conn = db.connect()
    except Exception:
        raise SystemExit("pgvector DB unreachable — docker compose up -d db")

    device = pick_device()
    from defectlens.rag.embed import CLIP_MODEL

    model_name = CLIP_MODEL
    print(f"Device: {device}; model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    if args.text is not None:
        hits = query_by_text(conn, lookup, model, processor, device, args.text, k=args.k)
    else:
        hits = query_by_image(conn, lookup, model, processor, device, args.image, k=args.k)

    for rank, hit in enumerate(hits, start=1):
        print(
            f"{rank}. [{hit.card.severity}] {hit.card.id} — {hit.card.title} "
            f"(distance={hit.distance:.4f})\n    {hit.card.citation}"
        )


if __name__ == "__main__":
    main()
