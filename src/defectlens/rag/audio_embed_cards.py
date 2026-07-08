"""CLAP text-embed the HVAC-maintenance guidance cards into audio_card_vectors.

Each hvac-* card's ``index_sentence`` is an audible-symptom description; we embed
it in CLAP *text* space so an uploaded clip (CLAP *audio* space) retrieves it
cross-modally at serve time. Mirrors rag/embed.py's indexer shape, but for the
512-dim CLAP table.

Usage: python -m defectlens.rag.audio_embed_cards   # needs pgvector DB up
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from defectlens.audio.embed import CLAP_MODEL, batched, load_clap
from defectlens.corpus import load_corpus_dir
from defectlens.eval.clip_zeroshot import pick_device
from defectlens.rag import audio_db


def embed_card_texts(model, processor, texts: list[str], device: str, batch_size: int = 16) -> np.ndarray:
    """CLAP text embeddings, L2-normalized; returns [N, 512] float32."""
    import torch

    out = []
    for batch in batched(list(texts), batch_size):
        inputs = processor(text=list(batch), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):  # transformers v5 output object
            feats = feats.pooler_output
        out.append(feats.cpu().numpy())
    embs = np.concatenate(out, axis=0).astype(np.float32)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    args = parser.parse_args(argv)

    cards = [c for c in load_corpus_dir(args.corpus_dir) if c.id.startswith("hvac-")]
    if not cards:
        raise SystemExit(
            f"no hvac-* cards found in {args.corpus_dir} — author "
            "corpus/hvac_maintenance.yaml (Task 1) before indexing audio cards"
        )

    try:
        conn = audio_db.connect()
    except Exception:
        raise SystemExit("pgvector DB unreachable — docker compose up -d db")
    audio_db.ensure_schema(conn)

    device = pick_device()
    print(f"Device: {device}; model: {CLAP_MODEL}; cards: {len(cards)}")
    model, processor = load_clap(device)

    embs = embed_card_texts(model, processor, [c.index_sentence for c in cards], device)
    for card, emb in zip(cards, embs):
        audio_db.upsert(conn, card.id, card.class_tags, emb)
    print(f"Indexed {len(cards)} hvac-* cards into audio_card_vectors")


if __name__ == "__main__":
    main()
