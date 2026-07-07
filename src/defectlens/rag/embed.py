"""CLIP embedding + pgvector indexing for the guidance-card corpus (spec §6)."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from defectlens.corpus import Card, load_corpus_dir
from defectlens.eval.clip_zeroshot import _features, pick_device
from defectlens.ingest import ManifestRow, read_manifest
from defectlens.rag import db

CLIP_MODEL = "openai/clip-vit-large-patch14"  # single source for RAG modules


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 1D or 2D array; guards zero vectors."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def sample_train_rows(
    rows: list[ManifestRow], per_class_cap: int, seed: int
) -> dict[str, list[ManifestRow]]:
    """Per-class seeded sample of train rows (order-independent per-class RNG)."""
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        grouped[r.unified_label].append(r)
    out: dict[str, list[ManifestRow]] = {}
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda r: r.image_path)
        rng = random.Random(f"{seed}:{label}")
        k = min(per_class_cap, len(group))
        out[label] = rng.sample(group, k)
    return out


def tag_centroid(centroids: dict[str, np.ndarray], tags: list[str]) -> np.ndarray:
    """Normalized mean of the class centroids for a card's tags."""
    stacked = np.stack([centroids[t] for t in tags])
    return normalize(stacked.mean(axis=0))


def embed_texts(
    model, processor, texts: list[str], device: str, batch_size: int = 32
) -> np.ndarray:
    """Embed a list of strings in batches; returns [N, 768] float32."""
    feats: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            emb = _features(model.get_text_features(**inputs))
        feats.append(emb.cpu().numpy())
    return normalize(np.concatenate(feats, axis=0).astype(np.float32))


def embed_images(
    model, processor, paths: list[str], device: str, batch_size: int = 32
) -> np.ndarray:
    """Embed a list of image paths in batches; returns [N, 768] float32."""
    feats: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="images"):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = _features(model.get_image_features(**inputs))
        feats.append(emb.cpu().numpy())
    return normalize(np.concatenate(feats, axis=0).astype(np.float32))


def class_centroids(
    model,
    processor,
    device,
    train_manifest: Path,
    per_class_cap: int = 200,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Per-class normalized-mean image centroid from a seeded sample of train rows."""
    rows = read_manifest(train_manifest)
    sampled = sample_train_rows(rows, per_class_cap=per_class_cap, seed=seed)
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(sampled):
        paths = [r.image_path for r in sampled[label]]
        embs = embed_images(model, processor, paths, device)
        centroids[label] = normalize(embs.mean(axis=0))
    return centroids


def index_corpus(
    conn, cards: list[Card], text_embs: np.ndarray, centroids: dict[str, np.ndarray]
) -> int:
    """Upsert a text row and an image_centroid row per card; returns vectors upserted."""
    count = 0
    for i, card in enumerate(cards):
        db.upsert_vector(conn, card.id, "text", card.class_tags, text_embs[i])
        count += 1
        centroid = tag_centroid(centroids, card.class_tags)
        db.upsert_vector(conn, card.id, "image_centroid", card.class_tags, centroid)
        count += 1
    return count


def main() -> None:
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument(
        "--train-manifest", type=Path, default=Path("data/manifests/train.csv")
    )
    parser.add_argument("--per-class-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cards = load_corpus_dir(args.corpus_dir)
    if not cards:
        raise SystemExit(f"no cards found in {args.corpus_dir}")

    try:
        conn = db.connect()
    except Exception:
        raise SystemExit("pgvector DB unreachable — docker compose up -d db")
    db.init_schema(conn)

    device = pick_device()
    model_name = CLIP_MODEL
    print(f"Device: {device}; model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    texts = [c.index_sentence for c in cards]
    text_embs = embed_texts(model, processor, texts, device)

    centroids = class_centroids(
        model,
        processor,
        device,
        args.train_manifest,
        per_class_cap=args.per_class_cap,
        seed=args.seed,
    )

    n = index_corpus(conn, cards, text_embs, centroids)
    print(f"Indexed {len(cards)} cards -> {n} vectors ({len(centroids)} class centroids)")


if __name__ == "__main__":
    main()
