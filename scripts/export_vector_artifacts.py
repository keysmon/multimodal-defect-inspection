"""Export the local pgvector card vectors to a cloud .npz artifact (Phase 5.5a).

The cloud Lambda has no database: it loads models/cloud_artifacts/card_vectors.npz
through rag.vector_store.ArrayVectorStore. This script reads the LOCAL pgvector
tables and writes that npz in exactly the ArrayVectorStore format:

- card_vectors        -> visual_ids / visual_tags_json / visual_embeddings_text
                         / visual_embeddings_centroid (one text + one centroid
                         vector per card, pivoted from the two `kind` rows; a
                         card missing either kind is dropped by design — the
                         npz format requires both, and the pipeline writes both)
- audio_card_vectors  -> audio_ids / audio_tags_json / audio_embeddings
- search-scoped index  -> search_ids / search_embeddings_text (format v2): CLIP
                         text embeddings of index_sentences for ALL cards —
                         visual cards reuse their DB `text` vectors, hvac-* audio
                         cards are freshly CLIP-text-embedded here so /search can
                         surface audible-symptom guidance. A `format_version` key
                         lets ArrayVectorStore reject a stale v1 npz loudly.

The other cloud-serving needs are corpus-independent of this export and are
baked directly by deploy/Dockerfile.lambda: the guidance corpus (corpus/), the
audio bank + calibration (models/audio_bank/), and the severity rules living in
code. This script only produces the vector npz.

Run against the local DB (both tables live in the same `defectlens` database):

    docker compose up -d db
    python scripts/export_vector_artifacts.py --verify
    docker compose stop db

--verify loads the npz back through ArrayVectorStore and cross-checks a handful
of random queries against the live DB (same top-5 ids for both the visual and
audio paths), catching a format / normalization / dedup drift before the
artifact is baked into the image. models/ is gitignored, so the npz never lands
in git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import yaml

from defectlens.rag import audio_db, db
from defectlens.rag.vector_store import (
    AUDIO_EMB,
    AUDIO_IDS,
    AUDIO_TAGS,
    EXEMPLAR_EMB,
    EXEMPLAR_IDS,
    EXEMPLAR_META,
    FORMAT,
    FORMAT_VERSION,
    SEARCH_IDS,
    SEARCH_TEXT,
    VISUAL_CENTROID,
    VISUAL_IDS,
    VISUAL_TAGS,
    VISUAL_TEXT,
    ArrayVectorStore,
)

DEFAULT_OUT = Path("models/cloud_artifacts/card_vectors.npz")
EXEMPLAR_MANIFEST = Path("data/exemplars/manifest.yaml")
EXEMPLAR_IMAGE_DIR = Path("frontend/public/exemplars")


def _to_np(emb) -> np.ndarray:
    """Materialize a pgvector column value as float32.

    A raw ``SELECT embedding`` returns a pgvector ``Vector`` (register_vector
    only auto-converts on some driver paths); ``.to_numpy()`` unwraps it. Plain
    lists/arrays fall through np.asarray.
    """
    if hasattr(emb, "to_numpy"):
        return emb.to_numpy().astype(np.float32)
    return np.asarray(emb, dtype=np.float32)


def read_visual(conn):
    """Pivot card_vectors' two `kind` rows into one text + one centroid per card.

    Cards ordered by id; a card missing either kind is skipped (the real
    pipeline writes both, so this only guards a half-indexed DB).
    """
    rows = conn.execute(
        "SELECT card_id, kind, class_tags, embedding FROM card_vectors"
    ).fetchall()
    by_card: dict[str, dict] = {}
    for card_id, kind, tags, emb in rows:
        entry = by_card.setdefault(card_id, {"tags": list(tags)})
        entry[kind] = _to_np(emb)

    ids, tags_json, text, centroid = [], [], [], []
    for card_id in sorted(by_card):
        e = by_card[card_id]
        if "text" not in e or "image_centroid" not in e:
            print(f"WARNING: {card_id} missing a text/centroid vector — skipped")
            continue
        ids.append(card_id)
        tags_json.append(json.dumps(e["tags"]))
        text.append(e["text"])
        centroid.append(e["image_centroid"])

    text_arr = np.stack(text) if text else np.zeros((0, db.DIM), np.float32)
    centroid_arr = np.stack(centroid) if centroid else np.zeros((0, db.DIM), np.float32)
    return ids, tags_json, text_arr, centroid_arr


def read_audio(conn):
    rows = conn.execute(
        "SELECT card_id, class_tags, embedding FROM audio_card_vectors"
    ).fetchall()
    ids, tags_json, emb = [], [], []
    for card_id, tags, e in sorted(rows, key=lambda r: r[0]):
        ids.append(card_id)
        tags_json.append(json.dumps(list(tags)))
        emb.append(_to_np(e))
    emb_arr = np.stack(emb) if emb else np.zeros((0, audio_db.DIM), np.float32)
    return ids, tags_json, emb_arr


def build_search(v_ids, v_text, corpus_dir: Path):
    """Search-scoped index over ALL cards: visual cards reuse their DB `text`
    vectors (already CLIP-text embeddings of their index_sentence); the hvac-*
    audio cards — absent from card_vectors — get freshly CLIP-text-embedded here
    so the /search box can surface audible-symptom guidance.
    """
    from defectlens.corpus import is_audio_card, load_corpus_dir

    hvac = [c for c in load_corpus_dir(corpus_dir) if is_audio_card(c)]
    hvac_ids = [c.id for c in hvac]
    if hvac_ids:
        from transformers import CLIPModel, CLIPProcessor

        from defectlens.eval.clip_zeroshot import pick_device
        from defectlens.rag.embed import CLIP_MODEL, embed_texts

        device = pick_device()
        model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        hvac_text = embed_texts(model, processor, [c.index_sentence for c in hvac], device)
    else:
        hvac_text = np.zeros((0, db.DIM), np.float32)

    search_ids = list(v_ids) + hvac_ids
    search_text = np.vstack([v_text, hvac_text]) if len(hvac_text) else v_text
    return search_ids, search_text


def build_exemplars(manifest_path: Path = EXEMPLAR_MANIFEST,
                    image_dir: Path = EXEMPLAR_IMAGE_DIR):
    """Exemplar-image index (format v3): CLIP-image-embed the SERVED 1024px
    derivatives (scripts/fetch_exemplars.py output — the exact pixels users
    see) and carry the manifest metadata as one JSON row per exemplar, with
    frontend-relative image/thumb URLs baked in.
    """
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["exemplars"]
    ids, meta_json, paths = [], [], []
    for e in entries:
        image = image_dir / f"{e['id']}.jpg"
        if not image.is_file():
            raise SystemExit(
                f"{image} missing — run scripts/fetch_exemplars.py before exporting"
            )
        ids.append(e["id"])
        paths.append(str(image))
        meta_json.append(json.dumps({
            "id": e["id"],
            "card_ids": e.get("card_ids", []),
            "class_tags": e["class_tags"],
            "license": e["license"],
            "credit": e["credit"],
            "source_url": e["source_url"],
            "caption": e.get("caption", ""),
            "image_url": f"/exemplars/{e['id']}.jpg",
            "thumb_url": f"/exemplars/thumbs/{e['id']}.jpg",
        }))

    from transformers import CLIPModel, CLIPProcessor

    from defectlens.eval.clip_zeroshot import pick_device
    from defectlens.rag.embed import CLIP_MODEL, embed_images

    device = pick_device()
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    emb = embed_images(model, processor, paths, device)
    return ids, meta_json, emb


def export(out: Path, corpus_dir: Path = Path("corpus")) -> tuple[int, int, int, int]:
    conn = db.connect()  # same DB holds card_vectors and audio_card_vectors
    v_ids, v_tags, v_text, v_centroid = read_visual(conn)
    a_ids, a_tags, a_emb = read_audio(conn)
    s_ids, s_text = build_search(v_ids, v_text, corpus_dir)
    e_ids, e_meta, e_emb = build_exemplars()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        **{
            VISUAL_IDS: np.array(v_ids),
            VISUAL_TAGS: np.array(v_tags),
            VISUAL_TEXT: v_text,
            VISUAL_CENTROID: v_centroid,
            AUDIO_IDS: np.array(a_ids),
            AUDIO_TAGS: np.array(a_tags),
            AUDIO_EMB: a_emb,
            SEARCH_IDS: np.array(s_ids),
            SEARCH_TEXT: s_text,
            EXEMPLAR_IDS: np.array(e_ids),
            EXEMPLAR_META: np.array(e_meta),
            EXEMPLAR_EMB: e_emb,
            FORMAT: np.array(FORMAT_VERSION),
        },
    )
    return len(v_ids), len(a_ids), len(s_ids), len(e_ids)


def verify(out: Path, queries: int = 5, seed: int = 11) -> bool:
    """Cross-check random queries: ArrayVectorStore top-5 == live-DB top-5.

    db.top_k is an exact full-scan sort; audio_db.top_k rides the HNSW index, so
    ef_search is raised well past the corpus size to make it exact for the
    comparison (otherwise an approximate-NN reorder would read as a false
    mismatch, not a real export bug).
    """
    store = ArrayVectorStore.load(out)
    conn = db.connect()
    conn.execute("SET hnsw.ef_search = 1000")  # exact NN over ~47 audio cards
    rng = np.random.default_rng(seed)
    ok = True

    for _ in range(queries):
        q = rng.standard_normal(db.DIM).astype(np.float32)
        q /= np.linalg.norm(q)
        db_ids = [r[0] for r in db.top_k(conn, q, 5, ("image_centroid",))]
        store_ids = [r[0] for r in store.visual_top_k(q, 5, ("image_centroid",))]
        if db_ids != store_ids:
            ok = False
            print(f"MISMATCH visual: db={db_ids} store={store_ids}")

    for _ in range(queries):
        q = rng.standard_normal(audio_db.DIM).astype(np.float32)
        q /= np.linalg.norm(q)
        db_ids = [r[0] for r in audio_db.top_k(conn, q, 5)]
        store_ids = [r[0] for r in store.audio_top_k(q, 5)]
        if db_ids != store_ids:
            ok = False
            print(f"MISMATCH audio: db={db_ids} store={store_ids}")

    # Search index has no DB counterpart (it covers hvac cards absent from
    # pgvector), so verify it structurally: hvac cards present, and each search
    # vector self-retrieves at rank 0 (ids ↔ rows aligned, cosine sane).
    if not any(cid.startswith("hvac") for cid in store.search_ids):
        ok = False
        print("MISMATCH search: no hvac ids in the search index")
    n = len(store.search_ids)
    for i in rng.choice(n, size=min(queries, n), replace=False):
        top = store.search_top_k(store._search[i], 1)[0][0]
        if top != store.search_ids[i]:
            ok = False
            print(f"MISMATCH search self-retrieval: {store.search_ids[i]} -> {top}")

    # Exemplar index (v3) is likewise DB-less: check self-retrieval alignment
    # and that metadata joins resolve (every meta row parses with an id).
    if store.exemplar_count() == 0:
        ok = False
        print("MISMATCH exemplar: index is empty")
    m = store.exemplar_count()
    for i in rng.choice(m, size=min(queries, m), replace=False):
        top_id, top_meta, _dist = store.exemplar_top_k(store._exemplar[i], 1)[0]
        if top_id != store.exemplar_ids[i] or top_meta.get("id") != top_id:
            ok = False
            print(f"MISMATCH exemplar self-retrieval: {store.exemplar_ids[i]} -> {top_id}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true", help="cross-check vs live DB")
    parser.add_argument("--queries", type=int, default=5)
    args = parser.parse_args()

    n_visual, n_audio, n_search, n_exemplar = export(args.out)
    print(
        f"Wrote {args.out} — {n_visual} visual + {n_audio} audio cards; "
        f"{n_search} search-scoped cards (visual + hvac); "
        f"{n_exemplar} exemplar images"
    )
    if n_visual == 0 and n_audio == 0:
        raise SystemExit(
            "DB has no card vectors — run the indexing pipeline "
            "(python -m defectlens.rag.embed / audio_embed_cards) first"
        )

    if args.verify:
        ok = verify(args.out, queries=args.queries)
        print("VERIFY:", "OK — store matches DB" if ok else "FAILED — see mismatches")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
