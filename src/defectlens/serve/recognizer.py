"""Recognition service: fused class ranking + guidance retrieval (spec §7).

Interim classifier (Phase 3 held): CLIP RRF fusion — the same math measured at
recall@5 0.863 in results/rag_recall_fused.json. The fine-tuned VLM replaces
class_ranking() later without changing the API contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from defectlens.corpus import Card, is_audio_card, load_corpus_dir
from defectlens.eval.clip_zeroshot import _features, pick_device
from defectlens.eval.rag_recall import rrf_fuse
from defectlens.rag import db
from defectlens.rag.embed import embed_texts, normalize
from defectlens.rag.retrieve import Hit, card_lookup, hits_from_rows
from defectlens.taxonomy import UNIFIED_CLASSES

# Most severe first; used to pick the max of (base band, tagged card severities).
SEVERITY_ORDER = ["structural", "urgent", "monitor", "cosmetic"]
SEVERITY_BANDS = {"exposed_rebar": "structural", "no_defect": "cosmetic"}


def fused_card_ranking(
    centroid_ranked_ids: list[str],
    prompt_class_sims: dict[str, float],
    cards_by_id: dict[str, Card],
    note_ranked_ids: list[str] | None = None,
) -> list[str]:
    """Reproduce rag_recall's fused image-mode ranking (RRF of two card rankings).

    ranking_prompt = card ids sorted desc by max(prompt sim over the card's tags),
    mirroring the `card_scores` / `sorted(zip(...), reverse=True)` logic in
    eval/rag_recall.py's main(). cards_by_id iteration order stands in for that
    module's `cards` list order (card_lookup is built from the same cards list).

    When an inspector note is present, note_ranked_ids (cards ranked by
    note-text similarity) joins the fusion as a third RRF ranking; absent a
    note it is None and the ranking stays the measured image-mode pair.
    """
    card_ids = list(cards_by_id.keys())
    card_scores = [
        max(prompt_class_sims[tag] for tag in cards_by_id[cid].class_tags)
        for cid in card_ids
    ]
    ranking_prompt = [
        cid for _score, cid in sorted(zip(card_scores, card_ids), reverse=True)
    ]
    rankings = [centroid_ranked_ids, ranking_prompt]
    if note_ranked_ids:
        rankings.append(note_ranked_ids)
    return rrf_fuse(rankings)


def class_ranking_from_cards(
    fused_ids: list[str], cards_by_id: dict[str, Card]
) -> list[tuple[str, float]]:
    """Aggregate a fused card ranking into a class ranking.

    Each class's score is 1/(rank+1) of the FIRST fused card carrying that tag
    (rank is 0-indexed position in fused_ids). Classes with no tagged card in
    fused_ids score 0.0. Returns all UNIFIED_CLASSES sorted desc by score;
    ties (including all the 0.0s) keep UNIFIED_CLASSES order (stable sort).
    """
    scores: dict[str, float] = dict.fromkeys(UNIFIED_CLASSES, 0.0)
    for rank, cid in enumerate(fused_ids):
        card = cards_by_id[cid]
        for tag in card.class_tags:
            if tag in scores and scores[tag] == 0.0:
                scores[tag] = 1.0 / (rank + 1)
    ranked_classes = sorted(UNIFIED_CLASSES, key=lambda c: -scores[c])
    return [(c, scores[c]) for c in ranked_classes]


def severity_for(top_class: str, top_cards: list[Card]) -> str:
    """Severity band for the top predicted class, escalated by tagged cards.

    Base band from SEVERITY_BANDS (default "monitor" for classes not listed),
    escalated to the most severe among (base, severities of top_cards tagged
    with top_class). Cards not tagged with top_class are ignored.
    """
    base = SEVERITY_BANDS.get(top_class, "monitor")
    candidates = [base] + [
        card.severity for card in top_cards if top_class in card.class_tags
    ]
    return min(candidates, key=SEVERITY_ORDER.index)


@dataclass
class RecognitionResult:
    classes: list[tuple[str, float]]  # (label, score) desc
    severity: str
    hits: list[Hit]  # top-k fused cards, metadata-joined


class Recognizer:
    """Loads CLIP + corpus + DB once, then serves fused recognition per image."""

    def __init__(
        self,
        corpus_dir: Path = Path("corpus"),
        text_templates: Path = Path("configs/clip_prompts.yaml"),
        vector_store=None,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.text_templates = text_templates
        self.model = None
        self.processor = None
        self.device: str | None = None
        self.prompt_feats: np.ndarray | None = None
        self.cards: list[Card] = []
        self.lookup: dict[str, Card] = {}  # visual cards only — drives fusion
        # All cards incl. hvac-* audio cards — drives /search metadata joins
        # (kept separate so hvac ids never enter the visual fusion lookup).
        self.search_lookup: dict[str, Card] = {}
        # When a vector_store is injected (cloud/no-DB path) retrieval goes
        # through it and conn stays None; otherwise the pgvector conn is the
        # default local-dev path. See rag.vector_store.
        self.vector_store = vector_store
        self.conn = None

    def load(self) -> None:
        import yaml
        from transformers import CLIPModel, CLIPProcessor

        from defectlens.rag.embed import CLIP_MODEL

        # Visual pipeline only: audio guidance cards (hvac-*) carry tags outside
        # the 9-class prompt-similarity dict and belong to AudioAnalyzer's lookup.
        # search_lookup keeps ALL cards so /search can surface hvac guidance;
        # self.lookup stays visual-only so fusion can never KeyError on hvac tags.
        all_cards = load_corpus_dir(self.corpus_dir)
        cards = [c for c in all_cards if not is_audio_card(c)]
        if not cards:
            raise RuntimeError(f"no cards found in {self.corpus_dir}")
        self.cards = cards
        self.lookup = card_lookup(cards)
        self.search_lookup = card_lookup(all_cards)

        if self.vector_store is None:
            self.conn = db.connect()

        cfg = yaml.safe_load(self.text_templates.read_text(encoding="utf-8"))
        class_phrases = cfg["class_phrases"]
        templates = cfg["templates"]
        model_name = cfg.get("model", CLIP_MODEL)

        self.device = pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        prompt_feats = []
        for cls in UNIFIED_CLASSES:
            prompts = [t.format(class_phrases[cls]) for t in templates]
            embs = embed_texts(self.model, self.processor, prompts, self.device)
            prompt_feats.append(normalize(embs.mean(axis=0)))
        self.prompt_feats = np.stack(prompt_feats)  # [9, 768]

    def _visual_top_k(self, embedding, k: int, kinds: tuple[str, ...]):
        """Dispatch to the injected vector_store, else the pgvector conn.

        Kept as one seam so both retrievals in analyze_image_bytes stay in sync.
        Calls module-level db.top_k in the conn branch (not an imported name) so
        tests that monkeypatch recognizer.db.top_k still land.
        """
        if self.vector_store is not None:
            return self.vector_store.visual_top_k(embedding, k, kinds)
        return db.top_k(self.conn, embedding, k, kinds)

    def search_text(self, query: str, k: int = 5) -> list[Hit]:
        """Text-query the guidance cards for the /search box.

        Cloud (the product surface): the injected vector_store carries a
        search-scoped index over ALL cards — including the hvac-* audio cards
        that are absent from the visual card_vectors table — so a query like
        "pump grinding noise" can surface bearing_wear. Rows join to full Card
        metadata via search_lookup (which contains hvac ids), so no KeyError.

        Local pgvector: falls back to the visual 'text' vectors only, because
        the hvac cards aren't in card_vectors. This is a known local/cloud
        difference; cloud is the product surface. Either way this must not touch
        recognizer.conn directly (it is None on the store path).
        """
        emb = normalize(embed_texts(self.model, self.processor, [query], self.device))[0]
        if self.vector_store is not None:
            rows = self.vector_store.search_top_k(emb, k)
            return hits_from_rows(rows, self.search_lookup)
        rows = db.top_k(self.conn, emb, k, ("text",))
        return hits_from_rows(rows, self.lookup)

    def analyze_image_bytes(
        self, data: bytes, k: int = 5, note: str | None = None
    ) -> RecognitionResult:
        img = Image.open(BytesIO(data)).convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            raw_emb = _features(self.model.get_image_features(**inputs))
        emb = normalize(raw_emb.cpu().numpy())[0]  # [768]

        centroid_rows = self._visual_top_k(emb, len(self.cards), ("image_centroid",))
        centroid_ranked_ids = [cid for cid, _tags, _dist in centroid_rows]

        class_sims = emb @ self.prompt_feats.T  # [9]
        prompt_class_sims = dict(zip(UNIFIED_CLASSES, class_sims))

        note_ranked_ids = None
        if note and note.strip():
            note_emb = normalize(
                embed_texts(self.model, self.processor, [note.strip()], self.device)
            )[0]
            note_rows = self._visual_top_k(note_emb, len(self.cards), ("text",))
            note_ranked_ids = [cid for cid, _tags, _dist in note_rows]

        fused_ids = fused_card_ranking(
            centroid_ranked_ids, prompt_class_sims, self.lookup, note_ranked_ids
        )

        top_ids = fused_ids[:k]
        hit_rows = [(cid, self.lookup[cid].class_tags, 0.0) for cid in top_ids]
        hits = hits_from_rows(hit_rows, self.lookup)

        classes = class_ranking_from_cards(fused_ids, self.lookup)
        top_class = classes[0][0]
        top_cards = [hit.card for hit in hits if top_class in hit.card.class_tags]
        severity = severity_for(top_class, top_cards)

        return RecognitionResult(classes=classes, severity=severity, hits=hits)
