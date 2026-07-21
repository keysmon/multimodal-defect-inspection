"""Card retrieval for grounding. CLIP is retrieval-ONLY here (locked design
decision): these wrappers surface candidate guidance cards for the reasoner
to cite; they never produce a headline classification.
"""
from __future__ import annotations


def retrieve_for_photo(recognizer, image_bytes: bytes, k: int = 5, note: str | None = None):
    """Top-k candidate cards for one photo via the fused image retrieval.

    Reuses Recognizer.analyze_image_bytes (image-centroid + prompt fusion,
    plus note-text fusion when a note rides along) but keeps ONLY the cited
    cards - the class ranking is deliberately ignored.
    """
    result = recognizer.analyze_image_bytes(image_bytes, k=k, note=note)
    return [hit.card for hit in result.hits]


def retrieve_for_text(recognizer, query: str, k: int = 3, class_tags: list[str] | None = None):
    """Top-k candidate cards for a free-text concern via text retrieval.

    class_tags optionally narrows to cards sharing at least one tag - the
    same off-class kill the agent applies to measured findings.
    """
    hits = recognizer.search_text(query, k=k)
    cards = [hit.card for hit in hits]
    if class_tags:
        wanted = set(class_tags)
        cards = [c for c in cards if wanted & set(c.class_tags)]
    return cards
