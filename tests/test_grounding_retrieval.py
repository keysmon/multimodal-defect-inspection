"""grounding.retrieval: CLIP-retrieval-only wrappers over the Recognizer."""
from dataclasses import dataclass

from defectlens.grounding.retrieval import retrieve_for_photo, retrieve_for_text


@dataclass(frozen=True)
class FakeCard:
    id: str
    class_tags: tuple


@dataclass(frozen=True)
class FakeHit:
    card: FakeCard
    distance: float = 0.0


@dataclass
class FakeResult:
    hits: list


class FakeRecognizer:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def analyze_image_bytes(self, data, k=5, note=None):
        self.calls.append(("image", k, note))
        return FakeResult(hits=self._hits[:k])

    def search_text(self, query, k=5):
        self.calls.append(("text", query, k))
        return self._hits[:k]


HITS = [
    FakeHit(FakeCard("crack-01", ("crack",))),
    FakeHit(FakeCard("spall-02", ("spalling",))),
    FakeHit(FakeCard("multi-03", ("crack", "spalling"))),
]


def test_photo_retrieval_returns_cards_and_passes_note():
    rec = FakeRecognizer(HITS)
    cards = retrieve_for_photo(rec, b"jpeg", k=2, note="damp smell")
    assert [c.id for c in cards] == ["crack-01", "spall-02"]
    assert rec.calls == [("image", 2, "damp smell")]


def test_text_retrieval_returns_cards():
    rec = FakeRecognizer(HITS)
    cards = retrieve_for_text(rec, "peeling paint remediation", k=3)
    assert [c.id for c in cards] == ["crack-01", "spall-02", "multi-03"]
    assert rec.calls == [("text", "peeling paint remediation", 3)]


def test_text_retrieval_class_tag_filter():
    rec = FakeRecognizer(HITS)
    cards = retrieve_for_text(rec, "q", k=3, class_tags=["spalling"])
    assert [c.id for c in cards] == ["spall-02", "multi-03"]
