import subprocess
import sys
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from defectlens.corpus import Card
from defectlens.rag.retrieve import Hit
from defectlens.serve.api import create_app
from defectlens.serve.audio_analyzer import AudioFinding
from defectlens.serve.recognizer import RecognitionResult


def make_card(cid, tags, severity="monitor"):
    return Card(
        id=cid,
        title=f"title-{cid}",
        class_tags=tags,
        severity=severity,
        index_sentence=f"index-{cid}",
        passage=f"passage-{cid}",
        citation=f"citation-{cid}",
        source_name=f"source-{cid}",
        source_url=f"https://example.com/{cid}",
        source_license="CC-BY-4.0",
    )


def make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    return buf.getvalue()


def make_wav_bytes() -> bytes:
    """A short, valid PCM wav that passes the /analyze decode-check."""
    import numpy as np
    import soundfile as sf

    buf = BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


class StubRecognizer:
    """Fixed-result stand-in for Recognizer — no model, no DB."""

    def __init__(self, result, expected_k=5):
        self.result = result
        self.expected_k = expected_k
        self.calls = []

    def analyze_image_bytes(self, data, k, note=None):
        assert k == self.expected_k
        self.calls.append(data)
        return self.result


class StubDescriber:
    """Fixed-text stand-in for Describer — no model."""

    def __init__(self, text="desc"):
        self.text = text
        self.calls = []
        self.audio_band = None

    def describe(self, image, top_classes, audio_band=None):
        self.calls.append((image, list(top_classes)))
        self.audio_band = audio_band
        return self.text


class StubAudioAnalyzer:
    """Fixed-finding stand-in for AudioAnalyzer — no model, no DB."""

    def __init__(self, finding, enabled=True):
        self.finding = finding
        self.enabled = enabled
        self.calls = []

    def analyze(self, wav_bytes):
        self.calls.append(wav_bytes)
        return self.finding


class StubTextSearcher:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k=5):
        self.calls.append((query, k))
        return self.hits


class FakeConn:
    """Fake DB connection: conn.execute(sql).fetchone() -> (count,)."""

    def __init__(self, count):
        self.count = count
        self.queries = []

    def execute(self, sql):
        self.queries.append(sql)
        return self

    def fetchone(self):
        return (self.count,)


class RaisingConn:
    def execute(self, sql):
        raise RuntimeError("db unreachable")


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


def _analyze_result():
    card_a = make_card("c1", ["crack"], severity="urgent")
    card_b = make_card("c2", ["spalling"], severity="monitor")
    hits = [Hit(card=card_a, distance=0.1), Hit(card=card_b, distance=0.2)]
    classes = [("crack", 0.9), ("spalling", 0.5), ("mold_algae", 0.1)]
    return RecognitionResult(classes=classes, severity="urgent", hits=hits)


def test_analyze_happy_path_full_response_shape():
    result = _analyze_result()
    recognizer = StubRecognizer(result)
    describer = StubDescriber(text="Visible diagonal cracking.")
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={"file": ("test.png", make_png_bytes(), "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["classes"] == [
        {"label": "crack", "score": 0.9},
        {"label": "spalling", "score": 0.5},
        {"label": "mold_algae", "score": 0.1},
    ]
    assert body["severity"] == "urgent"
    assert body["description"] == "Visible diagonal cracking."
    assert body["cards"] == [
        {
            "id": "c1",
            "title": "title-c1",
            "passage": "passage-c1",
            "severity": "urgent",
            "citation": "citation-c1",
            "source_name": "source-c1",
            "source_url": "https://example.com/c1",
        },
        {
            "id": "c2",
            "title": "title-c2",
            "passage": "passage-c2",
            "severity": "monitor",
            "citation": "citation-c2",
            "source_name": "source-c2",
            "source_url": "https://example.com/c2",
        },
    ]

    # recognizer was called with k=5 (asserted inside stub) and got raw bytes
    assert len(recognizer.calls) == 1
    assert isinstance(recognizer.calls[0], bytes)

    # describer got a PIL image and the top-3 class labels in score order
    assert len(describer.calls) == 1
    image, top_labels = describer.calls[0]
    assert isinstance(image, Image.Image)
    assert top_labels == ["crack", "spalling", "mold_algae"]


def test_analyze_non_image_payload_returns_400_with_message():
    def _boom(*a, **k):
        raise AssertionError("recognizer/describer must not run on bad payload")

    class ExplodingRecognizer:
        analyze_image_bytes = _boom

    class ExplodingDescriber:
        describe = _boom

    app = create_app(recognizer=ExplodingRecognizer(), describer=ExplodingDescriber())
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "image" in detail.lower()


def test_analyze_with_describer_disabled_returns_empty_description():
    result = _analyze_result()
    recognizer = StubRecognizer(result)
    describer = StubDescriber(text="")  # vlm disabled -> Describer.describe returns ""
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={"file": ("test.png", make_png_bytes(), "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json()["description"] == ""


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


def test_search_happy_path_returns_cards():
    card = make_card("s1", ["water_damage"], severity="monitor")
    text_searcher = StubTextSearcher(hits=[Hit(card=card, distance=0.05)])
    app = create_app(text_searcher=text_searcher)
    client = TestClient(app)

    resp = client.post("/search", json={"query": "dark stain under window"})

    assert resp.status_code == 200
    assert resp.json() == {
        "cards": [
            {
                "id": "s1",
                "title": "title-s1",
                "passage": "passage-s1",
                "severity": "monitor",
                "citation": "citation-s1",
                "source_name": "source-s1",
                "source_url": "https://example.com/s1",
            }
        ]
    }
    assert text_searcher.calls == [("dark stain under window", 5)]


def test_search_cloud_mode_no_db_returns_cards(monkeypatch):
    """Regression (H1): /search must not 500 when serving from a vector_store
    with no pgvector conn. Drives the real route -> TextSearcher ->
    Recognizer.search_text -> ArrayVectorStore (conn is None in cloud mode)."""
    import numpy as np

    from defectlens.rag.vector_store import ArrayVectorStore
    from defectlens.serve import recognizer as recognizer_mod
    from defectlens.serve.api import TextSearcher
    from defectlens.serve.recognizer import Recognizer

    card = make_card("s1", ["water_damage"])
    text = np.eye(1, 4, dtype=np.float32)  # one card, text vector [1,0,0,0]
    store = ArrayVectorStore(
        visual_ids=["s1"], visual_tags=[["water_damage"]],
        visual_text=text, visual_centroid=text,
        audio_ids=[], audio_tags=[], audio_emb=np.zeros((0, 4), np.float32),
        search_ids=["s1"], search_text=text,
    )
    rec = Recognizer(vector_store=store)
    rec.lookup = {"s1": card}
    rec.search_lookup = {"s1": card}
    rec.device = "cpu"
    rec.model = object()
    rec.processor = object()
    assert rec.conn is None  # cloud mode: no DB
    monkeypatch.setattr(
        recognizer_mod, "embed_texts", lambda *a, **k: np.ones((1, 4), dtype=np.float32)
    )

    app = create_app(recognizer=rec, text_searcher=TextSearcher(rec))
    client = TestClient(app)
    resp = client.post("/search", json={"query": "dark stain under window"})

    assert resp.status_code == 200  # not 500
    assert resp.json()["cards"][0]["id"] == "s1"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class RecognizerWithConn:
    def __init__(self, conn):
        self.conn = conn


class DescriberWithModel:
    def __init__(self, model):
        self.model = model


def test_health_ok_path_reports_db_and_vlm_state():
    recognizer = RecognizerWithConn(conn=FakeConn(count=410))
    describer = DescriberWithModel(model=object())
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "db": True,
        "cards_indexed": 410,
        "vlm_loaded": True,
        "classifier": "clip-fused",
    }


def test_health_degraded_path_when_db_query_raises():
    recognizer = RecognizerWithConn(conn=RaisingConn())
    describer = DescriberWithModel(model=None)
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "degraded",
        "db": False,
        "cards_indexed": 0,
        "vlm_loaded": False,
        "classifier": "clip-fused",
    }


def test_health_degraded_when_recognizer_never_wired():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["db"] is False
    assert body["cards_indexed"] == 0
    assert body["vlm_loaded"] is False


class RecognizerWithStore:
    """No-DB serving stand-in: exposes a vector_store, no conn."""

    def __init__(self, count):
        self.vector_store = _CountingStore(count)


class _CountingStore:
    def __init__(self, count):
        self._count = count

    def visual_count(self):
        return self._count


def test_health_store_path_is_ok_with_db_false():
    """Cloud/no-DB path: a loaded vector_store makes /health report status ok
    and cards_indexed from the store, while db stays honestly false (there is
    no pgvector). The 5.5b canary keys on status, not db."""
    recognizer = RecognizerWithStore(count=410)
    describer = DescriberWithModel(model=None)
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "db": False,
        "cards_indexed": 410,
        "vlm_loaded": False,
        "classifier": "clip-fused",
    }


# ---------------------------------------------------------------------------
# Import sanity — api module must stay cheap to import (spec §7)
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_torch_or_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.serve.api\n"
            "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
            "assert 'transformers' not in sys.modules, 'transformers imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_analyze_prefers_vlm_ranking_and_rekeys_severity():
    """A describer exposing non-empty rank_classes switches /analyze to the
    fine-tuned VLM classes + classifier tag and re-keys severity on the VLM
    top class; cards still come from the CLIP-RAG recognizer result."""
    card_rebar = make_card("c9", ["exposed_rebar"], severity="structural")
    hits = [Hit(card=card_rebar, distance=0.1)]
    clip_classes = [("crack", 0.9), ("spalling", 0.5)]
    result = RecognitionResult(classes=clip_classes, severity="urgent", hits=hits)

    describer = StubDescriber(text="Exposed reinforcement bar.")
    describer.rank_classes = lambda img, note=None: [
        ("exposed_rebar", 0.97), ("spalling", 0.02), ("crack", 0.01)
    ]
    app = create_app(recognizer=StubRecognizer(result), describer=describer)
    client = TestClient(app)

    resp = client.post("/analyze", files={"file": ("t.png", make_png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier"] == "vlm-qlora"
    assert body["classes"][0] == {"label": "exposed_rebar", "score": 0.97}
    assert body["severity"] != "urgent"  # re-keyed on exposed_rebar, not CLIP's crack
    assert body["cards"][0]["id"] == "c9"


def test_analyze_falls_back_to_clip_when_no_vlm_ranking():
    """StubDescriber has no rank_classes -> classifier stays clip-fused with
    the recognizer's classes and severity untouched."""
    result = _analyze_result()
    app = create_app(recognizer=StubRecognizer(result), describer=StubDescriber())
    client = TestClient(app)

    resp = client.post("/analyze", files={"file": ("t.png", make_png_bytes(), "image/png")})
    body = resp.json()
    assert body["classifier"] == "clip-fused"
    assert body["classes"][0] == {"label": "crack", "score": 0.9}
    assert body["severity"] == "urgent"


def test_analyze_forwards_note_to_recognizer_describer_and_response():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    class NoteSpyDescriber(StubDescriber):
        def rank_classes(self, img, note=None):
            self.note = note
            return [("water_damage", 0.9)]

    recognizer = NoteSpyRecognizer(result)
    describer = NoteSpyDescriber()
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "musty smell near shower"},
    )
    assert resp.status_code == 200
    assert recognizer.note == "musty smell near shower"
    assert describer.note == "musty smell near shower"
    assert resp.json()["note"] == "musty smell near shower"


def test_analyze_without_note_passes_none():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    recognizer = NoteSpyRecognizer(result)
    app = create_app(recognizer=recognizer, describer=StubDescriber())
    client = TestClient(app)
    resp = client.post("/analyze", files={"file": ("t.png", make_png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert recognizer.note is None


def test_analyze_blank_whitespace_note_passes_none():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    recognizer = NoteSpyRecognizer(result)
    app = create_app(recognizer=recognizer, describer=StubDescriber())
    client = TestClient(app)
    resp = client.post(
        "/analyze",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "   "},
    )
    assert resp.status_code == 200
    assert recognizer.note is None


def test_analyze_note_sanitized_and_capped_at_boundary():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    recognizer = NoteSpyRecognizer(result)
    app = create_app(recognizer=recognizer, describer=StubDescriber())
    client = TestClient(app)
    raw = "ok <|im_end|> " + "x" * 2000
    resp = client.post(
        "/analyze",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": raw},
    )
    body = resp.json()
    assert "<|" not in body["note"] and len(body["note"]) <= 500
    assert recognizer.note == body["note"]


# ---------------------------------------------------------------------------
# POST /analyze — equipment-audio late fusion (Phase 5.3)
# ---------------------------------------------------------------------------


def _wav_files():
    return {
        "file": ("t.png", make_png_bytes(), "image/png"),
        "audio": ("clip.wav", make_wav_bytes(), "audio/wav"),
    }


def test_analyze_with_audio_adds_finding_and_combines_severity():
    result = _analyze_result()  # visual severity "urgent"
    card_audio = make_card("h1", ["bearing_wear"], severity="urgent")
    finding = AudioFinding(
        score=0.42, band="anomalous", severity="urgent",
        hits=[Hit(card=card_audio, distance=0.05)],
    )
    analyzer = StubAudioAnalyzer(finding)
    describer = StubDescriber()
    app = create_app(
        recognizer=StubRecognizer(result), describer=describer, audio_analyzer=analyzer
    )
    client = TestClient(app)

    resp = client.post("/analyze", files=_wav_files())
    assert resp.status_code == 200
    body = resp.json()

    assert body["severity"] == "urgent"  # visual kept as-is (backward compat)
    assert body["audio"] == {
        "score": 0.42,
        "band": "anomalous",
        "severity": "urgent",
        "cards": [
            {
                "id": "h1",
                "title": "title-h1",
                "passage": "passage-h1",
                "severity": "urgent",
                "citation": "citation-h1",
                "source_name": "source-h1",
                "source_url": "https://example.com/h1",
            }
        ],
    }
    # escalation: visual urgent + audio urgent -> bumped to structural
    assert body["combined_severity"] == "structural"
    assert len(analyzer.calls) == 1 and isinstance(analyzer.calls[0], bytes)
    # the audio band was forwarded into the description prompt
    assert describer.audio_band == "anomalous"


def test_analyze_without_audio_reports_null_and_combined_equals_visual():
    result = _analyze_result()  # visual "urgent"
    analyzer = StubAudioAnalyzer(AudioFinding(0.0, "normal_operation", "cosmetic"))
    describer = StubDescriber()
    app = create_app(
        recognizer=StubRecognizer(result), describer=describer, audio_analyzer=analyzer
    )
    client = TestClient(app)

    resp = client.post("/analyze", files={"file": ("t.png", make_png_bytes(), "image/png")})
    body = resp.json()
    assert body["audio"] is None
    assert body["combined_severity"] == "urgent"  # equals visual when no audio
    assert analyzer.calls == []  # analyze() not called without an audio upload
    assert describer.audio_band is None


def test_analyze_audio_ignored_when_analyzer_disabled():
    result = _analyze_result()
    analyzer = StubAudioAnalyzer(AudioFinding(0.9, "anomalous", "urgent"), enabled=False)
    app = create_app(
        recognizer=StubRecognizer(result), describer=StubDescriber(), audio_analyzer=analyzer
    )
    client = TestClient(app)

    resp = client.post("/analyze", files=_wav_files())
    body = resp.json()
    assert body["audio"] is None
    assert body["combined_severity"] == "urgent"  # visual only
    assert analyzer.calls == []


def test_analyze_no_escalation_when_visual_below_monitor():
    card_none = make_card("cN", ["no_defect"], severity="cosmetic")
    hits = [Hit(card=card_none, distance=0.1)]
    result = RecognitionResult(
        classes=[("no_defect", 0.9)], severity="cosmetic", hits=hits
    )
    analyzer = StubAudioAnalyzer(AudioFinding(0.5, "anomalous", "urgent"))
    app = create_app(
        recognizer=StubRecognizer(result), describer=StubDescriber(), audio_analyzer=analyzer
    )
    client = TestClient(app)

    resp = client.post("/analyze", files=_wav_files())
    body = resp.json()
    assert body["severity"] == "cosmetic"
    assert body["audio"]["severity"] == "urgent"
    # worst-of = urgent; no escalation because visual is below monitor
    assert body["combined_severity"] == "urgent"


def test_analyze_unreadable_audio_returns_400():
    result = _analyze_result()
    analyzer = StubAudioAnalyzer(AudioFinding(0.1, "normal_operation", "cosmetic"))
    app = create_app(
        recognizer=StubRecognizer(result), describer=StubDescriber(), audio_analyzer=analyzer
    )
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={
            "file": ("t.png", make_png_bytes(), "image/png"),
            "audio": ("bad.wav", b"not a wav at all", "audio/wav"),
        },
    )
    assert resp.status_code == 400
    assert "audio" in resp.json()["detail"].lower()
    assert analyzer.calls == []  # decode-check 400s before analyze() is reached


def test_analyze_oversized_audio_returns_413():
    result = _analyze_result()
    analyzer = StubAudioAnalyzer(AudioFinding(0.1, "normal_operation", "cosmetic"))
    app = create_app(
        recognizer=StubRecognizer(result), describer=StubDescriber(), audio_analyzer=analyzer
    )
    client = TestClient(app)

    oversized = b"\x00" * (10 * 1024 * 1024 + 1)  # just over the 10MB cap
    resp = client.post(
        "/analyze",
        files={
            "file": ("t.png", make_png_bytes(), "image/png"),
            "audio": ("big.wav", oversized, "audio/wav"),
        },
    )
    assert resp.status_code == 413
    assert analyzer.calls == []  # size check 413s before decode/analyze


def test_analyze_audio_ignored_when_analyzer_not_wired():
    # audio uploaded but no analyzer wired (app.state.audio_analyzer is None):
    # 200 with "audio": null and combined == visual — locks the None guard.
    result = _analyze_result()  # visual severity "urgent"
    app = create_app(recognizer=StubRecognizer(result), describer=StubDescriber())
    client = TestClient(app)

    resp = client.post("/analyze", files=_wav_files())
    assert resp.status_code == 200
    body = resp.json()
    assert body["audio"] is None
    assert body["combined_severity"] == "urgent"


# --- describe_with_deadline: the cloud-path hang guard (added 2026-07-20) ---

import time as _time  # noqa: E402

from defectlens.serve.api import describe_with_deadline  # noqa: E402


class _FastDescriber:
    def describe(self, image, top_labels, audio_band=None):
        return "a crack runs diagonally across the surface"


class _SlowDescriber:
    def __init__(self, delay):
        self.delay = delay
        self.started = False

    def describe(self, image, top_labels, audio_band=None):
        self.started = True
        _time.sleep(self.delay)
        return "should never be seen within the deadline"


class _RaisingDescriber:
    def describe(self, image, top_labels, audio_band=None):
        raise RuntimeError("bedrock exploded")


def test_describe_deadline_returns_fast_result():
    out = describe_with_deadline(_FastDescriber(), "img", ["crack"], None, timeout_s=5)
    assert out == "a crack runs diagonally across the surface"


def test_describe_deadline_bounds_a_hang():
    slow = _SlowDescriber(delay=30)
    t0 = _time.perf_counter()
    out = describe_with_deadline(slow, "img", ["crack"], None, timeout_s=0.3)
    elapsed = _time.perf_counter() - t0
    assert out == ""  # abandoned; empty description, request continues
    assert slow.started  # it did run, we just stopped waiting
    assert elapsed < 2  # returned near the 0.3s budget, not the 30s hang


def test_describe_deadline_swallows_exceptions():
    out = describe_with_deadline(_RaisingDescriber(), "img", ["crack"], None, timeout_s=5)
    assert out == ""
