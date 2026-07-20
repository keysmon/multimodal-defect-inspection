"""FastAPI serving app (spec §7). Interim classifier: measured CLIP-fused pipeline."""
from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from defectlens.rag.retrieve import Hit
from defectlens.serve.audio_analyzer import combine_severity
from defectlens.train.qlora import MAX_NOTE_CHARS

# ---------------------------------------------------------------------------
# Config (env-driven; no hardcoded URLs elsewhere in this module)
# ---------------------------------------------------------------------------

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "DEFECTLENS_CORS_ORIGINS", "http://localhost:3000"
    ).split(",")
    if origin.strip()
]

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB; a 10s wav is ~1 MB, so this is generous

logger = logging.getLogger(__name__)

# The condition description is best-effort (describe() returns "" on any
# failure). A describer MAY advertise a wall-clock budget via a
# ``describe_budget_s`` attribute; the cloud BedrockDescriber does (a Bedrock
# call that, under concurrent-cold-start memory pressure, was observed NOT to
# honor botocore's cooperative read_timeout and ran to the Lambda's 120s
# ceiling — verified 2026-07-20). Describers WITHOUT the attribute (the local
# Qwen path, ~15s on MPS and self-bounded by max_new_tokens, no botocore, no
# gateway) are called directly so a valid slow description is never truncated.
#
# When a budget is set, describe runs in a throwaway daemon thread joined for
# at most the budget; on overrun the request ships classification + cited
# cards with an empty description instead of hanging past the 29s gateway cap.
# A bounded semaphore caps concurrently-stalled describe threads per warm
# process: once that many are stuck (still holding their Bedrock socket), new
# requests skip description immediately rather than leaking threads unbounded.
_MAX_INFLIGHT_DESCRIBES = 2
_DESCRIBE_INFLIGHT = threading.BoundedSemaphore(_MAX_INFLIGHT_DESCRIBES)


def _describe_safely(describer: Any, image: Any, top_labels: list[str], audio_band: str | None) -> str:
    try:
        return describer.describe(image, top_labels, audio_band=audio_band)
    except Exception:
        logger.warning("description failed", exc_info=True)
        return ""


def describe_with_deadline(
    describer: Any,
    image: Any,
    top_labels: list[str],
    audio_band: str | None,
) -> str:
    """Run describer.describe, wall-clock-bounded only if it advertises a budget."""
    budget = getattr(describer, "describe_budget_s", None)
    if not budget or budget <= 0:
        return _describe_safely(describer, image, top_labels, audio_band)

    if not _DESCRIBE_INFLIGHT.acquire(blocking=False):
        logger.warning("too many stalled descriptions in flight; skipping description")
        return ""

    result = {"text": ""}

    def _run() -> None:
        try:
            result["text"] = _describe_safely(describer, image, top_labels, audio_band)
        finally:
            _DESCRIBE_INFLIGHT.release()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(budget)
    if worker.is_alive():
        logger.warning(
            "description exceeded %.1fs budget; returning without it", budget
        )
    return result["text"]


class TextSearcher:
    """Text-vector retrieval that reuses the Recognizer's already-loaded CLIP
    model/processor/device.

    /search needs the same CLIP text-encoding path the Recognizer used to build
    its prompt features at load() time — rather than loading a second CLIP
    instance, TextSearcher wraps the Recognizer and delegates to its
    search_text, which routes through the shared retrieval seam (pgvector conn
    locally, injected vector_store in the no-DB cloud path). Production wiring
    constructs it in the lifespan handler after Recognizer.load(); tests bypass
    it entirely by injecting a stub with a `.search(query, k)` method.
    """

    def __init__(self, recognizer: Any) -> None:
        self._recognizer = recognizer

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self._recognizer.search_text(query, k=k)


class SearchRequest(BaseModel):
    query: str


def _card_to_dict(card: Any) -> dict:
    return {
        "id": card.id,
        "title": card.title,
        "passage": card.passage,
        "severity": card.severity,
        "citation": card.citation,
        "source_name": card.source_name,
        "source_url": card.source_url,
    }


def create_app(
    recognizer: Any = None,
    describer: Any = None,
    text_searcher: Any = None,
    audio_analyzer: Any = None,
    vlm_gateway: Any = None,
) -> FastAPI:
    """App factory. Pass stubs directly for tests (stored on app.state as-is,
    no lifespan-triggered loading needed since TestClient without `with` never
    runs lifespan). Production wiring (module-level `app = create_app()`)
    leaves all components None; the lifespan handler below loads the real,
    heavyweight ones exactly once on ASGI startup.

    vlm_gateway stays None on the CPU-only deploy (no SAGEMAKER_ENDPOINT), so the
    GPU endpoints below answer 503 rather than invoking a missing endpoint.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Cloud/no-DB path: when CARD_VECTORS_PATH points at a baked npz, load
        # it once and inject one ArrayVectorStore into both Recognizer and
        # AudioAnalyzer (they fall back to the pgvector conn when it's absent,
        # keeping local dev unchanged). See rag.vector_store.
        vector_store = None
        card_vectors_path = os.environ.get("CARD_VECTORS_PATH")
        if card_vectors_path:
            from defectlens.rag.vector_store import ArrayVectorStore

            vector_store = ArrayVectorStore.load(card_vectors_path)

        if app.state.recognizer is None:
            from defectlens.serve.recognizer import Recognizer

            r = Recognizer(vector_store=vector_store)
            r.load()
            app.state.recognizer = r
        if app.state.describer is None:
            from defectlens.serve.bedrock_describer import describer_is_bedrock

            if describer_is_bedrock():
                # Cloud path: Claude Haiku on Bedrock writes the description
                # (DEFECTLENS_NO_VLM=1, no local torch model in the image).
                from defectlens.serve.bedrock_describer import BedrockDescriber

                d = BedrockDescriber()
            else:
                from defectlens.serve.describer import Describer

                d = Describer()
            d.load()
            app.state.describer = d
        if app.state.text_searcher is None:
            app.state.text_searcher = TextSearcher(app.state.recognizer)
        if app.state.audio_analyzer is None:
            from defectlens.serve.audio_analyzer import AudioAnalyzer

            audio_kwargs: dict[str, Any] = {"vector_store": vector_store}
            audio_bank_dir = os.environ.get("AUDIO_BANK_DIR")
            if audio_bank_dir:
                audio_kwargs["bank_dir"] = Path(audio_bank_dir)
            a = AudioAnalyzer(**audio_kwargs)
            a.load()
            app.state.audio_analyzer = a
        if app.state.vlm_gateway is None:
            # None unless SAGEMAKER_ENDPOINT is set (5.5c GPU async path); the
            # CPU-only deploy leaves it None so /analyze-vlm returns 503.
            from defectlens.serve.vlm_gateway import build_gateway_from_env

            app.state.vlm_gateway = build_gateway_from_env()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.recognizer = recognizer
    app.state.describer = describer
    app.state.text_searcher = text_searcher
    app.state.audio_analyzer = audio_analyzer
    app.state.vlm_gateway = vlm_gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/analyze")
    async def analyze(
        request: Request,
        file: UploadFile = File(...),
        note: str = Form(""),
        audio: UploadFile | None = File(None),
    ) -> dict:
        note_text = re.sub(r"<\|[^>]*\|>", " ", note.strip())[:MAX_NOTE_CHARS] or None
        data = await file.read()
        try:
            img = Image.open(BytesIO(data))
            img.load()  # force full decode; catches truncated/corrupt payloads
            img = img.convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file is not a readable image: {exc}"
            ) from exc

        recognizer = request.app.state.recognizer
        describer = request.app.state.describer
        analyzer = request.app.state.audio_analyzer

        result = recognizer.analyze_image_bytes(data, k=5, note=note_text)

        # Phase 3 classifier: fine-tuned VLM ranking (macro top-1 0.851)
        # when the adapter is loaded; CLIP-fused ranking otherwise. Cards
        # retrieval stays CLIP-RAG either way; severity re-keys on the
        # final top class.
        classes = result.classes
        severity = result.severity
        classifier = "clip-fused"
        vlm_classes = getattr(describer, "rank_classes", lambda _img, note=None: [])(
            img, note=note_text
        )
        if vlm_classes:
            from defectlens.serve.recognizer import severity_for

            classes = vlm_classes
            classifier = "vlm-qlora"
            top_class = classes[0][0]
            top_cards = [
                hit.card for hit in result.hits if top_class in hit.card.class_tags
            ]
            severity = severity_for(top_class, top_cards)

        # Phase 5.3 late fusion: an optional equipment-audio clip adds a second
        # modality. Audio runs before description so the band can enter the prompt.
        audio_payload = None
        audio_band = None
        combined_severity = severity
        if audio is not None and analyzer is not None and getattr(analyzer, "enabled", False):
            audio_bytes = await audio.read()
            if len(audio_bytes) > MAX_AUDIO_BYTES:
                raise HTTPException(
                    status_code=413, detail="Uploaded audio exceeds the 10MB limit"
                )
            try:
                import soundfile as sf

                sf.read(BytesIO(audio_bytes))  # decode-check, mirrors img.load() above
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Uploaded audio is not a readable wav: {exc}"
                ) from exc
            finding = analyzer.analyze(audio_bytes)
            audio_band = finding.band
            audio_payload = {
                "score": finding.score,
                "band": finding.band,
                "severity": finding.severity,
                "cards": [_card_to_dict(hit.card) for hit in finding.hits],
            }
            combined_severity = combine_severity(severity, finding.severity)

        top_labels = [label for label, _score in classes[:3]]
        description = describe_with_deadline(describer, img, top_labels, audio_band)

        return {
            "classes": [{"label": label, "score": score} for label, score in classes],
            "severity": severity,
            "combined_severity": combined_severity,
            "classifier": classifier,
            "note": note_text,
            "description": description,
            "cards": [_card_to_dict(hit.card) for hit in result.hits],
            "audio": audio_payload,
        }

    @app.post("/search")
    async def search(payload: SearchRequest, request: Request) -> dict:
        text_searcher = request.app.state.text_searcher
        hits = text_searcher.search(payload.query, k=5)
        return {"cards": [_card_to_dict(hit.card) for hit in hits]}

    # -----------------------------------------------------------------------
    # GPU async path (Phase 5.5c): the fine-tuned VLM on a scale-to-zero
    # SageMaker async endpoint. /analyze-vlm submits a job (S3 in + async
    # invoke, returns immediately); /vlm-status polls the S3 result. Both 503
    # when the gateway isn't wired (CPU-only deploy — no SAGEMAKER_ENDPOINT).
    # -----------------------------------------------------------------------

    def _require_gateway(request: Request):
        gateway = request.app.state.vlm_gateway
        if gateway is None or not getattr(gateway, "enabled", False):
            raise HTTPException(status_code=503, detail="GPU path not deployed")
        return gateway

    @app.post("/analyze-vlm")
    async def analyze_vlm(
        request: Request,
        file: UploadFile = File(...),
        note: str = Form(""),
    ) -> dict:
        gateway = _require_gateway(request)
        note_text = re.sub(r"<\|[^>]*\|>", " ", note.strip())[:MAX_NOTE_CHARS] or None
        data = await file.read()
        try:
            img = Image.open(BytesIO(data))
            img.load()  # force full decode; catches truncated/corrupt payloads
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file is not a readable image: {exc}"
            ) from exc

        result = gateway.submit(data, note_text)
        return {
            "job_id": result["job_id"],
            "output_location": result["output_location"],
            "failure_location": result.get("failure_location"),
        }

    @app.get("/vlm-status")
    async def vlm_status(
        request: Request,
        response: Response,
        output_location: str,
        failure_location: str | None = None,
    ) -> dict:
        gateway = _require_gateway(request)
        state, classes = gateway.status(output_location, failure_location=failure_location)
        if state == "ready":
            return {"status": "ready", "classes": classes}
        if state == "failed":
            return {"status": "failed", "classes": None}
        response.status_code = 202  # still warming / running
        return {"status": "pending"}

    @app.get("/health")
    async def health(request: Request) -> dict:
        recognizer = request.app.state.recognizer
        describer = request.app.state.describer

        # "db" is an honest pgvector-reachability flag — legitimately false in
        # the cloud/no-DB path, where the baked npz vector_store is the index.
        # "status" keys on servability (DB reachable OR a store is loaded), so
        # the 5.5b canary must check "status", not "db".
        db_ok = False
        store_ok = False
        cards_indexed = 0
        if recognizer is not None:
            store = getattr(recognizer, "vector_store", None)
            if store is not None:
                cards_indexed = store.visual_count()
                store_ok = True
            else:
                try:
                    row = recognizer.conn.execute(
                        "SELECT count(*) FROM card_vectors"
                    ).fetchone()
                    cards_indexed = row[0]
                    db_ok = True
                except Exception:
                    db_ok = False
                    cards_indexed = 0

        vlm_loaded = bool(describer is not None and getattr(describer, "model", None) is not None)
        adapter_loaded = bool(getattr(describer, "adapter_loaded", False))

        return {
            "status": "ok" if (db_ok or store_ok) else "degraded",
            "db": db_ok,
            "cards_indexed": cards_indexed,
            "vlm_loaded": vlm_loaded,
            "classifier": "vlm-qlora" if adapter_loaded else "clip-fused",
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
