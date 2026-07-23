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

# Reject decompression-bomb images at the decode boundary. PIL reads dimensions
# from the header (no pixel decode), so a small, highly-compressible file that
# declares a huge canvas is caught here before it can allocate a multi-hundred-MB
# RGB buffer - which, on the async path, would OOM the worker out-of-band. ~50 MP
# covers any real inspection photo with room to spare.
MAX_IMAGE_PIXELS = 50_000_000

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
    *,
    budget_override: float | None = None,
) -> str:
    """Run describer.describe, wall-clock-bounded when a budget applies.

    The budget is ``budget_override`` when given (the async worker passes a
    generous one - it has no gateway cap, so a valid slow description is still
    included while a true hang stays bounded), else the describer's advertised
    ``describe_budget_s`` (the sync/gateway path), else unbounded (local Qwen,
    ~15s on MPS and self-bounded by max_new_tokens).
    """
    budget = (
        budget_override
        if budget_override is not None
        else getattr(describer, "describe_budget_s", None)
    )
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


async def _read_validated_audio(audio: UploadFile) -> bytes:
    """Read an uploaded audio clip, enforcing the 10MB cap (413) and a wav
    decode-check (400). Pure input validation - it does NOT gate on the
    analyzer being enabled, so callers decide when to invoke it: the sync
    /analyze route calls it only when the analyzer is loaded+enabled (behavior
    unchanged), while the model-free async submit route calls it whenever audio
    is present and lets the worker decide via ``enabled``.
    """
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded audio exceeds the 10MB limit")
    try:
        import soundfile as sf

        sf.read(BytesIO(audio_bytes))  # decode-check, mirrors img.load()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Uploaded audio is not a readable wav: {exc}"
        ) from exc
    return audio_bytes


def _validate_image(data: bytes) -> Image.Image:
    """Decode-and-validate an uploaded image, shared by the sync /analyze and
    async submit boundaries. Rejects (400): a non-image, a decompression bomb
    over MAX_IMAGE_PIXELS (checked from the header before any pixel decode), or a
    truncated/corrupt payload (via img.load()). Returns the loaded PIL image;
    callers convert to RGB as needed (the async submit route just validates)."""
    try:
        img = Image.open(BytesIO(data))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file is not a readable image: {exc}"
        ) from exc
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {img.width}x{img.height} exceeds the pixel limit",
        )
    try:
        img.load()  # force full decode; catches truncated/corrupt payloads
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file is not a readable image: {exc}"
        ) from exc
    return img


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


def _exemplar_to_dict(meta: dict) -> dict:
    """Public exemplar payload (KB track, plan C2): served-image URLs are
    frontend-relative (the derivatives ship with the site build)."""
    return {
        "id": meta["id"],
        "thumb_url": meta["thumb_url"],
        "image_url": meta["image_url"],
        "credit": meta["credit"],
        "source_url": meta["source_url"],
        "caption": meta.get("caption", ""),
    }


def _exemplar_store(app: FastAPI):
    """The ArrayVectorStore when one is wired (cloud path); None on the local
    pgvector path, where card payloads simply omit exemplars."""
    recognizer = getattr(app.state, "recognizer", None)
    store = getattr(recognizer, "vector_store", None)
    return store if hasattr(store, "exemplars_for_card") else None


def _cards_with_exemplars(hits, store) -> list[dict]:
    """Card payloads with an exemplar thumb strip joined by card_id (max 3)."""
    cards = []
    for hit in hits:
        card = _card_to_dict(hit.card)
        if store is not None:
            exemplars = store.exemplars_for_card(hit.card.id)
            if exemplars:
                card["exemplars"] = [_exemplar_to_dict(m) for m in exemplars]
        cards.append(card)
    return cards


def _similar_cases(store, query_embedding, k: int = 3) -> list[dict]:
    """Top-k exemplars by CLIP image similarity + their linked card ids."""
    if store is None or query_embedding is None:
        return []
    rows = store.exemplar_top_k(query_embedding, k)
    return [
        {**_exemplar_to_dict(meta), "card_ids": meta.get("card_ids", [])}
        for _eid, meta, _dist in rows
    ]


def _lazy_mode() -> bool:
    """True on the async cloud path: model loading is deferred out of the
    lifespan to the routes/worker that actually need it."""
    return os.environ.get("DEFECTLENS_LAZY_LOAD", "").strip() == "1"


_ENSURE_LOADED_LOCK = threading.Lock()


def ensure_loaded(app: FastAPI) -> None:
    """Idempotently load the heavyweight components onto app.state.

    Called eagerly by the lifespan (local/default) and lazily by the routes
    that actually need models (sync /analyze, /search, the async worker). The
    submit/status/health routes never call it, so they stay fast even on a cold
    env - the point of the async path. Each block is a no-op when its component
    is already set (injected stubs in tests, or a prior call), so this is safe
    to call on every request. The lock serializes concurrent callers - in
    Lambda each env handles one request, but LocalCpuJobStore's worker threads
    (local dev) would otherwise double-load models.
    """
    with _ENSURE_LOADED_LOCK:
        _ensure_loaded_locked(app)


def _ensure_loaded_locked(app: FastAPI) -> None:
    vector_store = None
    card_vectors_path = os.environ.get("CARD_VECTORS_PATH")
    if card_vectors_path and (
        app.state.recognizer is None or app.state.audio_analyzer is None
    ):
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
        from defectlens.serve.vlm_gateway import build_gateway_from_env

        app.state.vlm_gateway = build_gateway_from_env()


def run_analysis(
    app: FastAPI,
    data: bytes,
    img: Any,
    note_text: str | None,
    audio_bytes: bytes | None,
    *,
    skip_description: bool = False,
    describe_budget: float | None = None,
) -> dict:
    """The core image (+ optional audio) analysis, shared by the sync /analyze
    route and the async worker.

    Reads its already-loaded components off ``app.state`` - each caller loads
    models (``ensure_loaded``) and validates raw inputs (image decode, audio
    size/decode) at its own boundary first, so this function is pure pipeline:
    classify (CLIP-fused, or the fine-tuned VLM when the adapter is loaded) ->
    retrieve cited cards -> optional equipment-audio late fusion -> condition
    description -> assembled response dict.

    - ``skip_description``: the sync cold-start stopgap sets this so the first
      cold request warms the env without the extra Bedrock call. The worker
      never skips (async has no gateway cap).
    - ``describe_budget``: overrides the describer's advertised wall-clock
      budget. The worker passes a generous value so a valid slow description is
      included while a true hang stays bounded; ``None`` keeps the describer's
      own budget (the sync/gateway behavior).
    """
    recognizer = app.state.recognizer
    describer = app.state.describer
    analyzer = app.state.audio_analyzer

    result = recognizer.analyze_image_bytes(data, k=5, note=note_text)

    # Phase 3 classifier: fine-tuned VLM ranking (macro top-1 0.851) when the
    # adapter is loaded; CLIP-fused ranking otherwise. Cards retrieval stays
    # CLIP-RAG either way; severity re-keys on the final top class.
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
    if audio_bytes is not None and analyzer is not None and getattr(analyzer, "enabled", False):
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
    if skip_description:
        description = ""
    else:
        description = describe_with_deadline(
            describer, img, top_labels, audio_band, budget_override=describe_budget
        )

    store = _exemplar_store(app)
    return {
        "classes": [{"label": label, "score": score} for label, score in classes],
        "severity": severity,
        "combined_severity": combined_severity,
        "classifier": classifier,
        "note": note_text,
        "description": description,
        "cards": _cards_with_exemplars(result.hits, store),
        "similar_cases": _similar_cases(store, result.query_embedding),
        "audio": audio_payload,
    }


def create_app(
    recognizer: Any = None,
    describer: Any = None,
    text_searcher: Any = None,
    audio_analyzer: Any = None,
    vlm_gateway: Any = None,
    cpu_job_store: Any = None,
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
        # Eager load at startup unless DEFECTLENS_LAZY_LOAD=1 (the async cloud
        # path): lazy mode keeps submit/status/health model-free so they answer
        # fast on a cold env, and the async worker calls ensure_loaded() itself.
        # Local uvicorn keeps the eager behavior so the first request is warm.
        if not _lazy_mode():
            ensure_loaded(app)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.recognizer = recognizer
    app.state.describer = describer
    app.state.text_searcher = text_searcher
    app.state.audio_analyzer = audio_analyzer
    app.state.vlm_gateway = vlm_gateway
    app.state.cpu_job_store = cpu_job_store

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
        img = _validate_image(data).convert("RGB")

        if _lazy_mode():
            ensure_loaded(request.app)  # cloud lazy path loads on demand

        # Audio validation stays gated on the analyzer being loaded+enabled here
        # so the sync path's behavior is unchanged; the async submit route
        # captures audio ungated (it is model-free, so the analyzer isn't loaded
        # yet) and lets the worker decide via ``enabled``.
        analyzer = request.app.state.audio_analyzer
        audio_bytes = None
        if audio is not None and analyzer is not None and getattr(analyzer, "enabled", False):
            audio_bytes = await _read_validated_audio(audio)

        # Cold-start relief (sync/gateway path only): the FIRST /analyze on a
        # fresh env already pays the ~24s in-request model load, which alone
        # nears the 29s gateway cap. On the cloud path (describer advertises a
        # budget) skip the extra Bedrock call on that one request so it completes
        # under the cap and warms the env - classification + cited cards still
        # return; the description arrives on the next (warm) request. The async
        # worker never skips (it has no gateway cap). Local/ungated describers
        # and all subsequent requests are unaffected.
        describer = request.app.state.describer
        served_before = getattr(request.app.state, "served_analyze", False)
        request.app.state.served_analyze = True
        skip_description = (
            bool(getattr(describer, "describe_budget_s", None)) and not served_before
        )

        return run_analysis(
            request.app,
            data,
            img,
            note_text,
            audio_bytes,
            skip_description=skip_description,
        )

    @app.post("/search")
    async def search(payload: SearchRequest, request: Request) -> dict:
        if _lazy_mode():
            ensure_loaded(request.app)  # cloud lazy path loads on demand
        text_searcher = request.app.state.text_searcher
        hits = text_searcher.search(payload.query, k=5)
        return {"cards": _cards_with_exemplars(hits, _exemplar_store(request.app))}

    # -----------------------------------------------------------------------
    # CPU async path (async /analyze design, Phase 1): removes the 29s gateway
    # cap for /analyze. POST /analyze-jobs validates the image, writes the job
    # to S3 and async self-invokes the worker, returning 202 immediately - it is
    # model-free, so it's fast even on a cold env. GET /analyze-jobs/{job_id}
    # polls the S3 result. Both 503 when the async path isn't wired (no
    # CPU_JOBS_S3 / not running in Lambda); local dev uses sync /analyze.
    # -----------------------------------------------------------------------

    def _require_cpu_jobs(request: Request):
        store = request.app.state.cpu_job_store
        if store is None:
            from defectlens.serve.async_jobs import build_cpu_job_store_from_env

            store = build_cpu_job_store_from_env()
            request.app.state.cpu_job_store = store  # cache the boto3-backed store
        if store is None or not getattr(store, "enabled", False):
            raise HTTPException(status_code=503, detail="Async analyze path not deployed")
        binder = getattr(store, "bind", None)
        if binder is not None:
            binder(request.app)  # LocalCpuJobStore's worker needs the app; idempotent
        return store

    @app.post("/analyze-jobs", status_code=202)
    async def submit_analyze_job(
        request: Request,
        file: UploadFile = File(...),
        note: str = Form(""),
        audio: UploadFile | None = File(None),
    ) -> dict:
        store = _require_cpu_jobs(request)
        note_text = re.sub(r"<\|[^>]*\|>", " ", note.strip())[:MAX_NOTE_CHARS] or None
        data = await file.read()
        _validate_image(data)  # 400 on non-image / decompression bomb / corrupt
        # Model-free route: the analyzer isn't loaded here, so capture audio
        # whenever present (ungated) and let the worker decide via ``enabled``.
        audio_bytes = await _read_validated_audio(audio) if audio is not None else None
        return store.submit(data, note_text, audio_bytes)

    @app.get("/analyze-jobs/{job_id}")
    async def poll_analyze_job(
        request: Request, response: Response, job_id: str
    ) -> dict:
        store = _require_cpu_jobs(request)
        state, obj = store.status(job_id)
        if state == "ready":
            return obj
        if state == "failed":
            # The worker's raw error stays server-side (S3 err/ + CloudWatch via
            # logger.exception); return a generic message so an unauthenticated
            # client can't harvest bucket names / ARNs from botocore error text.
            raise HTTPException(status_code=500, detail="analysis failed")
        response.status_code = 202  # still running
        return {"status": "pending"}

    # -----------------------------------------------------------------------
    # Walkthrough diagnostic report (P2): N photos + a visit note -> the
    # Haiku-vision grounded report, on the same async job infra. Submit is
    # model-free (validate + S3 + self-invoke); the worker dispatches on the
    # payload's kind (async_jobs.run_worker -> serve.walkthrough).
    # -----------------------------------------------------------------------

    @app.post("/walkthrough-jobs", status_code=202)
    async def submit_walkthrough_job(
        request: Request,
        files: list[UploadFile] = File(...),
        visit_note: str = Form(""),
        photo_notes: list[str] = Form([]),
    ) -> dict:
        store = _require_cpu_jobs(request)
        from defectlens.report.synthesize import MAX_PHOTOS, MAX_VISIT_NOTE_CHARS
        from defectlens.serve.async_jobs import build_walkthrough_job_payload

        if len(files) > MAX_PHOTOS:
            raise HTTPException(
                status_code=400,
                detail=f"A walkthrough is capped at {MAX_PHOTOS} photos",
            )
        photos = []
        for i, upload in enumerate(files):
            data = await upload.read()
            _validate_image(data)  # 400 on non-image / bomb / corrupt
            raw_note = photo_notes[i] if i < len(photo_notes) else ""
            note_text = (
                re.sub(r"<\|[^>]*\|>", " ", raw_note.strip()).strip()[:MAX_NOTE_CHARS]
                or None
            )
            photos.append(
                {"photo_id": f"photo_{i + 1}", "image_bytes": data, "note": note_text}
            )
        visit = (
            re.sub(r"<\|[^>]*\|>", " ", visit_note.strip()).strip()[:MAX_VISIT_NOTE_CHARS]
            or None
        )
        return store.submit_payload(build_walkthrough_job_payload(photos, visit))

    @app.get("/walkthrough-jobs/{job_id}")
    async def poll_walkthrough_job(
        request: Request, response: Response, job_id: str
    ) -> dict:
        store = _require_cpu_jobs(request)
        state, obj = store.status(job_id)
        if state == "ready":
            return obj
        if state == "failed":
            # Raw worker error stays server-side (S3 err/ + CloudWatch); the
            # client gets a generic message (same posture as /analyze-jobs).
            raise HTTPException(status_code=500, detail="walkthrough failed")
        response.status_code = 202  # still running
        return {"status": "pending"}

    # -----------------------------------------------------------------------
    # Walkthrough GPU enrichment (P4): user-triggered fine-tuned-Qwen labels
    # through the consistency gate. POST fans one SageMaker async job per
    # photo (the endpoint sleeps at zero; the report NEVER waits on this);
    # GET polls and, once all photos settle, merges consistent labels into
    # the stored report. Both 503 without the GPU deploy.
    # -----------------------------------------------------------------------

    @app.post("/walkthrough-jobs/{job_id}/enrich", status_code=202)
    async def submit_walkthrough_enrichment(request: Request, job_id: str) -> dict:
        store = _require_cpu_jobs(request)
        gateway = _require_gateway(request)
        state, _report = store.status(job_id)
        if state != "ready":
            raise HTTPException(
                status_code=409, detail="walkthrough report not ready yet"
            )
        from defectlens.serve.async_jobs import parse_job_payload

        payload = parse_job_payload(store.get_input(job_id))
        if payload.get("kind") != "walkthrough":
            raise HTTPException(status_code=409, detail="not a walkthrough job")
        photos = payload["photos"]

        # Idempotent AND resumable: the mapping is claimed BEFORE the first GPU
        # submit and re-persisted after EVERY successful one, so a partial
        # fan-out failure (throttle, transient network) plus the frontend's
        # retry resumes the missing photos instead of re-waking - and
        # re-billing - the already-submitted ones.
        env = store.get_enrichment(job_id)
        mapping: dict[str, dict] = dict(env["photos"]) if env else {}
        if env is not None and len(mapping) >= len(photos):
            return {"status": "submitted", "photos": len(mapping)}
        if env is None:
            store.put_enrichment(job_id, {"photos": mapping, "merged": False})
        for photo in photos:
            pid = photo["photo_id"]
            if pid in mapping:
                continue  # resumed retry: this photo's GPU job is already live
            result = gateway.submit(photo["image_bytes"], photo.get("note"))
            mapping[pid] = {
                "output_location": result["output_location"],
                "failure_location": result.get("failure_location"),
            }
            store.put_enrichment(job_id, {"photos": mapping, "merged": False})
        return {"status": "submitted", "photos": len(mapping)}

    @app.get("/walkthrough-jobs/{job_id}/enrich")
    async def poll_walkthrough_enrichment(
        request: Request, response: Response, job_id: str
    ) -> dict:
        store = _require_cpu_jobs(request)
        gateway = _require_gateway(request)
        env = store.get_enrichment(job_id)
        if env is None:
            raise HTTPException(status_code=404, detail="enrichment not requested")
        mapping = env["photos"]
        if env.get("merged"):
            # Terminal: the merge already happened - serve the stored result
            # instead of re-reading every GPU status and re-merging per poll.
            state, report = store.status(job_id)
            if state == "ready":
                return {"status": "ready", "report": report, "gate": env.get("gate", {})}

        labels: dict[str, tuple[str, float]] = {}
        gpu_failures: list[str] = []
        pending = 0
        for photo_id, locs in mapping.items():
            state, classes = gateway.status(
                locs["output_location"], failure_location=locs.get("failure_location")
            )
            if state == "ready" and classes:
                top = classes[0]
                labels[photo_id] = (top["label"], top["score"])
            elif state == "failed" or (state == "ready" and not classes):
                gpu_failures.append(photo_id)
            else:
                pending += 1
        if pending:
            response.status_code = 202
            done = len(mapping) - pending
            return {"status": "pending", "done": done, "total": len(mapping)}

        state, report = store.status(job_id)
        if state != "ready":
            raise HTTPException(status_code=409, detail="walkthrough report missing")
        from defectlens.report.enrich import merge_enrichment

        merged, gate = merge_enrichment(report, labels)
        for photo_id in gpu_failures:
            gate["dropped"].append(
                {"photo_id": photo_id, "label": None, "confidence": None,
                 "reason": "gpu_failed"}
            )
        for finding in report.get("per_photo", []):
            # Photos the fan-out never reached (partial submit failure that was
            # not retried): accounted honestly, never silently absent.
            if finding["photo_id"] not in mapping:
                gate["dropped"].append(
                    {"photo_id": finding["photo_id"], "label": None,
                     "confidence": None, "reason": "not_submitted"}
                )
        store.put_output(job_id, merged)
        store.put_enrichment(job_id, {"photos": mapping, "merged": True, "gate": gate})
        return {"status": "ready", "report": merged, "gate": gate}

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

        # Lazy cloud path: the env answering /health may not have loaded models
        # yet (they load in the worker / on first use), but the service IS
        # servable when the async job path is wired and the baked vector artifact
        # is present. Key "ok" on servability, not this-env model state - else
        # /health flaps ok/degraded by which env answers and the canary
        # false-alarms after the DEFECTLENS_LAZY_LOAD switch.
        async_ready = (
            _lazy_mode()
            and bool(os.environ.get("CPU_JOBS_S3", "").strip())
            and Path(os.environ.get("CARD_VECTORS_PATH", "")).exists()
        )

        return {
            "status": "ok" if (db_ok or store_ok or async_ready) else "degraded",
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
