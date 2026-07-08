# Phase 4: Local Serving + UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally-running DefectLens product: upload a defect photo in a React UI → FastAPI returns ranked defect classes, a natural-language description, a severity band, and cited guidance cards; plus text search over the corpus and a markdown report export.

**Architecture (spec §7, adapted for held Phase 3):** Recognition uses the MEASURED pipeline — CLIP embeddings + RRF fusion (exemplar centroids + zero-shot prompts), the same math that scored recall@5 0.863 — exposed as a class ranking. Qwen2.5-VL-3B (base, fp16 on MPS) generates the description narrative. Severity = rule-based class band refined by retrieved-card severities. When Phase 3's fine-tune lands, it replaces the classifier without touching the API contract.

**Tech Stack:** FastAPI + uvicorn (new deps), existing CLIP/pgvector/RAG modules, Qwen2.5-VL-3B via transformers on MPS, React 19 (CRA scaffolding ported from the cooking-assistant repo), pytest + httpx TestClient with mocked models.

**Branch:** `feat/phase4-serving` off `feat/phase2-rag`.
**Constraints:** pgvector via `docker compose up -d db` must be running + indexed (it is). No cloud. All URLs/origins via env config — no hardcoding (the cooking app's sin).

---

## Task 1: Recognition service module (reuses measured components)

**Files:** `src/defectlens/serve/__init__.py` (empty), `src/defectlens/serve/recognizer.py`, `tests/test_recognizer.py`

`recognizer.py` responsibilities:
- `class_ranking(image_emb, prompt_feats, conn, cards_meta) -> list[tuple[str, float]]`: RRF-fused class scores exactly as `rag_recall`'s fused mode ranks cards, but aggregated to classes — rank classes by best fused card rank per class. Pure given embeddings (TDD with fakes).
- `Recognizer` class: `load()` (CLIP model/processor once, prompt feats once, DB conn), `analyze_embedding(emb) -> RecognitionResult(classes_ranked, hits)`; thin `analyze_image(path_or_bytes)`.
- SEVERITY_BANDS: `{structural: [exposed_rebar], urgent: [], monitor: [crack, spalling, efflorescence, corrosion_stain, mold_algae, water_damage, peeling_paint], cosmetic: [no_defect]}` — band = max(class band, max retrieved-card severity for the top class), documented as rule-based v1 (spec §3).
- TDD: rank aggregation, severity resolution (pure). Model paths exercised by the running server.

## Task 2: Description generator (Qwen2.5-VL-3B on MPS)

**Files:** `src/defectlens/serve/describer.py`, `tests/test_describer.py`

- `Describer.load()` — `Qwen2_5_VLForConditionalGeneration.from_pretrained(..., torch_dtype=torch.float16).to("mps")` + AutoProcessor; lazy import; ~7GB download on first run.
- `describe(image, top_classes) -> str` — chat-template prompt: "Describe the visible condition of this building surface in 2-3 sentences, focusing on {top classes}." max_new_tokens≈120. Deterministic (do_sample=False).
- Degrade gracefully: `DEFECTLENS_NO_VLM=1` env → return "" (UI hides the description panel); keeps the API usable on low-RAM machines and in tests.
- TDD: prompt construction pure fn; generation mocked.

## Task 3: FastAPI app

**Files:** `src/defectlens/serve/api.py`, `tests/test_api.py`; deps: add `fastapi>=0.115`, `uvicorn>=0.30`, `python-multipart`, `httpx` (dev) to pyproject.

Endpoints (spec §7):
- `POST /analyze` (multipart image) → `{classes: [{label, score}], description, severity, cards: [{id, title, passage, severity, citation, source_name, source_url}]}` (top-5 fused cards)
- `POST /search` (json `{query}`) → `{cards: [...]}` via text vectors
- `GET /health` → `{status, db, cards_indexed, vlm_loaded}` — REAL checks (SELECT count from card_vectors), not a bare 200.
- Config via env: `DEFECTLENS_CORS_ORIGINS` (default http://localhost:3000), `DEFECTLENS_DSN` (existing), `DEFECTLENS_NO_VLM`.
- Lifespan handler loads Recognizer (+Describer unless NO_VLM) once.
- TDD: httpx TestClient with Recognizer/Describer stubbed via dependency injection (app.state), including /health db-down path.

## Task 4: React frontend

**Files:** `frontend/` — port CRA scaffolding from `/Users/hangruan/Documents/GitHub/fullstack/frontend` (package.json renamed `defectlens-frontend`, App/RecipeAssistant → `DefectLens.js` rewritten):
- Upload + preview → results: class chips with scores, severity banner (color per band), description paragraph, guidance cards (title, passage, citation, source link, severity tag).
- Text search box → card list (same card component).
- "Export report" button → downloads markdown (client-side blob): photo filename, date, classes, severity, description, cards with citations.
- `REACT_APP_API_URL` env (default http://localhost:8000) — no hardcoded URLs in components.
- Basic RTL smoke tests (renders, mocked fetch happy path).

## Task 5: Real-flow verification + wrap

- Run server (`uvicorn defectlens.serve.api:app --port 8000`), `npm start`, drive the REAL flow per workflow rules: upload 2-3 known test-split images (a crack, an efflorescence, a no-defect) via the browser (Playwright MCP) → verify sensible classes/severity/cards; screenshot for README.
- `/search "foundation crack"` returns crack-tagged cards.
- README: Phase 4 section (run instructions, screenshot, interim-classifier note), lockfile refresh, push.

## Execution notes
- Subagent-driven, two-stage review per task (combined for Task 4 UI).
- Docker db must stay up. NO AWS.
- Qwen download (~7GB) happens once in Task 5's real run — controller runs it foreground with progress.

## Self-review
Spec §7 coverage: /analyze ✓ /search ✓ /health-real ✓ env-config ✓ MPS fp16 ✓ markdown export ✓ (PDF stays stretch); severity rule-based per §3 ✓; interim classifier honestly documented ✓. Placeholders: none. Types consistent with existing modules (reuses rrf_fuse pattern, Card, db.top_k).
