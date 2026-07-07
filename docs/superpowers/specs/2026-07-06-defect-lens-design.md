# DefectLens — Building-Defect Inspection Assistant (Design Spec)

**Date:** 2026-07-06
**Status:** Approved by user (design review 2026-07-06)
**Working name:** DefectLens (subject to change)

## 1. Summary

A professional tool for home inspectors and contractors: photograph a building
defect → fine-grained identification + severity framing + retrieved remediation
guidance and standards citations. The guidance corpus is also searchable by
plain-text query. Successor project to the Multimodal Cooking Assistant
(`fullstack` repo), reusing its architecture pattern in a domain where the
fine-tuning is justified and the product gap is real.

**Target resume bullets (the measurable outcomes this project must produce):**

> Fine-tuned a VLM (Qwen2.5-VL-3B) with LoRA on a unified multi-dataset corpus
> of building defects, lifting macro top-1 accuracy from X% (CLIP zero-shot) to
> Y% on a frozen held-out split; served full-stack via FastAPI + React.

> Grounded the VLM with cross-modal RAG over an inspection-standards corpus in
> a shared CLIP image-text space (pgvector), retrievable by defect photo or
> text query, reaching recall@5 of Z (target ≥ 0.85).

## 2. Context and key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Domain | Building-defect inspection | Public labeled data is abundant (unlike equipment maintenance, whose emptiness is caused by proprietary data); fine-tune justified (CLIP is weak on efflorescence/mold/spalling distinctions); broad, live-testable demo; real product whitespace (incumbent inspection software has no vision AI). |
| Persona (v1) | Home inspector / contractor | Pro tool → defensible severity framing (pros verify output), report generation is a natural killer feature. |
| Compute | AWS GPU (g5.xlarge spot) for training only; local M3 Pro (18GB) for serving; pgvector in local Docker Postgres | Budget <$100; GPU cost only during training runs. |
| ML stack | PyTorch + HuggingFace transformers/PEFT (drops MLX) | Standard, reproducible; MLX adapters aren't portable to PyTorch serving. |
| Model | Qwen2.5-VL-3B-Instruct | First-class HF + PEFT support; strong fine-grained vision; fp16 (~7GB) serves on 18GB M3 Pro via MPS. |
| Taxonomy | 8 defect classes + no-defect (broad building envelope) | Matches what inspectors photograph; multi-dataset unification is itself a portfolio artifact. |
| Repo | Fresh repo (this one) | Clean portfolio identity; cooking assistant remains a separate intact project. |
| Timeline | 4–6 weeks part-time, <$100 AWS | 2–3 training runs on spot instances. |

## 3. Architecture

```
photo ──► Fine-tuned Qwen2.5-VL-3B ──► defect class (top-k) + description
                │                            │
                ▼                            ▼
        CLIP embedding ──► pgvector ──► guidance cards (remediation + citation)
                                             │
text query ──► CLIP text embedding ──────────┘
                                             ▼
                              React UI: results panel → exportable report
```

Components (each independently testable):

- **Dataset pipeline** — merges public datasets into the unified taxonomy;
  produces frozen train/test splits.
- **Recognition service** — fine-tuned VLM; classify (top-k) + describe.
- **Retrieval service** — CLIP embeddings + pgvector; query by image or text.
- **Severity mapper** — rule-based class→severity-band table, refined by
  retrieved guidance. **Not a learned model in v1.** (e.g., exposed rebar →
  structural / refer to engineer; efflorescence → moisture indicator / monitor.)
- **API + UI** — FastAPI backend, React frontend, report export.

## 4. Unified dataset (Sub-project foundation, feeds both metrics)

**Taxonomy (9 labels):** crack, spalling, efflorescence, exposed rebar,
corrosion stain, mold/algae, water damage, peeling paint, no-defect.

| Source | Size | Contributes | Mapping notes |
|---|---|---|---|
| CODEBRIM | 1,590 imgs / 8,323 objects | crack, spalling, efflorescence, exposed rebar, corrosion stain | crop per bounding box |
| BD3 | 3,965 imgs | algae→mold/algae; major+minor crack→crack; peeling; spalling; stain→water damage | mapping documented per class |
| Roboflow building-defect-on-walls | 472 imgs | mold, water seepage→water damage, peeling paint, stairstep crack→crack | fills consumer-wall gap |
| SDNET2018 | 56,000+ imgs | crack + no-defect negatives | subsample to control imbalance |

Mechanics:

- **Versioned label-mapping table** — every mapping decision recorded with
  rationale; committed to the repo (portfolio artifact).
- **Frozen test split** — stratified by dataset and class, committed **before
  any training**; all reported numbers use this split.
- **Macro-averaged metrics** — cracks dominate raw counts.
- **Spot-check protocol** — manual review of ~30 images/class post-merge to
  catch mapping garbage.
- License review per dataset recorded alongside the mapping table.

## 5. Sub-project A — LoRA fine-tune + recognition eval

- **Training:** QLoRA (4-bit base, LoRA r=16 on attention + MLP projections)
  via transformers + peft + bitsandbytes on g5.xlarge spot (A10G 24GB).
  Instruction-formatted samples: image + defect question → templated structured
  answer. Est. $3–6 and a few hours per run; 2–3 runs within budget.
  Checkpoint/resume to survive spot interruption. After training, the LoRA
  adapter is merged into the base weights and exported fp16 for local serving.
- **Top-k metric methodology:** rank the 9 class names by sequence
  log-likelihood under the model; score top-1 / top-3. Deterministic and
  defensible — no free-text parsing.
- **Baseline ("money chart"):** CLIP ViT-L/14 zero-shot with prompt ensembling
  on the same frozen split. Expected ~40–60% macro top-1 baseline vs. ≥80%
  fine-tuned target. Deliverables include a per-class confusion matrix
  highlighting efflorescence / mold / water-stain confusions.

## 6. Sub-project B — Cross-modal RAG + retrieval eval

- **Corpus (~200–500 guidance cards):** InterNACHI Standards of Practice, EPA
  mold guidance (public domain), HUD/FEMA housing-inspection docs (public
  domain), enacted building codes via Public.Resource.org. **ICC code sections
  are cited by reference, never redistributed.** Card schema: title, guidance
  passage, severity note, source citation, defect-class tags.
- **Indexing (shared CLIP space, multi-vector):** CLIP's text tower truncates
  at 77 tokens and image→text retrieval across the modality gap is weak, so
  each card is indexed by multiple vectors in one pgvector table:
  1. a short *index sentence* embedded via CLIP-text;
  2. a *centroid of exemplar defect-image embeddings* via CLIP-image.
  Image queries hit exemplar vectors (image↔image is strong); text queries hit
  text vectors. One shared CLIP embedding space throughout.
- **Recall@5 methodology:** query with held-out test images and templated text
  queries; a retrieved card is relevant iff tagged with the query's true class.
  Target ≥ 0.85.
- **Fallbacks if stock CLIP underperforms:** boost retrieval by predicted
  class; light contrastive fine-tune of CLIP (stretch; budget allows).

## 7. Serving & UI

- **Backend (FastAPI):** `POST /analyze` (image → top-k classes, description,
  severity band, guidance cards), `POST /search` (text → cards), `GET /health`.
  Qwen2.5-VL-3B fp16 on MPS (~7GB; M3 Pro 18GB confirmed sufficient — quantized
  serving documented as contingency only). Model loaded once at startup.
  Postgres + pgvector via docker-compose. All URLs/origins via env config (no
  hardcoding).
- **Frontend (React, ported CRA scaffolding):** upload + preview → results
  panel (defect chips with confidence, severity band, guidance cards with
  citations) + text-search box. Markdown report export for the inspector
  persona (PDF export = stretch).

## 8. Build order & milestones (4–6 weeks part-time)

| Wk | Phase | Exit criterion |
|---|---|---|
| 1 | Repo scaffold + dataset unification + eval harness | frozen test split committed; CLIP zero-shot baseline number |
| 2 | RAG corpus + pgvector + retrieval eval | recall@5 measured |
| 3 | AWS QLoRA fine-tune (2–3 runs) | fine-tuned top-1/top-3 beats baseline on frozen split |
| 4 | Serving + React UI integration | end-to-end photo→guidance demo works locally |
| 5–6 | Report export, polish, README with charts | portfolio-ready |

Rationale: all dollar-costing steps (GPU) come after the free steps that
validate them; week 1's baseline gives Sub-project A its "before" number.

## 9. Testing

- Unit tests: label-mapping table application; card chunking/schema.
- Eval reproducibility: fixed seeds, committed splits; eval scripts rerunnable
  end-to-end.
- API integration tests with a mocked model (3B does not load in CI).
- Frontend: basic React Testing Library smoke tests.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cross-dataset label noise | spot-check protocol; per-dataset ablation |
| CLIP modality gap hurts recall@5 | exemplar-vector indexing (§6); class-boost fallback |
| Spot instance interruption | checkpoint + resume |
| ICC copyright | cite-by-reference only; corpus from public-domain / open sources |
| MPS memory pressure while serving | 18GB confirmed adequate; quantized fallback documented |
| Solo-project stall | weekly exit criteria (§8); each phase leaves a complete-looking artifact |

## 11. Out of scope for v1

- Learned severity model (severity is rule-based).
- Multi-defect detection/localization in a single photo (single primary label).
- PDF report export (markdown ships; PDF is stretch).
- Property-manager workflows (buildings/units/history data model).
- Cloud deployment of the serving stack (local demo; AWS used for training only).
