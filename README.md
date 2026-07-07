# DefectLens

Building-defect inspection assistant: photo → fine-grained defect ID + severity
framing + retrieved remediation guidance and standards citations.

Design spec: `docs/superpowers/specs/2026-07-06-defect-lens-design.md`

## Status

Phase 1 (dataset unification + CLIP zero-shot baseline) — **complete**.
Phase 2 (cross-modal RAG over inspection-standards corpus) — **complete**.
Next: Phase 3 (Qwen2.5-VL-3B LoRA fine-tune on AWS).

## Phase 2 Results — Cross-Modal RAG

**Corpus:** 205 cited guidance cards (`corpus/`) from EPA, HUD NSPIRE,
InterNACHI, and FHWA/NPS engineering references — every class ≥15 cards,
severity-labelled, licensing-clean (own-words passages; ICC/ACI cited by
reference only). Indexed in pgvector (`docker compose up -d db`) as one shared
CLIP space with two vectors per card: an index-sentence text embedding and a
train-split-only exemplar-image centroid.

**Retrieval eval** (frozen `data/manifests/test.csv`, 2,648 image queries +
36 templated text queries; a card is relevant iff tagged with the query's true
class):

| Query mode | recall@5 |
|---|---|
| Text → text vectors | **1.000** |
| Image → exemplar centroids | 0.773 |
| Image → **RRF fusion** (centroid + zero-shot prompt rankings) | **0.863** |

Fusion of the two decorrelated rankings lifts the fine-grained failures
(crack 0.65→0.92, efflorescence 0.85→0.95); spalling (0.60) remains the
weakest class in both signal paths — the measured motivation for Phase 3's
fine-tune. Reproduce: `python -m defectlens.rag.embed` then
`python -m defectlens.eval.rag_recall --image-mode fused`.

## Results

Unified dataset: 17,652 images / 9 classes merged from CODEBRIM, BD3, and
SDNET2018 (`docs/datasets.md`); frozen stratified test split of 2,648 images
(`data/manifests/test.csv`, seed 42). Spot-check QA on the label mapping passed
(30 images/class sampled; all classes ≥90% plausible).

| Model | Macro top-1 | Macro top-3 | Split |
|---|---|---|---|
| CLIP ViT-L/14 zero-shot (prompt ensemble) | 0.472 | 0.747 | frozen `data/manifests/test.csv` |
| Qwen2.5-VL-3B + LoRA (Phase 3) | — | — | same |

![CLIP zero-shot confusion matrix](results/clip_zeroshot_confusion.png)

Zero-shot CLIP is strong on commodity classes (crack 0.85, mold/algae 0.80,
peeling paint 0.77 top-1) but fails on exactly the fine-grained distinctions an
inspector needs — spalling 0.15, no-defect 0.17, exposed rebar 0.25, corrosion
stain 0.33 — which is the measured gap the Phase 3 fine-tune exists to close.
Environment for these numbers is pinned in `requirements-lock.txt`
(transformers 5.13.0, torch — see lockfile).

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
