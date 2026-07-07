# DefectLens

Building-defect inspection assistant: photo → fine-grained defect ID + severity
framing + retrieved remediation guidance and standards citations.

Design spec: `docs/superpowers/specs/2026-07-06-defect-lens-design.md`

## Status

Phase 1 (dataset unification + CLIP zero-shot baseline) — **complete**.
Next: Phase 2 (cross-modal RAG over inspection-standards corpus).

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
