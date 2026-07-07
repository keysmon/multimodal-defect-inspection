# DefectLens

Building-defect inspection assistant: photo → fine-grained defect ID + severity
framing + retrieved remediation guidance and standards citations.

Design spec: `docs/superpowers/specs/2026-07-06-defect-lens-design.md`

## Status

Phase 1 (dataset unification + CLIP zero-shot baseline) — in progress.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
