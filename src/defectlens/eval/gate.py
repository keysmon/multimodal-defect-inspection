"""Shared eval regression gate: a regressed run can never clobber the
baseline it failed against. Extracted from scripts/eval_agent.py so the
agent eval and the walkthrough eval enforce the identical discipline."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def regression_check(
    prev: dict, curr: dict, gated: tuple[str, ...], tolerance: float = 0.02
) -> list[str]:
    return [m for m in gated if curr.get(m, 0.0) < prev.get(m, 0.0) - tolerance]


def finalize_run(
    payload: dict,
    previous: dict | None,
    *,
    results_path: Path,
    rejected_path: Path,
    gated: tuple[str, ...],
    tolerance: float,
) -> int:
    """Gate against the previous baseline BEFORE persisting anything.

    A passing (or first-ever) run overwrites results_path; a regressed run is
    written to rejected_path and exits nonzero, so a bad run can never clobber
    the baseline it failed against.
    """
    metrics = payload["metrics"]
    print(json.dumps(metrics, indent=2))
    if previous:
        failed = regression_check(previous, metrics, gated, tolerance=tolerance)
        for m in gated:
            print(f"{m}: {previous.get(m):.3f} -> {metrics.get(m):.3f}")
        if failed:
            rejected_path.parent.mkdir(parents=True, exist_ok=True)
            rejected_path.write_text(json.dumps(payload, indent=2))
            print(
                f"REGRESSION: {failed}; baseline {results_path} kept, "
                f"run written to {rejected_path}",
                file=sys.stderr,
            )
            return 1
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2))
    return 0
