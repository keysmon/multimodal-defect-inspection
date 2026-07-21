"""Golden-set eval for the inspection workflow, with regression diffing.

Usage (from the repo root):
  .venv/bin/python scripts/eval_agent.py --provider local   # live run
  .venv/bin/python scripts/eval_agent.py --diff-only        # re-check gate
  .venv/bin/python scripts/eval_agent.py --provider local --limit 1  # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from defectlens.grounding.citations import citation_is_class_relevant

RESULTS = Path("results/agent_eval.json")
REJECTED = Path("results/agent_eval.rejected.json")
GOLDEN = Path("data/manifests/agent_golden.csv")
GATED_METRICS = ("findings_recall", "citation_validity")


def property_metrics(report: dict, expected: set[str], card_tags: dict[str, list[str]]) -> dict:
    measured = [f for f in report["findings"] if f["tier"] == "measured"]
    found = {f["defect_class"] for f in measured}
    recall = len(found & expected) / len(expected) if expected else 1.0
    precision = (
        len([f for f in measured if f["defect_class"] in expected]) / len(measured)
        if measured
        else 1.0
    )
    cites = [
        (f["defect_class"], c["card_id"])
        for f in measured
        for c in f.get("citations", [])
    ]
    valid = [citation_is_class_relevant(cid, cls, card_tags) for cls, cid in cites]
    citation_validity = sum(valid) / len(valid) if valid else 1.0
    return {
        "findings_recall": recall,
        "findings_precision": precision,
        "citation_validity": citation_validity,
    }


def regression_check(prev: dict, curr: dict, tolerance: float = 0.02) -> list[str]:
    return [m for m in GATED_METRICS if curr.get(m, 0.0) < prev.get(m, 0.0) - tolerance]


def aggregate_metrics(per_property: dict) -> dict:
    """Aggregate per-property metrics into run-level means.

    A property is a success when run_inspection produced a validated report
    (its entry has metric values); a failure carries an "error" key instead.
    schema_valid_rate = n_success / n_total measures pipeline reliability.
    The gated quality metrics (findings_recall / findings_precision /
    citation_validity) are averaged over SUCCESSFUL properties only: they
    measure agent quality on the reports that exist, while schema_valid_rate
    already accounts for the failures - counting a crashed property as 0.0
    quality would double-penalize it and blur the two signals.

    Raises ValueError when no property succeeded.
    """
    successes = {pid: m for pid, m in per_property.items() if "error" not in m}
    if not successes:
        raise ValueError("no property produced a validated report")
    metrics = {
        k: sum(p[k] for p in successes.values()) / len(successes)
        for k in ("findings_recall", "findings_precision", "citation_validity")
    }
    metrics["schema_valid_rate"] = len(successes) / len(per_property)
    return metrics


def finalize_run(payload: dict, previous: dict | None, tolerance: float) -> int:
    """Gate against the previous baseline BEFORE persisting anything.

    A passing (or first-ever) run overwrites RESULTS; a regressed run is
    written to REJECTED and exits nonzero, so a bad run can never clobber
    the baseline it failed against.
    """
    metrics = payload["metrics"]
    print(json.dumps(metrics, indent=2))
    if previous:
        failed = regression_check(previous, metrics, tolerance=tolerance)
        for m in GATED_METRICS:
            print(f"{m}: {previous.get(m):.3f} -> {metrics.get(m):.3f}")
        if failed:
            REJECTED.parent.mkdir(parents=True, exist_ok=True)
            REJECTED.write_text(json.dumps(payload, indent=2))
            print(
                f"REGRESSION: {failed}; baseline {RESULTS} kept, "
                f"run written to {REJECTED}",
                file=sys.stderr,
            )
            return 1
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2))
    return 0


def load_golden() -> dict[str, dict]:
    props: dict[str, dict] = defaultdict(lambda: {"images": [], "expected": set()})
    with GOLDEN.open() as f:
        for row in csv.DictReader(f):
            p = props[row["property_id"]]
            p["images"].append(row["image_path"])
            if row["unified_label"] != "no_defect":
                p["expected"].add(row["unified_label"])
    return dict(props)


def build_components(provider_name: str):
    """Load real components once. local = Describer-backed Qwen; mock for dry runs.

    Mirrors the serve/api.py lifespan wiring: Recognizer()/Describer() no-args
    then .load() (pgvector DB and the local model must be available - this
    only runs in live mode, never under the unit tests).

    Note: --provider bedrock swaps ONLY the reasoning LLM; the classifier
    (Describer) and retrieval (Recognizer/pgvector) remain local either way.
    """
    from defectlens.agent.providers import LocalQwenProvider, MockProvider
    from defectlens.corpus import load_corpus_dir
    from defectlens.serve.describer import Describer
    from defectlens.serve.recognizer import Recognizer

    recognizer = Recognizer()
    recognizer.load()
    describer = Describer()
    describer.load()
    if provider_name == "local":
        provider = LocalQwenProvider(describer=describer)
    elif provider_name == "bedrock":
        from defectlens.agent.providers import BedrockHaikuProvider

        provider = BedrockHaikuProvider()
    else:
        provider = MockProvider(responses=["[]"] * 200)
    cards = load_corpus_dir(Path("corpus"))
    # Card ids live on Card.id (corpus.py); keyed here for citation validity.
    card_tags = {c.id: list(c.class_tags) for c in cards}
    return describer, recognizer, provider, card_tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["local", "bedrock", "mock"], default="local")
    parser.add_argument("--out-dir", type=Path, default=Path("results/agent_runs"))
    parser.add_argument("--diff-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="run only the first N golden properties (sorted by id) - smoke "
        "runs; a limited run neither updates results/agent_eval.json nor "
        "applies the regression gate",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.limit is not None and args.diff_only:
        parser.error("--limit cannot be combined with --diff-only")
    if args.diff_only and not RESULTS.exists():
        parser.error(f"--diff-only needs an existing {RESULTS}; run the eval first")

    previous = json.loads(RESULTS.read_text())["metrics"] if RESULTS.exists() else None

    if args.diff_only:
        metrics = json.loads(RESULTS.read_text())["metrics"]
        print(json.dumps(metrics, indent=2))
        if previous:
            failed = regression_check(previous, metrics, tolerance=args.tolerance)
            for m in GATED_METRICS:
                print(f"{m}: {previous.get(m):.3f} -> {metrics.get(m):.3f}")
            if failed:
                print(f"REGRESSION: {failed}", file=sys.stderr)
                return 1
        return 0

    from defectlens.agent.inspect import run_inspection

    describer, recognizer, provider, card_tags = build_components(args.provider)
    golden = load_golden()
    items = sorted(golden.items())
    if args.limit is not None:
        items = items[: args.limit]
    per_property, latencies = {}, []
    for pid, spec in items:
        t0 = time.perf_counter()
        # Crash isolation: one bad property (model hiccup, unreadable image,
        # schema violation) must not sink the whole run.
        try:
            report, _usage, _trace = run_inspection(
                property_id=pid,
                image_paths=spec["images"],
                describer=describer,
                recognizer=recognizer,
                provider=provider,
                out_dir=args.out_dir,
            )
            elapsed = time.perf_counter() - t0
            per_property[pid] = property_metrics(
                json.loads(report.model_dump_json()), spec["expected"], card_tags
            )
            latencies.append(elapsed)
        except Exception as exc:
            per_property[pid] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{pid}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            # MPS allocations fragment across ~35 generate() calls per property;
            # a 15-property run swap-thrashed and was jetsam-killed on an 18GB
            # machine (observed 2026-07-10: 4min/property degrading to 19min,
            # then killed on the last property). Numerically a no-op.
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass
        print(f"{pid}: done in {time.perf_counter() - t0:.0f}s", flush=True)

    try:
        metrics = aggregate_metrics(per_property)
    except ValueError:
        print(
            f"all {len(per_property)} properties failed; no validated reports, "
            "nothing written",
            file=sys.stderr,
        )
        return 2
    n_success = len(latencies)
    metrics["latency_s_per_report"] = round(sum(latencies) / n_success, 2)
    metrics["cost_usd_per_report"] = round(provider.usage().cost_usd / n_success, 5)
    if args.limit is not None:
        # Partial run: not comparable to the frozen baseline, so neither
        # persist it nor gate on it.
        print(json.dumps(metrics, indent=2))
        print(
            f"--limit {args.limit}: smoke run only; results file and "
            "regression gate skipped",
            file=sys.stderr,
        )
        return 0
    payload = {
        "run_config": {"provider": provider.name, "n_properties": len(per_property)},
        "metrics": metrics,
        "per_property": per_property,
    }
    return finalize_run(payload, previous, tolerance=args.tolerance)


if __name__ == "__main__":
    raise SystemExit(main())
