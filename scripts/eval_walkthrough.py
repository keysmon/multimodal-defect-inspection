"""Golden-set eval for the walkthrough diagnostic report.

Usage (from the repo root):
  .venv/bin/python scripts/eval_walkthrough.py --provider bedrock   # live run
  .venv/bin/python scripts/eval_walkthrough.py --provider local     # $0 fallback
  .venv/bin/python scripts/eval_walkthrough.py --diff-only          # re-check gate
  .venv/bin/python scripts/eval_walkthrough.py --provider bedrock --limit 1  # smoke

Metrics (honest, mirrors agent_eval). IMPORTANT SEMANTICS: "grounded" here
means CITATION-PRESENT - the claim cites a card retrieved for this
walkthrough (per-photo claims: that photo's own retrieval + concern
retrievals; visit-level claims incl. the assessment: the walkthrough
union). It does NOT verify the card's content supports the claim; support
is covered only by the hand-rated spot-check.
- groundedness (post-gate): kept claims (per-photo findings, action items,
  cited answers, the shipped assessment) carrying citations / kept claims.
  1.0 by construction of the gate; measured, not assumed.
- raw_groundedness (pre-gate): kept / (kept + dropped no_valid_citation).
  The drift signal - if it slides, tighten the synthesis prompt.
- coverage: concerns Haiku answered on its own / concerns. Only a
  missing_answer (Haiku skipped the concern) lowers it; an answer dropped
  for bad citations lowers raw_groundedness and answered_with_evidence_rate
  instead.
- answered_with_evidence_rate: cited (non-not-observed) answers / concerns.
  GATED: the golden set is frozen, so a drop means degradation, not an
  honestly-unanswerable input.
- flagged_rate, latency, cost: reported only. flagged_claims reasons are
  heterogeneous (dropped claims, stripped invalid ids, overflow) - the rate
  measures total gate activity.
- Visual accuracy is NOT auto-measured (no labels for "did Haiku read the
  photo right"): results/walkthrough_spotcheck.md is the hand-rating
  template; the results file states the limitation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from defectlens.eval.gate import finalize_run, regression_check

RESULTS = Path("results/walkthrough_eval.json")
REJECTED = Path("results/walkthrough_eval.rejected.json")
GOLDEN = Path("data/manifests/walkthrough_golden.json")
SPOTCHECK = Path("results/walkthrough_spotcheck.md")
RUNS_DIR = Path("results/walkthrough_runs")
GATED_METRICS = ("raw_groundedness", "coverage", "answered_with_evidence_rate")
VISUAL_ACCURACY_NOTE = (
    "not auto-measured (no labels for visual correctness); "
    "hand-rated spot-check template at results/walkthrough_spotcheck.md"
)
GROUNDEDNESS_NOTE = (
    "groundedness = citation-presence within the walkthrough's retrieved set "
    "(per-photo scoped); card-content support is NOT auto-verified - see the "
    "hand-rated spot-check"
)


def report_metrics(report: dict) -> dict:
    per_photo = report["per_photo"]
    answers = report["summary"]["answers"]
    action_items = report["summary"]["action_items"]
    flagged = report.get("flagged_claims", [])
    concerns = report.get("concerns", [])

    kept = (
        [f for f in per_photo if not f.get("no_evidence")]
        + list(action_items)
        + [a for a in answers if not a.get("not_observed")]
    )
    # The assessment narrative is a gated claim too: it counts as kept only
    # when it shipped with citations (the uncited case is the deterministic
    # fallback, whose dropped original already sits in flagged_claims).
    if report["summary"].get("assessment_citations"):
        kept.append({"citations": report["summary"]["assessment_citations"]})

    def _cites(claim: dict) -> list:
        return claim.get("cited", claim.get("citations", []))

    groundedness = sum(1 for c in kept if _cites(c)) / len(kept) if kept else 1.0
    n_dropped = sum(1 for f in flagged if f.get("reason") == "no_valid_citation")
    raw_total = len(kept) + n_dropped
    raw_groundedness = len(kept) / raw_total if raw_total else 1.0
    n_missed = sum(1 for f in flagged if f.get("reason") == "missing_answer")
    coverage = (len(concerns) - n_missed) / len(concerns) if concerns else 1.0
    evidenced = sum(1 for a in answers if not a.get("not_observed"))
    return {
        "groundedness": groundedness,
        "raw_groundedness": raw_groundedness,
        "coverage": coverage,
        "answered_with_evidence_rate": evidenced / len(concerns) if concerns else 1.0,
        "flagged_rate": len(flagged) / (len(kept) + len(flagged)) if (kept or flagged) else 0.0,
    }


def aggregate_metrics(per_walkthrough: dict) -> dict:
    """Mean quality metrics over SUCCESSFUL walkthroughs; schema_valid_rate
    carries the failures (same two-signal split as the agent eval - a crashed
    walkthrough is not double-counted as 0.0 quality)."""
    successes = {wid: m for wid, m in per_walkthrough.items() if "error" not in m}
    if not successes:
        raise ValueError("no walkthrough produced a validated report")
    keys = (
        "groundedness", "raw_groundedness", "coverage",
        "answered_with_evidence_rate", "flagged_rate",
    )
    metrics = {k: sum(m[k] for m in successes.values()) / len(successes) for k in keys}
    metrics["schema_valid_rate"] = len(successes) / len(per_walkthrough)
    return metrics


def write_spotcheck_template(reports: dict[str, dict], path: Path) -> None:
    lines = [
        "# Walkthrough visual-accuracy spot-check (hand-rated)",
        "",
        "Visual accuracy is NOT auto-measured - there are no labels for whether",
        "the model read a photo correctly. Rate each observation against its",
        "photo: mark [x] when accurate; leave unchecked and add a note when not.",
        "",
    ]
    for wid, rep in sorted(reports.items()):
        lines.append(f"## {wid}")
        for f in rep["per_photo"]:
            tag = "no_evidence" if f.get("no_evidence") else ", ".join(f.get("cited", []))
            lines.append(f"- [ ] {f['photo_id']} ({tag}): {f['observation']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def load_golden() -> list[dict]:
    return json.loads(GOLDEN.read_text())["walkthroughs"]


def build_components(provider_name: str):
    """Load the real Recognizer once; pick the reasoning provider.

    Note: the provider choice swaps ONLY the reasoner; retrieval stays the
    local CLIP+index path either way (CLIP is retrieval-only by design).
    """
    from defectlens.serve.recognizer import Recognizer

    recognizer = Recognizer()
    recognizer.load()
    if provider_name == "local":
        from defectlens.agent.providers import LocalQwenProvider
        from defectlens.serve.describer import Describer

        describer = Describer()
        describer.load()
        provider = LocalQwenProvider(describer=describer)
    elif provider_name == "bedrock":
        from defectlens.agent.providers import BedrockHaikuProvider

        provider = BedrockHaikuProvider()
    else:
        from defectlens.agent.providers import MockProvider

        provider = MockProvider(
            responses=['["smoke concern"]', '{"per_photo": [], "summary": {}}'] * 20
        )
    return recognizer, provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["bedrock", "local", "mock"], default="bedrock")
    parser.add_argument("--diff-only", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="run only the first N walkthroughs (smoke; no gate, no persist)")
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
            failed = regression_check(previous, metrics, GATED_METRICS, tolerance=args.tolerance)
            if failed:
                print(f"REGRESSION: {failed}", file=sys.stderr)
                return 1
        return 0

    from defectlens.report.synthesize import run_walkthrough

    recognizer, provider = build_components(args.provider)
    walks = load_golden()
    if args.limit is not None:
        walks = walks[: args.limit]

    per_walkthrough: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    latencies: list[float] = []
    for walk in walks:
        wid = walk["walkthrough_id"]
        t0 = time.perf_counter()
        try:
            photos = [
                {
                    "photo_id": p["photo_id"],
                    "image_bytes": Path(p["image_path"]).read_bytes(),
                    "note": p.get("note"),
                }
                for p in walk["photos"]
            ]
            report = run_walkthrough(
                photos=photos,
                visit_note=walk["visit_note"],
                recognizer=recognizer,
                provider=provider,
            )
            report_dict = json.loads(report.model_dump_json())
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / f"report_{wid}.json").write_text(json.dumps(report_dict, indent=2))
            reports[wid] = report_dict
            per_walkthrough[wid] = report_metrics(report_dict)
            latencies.append(time.perf_counter() - t0)
        except Exception as exc:  # crash isolation: one bad walkthrough must not sink the run
            per_walkthrough[wid] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{wid}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"{wid}: done in {time.perf_counter() - t0:.0f}s", flush=True)

    try:
        metrics = aggregate_metrics(per_walkthrough)
    except ValueError:
        print("all walkthroughs failed; nothing written", file=sys.stderr)
        return 2
    n_success = len(latencies)
    metrics["latency_s_per_walkthrough"] = round(sum(latencies) / n_success, 2)
    # Total run spend over DELIVERED reports: spend on walkthroughs that later
    # crashed is included in the numerator, so this slightly overstates the
    # marginal cost of a successful report - deliberate (it is what a run costs).
    metrics["cost_usd_per_walkthrough"] = round(provider.usage().cost_usd / n_success, 5)

    if reports:
        write_spotcheck_template(reports, SPOTCHECK)

    if args.limit is not None:
        print(json.dumps(metrics, indent=2))
        print(f"--limit {args.limit}: smoke run only; results file and gate skipped",
              file=sys.stderr)
        return 0

    payload = {
        "run_config": {
            "provider": provider.name,
            "n_walkthroughs": len(per_walkthrough),
            "visual_accuracy": VISUAL_ACCURACY_NOTE,
            "groundedness_semantics": GROUNDEDNESS_NOTE,
        },
        "metrics": metrics,
        "per_walkthrough": per_walkthrough,
    }
    return finalize_run(
        payload, previous,
        results_path=RESULTS, rejected_path=REJECTED,
        gated=GATED_METRICS, tolerance=args.tolerance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
