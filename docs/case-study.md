# Case study: making a vision-language model trustworthy enough for field diagnostics

This project's central engineering problem was never "classify the defect."
It was: **how does a system built on models that can hallucinate produce a
report a technician can act on?** This document traces that answer through
the walkthrough diagnostic report - the feature where every earlier phase
(fine-tune, retrieval, serving, agent, evals) converges.

## The product shape

A technician's first site visit produces up to 10 photos, optional per-photo
notes, and a free-text concern note ("is the reinforcement at risk? is the
crack active?"). The system returns a structured draft diagnostic: per-photo
observations, a prioritized action list, an answer to every concern, and a
fixed disclaimer. It is explicitly a *draft to verify*, not a verdict.

Three model roles, deliberately separated:

- **Reasoner** - a general vision-language model receives ALL photos in one
  multimodal call, plus retrieved guidance cards and the concern checklist.
  Cross-photo reasoning is the point: staining in one photo changes the
  reading of a crack in another.
- **Retrieval** - CLIP embeddings over a 205-card cited standards corpus
  (per-photo image retrieval + per-concern text retrieval). Retrieval is
  demoted to supplying *candidates to cite*; it never produces the headline.
- **Narrow specialist** - the fine-tuned 9-class classifier (0.851 macro
  top-1) runs only as user-triggered enrichment on a scale-to-zero GPU, and
  its label merges only when consistent with what the reasoner observed.

## The trust mechanism: the LLM proposes, a gate disposes

Prompt instructions alone do not make grounding a property of the system;
they make it a hope. The walkthrough's grounding is enforced *after* the
model responds, in deterministic code (`grounding/` + `report/synthesize.py`):

1. **Citation gate.** Every claim must cite cards retrieved for this
   walkthrough. Invalid or hallucinated card ids are stripped and recorded;
   claims left uncited are dropped into a `flagged_claims` log - visible in
   the UI as "N claims were dropped by the citation gate," never silently
   kept.
2. **Scoped evidence.** A per-photo observation may cite only that photo's
   own retrieval (plus concern-driven retrieval). Cross-photo synthesis is
   allowed exactly where it belongs: visit-level action items, answers, and
   the assessment narrative - which is itself a gated claim with citations,
   not free text riding above the rules.
3. **Structural completeness.** The gate iterates the *input* photo list and
   the *extracted* concern list, so the model can neither drop a photo nor
   skip a concern. A concern the photos cannot answer gets an explicit
   "not observed - verify on-site" - the anti-hallucination rule is a
   first-class output, and the schema validators make an ungrounded claim
   unconstructible (a grounded claim without citations, or a not-observed
   claim with them, cannot be built).
4. **Gated enrichment.** The narrow classifier forces every photo into its
   9 classes, so it confidently mislabels out-of-scope scenes. Its label
   merges only above a confidence floor AND when keyword-consistent
   (negation-aware) with the observation text; a `no_defect` label can never
   land on a grounded finding. Kept-vs-dropped counts are logged - the gate's
   activity is itself an eval surface.

## Evaluation: honest numbers over impressive ones

Two frozen golden sets, each with a committed baseline and a regression gate
that writes failing runs aside rather than overwriting the baseline:

- **Dataset crops** (6 walkthroughs x 5 photos from the labeled test data):
  deliberately hard - 256px context-free crops.
- **Realistic field photos** (licensed Wikimedia Commons images, pinned-hash
  fetch script): full-context photos of real damp walls, mold interiors,
  spalled cover with exposed mesh, through-wall masonry cracks.

Automated metrics measure what they can honestly measure: groundedness is
*citation-presence within the retrieved set* (stated in the results files),
coverage is "did the model address every concern on its own." Both hold at
1.0. What automated metrics cannot measure - did the model read the photo
correctly? - is deliberately not faked with an LLM judge. It is covered by a
hand-rated spot-check committed next to the metrics, and the split it found
is the project's most instructive result:

| | Dataset crops | Realistic photos |
|---|---|---|
| accurate observations | 17/30 strict | 8/8 primary |

On context-free crops, the reasoner systematically over-calls hairline
cracks on clean textured concrete - a false-positive lean consistent with
the earlier cross-dataset OOD study, and the safer failure direction for
inspection. On realistic photos, every primary observation was correct. The
practical reading: visual accuracy is context-bound, the "verify on-site"
framing is load-bearing, and evaluation sets must include the easy-for-
humans/hard-for-models regime AND the deployment-realistic regime to see
either fact.

The realistic set also demonstrated why realistic eval data pays for itself
immediately: its first run exposed a serving bug no crop could trigger
(photographic PNG re-encoding exceeded the model API's 5 MB per-image cap),
fixed the same day with a regression test.

## Engineering discipline that made it hold

- **One trust story.** The citation-validity logic was extracted from the
  earlier inspection agent into a shared `grounding/` module both features
  call; the agent's behavior was regression-locked through the refactor.
- **Reviews as part of the build.** Two adversarial code reviews ran during
  development; their findings drove real changes - the assessment narrative
  turned out to be the one ungated claim and became a gated one; GPU
  enrichment fan-out became resumable so a partial failure plus a retry
  cannot re-wake (re-bill) already-submitted jobs.
- **Async-first serving.** The report runs on the same submit -> S3 ->
  worker -> poll path as single-photo analysis (one Lambda container
  reprocessing its own queue), so a cold start delays the poll instead of
  breaking the request; enrichment never blocks the report.
- **Determinism where it counts.** Temperature 0 for the eval provider,
  frozen manifests, committed baselines, and a gate that cannot be
  overwritten by a bad run.

## What it is not

Not a final inspection verdict, not a general home-inspection classifier,
and not a system that claims its narrative is "verified" - the disclaimer,
the flagged-claims log, and the stated limits of each metric are as much a
part of the product as the report itself.
