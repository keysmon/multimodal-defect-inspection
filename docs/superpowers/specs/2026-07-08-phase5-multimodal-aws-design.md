# Phase 5: Multimodal Field-Inspection Assistant + AWS Deployment - Design Spec

Approved via grilling session 2026-07-08.
Supersedes nothing; builds on the Phase 1-4 foundation (frozen split, CLIP baseline 0.472, RAG recall@5 0.863, fine-tuned VLM 0.851, local serving).

## Goal and CV thesis

Evolve DefectLens from an image-only building-defect assistant into a **multimodal field-inspection assistant** (image + text notes + equipment audio) with HVAC as a first-class domain, **deployed on AWS** with a live demo URL.

Target roles: AI Engineer, SDE, MLE, Full-stack.
The plan is deliberately balanced so each role finds its chapter: modeling depth (MLE), cloud architecture (AI/SDE), product UI (full-stack), GenAI integration via Bedrock + RAG (AI Engineer).

**Quality bar (user-mandated):** methodology-credibility over benchmark-chasing.
An expert reader must recognize the eval and training design as correct; absolute numbers are reported honestly, one good training run per model, no grinding.
Fine-tuned weights stay off GitHub (S3 only); the repo showcases methodology.

## Locked decisions

1. **Frame:** evolve DefectLens (one repo, one deepening story) - not a sibling project.
2. **New modalities:** inspector text notes (committed), equipment audio (committed, full product mode), thermal imagery (stretch: one time-boxed scouting day; enters scope only if a real labeled dataset exists).
3. **Field photos (user-collected):** photos of the existing 9 classes form a private out-of-distribution eval set (~100+ target). HVAC-visual defect photos (dirty coils, burnt contactors, etc.) are collected from day one as a **seed for a future phase** but do NOT expand the taxonomy now. Framing: tight on defect/equipment, nothing identifying; dataset stays in the private S3 bucket; only metrics are published.
4. **Audio methodology: unsupervised anomaly detection, DCASE-faithful.** Pretrained audio embeddings (CLAP; BEATs as alternate) + simple density scoring (kNN/GMM) fit on NORMAL clips only, per MIMII/DCASE protocol. Supervised training on abnormal clips is explicitly rejected (apples-to-oranges vs the published baseline; reads as methodology error to expert reviewers). Framing in all write-ups: "industrial-equipment audio (MIMII), HVAC-motivated" - user confirmed real HVAC sounds differ from the benchmark; no overclaim.
5. **Audio integration: Level 2 (full third mode).** Anomaly score + severity framing in the UI, plus audio-to-guidance retrieval (CLAP shared text/audio space, mirroring the CLIP image/text design). Fallback if CLAP retrieval disappoints: class-label card lookup. Requires ~40 new HVAC-maintenance corpus cards (user sanity-checks as domain expert; same citation discipline as the existing 205).
6. **Fusion: late fusion with a visible combined assessment.** Each modality goes to its expert model (image+note jointly through the fine-tuned VLM - that pair is already true joint fusion; audio through the anomaly scorer). A composition step produces a unified report: per-modality findings, combined severity (worst-of + escalation rules, e.g. moderate visual + abnormal audio on the same unit escalates), and one narrative paragraph composed by the description model from all signals. Deep/joint audio-visual fusion is rejected as research-project scope.
7. **AWS deployment: two-tier scale-to-zero (Option B), demo-first framing.**
   - Frontend: S3 + CloudFront (static React).
   - API: API Gateway + Lambda (orchestration, RAG retrieval, audio scoring - CPU-capable).
   - Always-on cheap path: CLIP-fused classifier fallback + **Bedrock** (Claude Haiku-class) descriptions - instant responses.
   - On-demand GPU path: fine-tuned VLM on a **SageMaker async endpoint autoscaling to zero**; SQS decoupling; explicit button, never auto-triggered; honest "GPU warming up (~3-5 min)" UI status.
   - IaC: **CDK in Python**. CI/CD: GitHub Actions with OIDC (no stored AWS keys) to ECR/CDK deploy.
   - Observability: CloudWatch dashboards, billing alarms, health canary.
   - One-command teardown and redeploy by design.
8. **Region: ca-central-1** for the whole stack, with cross-region Bedrock calls to us-east-1 for any model missing locally (build-time check).
9. **Public-demo protection + gallery:** open access (no login) with API Gateway per-IP throttling + global daily quota, upload size caps, $2/day spend kill-switch (CloudWatch alarm -> pause Lambda), CloudFront free-tier WAF rules. **One-click example gallery** (6-8 curated images, 3-4 audio clips, pre-filled notes) so a visitor with no defect photos sees full value in one click. Gallery asset licensing must be verified before publishing (research-dataset redistribution terms).
10. **Presentation: README overhaul + one architecture diagram + 60-90s demo GIF.** Diagram shows the two-tier AWS design with three modality paths. README leads with demo URL + headline numbers + diagram; per-phase detail collapses below. External blog/LinkedIn writeup deferred past phase end. No custom domain.
11. **Lifecycle:** scheduled health canary (hits /health + one gallery analysis; emails on failure), quarterly dependency/Bedrock-model touch, teardown-when-dormant (demo only needs to be live while applications are out).

## Success metrics (measure-and-report; hard gates marked)

| Workstream | Metric |
|---|---|
| Photo + note | Delta on the crack/no_defect ambiguous subset, reported. **HARD GATE: empty/irrelevant note never reduces accuracy.** |
| Audio (MIMII fan + pump) | AUC per machine ID vs the published DCASE 2020 Task 2 AE baseline; target beat-it, report honestly either way. |
| Audio -> guidance retrieval | recall@5 for correct fault-family cards on abnormal clips; ~0.8 aspiration, label-lookup fallback below it. |
| Field-photo OOD study | Measure the gap on n>=100 (or publish preliminary at lower n); recover >=50% of the gap with one augmentation/fine-tune round; report both numbers. |
| Deployment | Live URL; warm CPU-path answer <=10s; cold GPU path <=5min with honest status; **HARD GATE: measured idle bill <=$15/mo**; one-command teardown + redeploy. |
| Corpus | +40 HVAC-maintenance cards; every audio fault class covered by >=10 cards. |

## Budget

- **$25 one-time hard cap** (training/experiments; ask-before-every-GPU-launch rule continues; audio training likely ~$0 - CPU/MPS).
- **$15/month recurring cap** with 50/80/100% billing alarms.
- No domain purchase.

## Build order

1. Field-photo collection starts immediately (passive, both kinds: 9-class eval photos + HVAC-visual seed).
2. Photo + note joint input (days; conditions VLM prompt and RAG query).
3. Audio mode (MIMII download -> embeddings + density scorer -> DCASE-comparable eval -> corpus cards -> UI panel).
4. AWS deployment (deploy last, once the app's final shape is known).
5. Field-photo OOD eval when ~100 photos accumulate (slots in independently).
6. Thermal scout: one time-boxed day, opportunistic.

Seam discipline: every session ends with main in a phase-complete-looking state (tests green, README truthful); in-progress work lives on feature branches.

## De-scope triggers (cut, don't sink - pre-agreed)

| Trigger | Action |
|---|---|
| MIMII AUC below baseline after one run + one honest iteration | Report as-is with analysis; move on |
| CLAP audio->card retrieval below ~0.8 | Label-lookup fallback; note in write-up |
| Field photos under ~100 at phase end | Publish preliminary OOD study; phase closes anyway |
| Thermal scout finds no usable dataset | Drop; mark evaluated-and-deferred |
| GPU cold start worse than ~10min or flaky | GPU path demoted to demo GIF; live demo ships CPU-only |
| Recurring bill trends past $15/mo | Cut costliest optional component (GPU path first, then canary frequency) |

## Out of scope

Video, sensor time-series, production hardening (auth, multi-tenant, SLA), model-weight publishing, SOTA chasing, taxonomy expansion to HVAC-visual classes (seeded for a future phase), external blog writeup (deferred), custom domain.

## Known risks

- MIMII is CC BY-SA 4.0 (attribution required); gallery redistribution rights for CODEBRIM/BD3/SDNET images must be checked before the public gallery ships - fallback is self-taken photos.
- Bedrock model availability in ca-central-1 (mitigated by cross-region call).
- SageMaker async cold-start variance (mitigated by de-scope trigger).
- Real-world audio differs from MIMII (accepted and framed honestly; decision #4).
