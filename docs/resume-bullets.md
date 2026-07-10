# DefectLens — resume bullets (measured, Phase 5 close-out 2026-07-09)

Every number below is committed in `results/` or the README and reproducible from
the repo. Pick 3–4 per application; the deployment bullet plus the fine-tuning
bullet carry the most signal for AI-engineer/MLE roles, the deployment plus
full-stack bullets for SDE/full-stack roles.

## Core bullets

> Fine-tuned a vision-language model (Qwen2.5-VL-3B, QLoRA 4-bit) on 17,652
> building-defect images unified from three public datasets, lifting macro top-1
> accuracy from 0.472 (CLIP zero-shot baseline) to 0.851 (top-3 0.990) on a
> frozen 2,648-image test split — total GPU cost ~$3 on AWS spot instances with
> checkpoint auto-resume.

> Grounded the classifier with cross-modal RAG over a 252-card cited
> inspection-standards corpus (EPA/HUD/InterNACHI/FHWA/DOE) in shared CLIP/CLAP
> embedding spaces — pgvector locally, a dependency-free vector store in the
> cloud — reaching image-query recall@5 of 0.863 via reciprocal-rank fusion.

> Built and evaluated four input modalities with pre-registered protocols:
> photo, inspector notes (classification note-invariant at 0.900 — doubling as
> prompt-injection robustness — while retrieval is note-responsive), equipment
> audio (unsupervised CLAP-embedding anomaly scoring, pump AUC 0.801 vs the
> 0.726 DCASE 2020 baseline), and thermal imaging (a seed-replicated,
> confound-controlled negative result: a 3-seed CUDA replication for ~$1 of
> spot compute proved the RGB+IR fusion deficit was a stem-initialization
> artifact - hybrid-stem surgery recovered full parity with RGB,
> 0.472+-0.004 vs 0.463+-0.008 mean defect IoU).

> Deployed a two-tier scale-to-zero AWS stack with CDK (Python) and GitHub
> Actions: CloudFront + private S3 for the React SPA, API Gateway fronting a
> 7.8 GB Lambda container for instant CPU inference, and a SageMaker async
> endpoint (autoscaling 0↔1) serving the fine-tuned VLM on GPU — idle cost
> ~$2–3/month versus hundreds/month for an always-on GPU endpoint, guarded by
> budget alarms and an automated daily cost cutoff.

> Validated cross-dataset generalization on independently sourced
> out-of-distribution images: macro top-1 held at 0.877, with the error profile
> rotating toward false positives — the safer failure direction for inspection.

> Operated the stack like production: keyless GitHub Actions-to-AWS auth via
> OIDC with a least-privilege two-role split (read-only synth verification on
> every push, deploys behind manual dispatch and SHA-pinned actions, no stored
> AWS keys), plus a CloudWatch operations dashboard making scale-to-zero
> behavior observable (Lambda p95/errors, API 4xx/5xx, SageMaker 0-to-1
> instance drain, Bedrock throttles, cost guardrails).

## Supporting talking points (interview depth)

- **Methodology over benchmarks:** frozen splits committed as manifests with
  loud-fail guards; single prompt source shared between training and eval;
  length-normalized answer log-likelihood scoring; negative results reported
  as measured (thermal, fan-audio) with de-scope triggers honored.
- **Cost engineering:** Phase 3 fine-tune landed at ~$3.10 of a $10 cap (spot,
  smoke-run gate before full runs); demo runs at ~$2–3/month idle; a
  Cost-Explorer Lambda warns at half its daily limit and throttles at the cap.
- **Production debugging on AWS:** root-caused a 5× latency regression to
  boto3's default retry policy colliding with a zero-quota Bedrock account
  state (fail-fast client config, `total_max_attempts=1`); root-caused a
  CloudFormation race between SageMaker model validation and IAM policy
  attachment; two SageMaker inference toolkits have opposite `output_fn`
  contracts — caught pre-billing by desk review; diagnosed a CI keyless-auth
  failure as infrastructure drift (the deployed IAM stack predated the
  reviewed hardening commit) and added a post-deploy `cdk diff` no-drift
  gate.
- **Apple-Silicon ML:** trained locally on MPS where it was cheaper than GPU
  (SegFormer comparison, ~20 min/run); root-caused an MPS BatchNorm2d backward
  crash on non-contiguous inputs and shipped a numerically-neutral fix locked
  by a regression test.
- **Live demo:** https://d2wxjiu5re5mow.cloudfront.net (gallery → severity band,
  ranked classes, cited remediation guidance; GPU path on demand).
