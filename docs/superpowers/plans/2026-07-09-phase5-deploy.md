# Phase 5.5: AWS Deployment (CDK, two-tier scale-to-zero) - Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Sub-plans a/b/c land as separate milestones; b yields the live URL.

**Goal:** Live demo URL on AWS per spec decision 7: instant CPU path (CLIP-fused + CLAP audio + Bedrock descriptions), on-demand GPU path (fine-tuned VLM on SageMaker async, scale-to-zero), all in CDK-Python, ca-central-1, guarded ($2/day kill-switch, throttles, gallery).

**Verified facts (2026-07-09):** SageMaker `ml.g5.xlarge for endpoint usage` quota already 1.0 in ca-central-1 (L-1928E07B - no request needed). Bedrock ca-central-1 carries anthropic.claude-haiku-4-5 (use it; no cross-region). Budget: $15/mo recurring cap, alarms at 50/80/100%.

**Architecture decisions locked here:**
- **No database in the cloud.** 457 card vectors (410 visual x768 + 47 audio x512) become numpy .npz artifacts baked into the Lambda image at build time (exported from the local pgvector DB by a script). Brute-force cosine over 457 rows is microseconds - RDS deleted from the cost model. pgvector remains the LOCAL dev path; a VectorStore protocol with two impls (PgVectorStore, ArrayVectorStore) keeps one serving codebase.
- **One Lambda container image** (<= 10GB) holds: CLIP ViT-L/14, CLAP, card vectors, corpus YAML, audio bank + calibration, severity rules. DEFECTLENS_NO_VLM=1 mode; descriptions via Bedrock claude-haiku-4-5 (new BedrockDescriber implementing the Describer.describe contract; adapter OFF concept maps to "Bedrock never sees the fine-tune" - honest note in README).
- **GPU path**: SageMaker async endpoint, HF DLC container, inference.py loads Qwen base + adapter from the model artifact (packaged from S3 checkpoints), runs score_answers -> ranked classes; autoscaling min=0 max=1 on ApproximateBacklogSize; API publishes jobs via SQS-backed async invocation; frontend polls a /vlm-status endpoint with the honest "GPU warming up" banner.
- **Gallery**: 6 images (from test split - licensing checked: SDNET2018 is public-domain-ish/US-gov-adjacent, CODEBRIM research-only -> gallery uses BD3+SDNET images only + our own screenshots; verify per-source before shipping) + 3 MIMII clips (CC BY-NC-SA, attributed) + prefilled notes.
- **Protection**: API Gateway throttling (rate 5/s burst 10 per stage + usage-plan daily quota 500), Lambda reserved concurrency 5, $2/day CloudWatch billing alarm -> SNS -> kill Lambda (sets throttle to 0), WAF not needed at this budget (CloudFront + APIGW throttle suffice).
- **CI/CD**: GitHub Actions OIDC role (no stored keys), workflow: pytest -> cdk synth -> cdk deploy on main. Health canary: EventBridge 6h schedule -> Lambda hits /health + one gallery analyze -> SNS email on failure.

**Sub-plans:**
- **5.5a - Lambda-ready serving core (local, $0):** VectorStore protocol + ArrayVectorStore + export script (scripts/export_vector_artifacts.py); BedrockDescriber; lambda_handler (Mangum over the FastAPI app); Dockerfile.lambda; local container smoke test (docker run + curl). All TDD.
- **5.5b - CDK app + live URL (starts the ~$5-10/mo):** infra/ CDK-Python: FrontendStack (S3+CloudFront OAC), ApiStack (HTTP API + Lambda container + throttles + usage plan), OpsStack (billing alarm + kill-switch + canary + SNS email), GalleryStack assets; GitHub OIDC role + workflow. Milestone: URL serves the full CPU-path demo.
- **5.5c - GPU async path (adds ~$0 idle):** model packaging script (base+adapter -> model.tar.gz to S3), SageMaker async endpoint CDK + autoscale-to-zero, /analyze-vlm + /vlm-status endpoints, frontend GPU button + banner. First real request is the smoke test (ASK USER before the first cold-start bill - pennies but the rule stands).

**Money:** deploy-time ~$0; steady-state estimate: CloudFront+S3 ~<$1, APIGW+Lambda demo traffic ~<$1, Bedrock Haiku per-analysis ~$0.001, canary ~$0, SageMaker async idle $0 -> well under the $15/mo cap. HARD GATE: measured bill <= $15/mo.

**De-scope triggers (from spec):** GPU cold start > ~10min/flaky -> GPU path demoted to README GIF; bill trending past $15 -> cut GPU path first, then canary frequency.
