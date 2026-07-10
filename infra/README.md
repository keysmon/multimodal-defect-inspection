# Deployment (CDK, Phase 5.5b)

Standalone CDK v2 (Python) app that deploys the instant CPU-path demo to AWS:
account `002559670021`, region `ca-central-1`, profile `defectlens`.

## Stacks

| Stack | What it creates |
|-------|-----------------|
| `ApiStack` | Container Lambda (from `deploy/Dockerfile.lambda`, arm64, 3008 MB - account quota, 120 s) + HTTP API with a named `api` stage (throttle 5/s, burst 10). Bedrock (Haiku) IAM. |
| `FrontendStack` | Private S3 (OAC) + CloudFront. Default behaviour serves the React build; `/api/*` routes to the `api` stage. |
| `OpsStack` | SNS + email, `defectlens-deploy-15` Budget ($15/mo, 50/80/100%), 6h health canary, 6h Cost-Explorer `$5/day` kill-switch, `defectlens-ops` CloudWatch dashboard (Lambda/API/SageMaker-async/Bedrock - the scale-to-zero demo view). |
| `GitHubOidcStack` | GitHub OIDC provider + two keyless CI roles (synth: artifact-read-only; deploy: CDK bootstrap-assume), trust-pinned to `keysmon/multimodal-defect-inspection@main`. Deploy once to activate CI's AWS jobs (see below). |

## Single-origin routing (no CORS)

CloudFront is the only origin the browser sees. The React build is made with
`REACT_APP_API_URL=/api`, so it calls `/api/analyze` and `/api/search`
same-origin. The HTTP API uses a **named** stage `api`, whose name is part of the
invoke path, so `/api/analyze` at CloudFront reaches stage `api` route `/analyze`
with no rewriting. `ALL_VIEWER_EXCEPT_HOST_HEADER` on the `/api/*` behaviour keeps
API Gateway's Host check happy.

## Deploy

```bash
aws sts get-caller-identity --profile defectlens          # expect account 002559670021
cd frontend && REACT_APP_API_URL=/api npm run build && cd ..
# venv python must be on PATH so cdk.json's "python app.py" finds aws-cdk-lib:
export PATH="$PWD/.venv/bin:$PATH"
cd infra
npx aws-cdk@2 bootstrap aws://002559670021/ca-central-1 --profile defectlens   # first time only
npx aws-cdk@2 synth
npx aws-cdk@2 deploy ApiStack FrontendStack OpsStack --require-approval never --profile defectlens
```

The container image (~8 GB) is built and pushed to ECR inside `cdk deploy ApiStack`;
budget for a slow first push.

## CI (GitHub Actions, keyless)

`.github/workflows/deploy.yml` runs pytest + `cdk synth` on every push to `main` and keeps `cdk deploy` behind a manual `workflow_dispatch`.
AWS access is keyless: CI assumes a `GitHubOidcStack` role via GitHub's OIDC provider, trust-pinned to `repo:keysmon/multimodal-defect-inspection:ref:refs/heads/main`.
Blast radius is split across two roles: the every-push `synth-api` job assumes `defectlens-github-synth` (prefix-scoped S3 read ONLY), while `defectlens-github-deploy` (CDK bootstrap-role assume, region+account-scoped, 2h max session) is reserved for the manual deploy job.
No stored keys anywhere; workflow actions are pinned to full commit SHAs.

Activate once:

```bash
cd infra
npx aws-cdk@2 deploy GitHubOidcStack --profile defectlens
gh variable set AWS_OIDC_SYNTH_ROLE_ARN --body arn:aws:iam::002559670021:role/defectlens-github-synth
gh variable set AWS_OIDC_ROLE_ARN --body arn:aws:iam::002559670021:role/defectlens-github-deploy
```

Until the variables are set, the AWS-touching jobs are skipped (CI stays green) and a manual deploy dispatch fails with instructions.
ApiStack's gitignored image artifacts are synced by CI from `s3://defectlens-phase3-ca-002559670021/phase5/{cloud_artifacts,audio_bank}/`; publish/refresh them from a dev machine:

```bash
aws s3 cp models/cloud_artifacts/card_vectors.npz \
  s3://defectlens-phase3-ca-002559670021/phase5/cloud_artifacts/card_vectors.npz --profile defectlens
aws s3 sync models/audio_bank/ \
  s3://defectlens-phase3-ca-002559670021/phase5/audio_bank/ --profile defectlens
```

The CI `deploy` job builds the arm64 Lambda image under QEMU emulation on GitHub's x86 runners - budget an hour+ for a cold image build, or deploy `ApiStack` from an Apple Silicon machine and let CI deploy the rest.

## Known caveats (measured / documented)

- **Reserved concurrency is not set.** This account's Lambda concurrency limit is
  10 with a required unreserved floor of 10, so any positive reservation is
  rejected. The account cap (10) + the stage throttle (5/s) are the bound. Raise
  the Lambda concurrency quota to restore a per-function reservation.
- **Cold start vs. the 30 s API ceiling.** The serving Lambda scales to zero and
  its image is large; a cold request loads CLIP + CLAP before responding. HTTP
  API caps integration time at 30 s, so a cold `/api/analyze` can time out (504)
  where a warm one is a few seconds. The canary retries health to absorb this.
- **Descriptions are empty until Bedrock access clears.** The account's Anthropic
  use-case form is pending; `BedrockDescriber` returns `""` on `AccessDenied`, so
  `/analyze` still returns classes + cards. The canary keys on `classes`, not on
  a non-empty `description`.
