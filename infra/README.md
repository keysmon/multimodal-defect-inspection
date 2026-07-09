# DefectLens deployment (CDK, Phase 5.5b)

Standalone CDK v2 (Python) app that deploys the instant CPU-path demo to AWS:
account `002559670021`, region `ca-central-1`, profile `defectlens`.

## Stacks

| Stack | What it creates |
|-------|-----------------|
| `ApiStack` | Container Lambda (from `deploy/Dockerfile.lambda`, arm64, 6144 MB, 120 s) + HTTP API with a named `api` stage (throttle 5/s, burst 10). Bedrock (Haiku) IAM. |
| `FrontendStack` | Private S3 (OAC) + CloudFront. Default behaviour serves the React build; `/api/*` routes to the `api` stage. |
| `OpsStack` | SNS + email, `defectlens-deploy-15` Budget ($15/mo, 50/80/100%), 6h health canary, 6h Cost-Explorer `$2/day` kill-switch. |
| `GitHubOidcStack` | Keyless CI deploy role. **Authored, not deployed** by the demo milestone. |

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
