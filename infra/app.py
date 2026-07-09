#!/usr/bin/env python3
"""DefectLens deployment CDK app (Phase 5.5b).

Three deploy stacks (all ca-central-1, account 002559670021):

  ApiStack       HTTP API -> Lambda container (CPU serving image), stage `api`
                 with rate/burst throttling; Bedrock (Haiku) IAM.
  FrontendStack  private S3 (OAC) + CloudFront serving the React build, with an
                 `/api/*` behaviour routed to the HTTP API's `api` stage so the
                 SPA talks to a single origin (no CORS).
  OpsStack       SNS + email, a $15/mo Budget, a 6h health canary, and a daily
                 Cost-Explorer guard that throttles the API stage to zero if
                 spend crosses $2/day.

GitHubOidcStack is authored here but NOT deployed by the milestone (synth-only):
the demo deploy creates only Api/Frontend/Ops. Deploy the OIDC stack later when
wiring CI.
"""
from __future__ import annotations

import aws_cdk as cdk

from stacks.api_stack import ApiStack
from stacks.frontend_stack import FrontendStack
from stacks.github_oidc_stack import GitHubOidcStack
from stacks.gpu_stack import GpuStack
from stacks.ops_stack import OpsStack

# Pinned, NOT read from CDK_DEFAULT_ACCOUNT: this app is the defectlens account's
# demo. Pinning both bakes the right account into IAM ARNs regardless of which
# profile runs cdk, and makes CDK refuse a deploy with mismatched credentials.
ACCOUNT = "002559670021"
REGION = "ca-central-1"
ALERT_EMAIL = "hang@homewiseai.org"
GITHUB_REPO = "keysmon/defect-lens"

env = cdk.Environment(account=ACCOUNT, region=REGION)

app = cdk.App()

# GPU async path is off by default (CPU-only demo -> /analyze-vlm returns 503).
# Enable by redeploying ApiStack with `-c gpu_endpoint_name=defectlens-vlm-async`
# AFTER GpuStack is deployed; that sets SAGEMAKER_ENDPOINT + the invoke/S3 IAM.
gpu_endpoint_name = app.node.try_get_context("gpu_endpoint_name")

api = ApiStack(app, "ApiStack", env=env, gpu_endpoint_name=gpu_endpoint_name)

frontend = FrontendStack(
    app,
    "FrontendStack",
    env=env,
    http_api=api.http_api,
)

OpsStack(
    app,
    "OpsStack",
    env=env,
    alert_email=ALERT_EMAIL,
    distribution=frontend.distribution,
    http_api=api.http_api,
    api_stage_name=api.stage_name,
    daily_limit_usd="5",
    monthly_budget_usd=15,
)

# Authored for CI wiring; intentionally not part of the demo deploy.
GitHubOidcStack(app, "GitHubOidcStack", env=env, github_repo=GITHUB_REPO)

# Phase 5.5c GPU async path. Authored + synth-validated, but NOT auto-deployed:
# deploy explicitly (`cdk deploy GpuStack`) after the user signs off on the first
# GPU bill. No cross-stack ref to Api/Frontend/Ops, so it deploys standalone.
GpuStack(app, "GpuStack", env=env)

app.synth()
