"""GitHubOidcStack - keyless CI deploy role (Phase 5.5b, wired to CI in 5.6).

Creates the GitHub Actions OIDC provider (the account has none - verified via
``aws iam list-open-id-connect-providers`` on 2026-07-09) and the role that
``.github/workflows/deploy.yml`` assumes. No stored AWS keys anywhere.

Trust boundary (deliberate decisions):

* ``sub`` is an exact ``StringEquals`` on ``repo:<repo>:ref:refs/heads/main`` -
  no wildcards, no ``StringLike``. Only workflow runs on this repo's ``main``
  ref can assume the role; that covers both the ``push`` trigger and a
  ``workflow_dispatch`` run against ``main`` (same ``sub`` claim).
* Pull requests can NOT assume the role. The workflow has no ``pull_request``
  trigger, and PR tokens carry ``sub=repo:<repo>:pull_request`` which the
  trust policy rejects. If synth-on-PR is ever wanted, add that subject
  explicitly here rather than widening the existing one.

Permissions (least privilege for what CI actually does):

* ``sts:AssumeRole`` on the CDK bootstrap roles only - CloudFormation's exec
  role does the privileged work; the CI role just kicks off ``cdk deploy``.
* Read-only S3 on the two gitignored model-artifact prefixes that the ApiStack
  Lambda image COPYs (synced by CI before synth/deploy). Scoped to those
  prefixes, not the bucket, and never write.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import aws_iam as iam
from constructs import Construct

from stacks import _gpu_config as gpu

# The ApiStack image artifacts CI must sync before synth/deploy (mirrored by the
# `aws s3 sync` step in .github/workflows/deploy.yml - keep the two in step).
ARTIFACTS_BUCKET = gpu.BUCKET
CI_ARTIFACT_PREFIXES = ("phase5/cloud_artifacts/", "phase5/audio_bank/")


class GitHubOidcStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        github_repo: str,
        branch: str = "main",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        provider = iam.OpenIdConnectProvider(
            self,
            "GitHubOidcProvider",
            url="https://token.actions.githubusercontent.com",
            client_ids=["sts.amazonaws.com"],
        )

        role = iam.Role(
            self,
            "GitHubDeployRole",
            role_name="defectlens-github-deploy",
            description=f"CDK deploy role assumed by GitHub Actions ({github_repo}@{branch})",
            assumed_by=iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub": f"repo:{github_repo}:ref:refs/heads/{branch}",
                    },
                },
            ),
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[f"arn:aws:iam::{self.account}:role/cdk-hnb659fds-*"],
            )
        )

        # CI syncs the gitignored ApiStack model artifacts (card vectors + audio
        # bank) from S3 before synthing/deploying ApiStack. Read-only, prefix-
        # scoped: GetObject on the objects, ListBucket restricted to the same
        # prefixes (`aws s3 sync` needs the listing).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    f"arn:aws:s3:::{ARTIFACTS_BUCKET}/{prefix}*"
                    for prefix in CI_ARTIFACT_PREFIXES
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{ARTIFACTS_BUCKET}"],
                conditions={
                    "StringLike": {
                        "s3:prefix": [f"{prefix}*" for prefix in CI_ARTIFACT_PREFIXES]
                    }
                },
            )
        )

        CfnOutput(self, "DeployRoleArn", value=role.role_arn)
