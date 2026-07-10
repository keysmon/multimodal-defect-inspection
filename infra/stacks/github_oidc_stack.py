"""GitHubOidcStack - keyless CI roles (Phase 5.5b, wired to CI in 5.6).

Creates the GitHub Actions OIDC provider (the account has none - verified via
``aws iam list-open-id-connect-providers`` on 2026-07-09) and the roles that
``.github/workflows/deploy.yml`` assumes. No stored AWS keys anywhere.

Trust boundary (deliberate decisions):

* ``sub`` is an exact ``StringEquals`` on ``repo:<repo>:ref:refs/heads/main`` -
  no wildcards, no ``StringLike``. Only workflow runs on this repo's ``main``
  ref can assume either role; that covers both the ``push`` trigger and a
  ``workflow_dispatch`` run against ``main`` (same ``sub`` claim).
* Pull requests can NOT assume the roles. The workflow has no ``pull_request``
  trigger, and PR tokens carry ``sub=repo:<repo>:pull_request`` which the
  trust policy rejects. If synth-on-PR is ever wanted, add that subject
  explicitly (to the synth role only) rather than widening the existing one.

Blast-radius split (2026-07-09 security audit): assuming the CDK bootstrap
roles is effectively account admin via the cfn-exec role, so the every-push
job must not hold it. Two roles, same trust:

* ``defectlens-github-synth`` (synth-api job, every push): read-only S3 on the
  two gitignored model-artifact prefixes the ApiStack image COPYs - nothing
  else. No bootstrap-role assumption; no lookup-role assumption either, since
  the app contains no ``from_lookup`` (add the lookup-role assume here only if
  a stack ever gains one).
* ``defectlens-github-deploy`` (deploy job, manual dispatch only):
  ``sts:AssumeRole`` on the CDK bootstrap roles - scoped to THIS account and
  region (they are named ``cdk-hnb659fds-<purpose>-<account>-<region>``) -
  plus the same artifact reads (the deploy job syncs the artifacts before the
  image build). MaxSessionDuration is 2h so the workflow can request
  ``role-duration-seconds: 7200`` to outlast the QEMU-emulated image build.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
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

        def github_principal() -> iam.WebIdentityPrincipal:
            return iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub": f"repo:{github_repo}:ref:refs/heads/{branch}",
                    },
                },
            )

        # CI syncs the gitignored ApiStack model artifacts (card vectors + audio
        # bank) from S3 before synthing/deploying ApiStack. Read-only, prefix-
        # scoped: GetObject on the objects, ListBucket restricted to the same
        # prefixes (`aws s3 sync` needs the listing).
        def artifact_read_statements() -> list[iam.PolicyStatement]:
            return [
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[
                        f"arn:aws:s3:::{ARTIFACTS_BUCKET}/{prefix}*"
                        for prefix in CI_ARTIFACT_PREFIXES
                    ],
                ),
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[f"arn:aws:s3:::{ARTIFACTS_BUCKET}"],
                    conditions={
                        "StringLike": {
                            "s3:prefix": [
                                f"{prefix}*" for prefix in CI_ARTIFACT_PREFIXES
                            ]
                        }
                    },
                ),
            ]

        synth_role = iam.Role(
            self,
            "GitHubSynthRole",
            role_name="defectlens-github-synth",
            description=f"Artifact-read-only synth role assumed by GitHub Actions ({github_repo}@{branch})",
            assumed_by=github_principal(),
        )
        for statement in artifact_read_statements():
            synth_role.add_to_policy(statement)

        deploy_role = iam.Role(
            self,
            "GitHubDeployRole",
            role_name="defectlens-github-deploy",
            description=f"CDK deploy role assumed by GitHub Actions ({github_repo}@{branch})",
            assumed_by=github_principal(),
            # The workflow requests role-duration-seconds: 7200 to span the
            # QEMU-emulated arm64 image build (IAM default max is 1h).
            max_session_duration=Duration.hours(2),
        )
        deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[
                    f"arn:aws:iam::{self.account}:role/cdk-hnb659fds-*-{self.account}-{self.region}"
                ],
            )
        )
        for statement in artifact_read_statements():
            deploy_role.add_to_policy(statement)

        CfnOutput(self, "SynthRoleArn", value=synth_role.role_arn)
        CfnOutput(self, "DeployRoleArn", value=deploy_role.role_arn)
