"""GitHubOidcStack - keyless CI deploy role (Phase 5.5b, authored not deployed).

Creates the GitHub Actions OIDC provider and a role that
``.github/workflows/deploy.yml`` assumes (no stored AWS keys). The trust policy
is scoped to the repo's ``main`` branch, and the role may only assume the CDK
bootstrap roles - CloudFormation does the privileged work, the CI role just
kicks off ``cdk deploy``.

This stack is intentionally left OUT of the milestone deploy (``cdk deploy``
targets Api/Frontend/Ops only). Deploy it later when turning CI on.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import aws_iam as iam
from constructs import Construct


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
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{github_repo}:ref:refs/heads/{branch}"
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

        CfnOutput(self, "DeployRoleArn", value=role.role_arn)
