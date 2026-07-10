"""CDK-assertions tests for GitHubOidcStack (keyless CI role).

The trust policy is the security boundary for keyless CI: it must pin the role
to THIS repo's main branch (no wildcard subjects) and grant only what CI needs
(assume the CDK bootstrap roles + read the model-artifact S3 prefixes).

aws-cdk-lib is an infra dependency (infra/requirements.txt), not a package
dependency, so skip cleanly where it is absent (mirrors the boto3 skip in
test_bedrock_describer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk")

from aws_cdk import App, Environment  # noqa: E402
from aws_cdk.assertions import Match, Template  # noqa: E402

INFRA = Path(__file__).resolve().parents[1] / "infra"
if str(INFRA) not in sys.path:
    sys.path.insert(0, str(INFRA))

from stacks.github_oidc_stack import GitHubOidcStack  # noqa: E402

ACCOUNT = "002559670021"
REPO = "keysmon/defect-lens"
ARTIFACTS_BUCKET = "defectlens-phase3-ca-002559670021"


@pytest.fixture(scope="module")
def template() -> Template:
    app = App()
    stack = GitHubOidcStack(
        app,
        "GitHubOidcStack",
        github_repo=REPO,
        env=Environment(account=ACCOUNT, region="ca-central-1"),
    )
    return Template.from_stack(stack)


def _deploy_role(template: Template) -> dict:
    roles = template.find_resources(
        "AWS::IAM::Role",
        {"Properties": {"RoleName": "defectlens-github-deploy"}},
    )
    assert len(roles) == 1
    return next(iter(roles.values()))


def _deploy_role_statements(template: Template) -> list[dict]:
    """All inline-policy statements attached to the deploy role."""
    role_ids = list(
        template.find_resources(
            "AWS::IAM::Role",
            {"Properties": {"RoleName": "defectlens-github-deploy"}},
        )
    )
    assert len(role_ids) == 1
    role_id = role_ids[0]

    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        role_refs = [r.get("Ref") for r in policy["Properties"].get("Roles", [])]
        if role_id in role_refs:
            statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    assert statements, "deploy role has no inline policy statements"
    return statements


def test_oidc_provider_is_github(template: Template) -> None:
    template.resource_count_is("Custom::AWSCDKOpenIdConnectProvider", 1)
    template.has_resource_properties(
        "Custom::AWSCDKOpenIdConnectProvider",
        Match.object_like(
            {
                "Url": "https://token.actions.githubusercontent.com",
                "ClientIDList": ["sts.amazonaws.com"],
            }
        ),
    )


def test_trust_policy_pinned_to_repo_main_branch(template: Template) -> None:
    """sub must be an exact StringEquals on this repo's main ref - no wildcards."""
    role = _deploy_role(template)
    statements = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert len(statements) == 1
    stmt = statements[0]

    assert stmt["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert "Federated" in stmt["Principal"]

    conditions = stmt["Condition"]
    equals = conditions["StringEquals"]
    assert equals["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    assert (
        equals["token.actions.githubusercontent.com:sub"]
        == f"repo:{REPO}:ref:refs/heads/main"
    )
    # No pattern-matching escape hatch: an exact-match-only trust policy.
    assert "StringLike" not in conditions


def test_role_may_only_assume_cdk_bootstrap_roles(template: Template) -> None:
    statements = _deploy_role_statements(template)
    assume = [s for s in statements if "sts:AssumeRole" in _as_list(s["Action"])]
    assert len(assume) == 1
    resources = _as_list(assume[0]["Resource"])
    assert resources == [f"arn:aws:iam::{ACCOUNT}:role/cdk-hnb659fds-*"]


def test_role_reads_model_artifact_prefixes_only(template: Template) -> None:
    """CI syncs the gitignored ApiStack model artifacts from S3 - read-only,
    scoped to the two phase5 prefixes, never bucket-wide or account-wide."""
    statements = _deploy_role_statements(template)

    get_stmts = [s for s in statements if "s3:GetObject" in _as_list(s["Action"])]
    assert len(get_stmts) == 1
    assert set(_as_list(get_stmts[0]["Resource"])) == {
        f"arn:aws:s3:::{ARTIFACTS_BUCKET}/phase5/cloud_artifacts/*",
        f"arn:aws:s3:::{ARTIFACTS_BUCKET}/phase5/audio_bank/*",
    }
    assert set(_as_list(get_stmts[0]["Action"])) == {"s3:GetObject"}

    list_stmts = [s for s in statements if "s3:ListBucket" in _as_list(s["Action"])]
    assert len(list_stmts) == 1
    assert _as_list(list_stmts[0]["Resource"]) == [
        f"arn:aws:s3:::{ARTIFACTS_BUCKET}"
    ]
    prefixes = list_stmts[0]["Condition"]["StringLike"]["s3:prefix"]
    assert set(_as_list(prefixes)) == {
        "phase5/cloud_artifacts/*",
        "phase5/audio_bank/*",
    }


def test_no_wildcard_grants_on_deploy_role(template: Template) -> None:
    for stmt in _deploy_role_statements(template):
        assert "*" not in _as_list(stmt["Action"]), stmt
        for resource in _as_list(stmt["Resource"]):
            assert resource != "*", stmt


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]
