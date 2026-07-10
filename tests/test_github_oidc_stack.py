"""CDK-assertions tests for GitHubOidcStack (keyless CI roles).

The trust policy is the security boundary for keyless CI: it must pin both
roles to THIS repo's main branch (no wildcard subjects). Blast radius is split
across two roles (2026-07-09 security audit):

* ``defectlens-github-synth`` - every-push synth job: S3 read on the two
  model-artifact prefixes ONLY. No CDK bootstrap-role assumption (the app has
  no ``from_lookup``, so synth needs no lookup role either).
* ``defectlens-github-deploy`` - manual deploy job only: assume the CDK
  bootstrap roles (region+account-scoped) plus the same artifact reads (the
  deploy job syncs the artifacts before the image build).

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
REGION = "ca-central-1"
REPO = "keysmon/defect-lens"
ARTIFACTS_BUCKET = "defectlens-phase3-ca-002559670021"
SYNTH_ROLE = "defectlens-github-synth"
DEPLOY_ROLE = "defectlens-github-deploy"


@pytest.fixture(scope="module")
def template() -> Template:
    app = App()
    stack = GitHubOidcStack(
        app,
        "GitHubOidcStack",
        github_repo=REPO,
        env=Environment(account=ACCOUNT, region=REGION),
    )
    return Template.from_stack(stack)


def _role(template: Template, role_name: str) -> dict:
    roles = template.find_resources(
        "AWS::IAM::Role",
        {"Properties": {"RoleName": role_name}},
    )
    assert len(roles) == 1, f"expected exactly one role named {role_name}"
    return next(iter(roles.values()))


def _role_statements(template: Template, role_name: str) -> list[dict]:
    """All inline-policy statements attached to the named role."""
    role_ids = list(
        template.find_resources(
            "AWS::IAM::Role",
            {"Properties": {"RoleName": role_name}},
        )
    )
    assert len(role_ids) == 1
    role_id = role_ids[0]

    statements: list[dict] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        role_refs = [r.get("Ref") for r in policy["Properties"].get("Roles", [])]
        if role_id in role_refs:
            statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    assert statements, f"{role_name} has no inline policy statements"
    return statements


def _assert_artifact_read_only(statements: list[dict]) -> None:
    get_stmts = [s for s in statements if "s3:GetObject" in _as_list(s["Action"])]
    assert len(get_stmts) == 1
    assert set(_as_list(get_stmts[0]["Action"])) == {"s3:GetObject"}
    assert set(_as_list(get_stmts[0]["Resource"])) == {
        f"arn:aws:s3:::{ARTIFACTS_BUCKET}/phase5/cloud_artifacts/*",
        f"arn:aws:s3:::{ARTIFACTS_BUCKET}/phase5/audio_bank/*",
    }

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


@pytest.mark.parametrize("role_name", [SYNTH_ROLE, DEPLOY_ROLE])
def test_trust_policy_pinned_to_repo_main_branch(
    template: Template, role_name: str
) -> None:
    """sub must be an exact StringEquals on this repo's main ref - no wildcards."""
    role = _role(template, role_name)
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


def test_synth_role_cannot_assume_any_role(template: Template) -> None:
    """The every-push role must NOT hold the (admin-capable) bootstrap-assume."""
    statements = _role_statements(template, SYNTH_ROLE)
    for stmt in statements:
        assert "sts:AssumeRole" not in _as_list(stmt["Action"]), stmt
    # And nothing beyond the two artifact-read statements.
    assert len(statements) == 2
    _assert_artifact_read_only(statements)


def test_deploy_role_assumes_only_regional_bootstrap_roles(
    template: Template,
) -> None:
    statements = _role_statements(template, DEPLOY_ROLE)
    assume = [s for s in statements if "sts:AssumeRole" in _as_list(s["Action"])]
    assert len(assume) == 1
    assert set(_as_list(assume[0]["Action"])) == {"sts:AssumeRole"}
    assert _as_list(assume[0]["Resource"]) == [
        f"arn:aws:iam::{ACCOUNT}:role/cdk-hnb659fds-*-{ACCOUNT}-{REGION}"
    ]


def test_deploy_role_reads_model_artifact_prefixes(template: Template) -> None:
    """The deploy job syncs the artifacts before the image build."""
    _assert_artifact_read_only(_role_statements(template, DEPLOY_ROLE))


def test_deploy_role_session_can_span_the_qemu_build(template: Template) -> None:
    """role-duration-seconds: 7200 in the workflow needs MaxSessionDuration."""
    role = _role(template, DEPLOY_ROLE)
    assert role["Properties"]["MaxSessionDuration"] == 7200


@pytest.mark.parametrize("role_name", [SYNTH_ROLE, DEPLOY_ROLE])
def test_no_wildcard_grants(template: Template, role_name: str) -> None:
    for stmt in _role_statements(template, role_name):
        assert "*" not in _as_list(stmt["Action"]), stmt
        for resource in _as_list(stmt["Resource"]):
            assert resource != "*", stmt


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]
