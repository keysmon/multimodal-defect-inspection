"""CDK-assertions tests for the OpsStack CloudWatch dashboard.

The dashboard is the demo artifact that makes the scale-to-zero architecture
visible, so the tests lock the load-bearing names (namespaces, metric names,
endpoint/stage identifiers) without over-locking the widget JSON layout.

aws-cdk-lib is an infra dependency (infra/requirements.txt), not a package
dependency, so skip cleanly where it is absent (mirrors test_github_oidc_stack).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk")

from aws_cdk import App, Environment, Stack  # noqa: E402
from aws_cdk import aws_apigatewayv2 as apigwv2  # noqa: E402
from aws_cdk import aws_cloudfront as cloudfront  # noqa: E402
from aws_cdk import aws_cloudfront_origins as origins  # noqa: E402
from aws_cdk import aws_lambda as lambda_  # noqa: E402
from aws_cdk.assertions import Template  # noqa: E402

INFRA = Path(__file__).resolve().parents[1] / "infra"
if str(INFRA) not in sys.path:
    sys.path.insert(0, str(INFRA))

from stacks.ops_stack import OpsStack  # noqa: E402

ENV = Environment(account="002559670021", region="ca-central-1")


@pytest.fixture(scope="module")
def template() -> Template:
    """OpsStack synthed against lightweight stand-ins for the ApiStack /
    FrontendStack resources it consumes (no Docker/file assets needed)."""
    app = App()
    support = Stack(app, "Support", env=ENV)
    http_api = apigwv2.HttpApi(support, "Api", create_default_stage=False)
    distribution = cloudfront.Distribution(
        support,
        "Dist",
        default_behavior=cloudfront.BehaviorOptions(
            origin=origins.HttpOrigin("example.com")
        ),
    )
    serve_fn = lambda_.Function(
        support,
        "ServeFn",
        runtime=lambda_.Runtime.PYTHON_3_12,
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context):\n    pass\n"),
    )
    ops = OpsStack(
        app,
        "OpsStack",
        env=ENV,
        alert_email="ops@example.com",
        distribution=distribution,
        http_api=http_api,
        api_stage_name="api",
        serve_fn=serve_fn,
        daily_limit_usd="5",
        monthly_budget_usd=15,
    )
    return Template.from_stack(ops)


def _dashboard_body(template: Template) -> str:
    """Flatten the DashboardBody (an Fn::Join of literals and refs) to text."""
    dashboards = template.find_resources("AWS::CloudWatch::Dashboard")
    assert len(dashboards) == 1
    body = next(iter(dashboards.values()))["Properties"]["DashboardBody"]
    if isinstance(body, str):
        return body
    parts = body["Fn::Join"][1]
    return "".join(p if isinstance(p, str) else "<ref>" for p in parts)


def test_dashboard_exists_with_stable_name(template: Template) -> None:
    template.resource_count_is("AWS::CloudWatch::Dashboard", 1)
    dashboards = template.find_resources("AWS::CloudWatch::Dashboard")
    props = next(iter(dashboards.values()))["Properties"]
    assert props["DashboardName"] == "defectlens-ops"


def test_lambda_widgets(template: Template) -> None:
    body = _dashboard_body(template)
    assert "AWS/Lambda" in body
    for name in ("Duration", "Invocations", "Errors", "Throttles", "ConcurrentExecutions"):
        assert name in body, f"missing Lambda metric {name}"
    for stat in ("p50", "p95"):
        assert stat in body, f"missing duration statistic {stat}"


def test_api_gateway_widgets(template: Template) -> None:
    body = _dashboard_body(template)
    assert "AWS/ApiGateway" in body
    for name in ("Count", "4xx", "5xx", "IntegrationLatency"):
        assert name in body, f"missing API Gateway metric {name}"


def test_sagemaker_scale_to_zero_widgets(template: Template) -> None:
    body = _dashboard_body(template)
    assert "AWS/SageMaker" in body
    # The endpoint is referenced by NAME (GpuStack stays independently
    # deployable - no CFN cross-stack edge, matching stacks/_gpu_config).
    assert "defectlens-vlm-async" in body
    for name in (
        "ApproximateBacklogSize",
        "ApproximateBacklogSizePerInstance",
        "HasBacklogWithoutCapacity",
        "Invocations",
    ):
        assert name in body, f"missing SageMaker metric {name}"
    # Running-instance proxy: per-instance host metrics namespace.
    assert "/aws/sagemaker/Endpoints" in body


def test_bedrock_widgets(template: Template) -> None:
    body = _dashboard_body(template)
    # Account-level Bedrock metrics are per-ModelId; the dashboard aggregates
    # across models with SEARCH expressions.
    assert "AWS/Bedrock" in body
    assert "InvocationThrottles" in body


def test_invocation_spike_alarm_wired_to_sns(template: Template) -> None:
    """F3: a minutes-scale invocation-spike alarm (the CE cost guard lags ~1 day)
    on the serve Lambda's Invocations, wired to the ops SNS topic."""
    alarms = template.find_resources("AWS::CloudWatch::Alarm")
    spike = [
        a
        for a in alarms.values()
        if a["Properties"].get("AlarmName") == "defectlens-serve-invocation-spike"
    ]
    assert len(spike) == 1
    props = spike[0]["Properties"]
    assert props["MetricName"] == "Invocations"
    assert props["Namespace"] == "AWS/Lambda"
    assert props["Threshold"] == 1000
    assert props["ComparisonOperator"] == "GreaterThanThreshold"
    assert props.get("AlarmActions"), "alarm must notify the SNS topic"


def test_cost_guard_text_widget(template: Template) -> None:
    body = _dashboard_body(template)
    # The guards are CE/Budgets API driven, not CW metrics - documented on the
    # dashboard itself, with the actual configured limits.
    assert "Cost Explorer" in body
    assert "$5/day" in body
    assert "defectlens-deploy-15" in body
