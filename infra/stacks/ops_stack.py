"""OpsStack - budget, alerts, health canary, cost kill-switch, dashboard (5.5b).

Protections, from cheap to hard:
  * AWS Budget ``defectlens-deploy-15`` ($15/mo) with 50/80/100% ACTUAL alerts
    emailed directly (Budgets->email needs no SNS topic policy).
  * SNS topic (email subscription) carrying canary + kill-switch alerts.
  * Health canary Lambda every 6h: hits CloudFront ``/api/health`` and one baked
    ``/api/analyze``; publishes to SNS on failure. Keys on HTTP 200 / presence of
    ``classes`` - NOT on a non-empty ``description`` (Bedrock returns "" until the
    account's Anthropic use-case form clears, and that is a healthy deploy).
  * Cost guard Lambda every 6h: reads Cost Explorer daily spend and, if it
    crosses ``daily_limit_usd`` ($5/day as wired in app.py), PATCHes the HTTP
    API stage throttle to 0 (kills traffic) and alerts.

Plus the ``defectlens-ops`` CloudWatch dashboard: a portfolio/demo artifact
that makes the scale-to-zero architecture visible (Lambda cold/warm latency,
API traffic, the SageMaker async 0<->1 instance drain, the Bedrock
quota-activation moment) - see ``_build_dashboard``.

Why Cost Explorer and not a usage plan: HTTP APIs (apigatewayv2) do not support
API-key usage plans / daily quotas - that is a REST-API (v1) feature. CE data lags
~a day, so this guard is a next-day tripwire; the real-time bound is the stage
throttle (5/s) plus the account-wide Lambda concurrency cap (10).
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    Duration,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from constructs import Construct

from stacks import _gpu_config as gpu

LAMBDAS_DIR = Path(__file__).resolve().parents[1] / "lambdas"


class OpsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        alert_email: str,
        distribution: cloudfront.Distribution,
        http_api: apigwv2.HttpApi,
        api_stage_name: str,
        serve_fn: lambda_.IFunction,
        daily_limit_usd: str = "5",
        monthly_budget_usd: int = 15,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        topic = sns.Topic(self, "OpsAlerts", display_name="DefectLens ops alerts")
        topic.add_subscription(subs.EmailSubscription(alert_email))

        budgets.CfnBudget(
            self,
            "DeployBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="defectlens-deploy-15",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_budget_usd, unit="USD"
                ),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL",
                        comparison_operator="GREATER_THAN",
                        threshold=threshold,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=alert_email
                        )
                    ],
                )
                for threshold in (50, 80, 100)
            ],
        )

        cf_base_url = f"https://{distribution.distribution_domain_name}"

        canary = lambda_.Function(
            self,
            "CanaryFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(LAMBDAS_DIR / "canary")),
            # Must exceed the worst-case retry budget (health retries + one analyze,
            # each capped near the API's 29s ceiling) so a down endpoint still
            # reaches _notify instead of the canary timing out mid-retry.
            timeout=Duration.seconds(300),
            memory_size=256,
            environment={
                "CF_BASE_URL": cf_base_url,
                "SNS_TOPIC_ARN": topic.topic_arn,
            },
        )
        topic.grant_publish(canary)
        events.Rule(
            self,
            "CanarySchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(canary)],
        )

        stage_arn = (
            f"arn:aws:apigateway:{self.region}::/apis/{http_api.api_id}"
            f"/stages/{api_stage_name}"
        )
        cost_guard = lambda_.Function(
            self,
            "CostGuardFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(LAMBDAS_DIR / "cost_guard")),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                "SNS_TOPIC_ARN": topic.topic_arn,
                "API_ID": http_api.api_id,
                "STAGE_NAME": api_stage_name,
                "DAILY_LIMIT_USD": daily_limit_usd,
            },
        )
        topic.grant_publish(cost_guard)
        cost_guard.add_to_role_policy(
            # Cost Explorer has no resource-level permissions.
            iam.PolicyStatement(actions=["ce:GetCostAndUsage"], resources=["*"])
        )
        cost_guard.add_to_role_policy(
            iam.PolicyStatement(
                actions=["apigateway:PATCH", "apigateway:GET"],
                resources=[stage_arn],
            )
        )
        events.Rule(
            self,
            "CostGuardSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),
            targets=[targets.LambdaFunction(cost_guard)],
        )

        # Real-time invocation-spike tripwire. The Cost Explorer guard above lags
        # ~1 day, so a denial-of-wallet burst (each async analysis also fans out
        # into many cheap poll invocations) could run unmetered for hours. This
        # alarm fires in MINUTES on an abnormal invocation volume - well above
        # normal demo + polling traffic, below the ~1500/5min ceiling the 5/s
        # stage throttle allows - and emails via SNS. It complements the throttle
        # (real-time request bound) and the CE guard (next-day $ bound); it
        # alerts rather than auto-throttles, since the throttle already caps the
        # actual spend rate and a legitimate demo burst shouldn't kill traffic.
        invocation_spike = cloudwatch.Alarm(
            self,
            "InvocationSpikeAlarm",
            alarm_name="defectlens-serve-invocation-spike",
            metric=serve_fn.metric_invocations(period=Duration.minutes(5), statistic="Sum"),
            threshold=1000,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description=(
                "serve Lambda invocations spiked in a 5-min window (possible "
                "abuse / denial-of-wallet); the Cost Explorer guard lags ~1 day, "
                "so this is the minutes-scale tripwire."
            ),
        )
        invocation_spike.add_alarm_action(cw_actions.SnsAction(topic))

        self._build_dashboard(
            serve_fn=serve_fn,
            http_api=http_api,
            api_stage_name=api_stage_name,
            daily_limit_usd=daily_limit_usd,
            monthly_budget_usd=monthly_budget_usd,
        )

    # ------------------------------------------------------------------ helpers

    def _build_dashboard(
        self,
        *,
        serve_fn: lambda_.IFunction,
        http_api: apigwv2.HttpApi,
        api_stage_name: str,
        daily_limit_usd: str,
        monthly_budget_usd: int,
    ) -> None:
        """The ``defectlens-ops`` dashboard - scale-to-zero made visible.

        Cross-stack wiring: Lambda + HTTP API come in as CDK objects (ApiStack
        exports, same pattern as the rest of the app). The SageMaker async
        endpoint is deliberately referenced by NAME from stacks._gpu_config -
        GpuStack must stay independently deployable (no CFN edge; the widgets
        simply show no data until it exists). Bedrock metrics are per-ModelId,
        so SEARCH expressions aggregate across models (the inference-profile
        routing means the ModelId dimension value is not fully predictable).
        """
        five_min = Duration.minutes(5)
        one_min = Duration.minutes(1)

        header = cloudwatch.TextWidget(
            markdown=(
                "# DefectLens - scale-to-zero ops\n"
                "CPU path: CloudFront -> HTTP API -> container Lambda "
                "(keep-warm every 5 min; cold load ~55 s). GPU path: SageMaker "
                f"async endpoint `{gpu.ENDPOINT_NAME}` scaling 0<->1 - watch the "
                "backlog wake it and the drain put it back to sleep."
            ),
            width=24,
            height=2,
        )

        lambda_duration = cloudwatch.GraphWidget(
            title="Serve Lambda - duration",
            left=[
                serve_fn.metric_duration(statistic="p50", period=five_min, label="p50"),
                serve_fn.metric_duration(statistic="p95", period=five_min, label="p95"),
            ],
            width=8,
            height=6,
        )
        lambda_traffic = cloudwatch.GraphWidget(
            title="Serve Lambda - invocations & failures",
            left=[
                serve_fn.metric_invocations(period=five_min),
                serve_fn.metric_errors(period=five_min),
                serve_fn.metric_throttles(period=five_min),
            ],
            width=8,
            height=6,
        )
        lambda_concurrency = cloudwatch.GraphWidget(
            title="Serve Lambda - concurrent executions",
            left=[
                serve_fn.metric(
                    "ConcurrentExecutions", statistic="Maximum", period=one_min
                )
            ],
            left_annotations=[
                cloudwatch.HorizontalAnnotation(
                    value=10, label="account concurrency cap"
                )
            ],
            width=8,
            height=6,
        )

        api_traffic = cloudwatch.GraphWidget(
            title=f"HTTP API - requests & errors (stage {api_stage_name})",
            left=[
                http_api.metric_count(period=five_min),
                http_api.metric_client_error(period=five_min),
                http_api.metric_server_error(period=five_min),
            ],
            width=12,
            height=6,
        )
        api_latency = cloudwatch.GraphWidget(
            title="HTTP API - latency p95 (ms)",
            left=[
                http_api.metric_latency(statistic="p95", period=five_min),
                http_api.metric_integration_latency(statistic="p95", period=five_min),
            ],
            width=12,
            height=6,
        )

        def sm_metric(name: str, statistic: str, *, per_variant: bool = False):
            dims = {"EndpointName": gpu.ENDPOINT_NAME}
            if per_variant:
                dims["VariantName"] = gpu.VARIANT_NAME
            return cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name=name,
                dimensions_map=dims,
                statistic=statistic,
                period=one_min,
            )

        gpu_backlog = cloudwatch.GraphWidget(
            title="VLM async - backlog & wake signal",
            left=[
                sm_metric("ApproximateBacklogSize", "Maximum"),
                sm_metric("ApproximateBacklogSizePerInstance", "Average"),
            ],
            right=[sm_metric("HasBacklogWithoutCapacity", "Maximum")],
            width=12,
            height=6,
        )
        gpu_instances = cloudwatch.GraphWidget(
            title="VLM async - invocations & running instances (0<->1 drain)",
            left=[sm_metric("Invocations", "Sum", per_variant=True)],
            right=[
                # No stock "instance count" metric exists for SageMaker
                # endpoints. Proxy: each running instance emits one
                # CPUUtilization datapoint per minute into the host-metrics
                # namespace, so SampleCount at 1-min period == instances
                # running; the line vanishing IS the scale-to-zero drain.
                cloudwatch.Metric(
                    namespace="/aws/sagemaker/Endpoints",
                    metric_name="CPUUtilization",
                    dimensions_map={
                        "EndpointName": gpu.ENDPOINT_NAME,
                        "VariantName": gpu.VARIANT_NAME,
                    },
                    statistic="SampleCount",
                    period=one_min,
                    label="running instances (CPU samples/min)",
                )
            ],
            width=12,
            height=6,
        )

        bedrock = cloudwatch.GraphWidget(
            title="Bedrock - Haiku description calls (account-wide)",
            left=[
                cloudwatch.MathExpression(
                    expression=(
                        "SEARCH('{AWS/Bedrock,ModelId} "
                        'MetricName="Invocations"\', \'Sum\', 300)'
                    ),
                    using_metrics={},
                    label="invocations",
                    period=five_min,
                )
            ],
            right=[
                cloudwatch.MathExpression(
                    expression=(
                        "SEARCH('{AWS/Bedrock,ModelId} "
                        'MetricName="InvocationThrottles"\', \'Sum\', 300)'
                    ),
                    using_metrics={},
                    label="throttles",
                    period=five_min,
                )
            ],
            width=12,
            height=6,
        )

        cost_notes = cloudwatch.TextWidget(
            markdown=(
                "### Cost guardrails (Cost Explorer / Budgets APIs - no CW metrics)\n"
                f"- **${daily_limit_usd}/day kill-switch**: a 6h Lambda reads "
                "Cost Explorer daily spend; over the cap it PATCHes the API "
                "stage throttle to 0 and emails via SNS. CE data lags ~1 day - "
                "the real-time bound is the stage throttle (5/s) + the account "
                "Lambda concurrency cap (10).\n"
                f"- **Budget `defectlens-deploy-15`**: ${monthly_budget_usd}/mo "
                "with 50/80/100% ACTUAL email alerts.\n"
                "- **Canary**: every 6h hits CloudFront /api/health + one baked "
                "/api/analyze; SNS email on failure."
            ),
            width=12,
            height=6,
        )

        cloudwatch.Dashboard(
            self,
            "OpsDashboard",
            dashboard_name="defectlens-ops",
            widgets=[
                [header],
                [lambda_duration, lambda_traffic, lambda_concurrency],
                [api_traffic, api_latency],
                [gpu_backlog, gpu_instances],
                [bedrock, cost_notes],
            ],
        )
