"""OpsStack - budget, alerts, health canary, and cost kill-switch (Phase 5.5b).

Protections, from cheap to hard:
  * AWS Budget ``defectlens-deploy-15`` ($15/mo) with 50/80/100% ACTUAL alerts
    emailed directly (Budgets->email needs no SNS topic policy).
  * SNS topic (email subscription) carrying canary + kill-switch alerts.
  * Health canary Lambda every 6h: hits CloudFront ``/api/health`` and one baked
    ``/api/analyze``; publishes to SNS on failure. Keys on HTTP 200 / presence of
    ``classes`` - NOT on a non-empty ``description`` (Bedrock returns "" until the
    account's Anthropic use-case form clears, and that is a healthy deploy).
  * Cost guard Lambda every 6h: reads Cost Explorer daily spend and, if it crosses
    $2/day, PATCHes the HTTP API stage throttle to 0 (kills traffic) and alerts.

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
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from constructs import Construct

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
