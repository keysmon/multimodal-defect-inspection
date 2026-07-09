"""Cost kill-switch (Phase 5.5b). Stdlib + the runtime's bundled boto3 only.

Every 6h EventBridge invokes this. It reads the account's recent DAILY spend from
Cost Explorer; if any day crosses ``DAILY_LIMIT_USD`` it PATCHes the HTTP API's
stage throttle to 0 (rate+burst), stopping all traffic, and publishes to SNS.

HTTP APIs have no usage-plan/daily-quota, so this scheduled check is the daily
tripwire. Cost Explorer data lags ~a day, so this reacts next-day; the stage's
5/s throttle and the account Lambda concurrency cap bound cost in real time.
"""
from __future__ import annotations

import os
from datetime import date, timedelta


def max_daily_cost(results_by_time):
    """Return ``(date_str, amount)`` for the costliest day. Pure + testable.

    ``results_by_time`` is Cost Explorer's ``ResultsByTime`` list.
    """
    worst_date, worst_amount = None, 0.0
    for row in results_by_time:
        amount = float(row["Total"]["UnblendedCost"]["Amount"])
        if amount >= worst_amount:
            worst_amount = amount
            worst_date = row["TimePeriod"]["Start"]
    return worst_date, worst_amount


def _recent_daily_costs():
    import boto3  # bundled in the Lambda runtime; imported lazily so tests need no AWS

    # Cost Explorer's endpoint lives in us-east-1 regardless of the app region.
    ce = boto3.client("ce", region_name="us-east-1")
    today = date.today()
    resp = ce.get_cost_and_usage(
        TimePeriod={
            "Start": (today - timedelta(days=2)).isoformat(),
            "End": (today + timedelta(days=1)).isoformat(),  # End is exclusive
        },
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
    )
    return resp["ResultsByTime"]


def _throttle_stage_to_zero():
    import boto3

    boto3.client("apigatewayv2").update_stage(
        ApiId=os.environ["API_ID"],
        StageName=os.environ["STAGE_NAME"],
        DefaultRouteSettings={"ThrottlingRateLimit": 0, "ThrottlingBurstLimit": 0},
    )


def _notify(subject, message):
    import boto3

    boto3.client("sns").publish(
        TopicArn=os.environ["SNS_TOPIC_ARN"], Subject=subject[:100], Message=message
    )


def handler(event, context):
    limit = float(os.environ.get("DAILY_LIMIT_USD", "2"))
    worst_date, worst_amount = max_daily_cost(_recent_daily_costs())
    if worst_amount > limit:
        _throttle_stage_to_zero()
        message = (
            f"DefectLens cost guard TRIPPED: {worst_date} spend "
            f"${worst_amount:.2f} > ${limit:.2f}/day.\n"
            f"HTTP API {os.environ['API_ID']} stage {os.environ['STAGE_NAME']} "
            f"throttle set to 0 (traffic stopped). Investigate, then restore the "
            f"throttle (rate 5 / burst 10) to reopen the demo."
        )
        _notify("DefectLens cost guard TRIPPED", message)
        return {"tripped": True, "date": worst_date, "amount": worst_amount}
    return {"tripped": False, "date": worst_date, "amount": worst_amount}
