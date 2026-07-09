"""ApiStack - HTTP API in front of the CPU serving Lambda (Phase 5.5b).

The Lambda is the container image built from ``deploy/Dockerfile.lambda`` (arm64;
the base is public.ecr.aws/lambda/python:3.12 and Lambda runs it on Graviton).
It runs the no-VLM CPU path: CLIP-fused classifier + CLAP audio + baked card
vectors, with descriptions from Claude Haiku on Bedrock.

Routing shape (single-origin, see FrontendStack): the API has NO ``$default``
stage. A named stage ``api`` puts its name into the invoke path, so a request to
``/api/analyze`` at CloudFront reaches this API's ``api`` stage, route
``POST /analyze`` - no path rewriting needed anywhere.

Concurrency note: the plan asks for reserved concurrency 5, but this account's
Lambda concurrency limit is 10 with a required unreserved floor of 10, so ANY
positive reservation is rejected. The account-wide cap of 10 plus the stage
throttle (rate 5/s, burst 10) give a stricter real-time bound, so no reservation
is set here. Raise the Lambda concurrency quota to restore a per-function cap.
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_integrations as apigwv2_integrations
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bedrock: the global cross-region inference profile routes to the underlying
# foundation model in whatever region has capacity, so the policy grants invoke
# on BOTH the inference-profile ARN and the foundation-model ARNs, wildcarding
# the region segment (5.5a finding 1).
BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_MODEL_SLUG = "anthropic.claude-haiku-4-5-20251001-v1:0"

STAGE_NAME = "api"


class ApiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        fn = lambda_.DockerImageFunction(
            self,
            "ServeFn",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=str(REPO_ROOT),
                file="deploy/Dockerfile.lambda",
                platform=ecr_assets.Platform.LINUX_ARM64,
            ),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=6144,
            timeout=Duration.seconds(120),
            environment={
                "DEFECTLENS_NO_VLM": "1",
                "DEFECTLENS_DESCRIBER": "bedrock",
                "DEFECTLENS_BEDROCK_MODEL": BEDROCK_MODEL_ID,
                "DEFECTLENS_BEDROCK_REGION": self.region,
                # Absolute paths inside the image (LAMBDA_TASK_ROOT == /var/task).
                "CARD_VECTORS_PATH": "/var/task/models/cloud_artifacts/card_vectors.npz",
                "AUDIO_BANK_DIR": "/var/task/models/audio_bank",
            },
        )

        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/{BEDROCK_MODEL_ID}",
                    f"arn:aws:bedrock:*::foundation-model/{BEDROCK_MODEL_SLUG}",
                ],
            )
        )

        integration = apigwv2_integrations.HttpLambdaIntegration(
            "ServeIntegration",
            fn,
            # HTTP API caps integration timeout at 29s (CDK-validated ceiling);
            # give the cold-start model load the whole window (a warm request is a
            # few seconds).
            timeout=Duration.seconds(29),
        )

        http_api = apigwv2.HttpApi(
            self,
            "HttpApi",
            create_default_stage=False,
            description="DefectLens CPU serving API",
        )
        for path, method in (
            ("/analyze", apigwv2.HttpMethod.POST),
            ("/search", apigwv2.HttpMethod.POST),
            ("/health", apigwv2.HttpMethod.GET),
        ):
            http_api.add_routes(path=path, methods=[method], integration=integration)

        stage = apigwv2.HttpStage(
            self,
            "ApiStage",
            http_api=http_api,
            stage_name=STAGE_NAME,
            auto_deploy=True,
        )
        # Default-route throttling via the L1 escape hatch (the L2 stage does not
        # surface default_route_settings). The Ops cost-guard patches these to 0
        # to kill traffic if spend crosses the daily cap.
        cfn_stage = stage.node.default_child
        assert isinstance(cfn_stage, apigwv2.CfnStage)
        cfn_stage.default_route_settings = apigwv2.CfnStage.RouteSettingsProperty(
            throttling_rate_limit=5,
            throttling_burst_limit=10,
        )

        self.http_api = http_api
        self.stage_name = STAGE_NAME

        CfnOutput(self, "ApiEndpoint", value=http_api.api_endpoint)
        CfnOutput(self, "ApiStageUrl", value=f"{http_api.api_endpoint}/{STAGE_NAME}")
        CfnOutput(self, "ServeFunctionName", value=fn.function_name)
