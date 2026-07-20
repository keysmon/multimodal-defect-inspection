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
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from constructs import Construct

from stacks import _gpu_config as gpu

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bedrock: the global cross-region inference profile routes to the underlying
# foundation model in whatever region has capacity, so the policy grants invoke
# on BOTH the inference-profile ARN and the foundation-model ARNs, wildcarding
# the region segment (5.5a finding 1).
BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_MODEL_SLUG = "anthropic.claude-haiku-4-5-20251001-v1:0"

STAGE_NAME = "api"

# CPU async-job I/O lives under this prefix in the shared artifacts bucket
# (gpu.BUCKET): the submit route writes cpu-jobs/in/, the worker writes
# cpu-jobs/out/ or cpu-jobs/err/. A 1-day S3 lifecycle rule (applied out-of-band
# on the bucket - it is not CDK-managed) expires them.
CPU_JOBS_PREFIX = "phase5/cpu-jobs/"


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        gpu_endpoint_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        environment = {
            "DEFECTLENS_NO_VLM": "1",
            "DEFECTLENS_NO_AUDIO": "1",
            "DEFECTLENS_DESCRIBER": "bedrock",
            "DEFECTLENS_BEDROCK_MODEL": BEDROCK_MODEL_ID,
            "DEFECTLENS_BEDROCK_REGION": self.region,
            # Absolute paths inside the image (LAMBDA_TASK_ROOT == /var/task).
            "CARD_VECTORS_PATH": "/var/task/models/cloud_artifacts/card_vectors.npz",
            "AUDIO_BANK_DIR": "/var/task/models/audio_bank",
            # Async /analyze: lazy model load keeps submit/status/health
            # model-free (fast on a cold env); the worker calls ensure_loaded
            # itself. The submit route drops jobs under this s3://bucket/prefix
            # and async self-invokes this same Lambda (name from
            # AWS_LAMBDA_FUNCTION_NAME, set by the runtime).
            "DEFECTLENS_LAZY_LOAD": "1",
            "CPU_JOBS_S3": f"s3://{gpu.BUCKET}/{CPU_JOBS_PREFIX}",
        }
        # GPU async path (5.5c) is OFF by default: SAGEMAKER_ENDPOINT is the sole
        # on-switch, so a CPU-only deploy answers 503 rather than invoking a
        # missing endpoint. Enable by deploying GpuStack, then redeploying
        # ApiStack with `-c gpu_endpoint_name=defectlens-vlm-async`.
        if gpu_endpoint_name:
            environment["SAGEMAKER_ENDPOINT"] = gpu_endpoint_name
            environment["ASYNC_INPUT_S3"] = f"s3://{gpu.BUCKET}/{gpu.ASYNC_IN_PREFIX}"

        fn = lambda_.DockerImageFunction(
            self,
            "ServeFn",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=str(REPO_ROOT),
                file="deploy/Dockerfile.lambda",
                platform=ecr_assets.Platform.LINUX_ARM64,
            ),
            architecture=lambda_.Architecture.ARM_64,
            # Account quota caps new-account Lambda memory at 3008MB (increase
            # requested; bump to 8192 + re-enable audio when granted). At 3008
            # CLIP fits; CLAP is disabled via DEFECTLENS_NO_AUDIO to stay in RAM.
            memory_size=3008,
            # The async worker (Event self-invoke) is the SAME function, so it
            # runs under this timeout - but it has no 29s API-gateway cap, and a
            # cold worker (model load ~24-29s + classify + RAG + describe)
            # exceeds 30s. Give it headroom: 120s. Safe now because the Bedrock
            # hang - the original reason this was cut 120s->30s on 2026-07-20 -
            # is bounded by describe_with_deadline() (sync uses the describer's
            # 12s budget, the worker a generous-but-bounded one), so nothing in
            # the path stalls unboundedly: model load completes, classify/RAG are
            # CPU-fast. The HTTP path is unaffected in practice - the integration
            # still caps at 29s (below), so a client sees a 504 by then
            # regardless; the function only runs on to finish the cold load and
            # warm the env, not to 120s.
            timeout=Duration.seconds(120),
            environment=environment,
        )

        # When the GPU path is enabled, the serving Lambda invokes the async
        # endpoint and reads/writes the async S3 prefixes. Scoped to the known
        # endpoint + prefixes; attached only when wired (harmless when unused).
        if gpu_endpoint_name:
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["sagemaker:InvokeEndpointAsync"],
                    resources=[
                        f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{gpu_endpoint_name}"
                    ],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["s3:PutObject"],
                    resources=[f"arn:aws:s3:::{gpu.BUCKET}/{gpu.ASYNC_IN_PREFIX}*"],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[
                        f"arn:aws:s3:::{gpu.BUCKET}/{gpu.ASYNC_OUT_PREFIX}*",
                        f"arn:aws:s3:::{gpu.BUCKET}/{gpu.ASYNC_FAIL_PREFIX}*",
                    ],
                )
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

        # CPU async path: the submit route async self-invokes this SAME function
        # (InvocationType=Event) and reads/writes job payloads under cpu-jobs/.
        # Scoped to self + that prefix; a separate AWS::IAM::Policy on the role,
        # so referencing fn.function_arn here does not create a CFN cycle.
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[fn.function_arn],
            )
        )
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"arn:aws:s3:::{gpu.BUCKET}/{CPU_JOBS_PREFIX}*"],
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
            ("/analyze-jobs", apigwv2.HttpMethod.POST),
            ("/analyze-jobs/{job_id}", apigwv2.HttpMethod.GET),
            ("/analyze-vlm", apigwv2.HttpMethod.POST),
            ("/search", apigwv2.HttpMethod.POST),
            ("/health", apigwv2.HttpMethod.GET),
            ("/vlm-status", apigwv2.HttpMethod.GET),
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
        # Consumed by OpsStack's dashboard (cross-stack ref, like http_api).
        self.serve_fn = fn

        # Keep-warm: invoke the function every 5 minutes with a synthetic /health
        # HTTP-API event so the ~55s model load (3008MB account quota) almost
        # never lands on a real visitor. ~$0.04/mo of invocations.
        events.Rule(
            self,
            "KeepWarm",
            schedule=events.Schedule.rate(Duration.minutes(5)),
            targets=[
                targets.LambdaFunction(
                    fn,
                    event=events.RuleTargetInput.from_object(
                        {
                            "version": "2.0",
                            "routeKey": "GET /health",
                            "rawPath": "/api/health",
                            "rawQueryString": "",
                            "headers": {"host": "keepwarm"},
                            "requestContext": {
                                "http": {
                                    "method": "GET",
                                    "path": "/api/health",
                                    "protocol": "HTTP/1.1",
                                    "sourceIp": "127.0.0.1",
                                }
                            },
                            "isBase64Encoded": False,
                        }
                    ),
                )
            ],
        )

        CfnOutput(self, "ApiEndpoint", value=http_api.api_endpoint)
        CfnOutput(self, "ApiStageUrl", value=f"{http_api.api_endpoint}/{STAGE_NAME}")
        CfnOutput(self, "ServeFunctionName", value=fn.function_name)
