"""GpuStack - fine-tuned VLM on a SageMaker async endpoint, scale-to-zero (5.5c).

Authored and synth-validated here, but intentionally NOT part of the demo
milestone deploy (app.py builds it; the deploy commands target Api/Frontend/Ops
only). Deploy it explicitly when you want the GPU path:

    cd infra && npx --yes aws-cdk@2 deploy GpuStack

The first real request is the user-gated smoke test: async cold start is ~5-8 min
while the ~8GB Qwen2.5-VL-3B base downloads from the HF hub (the tarball ships
only the ~120MB adapter — see scripts/package_sagemaker_model.py).

Shape:
- Model: the HuggingFace PyTorch Inference DLC + model.tar.gz. SAGEMAKER_PROGRAM /
  SAGEMAKER_SUBMIT_DIRECTORY point the toolkit at code/inference.py inside the
  tarball, so the endpoint runs our score_answers handler, not the DLC default.
- EndpointConfig: one ml.g5.xlarge variant with AsyncInferenceConfig (results to
  s3://.../async-out/, failures to .../async-fail/). Async is what makes
  scale-to-zero usable: the client submits an S3 input and polls an S3 output, so
  requests queue while the endpoint is asleep instead of erroring.
- Auto scaling to zero: min 0 / max 1, via TWO policies that together are the
  standard SageMaker async scale-to-zero pattern:
    * target tracking on ApproximateBacklogSizePerInstance (target 1) — the
      steady-state control that also scales back down toward 0 as the queue drains;
    * a step policy on HasBacklogWithoutCapacity — the 0->1 wake-up. Target
      tracking CANNOT scale up from 0 (no per-instance metric is emitted with 0
      instances), so without this a slept endpoint stays asleep forever.
      (AWS async-inference "scale from zero" autoscaling guidance.)

DLC image + region rationale live in stacks._gpu_config.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
)
from aws_cdk import aws_applicationautoscaling as appscaling
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

from stacks import _gpu_config as gpu


class GpuStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        role = self._execution_role()

        model = sagemaker.CfnModel(
            self,
            "VlmModel",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=gpu.DLC_IMAGE,
                model_data_url=gpu.model_data_url(),
                environment={
                    # The toolkit finds code/inference.py inside the tarball via
                    # these two; without SAGEMAKER_PROGRAM it runs the DLC default
                    # handler instead of ours.
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": self.region,
                    # The ~8GB base downloads at start; point the HF cache at the
                    # instance's large local volume (/tmp) so it has room to land.
                    "HF_HOME": "/tmp/hf",
                    "TRANSFORMERS_CACHE": "/tmp/hf",
                },
            ),
        )
        # SageMaker validates S3 access to the model artifact AT CREATE TIME,
        # but CFN sees no edge between the Model and the role's DefaultPolicy
        # (add_to_policy statements) - the first deploy raced and failed with
        # "Could not access model data". Make the edge explicit.
        default_policy = role.node.try_find_child("DefaultPolicy")
        if default_policy is not None:
            model.node.add_dependency(default_policy)


        endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "VlmEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name=gpu.VARIANT_NAME,
                    model_name=model.attr_model_name,
                    # Must be >=1 at creation; auto scaling drops it to 0 once idle.
                    initial_instance_count=1,
                    instance_type=gpu.INSTANCE_TYPE,
                    initial_variant_weight=1.0,
                )
            ],
            async_inference_config=sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty(
                output_config=sagemaker.CfnEndpointConfig.AsyncInferenceOutputConfigProperty(
                    s3_output_path=f"s3://{gpu.BUCKET}/{gpu.ASYNC_OUT_PREFIX}",
                    s3_failure_path=f"s3://{gpu.BUCKET}/{gpu.ASYNC_FAIL_PREFIX}",
                ),
                client_config=sagemaker.CfnEndpointConfig.AsyncInferenceClientConfigProperty(
                    max_concurrent_invocations_per_instance=2,
                ),
            ),
        )

        endpoint = sagemaker.CfnEndpoint(
            self,
            "VlmEndpoint",
            endpoint_name=gpu.ENDPOINT_NAME,
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
        )

        self._autoscale_to_zero(endpoint)

        CfnOutput(self, "VlmEndpointName", value=gpu.ENDPOINT_NAME)
        CfnOutput(self, "VlmModelDataUrl", value=gpu.model_data_url())
        CfnOutput(self, "VlmDlcImage", value=gpu.DLC_IMAGE)
        CfnOutput(
            self, "VlmAsyncOutputPath", value=f"s3://{gpu.BUCKET}/{gpu.ASYNC_OUT_PREFIX}"
        )

    # ------------------------------------------------------------------ helpers

    def _execution_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "VlmEndpointRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="SageMaker execution role for the fine-tuned VLM async endpoint",
        )
        # Read the model artifact + async inputs; write async results + failures.
        role.add_to_policy(
            iam.PolicyStatement(
                # SageMaker's Endpoint-creation validation (observed 2026-07-09)
                # requires bucket-level s3:ListBucket AND broad PutObject on the
                # bucket, beyond the per-prefix grants - narrow grants fail with
                # "The provided role ... is invalid ... has s3:ListBucket and
                # s3:PutObject permissions for bucket".
                actions=["s3:GetObject"],
                resources=[
                    f"arn:aws:s3:::{gpu.BUCKET}/{gpu.MODEL_KEY}",
                    f"arn:aws:s3:::{gpu.BUCKET}/{gpu.ASYNC_IN_PREFIX}*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[f"arn:aws:s3:::{gpu.BUCKET}/*"],
            )
        )
        # Pull the DLC image. GetAuthorizationToken has no resource scope; the
        # layer/image pulls are scoped to the DLC repo in the AWS-owned account.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{gpu.BUCKET}"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"])
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                resources=[
                    f"arn:aws:ecr:{self.region}:{gpu.DLC_ACCOUNT}:repository/{gpu.DLC_REPO}"
                ],
            )
        )
        # CloudWatch logs (scoped to SageMaker's log groups) + endpoint metrics
        # (PutMetricData has no resource-level permissions).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"])
        )
        return role

    def _autoscale_to_zero(self, endpoint: sagemaker.CfnEndpoint) -> None:
        resource_id = f"endpoint/{gpu.ENDPOINT_NAME}/variant/{gpu.VARIANT_NAME}"
        scalable_target = appscaling.ScalableTarget(
            self,
            "VlmScalableTarget",
            service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
            resource_id=resource_id,
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            min_capacity=0,
            max_capacity=1,
        )
        # The endpoint must exist before Application Auto Scaling can register the
        # variant as a scalable target (resource_id references it by name).
        scalable_target.node.add_dependency(endpoint)

        # Steady-state: hold ~1 queued request per instance; scales back to 0 as
        # the backlog drains (with a cooldown to avoid thrashing on bursty demo use).
        scalable_target.scale_to_track_metric(
            "BacklogPerInstanceTracking",
            target_value=1,
            custom_metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="ApproximateBacklogSizePerInstance",
                dimensions_map={"EndpointName": gpu.ENDPOINT_NAME},
                statistic="Average",
                period=Duration.minutes(1),
            ),
            scale_in_cooldown=Duration.minutes(5),
            scale_out_cooldown=Duration.minutes(1),
        )

        # Wake from zero: HasBacklogWithoutCapacity is 1 when work is queued but no
        # instance is running. Target tracking can't act at 0 instances, so this
        # step policy provides the 0->1 bump.
        scalable_target.scale_on_metric(
            "ScaleFromZero",
            metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="HasBacklogWithoutCapacity",
                dimensions_map={"EndpointName": gpu.ENDPOINT_NAME},
                statistic="Average",
                period=Duration.minutes(1),
            ),
            scaling_steps=[
                appscaling.ScalingInterval(upper=0, change=0),
                appscaling.ScalingInterval(lower=1, change=+1),
            ],
            adjustment_type=appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
            cooldown=Duration.minutes(5),
            evaluation_periods=1,
            datapoints_to_alarm=1,
            metric_aggregation_type=appscaling.MetricAggregationType.MAXIMUM,
        )
