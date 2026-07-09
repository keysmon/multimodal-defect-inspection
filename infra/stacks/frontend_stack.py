"""FrontendStack - private S3 + CloudFront serving the React build (Phase 5.5b).

Single-origin design: CloudFront's default behaviour serves the SPA from a
private S3 bucket (Origin Access Control, bucket stays fully private), and an
``/api/*`` behaviour forwards to the HTTP API's ``api`` stage. Because the API
uses a named stage, ``/api/analyze`` at CloudFront maps straight to stage ``api``
route ``/analyze`` with no rewriting. The React build is made with
``REACT_APP_API_URL=/api`` so it calls same-origin relative paths - no CORS, and
no CloudFront URL is needed at build time.

DefectLens is a genuine single page (no client-side router), so no SPA 404->
index fallback is configured; that also keeps ``/api/*`` error statuses honest.
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_BUILD = REPO_ROOT / "frontend" / "build"


class FrontendStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        http_api: apigwv2.HttpApi,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self,
            "SiteBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        api_domain = f"{http_api.api_id}.execute-api.{self.region}.amazonaws.com"
        api_origin = origins.HttpOrigin(
            api_domain,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        )

        distribution = cloudfront.Distribution(
            self,
            "SiteDistribution",
            comment="DefectLens demo",
            default_root_object="index.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    # API Gateway rejects a mismatched Host header, so forward
                    # everything from the viewer EXCEPT Host.
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
        )

        s3deploy.BucketDeployment(
            self,
            "DeploySite",
            sources=[s3deploy.Source.asset(str(FRONTEND_BUILD))],
            destination_bucket=bucket,
            distribution=distribution,
            # Invalidate everything on deploy so viewers pick up a new build even
            # though CloudFront caches the SPA assets aggressively.
            distribution_paths=["/*"],
            prune=True,
        )

        self.distribution = distribution

        CfnOutput(self, "SiteURL", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(self, "DistributionId", value=distribution.distribution_id)
        CfnOutput(self, "SiteBucketName", value=bucket.bucket_name)
