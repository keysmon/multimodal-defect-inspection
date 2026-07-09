"""Shared constants for the Phase 5.5c GPU async path.

Both GpuStack (which creates the endpoint) and ApiStack (whose Lambda invokes it)
need the same endpoint name, bucket, and S3 prefixes. Keeping them here — plain
Python constants, imported by both — means ApiStack can reference the endpoint by
name WITHOUT a CloudFormation cross-stack dependency, so GpuStack stays
independently deployable (deploy the CPU stacks without it; add GPU later).
"""
from __future__ import annotations

# The Phase 3 artifacts bucket (also holds the training checkpoints). model.tar.gz
# is uploaded by scripts/package_sagemaker_model.py.
BUCKET = "defectlens-phase3-ca-002559670021"
MODEL_KEY = "phase5/sagemaker/model.tar.gz"

# Async I/O prefixes: the API Lambda PUTs the request payload under async-in/,
# SageMaker writes results under async-out/ and failures under async-fail/.
ASYNC_IN_PREFIX = "phase5/sagemaker/async-in/"
ASYNC_OUT_PREFIX = "phase5/sagemaker/async-out/"
ASYNC_FAIL_PREFIX = "phase5/sagemaker/async-fail/"

# Deterministic names so auto-scaling's resource_id and the API Lambda's env both
# reference the same endpoint/variant.
ENDPOINT_NAME = "defectlens-vlm-async"
VARIANT_NAME = "AllTraffic"

INSTANCE_TYPE = "ml.g5.xlarge"  # A10G 24GB — fits Qwen2.5-VL-3B bf16 comfortably

# HuggingFace PyTorch Inference DLC, verified present in ca-central-1 (account
# 763104351884) via `aws ecr describe-images` on 2026-07-09. transformers 5.5.3
# matches this repo's pyproject (>=5.0) and supports Qwen2.5-VL (>=4.49); py312 is
# the newest inference DLC — no transformers-5 py311 build exists. Region is
# hardcoded because the whole app is pinned to ca-central-1 (infra/app.py) and the
# DLC image must live in the endpoint's region.
# Source: https://aws.github.io/deep-learning-containers/reference/available_images/
DLC_ACCOUNT = "763104351884"
DLC_REPO = "huggingface-pytorch-inference"
DLC_TAG = "2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04"
DLC_REGION = "ca-central-1"
DLC_IMAGE = f"{DLC_ACCOUNT}.dkr.ecr.{DLC_REGION}.amazonaws.com/{DLC_REPO}:{DLC_TAG}"


def model_data_url() -> str:
    return f"s3://{BUCKET}/{MODEL_KEY}"
