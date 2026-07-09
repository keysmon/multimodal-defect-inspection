"""AWS Lambda entrypoint: Mangum over the FastAPI app (Phase 5.5 cloud path).

The container image / Lambda config drives everything through env vars, which
serve.api's lifespan reads on cold start:

- DEFECTLENS_NO_VLM=1          no local torch VLM (defaulted here)
- DEFECTLENS_DESCRIBER=bedrock description via Claude Haiku on Bedrock (defaulted here)
- CARD_VECTORS_PATH            npz for ArrayVectorStore, baked into the image
- AUDIO_BANK_DIR               audio bank dir, baked into the image
- DEFECTLENS_NO_AUDIO=1        (optional) disable the audio path

This handler only sets sane defaults for the no-VLM Bedrock cloud shape, then
wraps the app with Mangum. All component wiring lives in the lifespan, so a
local `uvicorn` run and the Lambda share one code path.
"""
from __future__ import annotations

import os

# Defaults for the cloud shape; a real env value always wins (setdefault).
# CARD_VECTORS_PATH / AUDIO_BANK_DIR here are RELATIVE, and so are the corpus/
# configs paths Recognizer defaults to (Path("corpus"), Path("configs/...")).
# They resolve against the process cwd, which is /var/task (LAMBDA_TASK_ROOT) in
# Lambda — where deploy/Dockerfile.lambda COPYs the artifacts. The Dockerfile
# also sets these two vars to absolute /var/task paths, so in-container these
# relative defaults are overridden; they only apply to a non-container run
# started from the repo root.
os.environ.setdefault("DEFECTLENS_NO_VLM", "1")
os.environ.setdefault("DEFECTLENS_DESCRIBER", "bedrock")
os.environ.setdefault("CARD_VECTORS_PATH", "models/cloud_artifacts/card_vectors.npz")
os.environ.setdefault("AUDIO_BANK_DIR", "models/audio_bank")

from mangum import Mangum  # noqa: E402  (after env defaults)

from defectlens.serve.api import create_app  # noqa: E402

app = create_app()
# The HTTP API named stage prefixes every path with /api (CloudFront routes
# /api/* straight through); strip it so FastAPI sees its own routes.
handler = Mangum(app, api_gateway_base_path=os.environ.get("API_GATEWAY_BASE_PATH", "/api"))
