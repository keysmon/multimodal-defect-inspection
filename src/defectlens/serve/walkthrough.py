"""Serve-layer glue for the walkthrough report: provider selection + worker.

The report layer (defectlens.report) is provider-agnostic; this module picks
the reasoning LLM behind the existing provider seam and runs one walkthrough
job on the async worker path. Kept separate from serve.api so the routes stay
thin and the worker import stays lazy (async_jobs dispatches here by payload
kind).
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def build_report_provider(app):
    """The reasoning LLM for walkthrough synthesis, behind the provider seam.

    Precedence: an injected app.state.report_provider (tests / future config)
    -> Bedrock Haiku when the cloud describer gate is on (model/region env
    names reused from bedrock_describer) -> the local Qwen when a
    chat-capable describer is loaded -> None (the worker fails the job
    gracefully; the cloud BedrockDescriber has no chat(), so without the
    bedrock gate there is nothing to reason with).
    """
    injected = getattr(app.state, "report_provider", None)
    if injected is not None:
        return injected
    from defectlens.serve.bedrock_describer import (
        DEFAULT_MODEL_ID,
        DEFAULT_REGION,
        describer_is_bedrock,
    )

    if describer_is_bedrock():
        from defectlens.agent.providers import BedrockHaikuProvider

        return BedrockHaikuProvider(
            model_id=os.environ.get("DEFECTLENS_BEDROCK_MODEL", DEFAULT_MODEL_ID),
            region=os.environ.get("DEFECTLENS_BEDROCK_REGION", DEFAULT_REGION),
        )
    describer = getattr(app.state, "describer", None)
    if describer is not None and callable(getattr(describer, "chat", None)):
        from defectlens.agent.providers import LocalQwenProvider

        return LocalQwenProvider(describer=describer)
    return None


def run_walkthrough_job(app, payload: dict) -> dict:
    """Run one walkthrough job (worker side): load components, synthesize.

    Raises on missing provider / synthesis failure - the worker's crash
    isolation writes the err/ object so the poll surfaces a generic 500.
    """
    from defectlens.serve.api import ensure_loaded

    ensure_loaded(app)
    provider = build_report_provider(app)
    if provider is None:
        raise RuntimeError(
            "walkthrough needs a reasoning provider: set DEFECTLENS_DESCRIBER=bedrock "
            "or run with the local VLM loaded (DEFECTLENS_NO_VLM unset)"
        )
    from defectlens.report.synthesize import run_walkthrough

    report = run_walkthrough(
        photos=payload["photos"],
        visit_note=payload.get("visit_note"),
        recognizer=app.state.recognizer,
        provider=provider,
    )
    return json.loads(report.model_dump_json())
