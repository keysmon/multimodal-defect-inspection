"""Shared grounding: retrieval + citation validity, used by the agent AND the
walkthrough report layer. One module, one trust story."""
from defectlens.grounding.citations import (  # noqa: F401
    citation_is_class_relevant,
    on_class_citations,
    validate_citations,
)
