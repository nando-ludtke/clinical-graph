"""Guardrails package for clinical-guard prompt-injection integration."""

from clinical_graph.guardrails.prompt_guard import (
    ClinicalGuardClient,
    GuardCheckResult,
    filter_web_snippets,
    scan_text_for_injection,
)

__all__ = [
    "ClinicalGuardClient",
    "GuardCheckResult",
    "filter_web_snippets",
    "scan_text_for_injection",
]
