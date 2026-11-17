"""Pydantic v2 schemas for step-level latency and LLM token/cost telemetry."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Aggregated token counts and estimated USD cost for one or more LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: LLMUsage) -> LLMUsage:
        """Return a new ``LLMUsage`` summing token counts and estimated cost."""
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


class PipelineMetrics(BaseModel):
    """Step-level execution latencies (ms) and aggregated LLM usage for a pipeline run."""

    latencies_ms: dict[str, float] = Field(default_factory=dict)
    usage: LLMUsage = Field(default_factory=LLMUsage)
