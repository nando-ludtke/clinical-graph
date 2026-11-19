"""OpenAI Decisions API (``gpt-6-luna``) client for fast GraphRAG sufficiency & tier routing."""

from dataclasses import dataclass
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.schemas.metrics import LLMUsage

logger = logging.getLogger(__name__)

_OPENAI_DECISIONS_ENDPOINT = "https://api.openai.com/v1/decisions"
SearchTierChoice = Literal["dominant", "regional", "both"]


@dataclass(frozen=True, slots=True)
class SufficiencyDecision:
    """Result of evaluating whether local GraphRAG context suffices or web search is needed."""

    sufficient: bool
    sufficiency_prob: float
    search_tier: SearchTierChoice
    used_decisions_api: bool
    usage: LLMUsage


class _FallbackDecisionOutput(BaseModel):
    sufficient: bool = True
    search_tier: SearchTierChoice = "both"


class OpenAIDecisionClient:
    """Evaluates local context sufficiency via OpenAI ``/v1/decisions`` (``gpt-6-luna``)."""

    def __init__(
        self,
        *,
        decision_model: str = "gpt-6-luna",
        fallback_llm_client: LiteLLMClient | None = None,
        fallback_model: str = "gpt-4o-mini",
        sufficiency_threshold: float = 0.65,
        timeout: float = 5.0,
    ) -> None:
        self.decision_model = decision_model
        self.fallback_llm_client = fallback_llm_client
        self.fallback_model = fallback_model
        self.sufficiency_threshold = sufficiency_threshold
        self.timeout = timeout

    @staticmethod
    def _parse_decisions_response(
        payload: dict[str, Any],
        *,
        threshold: float,
    ) -> tuple[bool, float, SearchTierChoice] | None:
        answers = payload.get("answers")
        if not isinstance(answers, list):
            return None

        by_name: dict[str, dict[str, Any]] = {
            str(item.get("name")): item
            for item in answers
            if isinstance(item, dict) and item.get("name")
        }
        suf_item = by_name.get("is_sufficient")
        tier_item = by_name.get("search_tier")
        if suf_item is None:
            return None

        raw_prob = suf_item.get("probability", suf_item.get("score", suf_item.get("value")))
        if isinstance(raw_prob, bool):
            prob = 1.0 if raw_prob else 0.0
        else:
            try:
                prob = float(raw_prob)
            except (TypeError, ValueError):
                return None

        tier_val = str((tier_item or {}).get("value") or "both").lower()
        tier: SearchTierChoice = (
            tier_val if tier_val in ("dominant", "regional", "both") else "both"  # type: ignore[assignment]
        )
        return prob >= threshold, prob, tier

    async def evaluate_sufficiency(
        self,
        *,
        query: str,
        local_context: str,
        api_key: str | None,
    ) -> SufficiencyDecision:
        """Call OpenAI ``POST /v1/decisions`` (``gpt-6-luna``), falling back to ``LiteLLMClient``."""
        shared_input = f"User Query: {query}\n\nRetrieved Local GraphRAG Context:\n{local_context}"

        if api_key:
            request_body = {
                "model": self.decision_model,
                "input": shared_input,
                "questions": [
                    {
                        "type": "predicate",
                        "name": "is_sufficient",
                        "instructions": (
                            "Is the retrieved local GraphRAG context sufficient to accurately, "
                            "safely, and completely answer the user's clinical query—including any "
                            "requested specific drug, dosing, eGFR/age cutoff, or guideline—without "
                            "needing external web search? Context written in another language "
                            "(e.g., German) is sufficient if it contains the clinical facts needed "
                            "to answer the query."
                        ),
                    },
                    {
                        "type": "choice",
                        "name": "search_tier",
                        "instructions": (
                            "If external web search is needed, which allowlisted source tier "
                            "should be queried?"
                        ),
                        "choices": [
                            {
                                "value": "dominant",
                                "description": (
                                    "Global/US/UK clinical authorities (FDA, DailyMed, NIH, "
                                    "MedlinePlus, CDC, NHS, NICE, WHO, PubMed/PMC, Merck Manuals)."
                                ),
                            },
                            {
                                "value": "regional",
                                "description": (
                                    "EU, Canada, Australia, India, or China authorities (EMA, "
                                    "ECDC, AWMF, RKI, HAS, Health Canada, TGA, ICMR, MoHFW, NHC, NMPA)."
                                ),
                            },
                            {
                                "value": "both",
                                "description": (
                                    "Query both dominant and regional tiers in parallel (e.g., "
                                    "when regional or comparative guidelines may be needed)."
                                ),
                            },
                        ],
                    },
                ],
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        _OPENAI_DECISIONS_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_body,
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    parsed = self._parse_decisions_response(
                        data,
                        threshold=self.sufficiency_threshold,
                    )
                    if parsed is not None:
                        sufficient, prob, tier = parsed
                        usage_raw = data.get("usage") or {}
                        usage = LLMUsage(
                            prompt_tokens=int(usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0),
                            completion_tokens=0,
                            total_tokens=int(usage_raw.get("total_tokens") or usage_raw.get("input_tokens") or 0),
                        )
                        return SufficiencyDecision(
                            sufficient=sufficient,
                            sufficiency_prob=prob,
                            search_tier=tier,
                            used_decisions_api=True,
                            usage=usage,
                        )
            except Exception as exc:
                logger.warning("OpenAI Decisions API failed (%s); falling back to LLM.", exc)

        if self.fallback_llm_client is not None:
            try:
                raw, usage = await complete_with_usage_compat(
                    self.fallback_llm_client,
                    model=self.fallback_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Decide if the retrieved local clinical context is sufficient to "
                                "completely answer the user's query without web search. Return JSON "
                                '{"sufficient": bool, "search_tier": "dominant"|"regional"|"both"}.'
                            ),
                        },
                        {"role": "user", "content": shared_input},
                    ],
                    temperature=0.0,
                    response_format=_FallbackDecisionOutput,
                    api_key=api_key,
                )
                parsed_fb = _FallbackDecisionOutput.model_validate_json(raw)
                return SufficiencyDecision(
                    sufficient=parsed_fb.sufficient,
                    sufficiency_prob=1.0 if parsed_fb.sufficient else 0.0,
                    search_tier=parsed_fb.search_tier,
                    used_decisions_api=False,
                    usage=usage,
                )
            except Exception as exc:
                logger.warning("Fallback sufficiency decision failed: %s", exc)

        has_local = "No local chunks" not in local_context
        return SufficiencyDecision(
            sufficient=has_local,
            sufficiency_prob=1.0 if has_local else 0.0,
            search_tier="both",
            used_decisions_api=False,
            usage=LLMUsage(),
        )
