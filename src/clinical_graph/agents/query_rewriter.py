"""History-aware clinical query rewriter for multi-query seed retrieval."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.prompts import load_prompt
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.schemas.query import QueryRequest

logger = logging.getLogger(__name__)

RetrievalMode = Literal["rag", "hybrid"]


class RewrittenQueries(BaseModel):
    """Structured output containing standalone clinical search queries and routing mode."""

    search_queries: list[str] = Field(min_length=1, max_length=5)
    retrieval_mode: Literal["rag", "hybrid"] = "hybrid"


class QueryRewriter:
    """Rewrites conversational clinical queries into standalone search queries."""

    def __init__(
        self,
        llm_client: LiteLLMClient | None = None,
        model: str = "gpt-4o-mini",
        prompt_version: str | None = None,
    ) -> None:
        self.llm_client: LiteLLMClient | None = llm_client
        self.model = model
        template = load_prompt("query_rewrite", version=prompt_version)
        self.prompt_version = prompt_version or template.version
        self._system_prompt = template.system_prompt

    async def rewrite_with_routing_async(
        self,
        request: QueryRequest,
        api_key: str | None = None,
    ) -> tuple[list[str], Literal["rag", "hybrid"], LLMUsage]:
        """Generate deduplicated search queries, routing mode, and ``LLMUsage``."""
        if self.llm_client is None:
            return [request.query], "hybrid", LLMUsage()

        user_prompt = self._build_user_prompt(request)
        usage = LLMUsage()
        try:
            raw, usage = await complete_with_usage_compat(
                self.llm_client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format=RewrittenQueries,
                api_key=api_key,
            )
            parsed = RewrittenQueries.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Query rewriting failed; falling back to original query: %s", exc)
            return [request.query], "hybrid", usage

        deduped = _dedup_queries(parsed.search_queries)[:4]
        return (deduped or [request.query]), parsed.retrieval_mode, usage

    async def rewrite_with_usage_async(
        self,
        request: QueryRequest,
        api_key: str | None = None,
    ) -> tuple[list[str], LLMUsage]:
        """Generate deduplicated standalone search queries and return ``(queries, LLMUsage)``."""
        queries, _, usage = await self.rewrite_with_routing_async(request, api_key=api_key)
        return queries, usage

    async def rewrite_async(
        self,
        request: QueryRequest,
        api_key: str | None = None,
    ) -> list[str]:
        """Generate deduplicated standalone search queries, falling back to ``[request.query]``."""
        queries, _ = await self.rewrite_with_usage_async(request, api_key=api_key)
        return queries

    @staticmethod
    def _build_user_prompt(request: QueryRequest) -> str:
        sections: list[str] = []

        if request.messages:
            history_lines = [
                f"{msg.role}: {msg.content}"
                for msg in request.messages
            ]
            sections.append("Conversation history:\n" + "\n".join(history_lines))

        if request.patient_context is not None:
            ctx_parts: list[str] = []
            if request.patient_context.age_group:
                ctx_parts.append(f"age_group={request.patient_context.age_group}")
            if request.patient_context.pregnancy_status is not None:
                ctx_parts.append(f"pregnancy={request.patient_context.pregnancy_status}")
            if request.patient_context.renal_impairment is not None:
                ctx_parts.append(f"renal_impairment={request.patient_context.renal_impairment}")
            if request.patient_context.allergies:
                ctx_parts.append(f"allergies={', '.join(request.patient_context.allergies)}")
            if ctx_parts:
                sections.append(f"Patient context: {'; '.join(ctx_parts)}.")

        sections.append(f"Current query: {request.query}")
        return "\n\n".join(sections)


def _dedup_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for query in queries:
        cleaned = query.strip()
        norm = cleaned.lower()
        if cleaned and norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result
