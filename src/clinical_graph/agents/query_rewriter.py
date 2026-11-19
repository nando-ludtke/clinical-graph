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
    """Structured output containing standalone clinical search queries, routing mode, and scope gate."""

    search_queries: list[str] = Field(min_length=1, max_length=5)
    retrieval_mode: Literal["rag", "hybrid"] = "hybrid"
    is_medical: bool = True
    off_topic_reason: str | None = None


class RewrittenQueriesWithWeb(RewrittenQueries):
    """Extended output adding 1 to 2 concise keyword queries for Tavily when web search is enabled."""

    web_search_queries: list[str] = Field(default_factory=list, max_length=3)


QueryRewriteOutput = RewrittenQueries


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

    async def rewrite_with_gate_async(
        self,
        request: QueryRequest,
        api_key: str | None = None,
    ) -> tuple[list[str], Literal["rag", "hybrid"], bool, str | None, LLMUsage]:
        """Generate search queries, routing mode, ``is_medical`` gate, and ``LLMUsage``."""
        if self.llm_client is None:
            return [request.query], "hybrid", True, None, LLMUsage()

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
            return [request.query], "hybrid", True, None, usage

        deduped = _dedup_queries(parsed.search_queries)[:4]
        return (
            deduped or [request.query],
            parsed.retrieval_mode,
            parsed.is_medical,
            parsed.off_topic_reason,
            usage,
        )

    async def rewrite_with_web_async(
        self,
        request: QueryRequest,
        *,
        include_web_query: bool = False,
        api_key: str | None = None,
    ) -> tuple[list[str], Literal["rag", "hybrid"], bool, str | None, list[str], LLMUsage]:
        """Generate search queries, routing mode, gate, optional ``web_search_queries``, and usage."""
        if not include_web_query:
            queries, mode, is_med, reason, usage = await self.rewrite_with_gate_async(
                request, api_key=api_key
            )
            return queries, mode, is_med, reason, [], usage

        if self.llm_client is None:
            return [request.query], "hybrid", True, None, [request.query], LLMUsage()

        user_prompt = (
            self._build_user_prompt(request)
            + "\n\nAlso generate `web_search_queries`: 1 to 2 short, keyword-focused clinical web "
            "search queries (3 to 7 words each, no `site:` operators) suitable for searching trusted "
            "medical websites. Use 1 query for single-topic questions, or 2 queries only when the "
            "question asks about distinct sub-topics (e.g. ['apixaban dose reduction renal', "
            "'apixaban reversal agent bleeding'])."
        )
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
                response_format=RewrittenQueriesWithWeb,
                api_key=api_key,
            )
            parsed = RewrittenQueriesWithWeb.model_validate_json(raw)
        except Exception as exc:
            logger.warning("Query rewriting (with web) failed; falling back: %s", exc)
            return [request.query], "hybrid", True, None, [request.query], usage

        deduped = _dedup_queries(parsed.search_queries)[:4]
        web_qs = _dedup_queries(parsed.web_search_queries)[:2] or [request.query]
        return (
            deduped or [request.query],
            parsed.retrieval_mode,
            parsed.is_medical,
            parsed.off_topic_reason,
            web_qs,
            usage,
        )

    async def rewrite_with_routing_async(
        self,
        request: QueryRequest,
        api_key: str | None = None,
    ) -> tuple[list[str], Literal["rag", "hybrid"], LLMUsage]:
        """Generate deduplicated search queries, routing mode, and ``LLMUsage``."""
        queries, mode, _, _, usage = await self.rewrite_with_gate_async(
            request, api_key=api_key
        )
        return queries, mode, usage

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
