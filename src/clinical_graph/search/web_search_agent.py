"""1-shot verified web search agent with OpenAI Decisions (``gpt-6-luna``) & post-retrieval guard."""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel

from clinical_graph.guardrails.prompt_guard import (
    ClinicalGuardClient,
    scan_text_for_injection,
)
from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.prompts import load_prompt
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.schemas.query import QueryRequest
from clinical_graph.search.auditor import SourceAuditorAgent
from clinical_graph.search.decision_client import OpenAIDecisionClient, SearchTierChoice
from clinical_graph.search.fetcher import FetchedPage, fetch_allowlisted_url, is_js_gated_text
from clinical_graph.search.schemas import (
    AuditFinding,
    WebEvidenceItem,
    WebSearchOutcome,
    WebSearchResult,
)
from clinical_graph.search.tavily_client import TavilySearchClient

logger = logging.getLogger(__name__)


class _SearchQueryPlan(BaseModel):
    search_query: str
    tier: SearchTierChoice = "both"


def _format_local_evidence_summary(
    *,
    retrieved_chunks: list[dict[str, Any]],
    retrieved_tables: list[dict[str, Any]],
    seed_node_details: list[dict[str, Any]],
    ranked_edges: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    if seed_node_details:
        node_lines = []
        for node in seed_node_details[:10]:
            name = node.get("canonical_name") or node.get("node_id", "")
            ntype = node.get("node_type", "")
            desc = (node.get("description") or "").strip()[:160]
            line = f"- {name} [{ntype}]" if ntype else f"- {name}"
            if desc:
                line += f": {desc}"
            node_lines.append(line)
        parts.append("Graph Nodes:\n" + "\n".join(node_lines))

    if ranked_edges:
        edge_lines = []
        for edge in ranked_edges[:12]:
            u = edge.get("source", "")
            rel = edge.get("relation", "")
            v = edge.get("target", "")
            desc = (edge.get("relationship_description") or "").strip()[:160]
            line = f"- {u} --[{rel}]--> {v}"
            if desc:
                line += f": {desc}"
            edge_lines.append(line)
        parts.append("Graph Edges:\n" + "\n".join(edge_lines))

    if retrieved_chunks:
        chunk_lines = []
        for idx, chunk in enumerate(retrieved_chunks[:8], start=1):
            text = (chunk.get("text") or chunk.get("passage") or chunk.get("content") or "").strip()
            if text:
                chunk_lines.append(f"[Local Chunk {idx}] {text[:450]}")
        if chunk_lines:
            parts.append("Retrieved Chunks:\n" + "\n".join(chunk_lines))

    if retrieved_tables:
        table_lines = []
        for idx, tbl in enumerate(retrieved_tables[:4], start=1):
            text = (tbl.get("text") or tbl.get("markdown") or "").strip()
            if text:
                table_lines.append(f"[Local Table {idx}] {text[:400]}")
        if table_lines:
            parts.append("Retrieved Tables:\n" + "\n".join(table_lines))

    return "\n\n".join(parts) if parts else "No local chunks, tables, or graph evidence found."


class WebSearchAgent:
    """1-shot verified web fallback using label fast-path or OpenAI Decisions (``gpt-6-luna``)."""

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient | None = None,
        search_client: TavilySearchClient | None = None,
        auditor: SourceAuditorAgent | None = None,
        decision_client: OpenAIDecisionClient | None = None,
        clinical_guard_client: ClinicalGuardClient | None = None,
        model: str = "gpt-4o-mini",
        prompt_version: str | None = None,
        allow_pdf: bool = False,
        include_regional: bool = False,
        results_per_tier: int = 5,
        results_per_tier_with_pdf: int = 3,
        max_audit_results: int = 3,
        fetch_timeout: float = 8.0,
        max_pdf_pages: int = 40,
        fetcher: Callable[..., Awaitable[FetchedPage]] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client
        self.clinical_guard_client = clinical_guard_client
        self.auditor = auditor or SourceAuditorAgent(
            llm_client=llm_client,
            clinical_guard_client=clinical_guard_client,
            model=model,
        )
        self.decision_client = decision_client or OpenAIDecisionClient(
            fallback_llm_client=llm_client,
            fallback_model=model,
        )
        self.model = model
        self.allow_pdf = allow_pdf
        self.include_regional = include_regional
        self.results_per_tier = results_per_tier
        self.results_per_tier_with_pdf = results_per_tier_with_pdf
        self.max_audit_results = max_audit_results
        self.fetch_timeout = fetch_timeout
        self.max_pdf_pages = max_pdf_pages
        self._fetcher = fetcher or fetch_allowlisted_url
        template = load_prompt("web_search_agent", version=prompt_version)
        self.prompt_version = prompt_version or template.version
        self._system_prompt = template.system_prompt

    async def _plan_query(
        self,
        *,
        query: str,
        local_summary: str,
        default_tier: SearchTierChoice,
        api_key: str | None,
    ) -> tuple[_SearchQueryPlan, LLMUsage]:
        if self.llm_client is None:
            return _SearchQueryPlan(search_query=query, tier=default_tier), LLMUsage()

        user_prompt = (
            f"User Query: {query}\n\n"
            f"Local Context Summary:\n{local_summary[:1800]}\n\n"
            "Return JSON with a concise `search_query` (3-7 words, no `site:`) and `tier`."
        )
        try:
            raw, usage = await complete_with_usage_compat(
                self.llm_client,
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format=_SearchQueryPlan,
                api_key=api_key,
            )
            return _SearchQueryPlan.model_validate_json(raw), usage
        except Exception as exc:
            logger.warning("Web query planning failed; using raw query: %s", exc)
            return _SearchQueryPlan(search_query=query, tier=default_tier), LLMUsage()

    async def _resolve_page_one(
        self,
        *,
        res: WebSearchResult,
    ) -> tuple[WebSearchResult, FetchedPage, float]:
        fetch_ms = 0.0
        if not res.is_pdf and res.raw_content and not is_js_gated_text(res.raw_content):
            page = FetchedPage(
                url=res.url,
                domain=res.domain,
                title=res.title,
                content_type="html",
                text=res.raw_content,
                readable=True,
            )
        else:
            t_fetch = time.perf_counter()
            page = await self._fetcher(
                res.url,
                fallback_title=res.title,
                timeout=self.fetch_timeout,
                max_pdf_pages=self.max_pdf_pages,
            )
            fetch_ms = (time.perf_counter() - t_fetch) * 1000.0
        return res, page, fetch_ms

    async def _guard_selected_evidence(
        self,
        *,
        findings: list[AuditFinding],
        session_id: str,
        client_ip: str | None,
    ) -> tuple[list[WebEvidenceItem], bool, bool | None, int]:
        relevant = [f for f in findings if f.relevant and f.guard_passed][:4]
        selected_evidence: list[WebEvidenceItem] = []
        guard_checked = False
        guard_reachable: bool | None = None
        blocked_count = sum(1 for f in findings if not f.guard_passed)

        for finding in relevant:
            combined_chunks = "\n\n".join(finding.extracted_chunks)
            guard_checked = True
            hit = False
            for candidate in (combined_chunks, finding.title):
                candidate_hit, _ = scan_text_for_injection(candidate)
                if candidate_hit:
                    hit = True
                    break
            if hit:
                blocked_count += 1
                continue

            selected_evidence.append(
                WebEvidenceItem(
                    result_id=finding.result_id,
                    url=finding.url,
                    title=finding.title,
                    domain=finding.domain,
                    content_type=finding.content_type,
                    tier=finding.tier,
                    chunks=finding.extracted_chunks,
                )
            )

        return selected_evidence, guard_checked, guard_reachable, blocked_count

    async def run_one_shot_search_async(
        self,
        *,
        request: QueryRequest,
        search_query: str | None = None,
        search_queries: list[str] | None = None,
        query_vectors: list[Any] | None = None,
        tier: Literal["dominant", "regional", "both"] = "dominant",
        session_id: str,
        client_ip: str | None = None,
        api_key: str | None = None,
        reason: str = "Fast-path label mismatch; parallel 1-shot web search.",
    ) -> WebSearchOutcome:
        """Execute parallel 1-shot Tavily search(es) + parallel audit + post-retrieval guard."""
        if self.search_client is None or not self.search_client.is_configured:
            return WebSearchOutcome(
                triggered=False,
                sufficient_locally=True,
                sufficiency_reason="Tavily search API key not configured.",
            )

        total_usage = LLMUsage()
        effective_queries = [
            q.strip()
            for q in (search_queries or ([search_query] if search_query else []))
            if q and q.strip()
        ]
        if not effective_queries:
            plan, plan_usage = await self._plan_query(
                query=request.query,
                local_summary="No matching entity labels found in local graph.",
                default_tier="both" if self.include_regional else "dominant",
                api_key=api_key,
            )
            total_usage = total_usage.add(plan_usage)
            effective_queries = [plan.search_query]
            tier = plan.tier

        effective_tier: Literal["dominant", "regional", "both"] = (
            tier if self.include_regional else "dominant"
        )
        effective_results = (
            self.results_per_tier_with_pdf if self.allow_pdf else self.results_per_tier
        )

        async def _run_query_fetch(
            eq: str,
            q_idx: int,
        ) -> tuple[list[tuple[WebSearchResult, FetchedPage, float]], float]:
            t_tav = time.perf_counter()
            batch = await self.search_client.search_parallel(  # type: ignore[union-attr]
                eq,
                tier=effective_tier,
                allow_pdf=self.allow_pdf,
                results_per_tier=effective_results,
                result_id_prefix=f"w{q_idx}_",
                start_index=1,
            )
            tavily_ms = (time.perf_counter() - t_tav) * 1000.0
            top_batch = batch[: self.max_audit_results]
            if not top_batch:
                return [], tavily_ms
            resolved = await asyncio.gather(*[self._resolve_page_one(res=res) for res in top_batch])
            return list(resolved), tavily_ms

        per_query_fetches = await asyncio.gather(
            *[
                _run_query_fetch(eq, idx)
                for idx, eq in enumerate(effective_queries[:3], start=1)
            ]
        )

        resolved_pages: list[tuple[str, FetchedPage]] = []
        seen_urls: set[str] = set()
        tavily_ms_max = 0.0
        fetch_ms_max = 0.0
        for items, q_tav_ms in per_query_fetches:
            tavily_ms_max = max(tavily_ms_max, q_tav_ms)
            for res, page, f_ms in items:
                fetch_ms_max = max(fetch_ms_max, f_ms)
                if res.url in seen_urls:
                    continue
                seen_urls.add(res.url)
                rid = f"w{len(resolved_pages) + 1}"
                resolved_pages.append((rid, page))

        t_audit = time.perf_counter()
        findings, page_cache, audit_usage = await self.auditor.audit_pages_batch(
            query=request.query,
            pages=resolved_pages,
            query_vectors=query_vectors,
            api_key=api_key,
        )
        audit_ms = (time.perf_counter() - t_audit) * 1000.0
        total_usage = total_usage.add(audit_usage)

        t_guard = time.perf_counter()
        selected_evidence, g_checked, g_reach, blocked = await self._guard_selected_evidence(
            findings=findings,
            session_id=session_id,
            client_ip=client_ip,
        )
        guard_ms = (time.perf_counter() - t_guard) * 1000.0

        return WebSearchOutcome(
            triggered=True,
            sufficient_locally=False,
            sufficiency_reason=reason,
            searches_performed=len(effective_queries[:3]),
            queries_used=effective_queries[:3],
            findings=findings,
            selected_evidence=selected_evidence,
            page_cache=page_cache,
            guard_checked=g_checked,
            guard_reachable=g_reach,
            guard_blocked_count=blocked,
            stage_latencies_ms={
                "tavily_ms": round(tavily_ms_max, 2),
                "fetch_fallback_ms": round(fetch_ms_max, 2),
                "audit_ms": round(audit_ms, 2),
                "guard_ms": round(guard_ms, 2),
            },
            usage=total_usage,
        )

    async def run_decision_and_search_async(
        self,
        *,
        request: QueryRequest,
        retrieved_chunks: list[dict[str, Any]],
        retrieved_tables: list[dict[str, Any]],
        seed_node_details: list[dict[str, Any]],
        ranked_edges: list[dict[str, Any]],
        search_query: str | None = None,
        search_queries: list[str] | None = None,
        query_vectors: list[Any] | None = None,
        session_id: str,
        client_ip: str | None = None,
        api_key: str | None = None,
    ) -> WebSearchOutcome:
        """Run OpenAI Decisions (``gpt-6-luna``) on local context; if insufficient, run 1-shot search."""
        if self.search_client is None or not self.search_client.is_configured:
            return WebSearchOutcome(
                triggered=False,
                sufficient_locally=True,
                sufficiency_reason="Tavily search API key not configured.",
            )

        local_summary = _format_local_evidence_summary(
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            seed_node_details=seed_node_details,
            ranked_edges=ranked_edges,
        )
        decision = await self.decision_client.evaluate_sufficiency(
            query=request.query,
            local_context=local_summary,
            api_key=api_key,
        )
        if decision.sufficient:
            return WebSearchOutcome(
                triggered=False,
                sufficient_locally=True,
                sufficiency_reason=(
                    f"Decision model ({self.decision_client.decision_model}) marked local context "
                    f"sufficient (p={decision.sufficiency_prob:.2f})."
                ),
                usage=decision.usage,
            )

        effective_queries = [
            q.strip()
            for q in (search_queries or ([search_query] if search_query else []))
            if q and q.strip()
        ]
        plan_usage = LLMUsage()
        if not effective_queries:
            plan, plan_usage = await self._plan_query(
                query=request.query,
                local_summary=local_summary,
                default_tier=decision.search_tier if self.include_regional else "dominant",
                api_key=api_key,
            )
            effective_queries = [plan.search_query]

        outcome = await self.run_one_shot_search_async(
            request=request,
            search_queries=effective_queries,
            query_vectors=query_vectors,
            tier=decision.search_tier if self.include_regional else "dominant",
            session_id=session_id,
            client_ip=client_ip,
            api_key=api_key,
            reason=(
                f"Decision model marked local context insufficient "
                f"(p={decision.sufficiency_prob:.2f}, tier={decision.search_tier})."
            ),
        )
        outcome.usage = outcome.usage.add(decision.usage).add(plan_usage)
        return outcome

    async def run_async(
        self,
        *,
        request: QueryRequest,
        retrieved_chunks: list[dict[str, Any]],
        retrieved_tables: list[dict[str, Any]],
        seed_node_details: list[dict[str, Any]],
        ranked_edges: list[dict[str, Any]],
        session_id: str,
        client_ip: str | None = None,
        api_key: str | None = None,
    ) -> WebSearchOutcome:
        """Alias for ``run_decision_and_search_async``."""
        return await self.run_decision_and_search_async(
            request=request,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            seed_node_details=seed_node_details,
            ranked_edges=ranked_edges,
            session_id=session_id,
            client_ip=client_ip,
            api_key=api_key,
        )
