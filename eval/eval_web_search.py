"""Unit eval for isolated 1-shot Web Search + Source Auditor + optional Prompt Guard."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

from clinical_graph.agents import QueryRewriter
from clinical_graph.config import get_settings
from clinical_graph.graph import LiteLLMEmbedder
from clinical_graph.guardrails.prompt_guard import ClinicalGuardClient
from clinical_graph.llm import LiteLLMClient
from clinical_graph.schemas.query import QueryRequest
from clinical_graph.search import SourceAuditorAgent, TavilySearchClient, WebSearchAgent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate isolated 1-shot web search & auditor.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("eval/unit_web_search_cases.json"),
        help='Path to queries JSON (default: "eval/unit_web_search_cases.json").',
    )
    parser.add_argument("--allow-pdf", action="store_true", help="Enable PDF fetching & parsing.")
    parser.add_argument("--prompt-guard", action="store_true", help="Run ClinicalGuardClient on final chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max queries to run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help='Directory for JSON results (default: "eval/results").',
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]
    settings = get_settings()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    rewriter = QueryRewriter(llm_client=llm_client, model=settings.query_rewrite_model)
    guard_client = (
        ClinicalGuardClient(
            base_url=settings.clinical_guard_url,
            timeout=settings.clinical_guard_timeout_seconds,
        )
        if args.prompt_guard and settings.clinical_guard_url
        else None
    )
    search_client = TavilySearchClient(
        api_key=settings.tavily_api_key,
        timeout=settings.web_fetch_timeout_seconds,
    )
    embedder = LiteLLMEmbedder(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    auditor = SourceAuditorAgent(
        embedder=embedder,
        pdf_preview_chars=settings.web_pdf_preview_chars,
    )
    agent = WebSearchAgent(
        llm_client=llm_client,
        search_client=search_client,
        auditor=auditor,
        clinical_guard_client=guard_client,
        model=settings.web_search_model,
        allow_pdf=args.allow_pdf or settings.enable_web_pdf_fetch,
        include_regional=settings.include_regional_domains,
        results_per_tier=settings.web_search_results_per_tier,
        results_per_tier_with_pdf=settings.web_search_results_per_tier_with_pdf,
        fetch_timeout=settings.web_fetch_timeout_seconds,
        max_pdf_pages=settings.web_pdf_max_pages,
    )

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        query = case["query"]
        t0 = time.perf_counter()
        req = QueryRequest(session_id=f"web-{cid}", query=query)
        t_rw = time.perf_counter()
        retrieval_queries, _, _, _, web_queries, _ = await rewriter.rewrite_with_web_async(
            req, include_web_query=True, api_key=settings.openai_api_key
        )
        rewrite_ms = round((time.perf_counter() - t_rw) * 1000.0, 2)
        qvecs = embedder.embed_many(retrieval_queries, api_key=settings.openai_api_key)
        outcome = await agent.run_one_shot_search_async(
            request=req,
            search_queries=web_queries,
            query_vectors=qvecs,
            session_id=f"web-{cid}",
            api_key=settings.openai_api_key,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        stage_latencies = {"rewrite_ms": rewrite_ms, **outcome.stage_latencies_ms, "total_ms": round(ms, 2)}
        print(
            f"[{idx}/{len(cases)}] {cid}: query={outcome.queries_used} "
            f"sources={len(outcome.selected_evidence)}/{len(outcome.findings)} "
            f"latencies={stage_latencies}",
            flush=True,
        )
        results.append(
            {
                "id": cid,
                "query": query,
                "planned_queries": outcome.queries_used,
                "findings_count": len(outcome.findings),
                "selected_count": len(outcome.selected_evidence),
                "guard_checked": outcome.guard_checked,
                "guard_blocked_count": outcome.guard_blocked_count,
                "selected_evidence": [item.model_dump(mode="json") for item in outcome.selected_evidence],
                "findings": [f.model_dump(mode="json") for f in outcome.findings],
                "stage_latencies_ms": stage_latencies,
                "latency_ms": round(ms, 2),
                "tokens": outcome.usage.total_tokens,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"unit_web_search_{stamp}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    md_lines = [
        f"# Unit Web Search Evaluation (`{stamp}`)",
        "",
        f"- **Cases:** {len(results)}",
        "",
        "| ID | Planned Queries | Selected / Findings | Guard Blocked | Tokens | Total (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        q_str = "; ".join(r["planned_queries"]) or "—"
        md_lines.append(
            f"| `{r['id']}` | {q_str} | {r['selected_count']}/{r['findings_count']} | "
            f"{r['guard_blocked_count']} | {r['tokens']} | {r['latency_ms']:.1f} |"
        )
    md_path = args.output_dir / f"unit_web_search_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"=== WEB SEARCH EVAL COMPLETE -> {out_path} ({md_path}) ===", flush=True)


def main() -> None:
    asyncio.run(_run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
