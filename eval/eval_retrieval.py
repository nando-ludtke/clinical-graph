"""Unit eval for local GraphRAG retrieval (no AnswerAgent synthesis or web search)."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

from clinical_graph.agents import AnswerAgent, QueryRewriter
from clinical_graph.config import get_settings
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    KnowledgeManager,
    LiteLLMEmbedder,
    LocalVectorStore,
)
from clinical_graph.llm import LiteLLMClient
from clinical_graph.pipelines import QueryPipeline
from clinical_graph.schemas.query import QueryRequest, QueryResponse


class _NoOpAnswerAgent(AnswerAgent):
    """Skips LLM answer generation so retrieval can be benchmarked in isolation."""

    async def synthesize_with_usage_async(self, request: QueryRequest, **kwargs: Any) -> tuple[QueryResponse, Any]:
        from clinical_graph.schemas.metrics import LLMUsage

        return (
            QueryResponse(
                session_id=request.session_id,
                answer="[retrieval-only]",
                chosen_mode=kwargs.get("chosen_mode"),
                citations=kwargs.get("citations") or [],
                trace=kwargs.get("trace"),
            ),
            LLMUsage(),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local GraphRAG retrieval in isolation.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/unit_retrieval_cases.json"),
        help='Path to retrieval test cases JSON (default: "eval/unit_retrieval_cases.json").',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help='Directory for JSON results (default: "eval/results").',
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = get_settings()
    embedder = LiteLLMEmbedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    vector_store = LocalVectorStore(embedder=embedder)
    graph = ClinicalKnowledgeGraph(vector_store=vector_store)
    manager = KnowledgeManager(
        graph=graph,
        storage_dir=settings.document_storage_dir or "data/documents",
        graph_dir=settings.graph_storage_dir or "data/graph",
    )
    manager.load()

    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(llm_client=llm_client, model=settings.query_rewrite_model),
        answer_agent=_NoOpAnswerAgent(),
    )

    results: list[dict[str, Any]] = []
    passed = 0
    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        query = case["query"]
        mode = case.get("mode", "auto")
        t0 = time.perf_counter()
        resp = await pipeline.run(
            QueryRequest(
                session_id=f"ret-{cid}",
                query=query,
                k_hops=settings.graph_k_hops,
                allow_web_fallback=False,
                include_trace=True,
                retrieval_mode=mode,
            ),
            api_key=settings.openai_api_key,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        ctx = (resp.trace.assembled_context if resp.trace else "") or ""
        ctx_low = ctx.lower()
        terms = case.get("expected_terms") or []
        hits = [t for t in terms if t.lower() in ctx_low]
        ok = len(hits) == len(terms)
        if ok:
            passed += 1

        stage_latencies = resp.metrics.latencies_ms if resp.metrics else {}
        print(
            f"[{idx}/{len(cases)}] {cid} (mode={mode} -> {resp.chosen_mode}): "
            f"ok={ok} term_hits={len(hits)}/{len(terms)} "
            f"seeds={len(resp.trace.seed_nodes) if resp.trace else 0} "
            f"edges={len(resp.trace.traversed_edges) if resp.trace else 0} "
            f"latencies={stage_latencies}",
            flush=True,
        )
        results.append(
            {
                "id": cid,
                "query": query,
                "requested_mode": mode,
                "chosen_mode": resp.chosen_mode,
                "passed": ok,
                "matched_terms": hits,
                "seed_nodes": resp.trace.seed_nodes if resp.trace else [],
                "edge_count": len(resp.trace.traversed_edges) if resp.trace else 0,
                "stage_latencies_ms": stage_latencies,
                "latency_ms": round(ms, 2),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"unit_retrieval_{stamp}.json"
    out_path.write_text(
        json.dumps({"passed": passed, "total": len(cases), "results": results}, indent=2),
        encoding="utf-8",
    )
    md_lines = [
        f"# Unit Retrieval Evaluation (`{stamp}`)",
        "",
        f"- **Score:** {passed}/{len(cases)} passed",
        "",
        "| ID | Status | Mode | Seeds | Edges | Matched Terms | Total (ms) |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        mode_str = f"{r['requested_mode']} -> {r['chosen_mode']}"
        terms_str = ", ".join(r["matched_terms"]) or "—"
        md_lines.append(
            f"| `{r['id']}` | {status} | {mode_str} | {len(r['seed_nodes'])} | "
            f"{r['edge_count']} | {terms_str} | {r['latency_ms']:.1f} |"
        )
    md_path = args.output_dir / f"unit_retrieval_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"=== RETRIEVAL EVAL: {passed}/{len(cases)} passed -> {out_path} ({md_path}) ===", flush=True)


def main() -> None:
    asyncio.run(_run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
