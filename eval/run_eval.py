"""Benchmark evaluation runner comparing rag, hybrid, and auto retrieval modes."""

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
from clinical_graph.schemas.query import QueryRequest, RetrievalStrategy

MODES: tuple[RetrievalStrategy, ...] = ("rag", "hybrid", "auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval mode evaluation (rag, hybrid, auto) against persisted data/graph/."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("eval/diabetes_queries.json"),
        help='Path to evaluation queries JSON (default: "eval/diabetes_queries.json").',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help='Directory for eval JSON and Markdown outputs (default: "eval/results").',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries to evaluate.",
    )
    return parser


def _build_manager() -> KnowledgeManager:
    t0 = time.perf_counter()
    settings = get_settings()
    embedder = LiteLLMEmbedder(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = LocalVectorStore(embedder=embedder)
    graph = ClinicalKnowledgeGraph(vector_store=vector_store)
    graph_dir = settings.graph_storage_dir or "data/graph"
    manager = KnowledgeManager(
        graph=graph,
        storage_dir=settings.document_storage_dir or "data/documents",
        graph_dir=graph_dir,
    )
    manager.load()
    ms = (time.perf_counter() - t0) * 1000.0
    print(f"[init] Loaded graph from {graph_dir}/ ({ms:.2f} ms)", flush=True)
    return manager


def _load_queries(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        items = items[:limit]
    return items


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines: list[str] = [
        "# Clinical Graph Retrieval Mode Evaluation",
        "",
        f"- **Timestamp (UTC):** {report['timestamp']}",
        f"- **Queries File:** `{report['queries_file']}`",
        f"- **Total Queries:** {summary['total_queries']}",
        f"- **Auto Routing Matches:** {summary['auto_matches']} / {summary['total_queries']} ({summary['auto_accuracy_pct']:.1f}%)",
        "",
        "## Summary by Mode",
        "",
        "| Mode | Avg Latency (ms) | Total Tokens | Total Cost ($) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        m_sum = summary["by_mode"][mode]
        lines.append(
            f"| `{mode}` | {m_sum['avg_latency_ms']:.2f} | {m_sum['total_tokens']} | ${m_sum['total_cost_usd']:.6f} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Per-Query Routing & Metrics",
            "",
            "| ID | Expected | Auto Chosen | Match | RAG (ms / tok / $) | Hybrid (ms / tok / $) | Auto (ms / tok / $) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["results"]:
        auto_res = row["runs"]["auto"]
        rag_res = row["runs"]["rag"]
        hybrid_res = row["runs"]["hybrid"]
        match_str = "YES" if row["auto_matches_expected"] else "NO"
        lines.append(
            f"| `{row['id']}` | `{row['expected_mode']}` | `{auto_res['chosen_mode']}` | {match_str} | "
            f"{rag_res['latency_ms']:.1f} / {rag_res['total_tokens']} / ${rag_res['cost_usd']:.5f} | "
            f"{hybrid_res['latency_ms']:.1f} / {hybrid_res['total_tokens']} / ${hybrid_res['cost_usd']:.5f} | "
            f"{auto_res['latency_ms']:.1f} / {auto_res['total_tokens']} / ${auto_res['cost_usd']:.5f} |"
        )
    lines.append("")

    lines.append("## Answers by Query")
    lines.append("")
    for row in report["results"]:
        lines.extend(
            [
                f"### `{row['id']}` (expected: `{row['expected_mode']}`, auto chosen: `{row['runs']['auto']['chosen_mode']}`)",
                "",
                f"**Query:** {row['query']}",
                "",
                f"**Rationale:** {row.get('rationale', '')}",
                "",
            ]
        )
        for mode in MODES:
            run_data = row["runs"][mode]
            seed_nodes = run_data.get("seed_nodes") or []
            traversed_edges = run_data.get("traversed_edges") or []
            citations = run_data.get("citations") or []
            assembled_context = run_data.get("assembled_context")

            seed_str = ", ".join(f"`{n}`" for n in seed_nodes) if seed_nodes else "None"
            lines.extend(
                [
                    f"#### Mode: `{mode}` (chosen: `{run_data['chosen_mode']}`, {run_data['latency_ms']:.2f} ms, {run_data['total_tokens']} tok, ${run_data['cost_usd']:.6f})",
                    "",
                    run_data["answer"],
                    "",
                    f"**Seed Nodes:** {seed_str}",
                    "",
                    f"**Traversed Edges:** {len(traversed_edges)}",
                    "",
                ]
            )
            if citations:
                lines.append("**Citations:**")
                for c in citations:
                    passages = c.get("passages") or []
                    if passages:
                        lines.append(f"- **{c['title']}** (`{c['doc_id']}`):")
                        for p in passages:
                            cid_label = f"`{p['chunk_id']}` " if p.get("chunk_id") else ""
                            lines.append(f"  - {cid_label}\"{p['quote']}\"")
                            if p.get("highlighted_chunk"):
                                quoted_chunk = "\n    > ".join(
                                    p["highlighted_chunk"].splitlines()
                                )
                                lines.extend(
                                    [
                                        "    <details>",
                                        "    <summary>Highlighted chunk</summary>",
                                        "",
                                        f"    > {quoted_chunk}",
                                        "",
                                        "    </details>",
                                    ]
                                )
                    else:
                        excerpts = c.get("excerpts") or (
                            [c["excerpt"]] if c.get("excerpt") else []
                        )
                        if len(excerpts) > 1:
                            lines.append(f"- **{c['title']}** (`{c['doc_id']}`):")
                            for ex in excerpts:
                                lines.append(f"  - {ex}")
                        elif excerpts:
                            lines.append(f"- **{c['title']}** (`{c['doc_id']}`): {excerpts[0]}")
                        else:
                            lines.append(f"- **{c['title']}** (`{c['doc_id']}`)")
                lines.append("")
            else:
                lines.extend(["**Citations:** None", ""])

            if assembled_context:
                lines.extend(
                    [
                        "<details>",
                        "<summary>Context Passed to AnswerAgent (Nodes, Edges & Passages)</summary>",
                        "",
                        "```text",
                        assembled_context,
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> None:
    queries = _load_queries(args.queries, args.limit)
    settings = get_settings()
    manager = _build_manager()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    pipeline = QueryPipeline(
        manager=manager,
        query_rewriter=QueryRewriter(
            llm_client=llm_client,
            model=settings.query_rewrite_model,
        ),
        answer_agent=AnswerAgent(
            llm_client=llm_client,
            model=settings.answer_generation_model,
        ),
    )

    results: list[dict[str, Any]] = []
    total = len(queries)
    for idx, item in enumerate(queries, start=1):
        qid = item["id"]
        query_text = item["query"]
        expected_mode = item["expected_mode"]
        print(f"[{idx}/{total}] {qid} (expected={expected_mode}): {query_text}", flush=True)

        runs: dict[str, Any] = {}
        for mode in MODES:
            response = await pipeline.run(
                QueryRequest(
                    session_id=f"eval-{qid}-{mode}",
                    query=query_text,
                    k_hops=settings.graph_k_hops,
                    include_trace=True,
                    retrieval_mode=mode,
                ),
                api_key=settings.openai_api_key,
            )
            metrics = response.metrics
            latency_ms = metrics.latencies_ms.get("total", 0.0) if metrics else 0.0
            usage = metrics.usage if metrics else None
            total_tokens = usage.total_tokens if usage else 0
            cost_usd = usage.estimated_cost_usd if usage else 0.0
            runs[mode] = {
                "requested_mode": mode,
                "chosen_mode": response.chosen_mode,
                "answer": response.answer,
                "citation_count": len(response.citations),
                "citations": [c.model_dump(mode="json") for c in response.citations],
                "search_queries": response.trace.search_queries if response.trace else [],
                "seed_nodes": response.trace.seed_nodes if response.trace else [],
                "traversed_edges": response.trace.traversed_edges if response.trace else [],
                "assembled_context": response.trace.assembled_context if response.trace else None,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_usd, 6),
            }
            print(
                f"  - {mode}: chosen={response.chosen_mode}, "
                f"latency={latency_ms:.1f} ms, tokens={total_tokens}, cost=${cost_usd:.6f}",
                flush=True,
            )

        auto_match = runs["auto"]["chosen_mode"] == expected_mode
        results.append(
            {
                "id": qid,
                "query": query_text,
                "expected_mode": expected_mode,
                "rationale": item.get("rationale", ""),
                "auto_matches_expected": auto_match,
                "runs": runs,
            }
        )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        latencies = [r["runs"][mode]["latency_ms"] for r in results]
        tokens = sum(r["runs"][mode]["total_tokens"] for r in results)
        costs = sum(r["runs"][mode]["cost_usd"] for r in results)
        by_mode[mode] = {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "total_tokens": tokens,
            "total_cost_usd": round(costs, 6),
        }

    auto_matches = sum(1 for r in results if r["auto_matches_expected"])
    now = datetime.now(UTC)
    report: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "queries_file": str(args.queries),
        "summary": {
            "total_queries": len(results),
            "auto_matches": auto_matches,
            "auto_accuracy_pct": round((auto_matches / len(results)) * 100.0, 2) if results else 0.0,
            "by_mode": by_mode,
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"eval_{stamp}.json"
    md_path = args.output_dir / f"eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print("=== EVAL COMPLETE ===", flush=True)
    print(f"auto_routing_accuracy: {auto_matches}/{len(results)} ({report['summary']['auto_accuracy_pct']:.1f}%)", flush=True)
    print(f"json:     {json_path}", flush=True)
    print(f"markdown: {md_path}", flush=True)


def main() -> None:
    """Parse CLI arguments and run the retrieval mode evaluation."""
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
