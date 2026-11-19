"""Benchmark evaluation runner comparing rag, hybrid, and auto retrieval modes."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from clinical_graph.agents import AnswerAgent, DocumentClassifier, GraphExtractor, QueryRewriter
from clinical_graph.config import get_settings
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    KnowledgeManager,
    LiteLLMEmbedder,
    LocalVectorStore,
)
from clinical_graph.guardrails.prompt_guard import ClinicalGuardClient
from clinical_graph.llm import LiteLLMClient
from clinical_graph.parser import DocumentParser
from clinical_graph.pipelines import DocumentRejectedError, IngestionPipeline, QueryPipeline
from clinical_graph.schemas.document import (
    DocType,
    DocumentCreate,
    DocumentMetadata,
    LocalRestriction,
    SourceType,
)
from clinical_graph.schemas.query import QueryRequest, RetrievalStrategy

MODES: tuple[RetrievalStrategy, ...] = ("rag", "hybrid", "auto")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end clinical-graph evaluation (--query or --ingestion)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--query",
        action="store_true",
        help="Run end-to-end query evaluation (default).",
    )
    mode_group.add_argument(
        "--ingestion",
        action="store_true",
        help="Run end-to-end document upload & ingestion evaluation in an isolated temp store.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("eval/e2e_diabetes_queries.json"),
        help='Path to evaluation queries JSON (default: "eval/e2e_diabetes_queries.json").',
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/unit_ingestion_cases.json"),
        help='Path to ingestion cases JSON when --ingestion is set (default: "eval/unit_ingestion_cases.json").',
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
        help="Optional maximum number of queries/cases to evaluate.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["rag", "hybrid", "auto"],
        default=["auto"],
        help='Retrieval modes to evaluate (default: "auto").',
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run all three modes (rag, hybrid, auto) for side-by-side comparison.",
    )
    parser.add_argument(
        "--web-search",
        "--web-fallback",
        dest="web_fallback",
        action="store_true",
        help="Force allow_web_fallback=True on every evaluation query.",
    )
    parser.add_argument(
        "--prompt-guard",
        action="store_true",
        help="Enable ClinicalGuardClient checks during --ingestion or --query.",
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
    modes: list[str] = report.get("modes") or list(summary["by_mode"].keys())
    has_auto = "auto" in modes
    lines: list[str] = [
        "# Clinical Graph Retrieval Mode Evaluation",
        "",
        f"- **Timestamp (UTC):** {report['timestamp']}",
        f"- **Queries File:** `{report['queries_file']}`",
        f"- **Modes:** {', '.join(f'`{m}`' for m in modes)}",
        f"- **Total Queries:** {summary['total_queries']}",
    ]
    if has_auto:
        lines.append(
            f"- **Auto Routing Matches:** {summary['auto_matches']} / {summary['total_queries']} ({summary['auto_accuracy_pct']:.1f}%)"
        )
    lines.extend(
        [
            "",
            "## Summary by Mode",
            "",
            "| Mode | Avg Latency (ms) | Total Tokens | Total Cost ($) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for mode in modes:
        m_sum = summary["by_mode"][mode]
        lines.append(
            f"| `{mode}` | {m_sum['avg_latency_ms']:.2f} | {m_sum['total_tokens']} | ${m_sum['total_cost_usd']:.6f} |"
        )
    lines.append("")

    mode_headers = " | ".join(f"{m.upper()} (ms / tok / $)" for m in modes)
    mode_seps = " | ".join("---" for _ in modes)
    auto_cols = " Auto Chosen | Match |" if has_auto else ""
    auto_seps = " --- | --- |" if has_auto else ""
    lines.extend(
        [
            "## Per-Query Routing & Metrics",
            "",
            f"| ID | Expected |{auto_cols} {mode_headers} |",
            f"| --- | --- |{auto_seps} {mode_seps} |",
        ]
    )
    for row in report["results"]:
        mode_cells = " | ".join(
            f"{row['runs'][m]['latency_ms']:.1f} / {row['runs'][m]['total_tokens']} / ${row['runs'][m]['cost_usd']:.5f}"
            for m in modes
        )
        if has_auto:
            auto_res = row["runs"]["auto"]
            match_str = "YES" if row["auto_matches_expected"] else "NO"
            auto_str = f" `{auto_res['chosen_mode']}` | {match_str} |"
        else:
            auto_str = ""
        lines.append(f"| `{row['id']}` | `{row['expected_mode']}` |{auto_str} {mode_cells} |")
    lines.append("")

    lines.append("## Answers by Query")
    lines.append("")
    for row in report["results"]:
        auto_note = (
            f", auto chosen: `{row['runs']['auto']['chosen_mode']}`"
            if has_auto
            else ""
        )
        lines.extend(
            [
                f"### `{row['id']}` (expected: `{row['expected_mode']}`{auto_note})",
                "",
                f"**Query:** {row['query']}",
                "",
                f"**Rationale:** {row.get('rationale', '')}",
                "",
            ]
        )
        for mode in modes:
            run_data = row["runs"][mode]
            seed_nodes = run_data.get("seed_nodes") or []
            traversed_edges = run_data.get("traversed_edges") or []
            chunk_source_counts = run_data.get("chunk_source_counts") or {}
            citations = run_data.get("citations") or []
            assembled_context = run_data.get("assembled_context")
            used_web = bool(run_data.get("used_web_fallback", False))
            web_queries = run_data.get("web_search_queries") or []
            web_sources = run_data.get("web_sources") or []

            seed_str = ", ".join(f"`{n}`" for n in seed_nodes) if seed_nodes else "None"
            chunk_sources_str = (
                f"rag={chunk_source_counts.get('rag', 0)}, "
                f"graph={chunk_source_counts.get('graph', 0)}, "
                f"both={chunk_source_counts.get('both', 0)}"
            )
            web_queries_str = ", ".join(f"`{q}`" for q in web_queries) if web_queries else "None"
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
                    f"**Chunk Sources:** {chunk_sources_str}",
                    "",
                    f"**Web Fallback:** `{used_web}` (search queries: {web_queries_str})",
                    "",
                ]
            )
            if web_sources:
                lines.append("**Web Sources:**")
                for ws in web_sources:
                    lines.append(
                        f"- [{ws.get('title', ws.get('url', ''))}]({ws.get('url', '')}) "
                        f"(`{ws.get('domain', '')}`, `{ws.get('content_type', 'html')}`)"
                    )
                lines.append("")
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


async def _run_ingestion(args: argparse.Namespace) -> None:
    cases = _load_queries(args.cases, args.limit)
    settings = get_settings()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    parser_tool = DocumentParser()
    guard_client = (
        ClinicalGuardClient(
            base_url=settings.clinical_guard_url,
            timeout=settings.clinical_guard_timeout_seconds,
        )
        if args.prompt_guard and settings.clinical_guard_url
        else None
    )

    results: list[dict[str, Any]] = []
    passed = 0
    with tempfile.TemporaryDirectory(prefix="clinical_graph_ingest_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        embedder = LiteLLMEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        vector_store = LocalVectorStore(embedder=embedder)
        graph = ClinicalKnowledgeGraph(vector_store=vector_store)
        manager = KnowledgeManager(
            graph=graph,
            storage_dir=str(tmp_path / "documents"),
            graph_dir=str(tmp_path / "graph"),
        )
        pipeline = IngestionPipeline(
            manager=manager,
            classifier=DocumentClassifier(llm_client=llm_client, model=settings.classifier_model),
            extractor=GraphExtractor(llm_client=llm_client),
        )

        for idx, case in enumerate(cases, start=1):
            cid = case["id"]
            file_path = Path(case["path"])
            parsed = parser_tool.parse_file(file_path)
            t0 = time.perf_counter()
            allowed = True
            blocked_reason: str | None = None
            node_count = 0
            edge_count = 0

            if guard_client is not None:
                guard_res = await guard_client.scan_document(
                    parsed.text,
                    extra_texts=[case.get("title", file_path.name)],
                    session_id=f"ingest-{cid}",
                )
                if guard_res.blocked:
                    allowed = False
                    blocked_reason = guard_res.reason or "Blocked by clinical-guard"

            if allowed:
                payload = DocumentCreate(
                    content=parsed.text,
                    metadata=DocumentMetadata(
                        title=case.get("title", file_path.stem),
                        doc_type=DocType(case.get("doc_type", "guideline")),
                        source_type=parsed.source_type or SourceType.RAW_TEXT,
                        local_restriction=LocalRestriction(case.get("local_restriction", "NONE")),
                    ),
                )
                try:
                    res = await pipeline.ingest_async(
                        payload,
                        classify=True,
                        extract=bool(case.get("extract", False)),
                        parsed_tables=parsed.table_entries,
                        api_key=settings.openai_api_key,
                    )
                    node_count = res.extracted_node_count
                    edge_count = res.extracted_edge_count
                except DocumentRejectedError as exc:
                    allowed = False
                    blocked_reason = exc.detail

            ms = (time.perf_counter() - t0) * 1000.0
            expected_allowed = bool(case.get("expected_allowed", True))
            ok = (allowed == expected_allowed) if args.prompt_guard else True
            if ok:
                passed += 1
            print(
                f"[{idx}/{len(cases)}] {cid}: allowed={allowed} (expected={expected_allowed}) "
                f"ok={ok} nodes={node_count} edges={edge_count} ({ms:.1f} ms)",
                flush=True,
            )
            results.append(
                {
                    "id": cid,
                    "path": str(file_path),
                    "allowed": allowed,
                    "expected_allowed": expected_allowed,
                    "passed": ok,
                    "blocked_reason": blocked_reason,
                    "extracted_node_count": node_count,
                    "extracted_edge_count": edge_count,
                    "latency_ms": round(ms, 2),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"eval_ingestion_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "prompt_guard": bool(args.prompt_guard),
                "passed": passed,
                "total": len(cases),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"=== INGESTION EVAL COMPLETE: {passed}/{len(cases)} -> {json_path} ===", flush=True)


async def _run(args: argparse.Namespace) -> None:
    if args.ingestion:
        await _run_ingestion(args)
        return
    modes: tuple[RetrievalStrategy, ...] = (
        MODES if args.compare_modes else tuple(dict.fromkeys(args.modes))  # type: ignore[assignment]
    )
    queries = _load_queries(args.queries, args.limit)
    settings = get_settings()
    manager = _build_manager()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    guard_client = (
        ClinicalGuardClient(
            base_url=settings.clinical_guard_url,
            timeout=settings.clinical_guard_timeout_seconds,
        )
        if args.prompt_guard and settings.clinical_guard_url
        else None
    )
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
        clinical_guard_client=guard_client,
    )

    results: list[dict[str, Any]] = []
    total = len(queries)
    for idx, item in enumerate(queries, start=1):
        qid = item["id"]
        query_text = item["query"]
        expected_mode = item["expected_mode"]
        print(f"[{idx}/{total}] {qid} (expected={expected_mode}): {query_text}", flush=True)

        runs: dict[str, Any] = {}
        for mode in modes:
            response = await pipeline.run(
                QueryRequest(
                    session_id=f"eval-{qid}-{mode}",
                    query=query_text,
                    k_hops=settings.graph_k_hops,
                    allow_web_fallback=True if args.web_fallback else None,
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
            chunk_sources = dict(response.trace.chunk_sources) if response.trace else {}
            chunk_source_counts = {
                src: sum(1 for v in chunk_sources.values() if v == src)
                for src in ("rag", "graph", "both")
            }
            runs[mode] = {
                "requested_mode": mode,
                "chosen_mode": response.chosen_mode,
                "answer": response.answer,
                "citation_count": len(response.citations),
                "citations": [c.model_dump(mode="json") for c in response.citations],
                "search_queries": response.trace.search_queries if response.trace else [],
                "seed_nodes": response.trace.seed_nodes if response.trace else [],
                "traversed_edges": response.trace.traversed_edges if response.trace else [],
                "chunk_sources": chunk_sources,
                "chunk_source_counts": chunk_source_counts,
                "used_web_fallback": response.used_web_fallback,
                "web_search_queries": response.trace.web_search_queries if response.trace else [],
                "web_sources": [ws.model_dump(mode="json") for ws in response.web_sources],
                "assembled_context": response.trace.assembled_context if response.trace else None,
                "stage_latencies_ms": dict(metrics.latencies_ms) if metrics else {},
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_usd, 6),
            }
            print(
                f"  - {mode}: chosen={response.chosen_mode}, "
                f"chunks(rag={chunk_source_counts['rag']}, graph={chunk_source_counts['graph']}, both={chunk_source_counts['both']}), "
                f"web={response.used_web_fallback} ({len(response.web_sources)} sources), "
                f"latency={latency_ms:.1f} ms, tokens={total_tokens}, cost=${cost_usd:.6f}",
                flush=True,
            )

        auto_match = (
            runs["auto"]["chosen_mode"] == expected_mode
            if "auto" in runs
            else None
        )
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
    for mode in modes:
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
        "modes": list(modes),
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
