"""Live end-to-end CLI script for ingesting documents and querying the clinical knowledge graph."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
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
from clinical_graph.llm import LiteLLMClient
from clinical_graph.pipelines import IngestionPipeline, QueryPipeline
from clinical_graph.schemas.query import QueryRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live ingestion and/or GraphRAG queries against local data/ persistence."
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        default=None,
        help="Path to a .pdf, .md, or .txt file to classify, extract, and persist.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for --ingest (defaults to derived filename stem).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Clinical query string to run against the loaded knowledge graph.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "rag", "hybrid"],
        default="auto",
        help='Retrieval mode for --query (default: "auto").',
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Write JSON and Markdown dev reports into --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/results"),
        help='Directory for --dev report outputs (default: "test_data/results").',
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
    print(f"[init] Loading persisted graph from {graph_dir}/... done ({ms:.2f} ms)", flush=True)
    return manager


def _render_metrics_markdown(metrics: dict[str, Any], heading: str) -> list[str]:
    """Render step-level latencies (with ms units) and LLM usage as Markdown lines."""
    usage = metrics["usage"]
    lines: list[str] = [
        heading,
        "",
        "#### Latencies (ms)",
        "",
    ]
    for step, latency in metrics["latencies_ms"].items():
        lines.append(f"- `{step}`: {latency:.2f} ms")
    lines.extend(
        [
            "",
            "#### LLM Usage",
            "",
            f"- **Prompt Tokens:** {usage['prompt_tokens']}",
            f"- **Completion Tokens:** {usage['completion_tokens']}",
            f"- **Total Tokens:** {usage['total_tokens']}",
            f"- **Estimated Cost (USD):** ${usage['estimated_cost_usd']:.6f}",
            "",
        ]
    )
    return lines


def _render_markdown_report(report: dict[str, Any]) -> str:
    """Render a structured Markdown summary from a dev E2E report payload."""
    lines: list[str] = [
        "# Clinical Graph Live E2E Report",
        "",
        f"- **Timestamp (UTC):** {report['timestamp']}",
        "",
    ]

    ingest = report.get("ingest")
    if ingest is not None:
        meta = ingest["metadata"]
        counts = ingest["counts"]
        lines.extend(
            [
                "## Ingestion",
                "",
                f"- **Input File:** `{ingest['input_file']}`",
                f"- **Doc ID:** `{ingest['doc_id']}`",
                f"- **Slug:** `{ingest['slug']}`",
                f"- **Title:** {meta['title']}",
                f"- **Doc Type:** `{meta['doc_type']}`",
                f"- **Source Type:** `{meta['source_type']}`",
                f"- **Local Restriction:** `{meta['local_restriction']}`",
                f"- **Chunks:** {counts['chunks']}",
                f"- **Omissions:** {counts['omissions']}",
                f"- **Extracted Nodes:** {ingest.get('extracted_node_count', counts.get('extracted_node_count', 0))}",
                f"- **Extracted Edges:** {ingest.get('extracted_edge_count', counts.get('extracted_edge_count', 0))}",
                f"- **Graph Nodes:** {ingest.get('graph_nodes', counts['graph_nodes'])}",
                f"- **Graph Edges:** {ingest.get('graph_edges', counts['graph_edges'])}",
                "",
            ]
        )
        if ingest.get("metrics") is not None:
            lines.extend(_render_metrics_markdown(ingest["metrics"], "### Ingestion Metrics"))

    query = report.get("query")
    if query is not None:
        lines.extend(
            [
                "## Query",
                "",
                f"- **Query:** {query['query']}",
                f"- **Session ID:** `{query['session_id']}`",
                f"- **Requested Mode:** `{query.get('requested_mode', 'auto')}`",
                f"- **Chosen Mode:** `{query.get('chosen_mode')}`",
                f"- **Used Web Fallback:** {query['used_web_fallback']}",
                f"- **Audit Passed:** {query['audit_passed']}",
                "",
                "### Answer",
                "",
                query["answer"],
                "",
                f"### Citations ({len(query['citations'])})",
                "",
            ]
        )
        for idx, citation in enumerate(query["citations"], start=1):
            lines.append(
                f"{idx}. **{citation['title']}** (`{citation['doc_id']}`, "
                f"type=`{citation['doc_type']}`, restriction=`{citation['local_restriction']}`)"
            )
            passages = citation.get("passages") or []
            if passages:
                for p in passages:
                    cid_label = f"`{p['chunk_id']}` " if p.get("chunk_id") else ""
                    lines.append(f"   - {cid_label}\"{p['quote']}\"")
                    if p.get("highlighted_chunk"):
                        quoted_chunk = "\n     > ".join(
                            p["highlighted_chunk"].splitlines()
                        )
                        lines.extend(
                            [
                                "     <details>",
                                "     <summary>Highlighted chunk</summary>",
                                "",
                                f"     > {quoted_chunk}",
                                "",
                                "     </details>",
                            ]
                        )
            else:
                excerpts = citation.get("excerpts") or (
                    [citation["excerpt"]] if citation.get("excerpt") else []
                )
                for ex in excerpts:
                    lines.append(f"   - Excerpt: {ex}")
        lines.append("")

        trace = query.get("trace")
        if trace is not None:
            lines.extend(
                [
                    "### Trace",
                    "",
                    f"- **Search Queries:** {trace['search_queries']}",
                    f"- **Seed Nodes:** {trace['seed_nodes']}",
                    f"- **Traversed Edges:** {len(trace['traversed_edges'])}",
                    f"- **Matched Doc IDs:** {trace['matched_doc_ids']}",
                    "",
                ]
            )
            if trace.get("assembled_context"):
                lines.extend(
                    [
                        "## Context Passed to AnswerAgent",
                        "",
                        "```text",
                        trace["assembled_context"],
                        "```",
                        "",
                    ]
                )

        metrics = query.get("metrics")
        if metrics is not None:
            lines.extend(_render_metrics_markdown(metrics, "### Pipeline Metrics"))

    return "\n".join(lines)


def _write_dev_reports(output_dir: Path, report: dict[str, Any], file_stamp: str) -> tuple[Path, Path]:
    """Write JSON and Markdown dev reports to output_dir and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"e2e_{file_stamp}.json"
    md_path = output_dir / f"e2e_{file_stamp}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    manager = _build_manager()
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    now = datetime.now(UTC)
    ingest_report: dict[str, Any] | None = None
    query_report: dict[str, Any] | None = None

    if args.ingest is not None:
        input_path: Path = args.ingest
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        classifier = DocumentClassifier(
            llm_client=llm_client,
            model=settings.classifier_model,
        )
        extractor = GraphExtractor(
            llm_client=llm_client,
            node_model=settings.node_extraction_model,
            edge_model=settings.edge_extraction_model,
        )
        ingestion = IngestionPipeline(
            manager=manager,
            classifier=classifier,
            extractor=extractor,
        )
        title = args.title or input_path.stem.replace("_", " ").replace("-", " ").title()
        ingest_result = await ingestion.ingest_file_bytes_with_report(
            input_path.read_bytes(),
            input_path.name,
            title=title,
            classify=True,
            extract=True,
            api_key=settings.openai_api_key,
            on_step=lambda msg: print(msg, flush=True),
        )
        record = ingest_result.record
        graph_nodes = manager.graph.graph.number_of_nodes()
        graph_edges = manager.graph.graph.number_of_edges()
        counts = {
            "chunks": len(record.chunk_ids),
            "omissions": len(record.omissions),
            "extracted_node_count": ingest_result.extracted_node_count,
            "extracted_edge_count": ingest_result.extracted_edge_count,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
        }
        ingest_report = {
            "input_file": str(input_path),
            "doc_id": record.doc_id,
            "slug": record.slug,
            "metadata": record.metadata.model_dump(mode="json"),
            "record": record.model_dump(mode="json"),
            "counts": counts,
            "omissions": [o.model_dump(mode="json") for o in record.omissions],
            "extracted_node_count": ingest_result.extracted_node_count,
            "extracted_edge_count": ingest_result.extracted_edge_count,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "metrics": ingest_result.metrics.model_dump(mode="json"),
        }
        print("=== INGESTION COMPLETE ===")
        print(f"doc_id:          {record.doc_id}")
        print(f"slug:            {record.slug}")
        print(f"doc_type:        {record.metadata.doc_type.value}")
        print(f"chunks:          {counts['chunks']}")
        print(f"tables:          {len(record.tables)}")
        print(f"omissions:       {counts['omissions']}")
        print(f"extracted_nodes: {ingest_result.extracted_node_count}")
        print(f"extracted_edges: {ingest_result.extracted_edge_count}")
        print(f"graph_nodes:     {graph_nodes}")
        print(f"graph_edges:     {graph_edges}")
        print(f"vectors:         {manager.graph.vector_store.count()}")
        ingest_latencies = ", ".join(
            f"{k}={v:.2f} ms" for k, v in ingest_result.metrics.latencies_ms.items()
        )
        print(f"latencies:       {ingest_latencies}")
        print(f"usage:           {ingest_result.metrics.usage.model_dump()}")
        print()

    if args.query is not None:
        query_pipeline = QueryPipeline(
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
        response = await query_pipeline.run(
            QueryRequest(
                session_id="live-e2e",
                query=args.query,
                k_hops=settings.graph_k_hops,
                include_trace=True,
                retrieval_mode=args.mode,
            ),
            api_key=settings.openai_api_key,
            on_step=lambda msg: print(msg, flush=True),
        )
        query_report = {
            "query": args.query,
            "session_id": response.session_id,
            "requested_mode": args.mode,
            "chosen_mode": response.chosen_mode,
            "answer": response.answer,
            "used_web_fallback": response.used_web_fallback,
            "audit_passed": response.audit_passed,
            "citations": [c.model_dump(mode="json") for c in response.citations],
            "trace": response.trace.model_dump(mode="json") if response.trace is not None else None,
            "metrics": response.metrics.model_dump(mode="json") if response.metrics is not None else None,
        }
        print("=== QUERY RESPONSE ===")
        print(f"chosen_mode: {response.chosen_mode}")
        print(response.answer)
        print()
        print(f"=== CITATIONS ({len(response.citations)}) ===")
        for idx, citation in enumerate(response.citations, start=1):
            print(
                f"{idx}. {citation.title} ({citation.doc_id}, "
                f"type={citation.doc_type.value}, restriction={citation.local_restriction.value})"
            )
            excerpts = citation.excerpts or ([citation.excerpt] if citation.excerpt else [])
            for ex in excerpts:
                print(f"   excerpt: {ex}")
        print()
        if response.trace is not None:
            print("=== TRACE SUMMARY ===")
            print(f"search_queries:  {response.trace.search_queries}")
            print(f"seed_nodes:      {response.trace.seed_nodes}")
            print(f"traversed_edges: {len(response.trace.traversed_edges)}")
            print(f"matched_doc_ids: {response.trace.matched_doc_ids}")
            print()
        if response.metrics is not None:
            print("=== PIPELINE METRICS ===")
            query_latencies = ", ".join(
                f"{k}={v:.2f} ms" for k, v in response.metrics.latencies_ms.items()
            )
            print(f"latencies: {query_latencies}")
            print(f"usage:     {response.metrics.usage.model_dump()}")
            print()

    if args.dev:
        report = {
            "timestamp": now.isoformat(),
            "ingest": ingest_report,
            "query": query_report,
        }
        file_stamp = now.strftime("%Y%m%d_%H%M%S")
        json_path, md_path = _write_dev_reports(args.output_dir, report, file_stamp)
        print("=== DEV REPORTS WRITTEN ===")
        print(f"json:     {json_path}")
        print(f"markdown: {md_path}")


def main() -> None:
    """Parse CLI arguments and run the live end-to-end pipeline."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.ingest is None and args.query is None:
        parser.error("Provide at least one of --ingest or --query.")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
