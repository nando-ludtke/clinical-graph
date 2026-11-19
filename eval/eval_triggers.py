"""Unit eval for Web Search Triggers: Trigger 1 (node_label similarity) & Trigger 2 (gpt-6-luna)."""

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
from clinical_graph.graph import (
    ClinicalKnowledgeGraph,
    KnowledgeManager,
    LiteLLMEmbedder,
    LocalVectorStore,
)
from clinical_graph.llm import LiteLLMClient
from clinical_graph.schemas.query import QueryRequest
from clinical_graph.search import OpenAIDecisionClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate web search trigger paths.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/unit_triggers_cases.json"),
        help='Path to trigger test cases JSON (default: "eval/unit_triggers_cases.json").',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override web_label_similarity_threshold.",
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
    threshold = args.threshold if args.threshold is not None else settings.web_label_similarity_threshold

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
    rewriter = QueryRewriter(llm_client=llm_client, model=settings.query_rewrite_model)
    decision_client = OpenAIDecisionClient(
        decision_model=settings.decision_model,
        fallback_llm_client=llm_client,
        fallback_model=settings.web_search_model,
    )

    results: list[dict[str, Any]] = []
    passed = 0
    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        query = case["query"]
        req = QueryRequest(session_id=f"trig-{cid}", query=query)
        t0 = time.perf_counter()
        t_rw = time.perf_counter()
        search_queries, mode, _, _, _ = await rewriter.rewrite_with_gate_async(
            req, api_key=settings.openai_api_key
        )
        rewrite_ms = round((time.perf_counter() - t_rw) * 1000.0, 2)

        t_emb = time.perf_counter()
        qvecs = vector_store.embed_queries(search_queries, api_key=settings.openai_api_key)
        embed_ms = round((time.perf_counter() - t_emb) * 1000.0, 2)

        t_lbl = time.perf_counter()
        covered, min_score = graph.check_label_coverage(
            qvecs, threshold=threshold, api_key=settings.openai_api_key
        )
        label_check_ms = round((time.perf_counter() - t_lbl) * 1000.0, 2)

        actual_path = "none"
        actual_trigger = False
        decision_prob: float | None = None
        decision_tier: str | None = None
        used_decisions_api: bool | None = None
        retrieve_ctx_ms = 0.0
        decision_ms = 0.0

        if not covered:
            actual_trigger = True
            actual_path = "label_mismatch"
        else:
            t_ctx = time.perf_counter()
            seen_ids: set[str] = set()
            chunk_hits: list[str] = []
            for sq, qvec in zip(search_queries, qvecs, strict=True):
                for entry_type in ("chunk", "table"):
                    for hit_id, _ in vector_store.search(
                        sq, top_k=4, entry_types={entry_type}, query_vector=qvec
                    ):
                        if hit_id in seen_ids:
                            continue
                        seen_ids.add(hit_id)
                        text = vector_store._texts.get(hit_id, "")
                        if text:
                            chunk_hits.append(f"[{hit_id}] {text}")
            local_ctx = "\n\n".join(chunk_hits[:8]) or "No local chunks found."
            retrieve_ctx_ms = round((time.perf_counter() - t_ctx) * 1000.0, 2)

            t_dec = time.perf_counter()
            dec = await decision_client.evaluate_sufficiency(
                query=query, local_context=local_ctx, api_key=settings.openai_api_key
            )
            decision_ms = round((time.perf_counter() - t_dec) * 1000.0, 2)
            decision_prob = dec.sufficiency_prob
            decision_tier = dec.search_tier
            used_decisions_api = dec.used_decisions_api
            if not dec.sufficient:
                actual_trigger = True
                actual_path = "decision_model"

        ms = (time.perf_counter() - t0) * 1000.0
        stage_latencies = {
            "rewrite_ms": rewrite_ms,
            "embed_ms": embed_ms,
            "label_check_ms": label_check_ms,
            "retrieve_ctx_ms": retrieve_ctx_ms,
            "decision_ms": decision_ms,
            "total_ms": round(ms, 2),
        }
        ok = actual_trigger == case["expected_trigger"] and (
            case.get("expected_path") in (None, actual_path)
        )
        if ok:
            passed += 1
        print(
            f"[{idx}/{len(cases)}] {cid}: ok={ok} trigger={actual_trigger} path={actual_path} "
            f"min_label_score={min_score:.3f} decision_p={decision_prob} latencies={stage_latencies}",
            flush=True,
        )
        results.append(
            {
                "id": cid,
                "query": query,
                "expected_trigger": case["expected_trigger"],
                "expected_path": case.get("expected_path"),
                "actual_trigger": actual_trigger,
                "actual_path": actual_path,
                "passed": ok,
                "min_label_score": round(min_score, 4),
                "decision_prob": decision_prob,
                "decision_tier": decision_tier,
                "used_decisions_api": used_decisions_api,
                "search_queries": search_queries,
                "rewriter_mode": mode,
                "stage_latencies_ms": stage_latencies,
                "latency_ms": round(ms, 2),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"unit_triggers_{stamp}.json"
    out_path.write_text(
        json.dumps({"threshold": threshold, "passed": passed, "total": len(cases), "results": results}, indent=2),
        encoding="utf-8",
    )
    md_lines = [
        f"# Unit Triggers Evaluation (`{stamp}`)",
        "",
        f"- **Score:** {passed}/{len(cases)} passed",
        f"- **Label threshold:** `{threshold}`",
        "",
        "| ID | Status | Expected | Actual | Path | Min Label | Decision P | Total (ms) |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        exp = f"{r['expected_trigger']} ({r['expected_path'] or 'any'})"
        dec_p = f"{r['decision_prob']:.2f}" if r["decision_prob"] is not None else "—"
        md_lines.append(
            f"| `{r['id']}` | {status} | {exp} | {r['actual_trigger']} | {r['actual_path']} | "
            f"{r['min_label_score']:.3f} | {dec_p} | {r['latency_ms']:.1f} |"
        )
    md_path = args.output_dir / f"unit_triggers_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"=== TRIGGERS EVAL: {passed}/{len(cases)} passed -> {out_path} ({md_path}) ===", flush=True)


def main() -> None:
    asyncio.run(_run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
