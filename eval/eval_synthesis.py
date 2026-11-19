"""Unit eval for AnswerAgent grounding, citations, and pretraining-leak prevention."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

from clinical_graph.agents import AnswerAgent
from clinical_graph.config import get_settings
from clinical_graph.llm import LiteLLMClient
from clinical_graph.schemas.document import (
    DocType,
    DocumentMetadata,
    DocumentRecord,
    LocalRestriction,
    SourceType,
)
from clinical_graph.schemas.query import QueryRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AnswerAgent grounding & abstention.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval/unit_synthesis_cases.json"),
        help='Path to synthesis test cases JSON (default: "eval/unit_synthesis_cases.json").',
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
    llm_client = LiteLLMClient(max_concurrent_calls=settings.max_concurrent_llm_calls)
    agent = AnswerAgent(llm_client=llm_client, model=settings.answer_generation_model)

    results: list[dict[str, Any]] = []
    passed = 0
    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        query = case["query"]
        chunks = case.get("chunks") or []
        web_evidence = case.get("web_evidence") or []

        docs_by_id: dict[str, DocumentRecord] = {}
        chunk_lookup: dict[str, str] = {}
        for ch in chunks:
            did = ch["doc_id"]
            chunk_lookup[ch["chunk_id"]] = ch["text"]
            if did not in docs_by_id:
                now = datetime.now(UTC)
                docs_by_id[did] = DocumentRecord(
                    doc_id=did,
                    slug=ch["slug"],
                    content=ch["text"],
                    metadata=DocumentMetadata(
                        title=ch["slug"],
                        doc_type=DocType.GUIDELINE,
                        source_type=SourceType.RAW_TEXT,
                        local_restriction=LocalRestriction.NONE,
                    ),
                    created_at=now,
                    updated_at=now,
                )

        t0 = time.perf_counter()
        resp, usage = await agent.synthesize_with_usage_async(
            request=QueryRequest(session_id=f"synth-{cid}", query=query),
            seed_nodes=[],
            traversed_edges=[],
            documents=list(docs_by_id.values()),
            citations=[],
            api_key=settings.openai_api_key,
            retrieved_chunks=chunks,
            chunk_lookup=chunk_lookup,
            web_evidence=web_evidence,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        ans_low = resp.answer.lower()
        must_ok = all(term.lower() in ans_low for term in case.get("must_contain", []))
        forbid_ok = all(term.lower() not in ans_low for term in case.get("must_not_contain", []))
        abstain_ok = (len(resp.citations) == 0) if case.get("expected_abstain") else True
        ok = must_ok and forbid_ok and abstain_ok
        if ok:
            passed += 1

        print(
            f"[{idx}/{len(cases)}] {cid}: ok={ok} must_ok={must_ok} forbid_ok={forbid_ok} "
            f"citations={len(resp.citations)} ({ms:.1f} ms)",
            flush=True,
        )
        results.append(
            {
                "id": cid,
                "query": query,
                "expected_abstain": case.get("expected_abstain", False),
                "passed": ok,
                "answer": resp.answer,
                "citations": [c.model_dump(mode="json") for c in resp.citations],
                "stage_latencies_ms": {"synthesize_ms": round(ms, 2)},
                "latency_ms": round(ms, 2),
                "tokens": usage.total_tokens,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"unit_synthesis_{stamp}.json"
    out_path.write_text(
        json.dumps({"passed": passed, "total": len(cases), "results": results}, indent=2),
        encoding="utf-8",
    )
    md_lines = [
        f"# Unit Synthesis Evaluation (`{stamp}`)",
        "",
        f"- **Score:** {passed}/{len(cases)} passed",
        "",
        "| ID | Status | Abstain Expected | Citations | Tokens | Total (ms) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        md_lines.append(
            f"| `{r['id']}` | {status} | {r['expected_abstain']} | "
            f"{len(r['citations'])} | {r['tokens']} | {r['latency_ms']:.1f} |"
        )
    md_path = args.output_dir / f"unit_synthesis_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"=== SYNTHESIS EVAL: {passed}/{len(cases)} passed -> {out_path} ({md_path}) ===", flush=True)


def main() -> None:
    asyncio.run(_run(_build_parser().parse_args()))


if __name__ == "__main__":
    main()
