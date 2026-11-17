"""CLI script to test local PDF/text parsing, cleaning, chunking, and optional LLM steps."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any

from clinical_graph.agents.classifier import DocumentClassifier
from clinical_graph.agents.extractor import GraphExtractor
from clinical_graph.config import get_settings
from clinical_graph.llm.client import LiteLLMClient
from clinical_graph.parser.chunker import chunk_text
from clinical_graph.parser.cleaner import strip_references_and_boilerplate
from clinical_graph.parser.document_parser import DocumentParser, ParsedDocument
from clinical_graph.schemas.document import OmissionRecord


def _resolve_input_path(raw_path: str) -> Path:
    """Resolve input path, falling back to ``{input_path}.pdf`` when no suffix is given."""
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        pdf_candidate = candidate.with_suffix(".pdf")
        if pdf_candidate.exists():
            return pdf_candidate
    raise FileNotFoundError(f"Input file not found: {raw_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test local clinical document parsing, cleaning, chunking, and optional LLM steps."
    )
    parser.add_argument(
        "input_path",
        help="Path to a .pdf, .md, or .txt file (auto-resolves <path>.pdf if extension omitted).",
    )
    parser.add_argument(
        "--output-dir",
        default="test_data/results",
        help="Directory for output artifacts (default: test_data/results).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=5000,
        help="Maximum characters per chunk (default: 5000).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to process during extraction (default: all).",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run DocumentClassifier using configured CLASSIFIER_MODEL and OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run GraphExtractor.extract() using configured extraction models and OPENAI_API_KEY.",
    )
    return parser


def _format_summary_markdown(
    *,
    input_path: Path,
    parsed: ParsedDocument,
    raw_chars: int,
    cleaned_chars: int,
    reduction_pct: float,
    omissions: list[OmissionRecord],
    chunks: list[str],
    chunk_lengths: list[int],
    latencies_ms: dict[str, float],
    classification_data: dict[str, Any] | None,
    extraction_data: dict[str, Any] | None,
) -> str:
    """Render a human-readable Markdown summary report."""
    lines: list[str] = [
        f"# Pipeline Summary: `{input_path.name}`",
        "",
        "## Document & Cleaning Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Source Type | `{parsed.source_type.value}` |",
        f"| Page Count | {parsed.page_count if parsed.page_count is not None else 'N/A'} |",
        f"| Table Count | {len(parsed.tables)} |",
        f"| Raw Chars | {raw_chars:,} |",
        f"| Cleaned Chars | {cleaned_chars:,} |",
        f"| Reduction | {reduction_pct:.2f}% |",
        f"| Chunk Count | {len(chunks)} |",
        "",
        "## Latencies (ms)",
        "",
        "| Stage | Latency (ms) |",
        "| --- | --- |",
    ]
    for stage, ms in latencies_ms.items():
        lines.append(f"| `{stage}` | {ms:.2f} ms |")

    lines.extend(["", "## Omissions", ""])
    if omissions:
        lines.extend(
            [
                "| ID | Reason | Heading | Chars | Preview |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for om in omissions:
            preview = om.preview.replace("\n", " ").replace("|", "\\|")
            heading = (om.heading or "-").replace("|", "\\|")
            lines.append(
                f"| `{om.omission_id}` | `{om.reason}` | {heading} | {om.char_count:,} | {preview} |"
            )
    else:
        lines.append("_No omissions detected._")

    lines.extend(["", "## Chunk Stats", ""])
    if chunk_lengths:
        min_len = min(chunk_lengths)
        max_len = max(chunk_lengths)
        avg_len = sum(chunk_lengths) / len(chunk_lengths)
        lines.extend(
            [
                f"- **Count:** {len(chunk_lengths)}",
                f"- **Min / Avg / Max chars:** {min_len:,} / {avg_len:,.1f} / {max_len:,}",
                f"- **Lengths:** {', '.join(str(n) for n in chunk_lengths)}",
            ]
        )
    else:
        lines.append("_No chunks produced._")

    if parsed.tables:
        lines.extend(["", "## Sample Extracted Tables (up to 2)", ""])
        for idx, table in enumerate(parsed.tables[:2], start=1):
            lines.extend([f"### Table {idx}", "", table, ""])

    if classification_data is not None:
        res = classification_data["result"]
        usage = classification_data["usage"]
        lines.extend(
            [
                "",
                "## Classification",
                "",
                f"- **Model:** `{classification_data['model']}`",
                f"- **Allowed:** {res['is_allowed']}",
                f"- **Predicted Doc Type:** `{res.get('predicted_doc_type')}`",
                f"- **Blocked Category:** `{res.get('blocked_category')}`",
                f"- **Contains PII/PHI:** {res['contains_pii_or_phi']}",
                f"- **Confidence:** {res['confidence']:.2f}",
                f"- **Reason:** {res['reason']}",
                (
                    f"- **Token Usage:** prompt={usage['prompt_tokens']}, "
                    f"completion={usage['completion_tokens']}, total={usage['total_tokens']}"
                ),
            ]
        )

    if extraction_data is not None:
        lines.extend(
            [
                "",
                "## Graph Extraction",
                "",
                f"- **Node Model:** `{extraction_data['node_model']}`",
                f"- **Edge Model:** `{extraction_data['edge_model']}`",
                f"- **Chunks Extracted:** {extraction_data['chunks_extracted']} / {len(chunks)}",
                f"- **Nodes Extracted:** {extraction_data['node_count']}",
                f"- **Edges Extracted:** {extraction_data['edge_count']}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


async def _run_pipeline(args: argparse.Namespace) -> int:
    input_path = _resolve_input_path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # 1. Parse
    t0 = time.perf_counter()
    parsed = DocumentParser().parse_file(input_path)
    parse_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Clean
    t0 = time.perf_counter()
    cleaned_text, omissions = strip_references_and_boilerplate(parsed.text)
    clean_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Chunk
    t0 = time.perf_counter()
    chunks = chunk_text(cleaned_text, max_chars=args.max_chars)
    chunk_ms = (time.perf_counter() - t0) * 1000.0

    raw_chars = len(parsed.text)
    cleaned_chars = len(cleaned_text)
    reduction_pct = (
        ((raw_chars - cleaned_chars) / raw_chars) * 100.0 if raw_chars > 0 else 0.0
    )
    chunk_lengths = [len(c) for c in chunks]

    latencies_ms: dict[str, float] = {
        "parse_ms": round(parse_ms, 2),
        "clean_ms": round(clean_ms, 2),
        "chunk_ms": round(chunk_ms, 2),
    }

    settings = get_settings() if (args.classify or args.extract) else None
    llm_client = LiteLLMClient() if (args.classify or args.extract) else None

    classification_data: dict[str, Any] | None = None
    if args.classify and settings is not None:
        classifier = DocumentClassifier(
            llm_client=llm_client,
            model=settings.classifier_model,
        )
        t0 = time.perf_counter()
        classification, usage = await classifier.classify_async(
            cleaned_text,
            title=stem,
            api_key=settings.openai_api_key,
        )
        latencies_ms["classify_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        classification_data = {
            "model": settings.classifier_model,
            "result": classification.model_dump(mode="json"),
            "usage": usage.model_dump(mode="json"),
        }

    extraction_data: dict[str, Any] | None = None
    if args.extract and settings is not None:
        extractor = GraphExtractor(
            llm_client=llm_client,
            node_model=settings.node_extraction_model,
            edge_model=settings.edge_extraction_model,
        )
        t0 = time.perf_counter()
        extraction = await extractor.extract(
            cleaned_text,
            api_key=settings.openai_api_key,
            max_chars=args.max_chars,
            max_chunks=args.max_chunks,
        )
        latencies_ms["extract_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        extraction_data = {
            "node_model": settings.node_extraction_model,
            "edge_model": settings.edge_extraction_model,
            "chunks_extracted": len(extraction.chunk_ids),
            "node_count": len(extraction.nodes),
            "edge_count": len(extraction.edges),
            "nodes": [n.model_dump(mode="json") for n in extraction.nodes],
            "edges": [e.model_dump(mode="json") for e in extraction.edges],
            "metrics": extraction.metrics.model_dump(mode="json"),
        }

    # Save outputs
    parsed_path = output_dir / f"{stem}_parsed.txt"
    cleaned_path = output_dir / f"{stem}_cleaned.txt"
    report_path = output_dir / f"{stem}_report.json"
    summary_path = output_dir / f"{stem}_summary.md"

    parsed_path.write_text(parsed.text, encoding="utf-8")
    cleaned_path.write_text(cleaned_text, encoding="utf-8")

    report: dict[str, Any] = {
        "input_file": str(input_path),
        "source_type": parsed.source_type.value,
        "page_count": parsed.page_count,
        "raw_chars": raw_chars,
        "table_count": len(parsed.tables),
        "cleaned_chars": cleaned_chars,
        "reduction_pct": round(reduction_pct, 2),
        "omissions": [om.model_dump(mode="json") for om in omissions],
        "chunk_count": len(chunks),
        "chunk_char_lengths": chunk_lengths,
        "latencies_ms": latencies_ms,
    }
    if classification_data is not None:
        report["classification"] = classification_data
    if extraction_data is not None:
        report["extraction"] = extraction_data

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_md = _format_summary_markdown(
        input_path=input_path,
        parsed=parsed,
        raw_chars=raw_chars,
        cleaned_chars=cleaned_chars,
        reduction_pct=reduction_pct,
        omissions=omissions,
        chunks=chunks,
        chunk_lengths=chunk_lengths,
        latencies_ms=latencies_ms,
        classification_data=classification_data,
        extraction_data=extraction_data,
    )
    summary_path.write_text(summary_md, encoding="utf-8")

    print(
        f"Processed {input_path} ({parsed.source_type.value}, "
        f"pages={parsed.page_count if parsed.page_count is not None else 'N/A'}, "
        f"tables={len(parsed.tables)})"
    )
    print(
        f"Chars: {raw_chars:,} -> {cleaned_chars:,} "
        f"(-{reduction_pct:.2f}%, omissions={len(omissions)}), "
        f"Chunks: {len(chunks)} (max_chars={args.max_chars})"
    )
    print(
        "Latencies (ms): "
        + ", ".join(f"{k}={v:.2f} ms" for k, v in latencies_ms.items())
    )
    if classification_data is not None:
        res = classification_data["result"]
        print(
            f"Classification ({classification_data['model']}): "
            f"allowed={res['is_allowed']}, type={res.get('predicted_doc_type')}, "
            f"confidence={res['confidence']:.2f}"
        )
    if extraction_data is not None:
        print(
            f"Extraction ({extraction_data['node_model']} / {extraction_data['edge_model']}): "
            f"chunks={extraction_data['chunks_extracted']}/{len(chunks)}, "
            f"nodes={extraction_data['node_count']}, edges={extraction_data['edge_count']}"
        )
    print(f"Outputs written to: {output_dir}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_chunks is not None and args.max_chunks < 0:
        parser.error("--max-chunks must be >= 0")
    return asyncio.run(_run_pipeline(args))


if __name__ == "__main__":
    raise SystemExit(main())
