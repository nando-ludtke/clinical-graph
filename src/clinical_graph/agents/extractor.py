"""Single-pass LLM-powered clinical node, edge, and table extractor for KnowledgeGraph ingestion."""

import asyncio
from collections.abc import Callable
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.parser.chunker import (
    chunk_text_with_spans,
    group_chunks_for_extraction,
)
from clinical_graph.parser.cleaner import strip_references_and_boilerplate
from clinical_graph.parser.document_parser import ParsedTable
from clinical_graph.prompts import load_prompt
from clinical_graph.schemas.document import (
    ChunkSpan,
    DocumentCreate,
    DocumentRecord,
    OmissionRecord,
    TableRecord,
)
from clinical_graph.schemas.metrics import LLMUsage, PipelineMetrics
from clinical_graph.graph.document_db import slugify
from clinical_graph.graph.manager import KnowledgeManager


ClinicalNodeType = Literal[
    "drug",
    "drug_class",
    "condition",
    "symptom",
    "biomarker",
    "procedure",
    "population",
]

RelationType = Literal[
    "TREATS",
    "CAUSES",
    "CONTRAINDICATED_IN",
    "INTERACTS_WITH",
    "REQUIRES_MONITORING",
    "DIAGNOSES",
    "PREVENTS",
    "RISK_FACTOR_FOR",
    "RECOMMENDED_FOR",
]


class ExtractedNode(BaseModel):
    """Canonical clinical entity extracted from a document."""

    canonical_name: str
    node_type: ClinicalNodeType = "condition"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class ExtractedEdge(BaseModel):
    """Directed clinical relation with verbatim source-text provenance."""

    source: str
    target: str
    relation: RelationType
    relationship_description: str = ""
    relationship_strength: float = Field(default=5.0, ge=1.0, le=10.0)
    quote: str
    chunk_index: int | None = None


class ExtractionResult(BaseModel):
    """Combined node and edge extraction output for a single document."""

    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_spans: list[tuple[int, int, int]] = Field(default_factory=list)
    omissions: list[OmissionRecord] = Field(default_factory=list)
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)


class _NodeExtractionPayload(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)


class _LLMEdgeItem(BaseModel):
    source: str
    target: str
    relation: RelationType
    relationship_description: str = ""
    relationship_strength: float = Field(default=5.0, ge=1.0, le=10.0)
    quote: str


class _EdgeExtractionPayload(BaseModel):
    edges: list[_LLMEdgeItem] = Field(default_factory=list)


class _SinglePassExtractionPayload(BaseModel):
    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[_LLMEdgeItem] = Field(default_factory=list)


class GraphExtractor:
    """Single-pass clinical entity and relation extractor backed by ``LiteLLMClient``."""

    def __init__(
        self,
        llm_client: LiteLLMClient | None = None,
        node_model: str = "gpt-4o-mini",
        edge_model: str = "gpt-4o-mini",
        prompt_version: str | None = None,
    ) -> None:
        self.llm_client: LiteLLMClient = llm_client or LiteLLMClient()
        self.node_model = node_model
        self.edge_model = edge_model
        graph_prompt = load_prompt("graph_extraction", version=prompt_version)
        table_prompt = load_prompt("table_summary")
        node_prompt = load_prompt("node_extraction")
        edge_prompt = load_prompt("edge_extraction")
        self.prompt_version = prompt_version or graph_prompt.version
        self._graph_system_prompt = graph_prompt.system_prompt
        self._table_summary_system_prompt = table_prompt.system_prompt
        self._node_system_prompt = node_prompt.system_prompt
        self._edge_system_prompt = edge_prompt.system_prompt

    async def _extract_nodes_with_usage(
        self,
        text: str,
        api_key: str | None = None,
    ) -> tuple[list[ExtractedNode], LLMUsage]:
        raw, usage = await complete_with_usage_compat(
            self.llm_client,
            model=self.node_model,
            messages=[
                {"role": "system", "content": self._node_system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            response_format=_NodeExtractionPayload,
            api_key=api_key,
        )
        parsed = _NodeExtractionPayload.model_validate_json(raw)
        return parsed.nodes, usage

    async def extract_nodes(
        self,
        text: str,
        api_key: str | None = None,
    ) -> list[ExtractedNode]:
        """Extract canonical clinical nodes from ``text`` using ``node_model``."""
        nodes, _ = await self._extract_nodes_with_usage(text, api_key=api_key)
        return nodes

    async def _extract_edges_with_usage(
        self,
        text: str,
        nodes: list[ExtractedNode],
        api_key: str | None = None,
    ) -> tuple[list[ExtractedEdge], LLMUsage]:
        entity_summary = ", ".join(
            f"{node.canonical_name} ({node.node_type})" for node in nodes
        )
        user_prompt = (
            f"Canonical entities: [{entity_summary}]\n\n"
            f"Document text:\n{text}"
        )
        raw, usage = await complete_with_usage_compat(
            self.llm_client,
            model=self.edge_model,
            messages=[
                {"role": "system", "content": self._edge_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format=_EdgeExtractionPayload,
            api_key=api_key,
        )
        parsed = _EdgeExtractionPayload.model_validate_json(raw)
        edges = [
            ExtractedEdge(
                source=item.source,
                target=item.target,
                relation=item.relation,
                relationship_description=item.relationship_description,
                relationship_strength=item.relationship_strength,
                quote=item.quote,
            )
            for item in parsed.edges
            if item.source.strip().lower() != item.target.strip().lower()
        ]
        return edges, usage

    async def extract_edges(
        self,
        text: str,
        nodes: list[ExtractedNode],
        api_key: str | None = None,
    ) -> list[ExtractedEdge]:
        """Extract directed relations with verbatim quotes using ``edge_model``."""
        edges, _ = await self._extract_edges_with_usage(text, nodes, api_key=api_key)
        return edges

    async def _extract_window_with_usage(
        self,
        window_text: str,
        api_key: str | None = None,
    ) -> tuple[list[ExtractedNode], list[ExtractedEdge], LLMUsage]:
        delimited_window = f"<source_data>\n{window_text}\n</source_data>"
        raw, usage = await complete_with_usage_compat(
            self.llm_client,
            model=self.node_model,
            messages=[
                {"role": "system", "content": self._graph_system_prompt},
                {"role": "user", "content": delimited_window},
            ],
            temperature=0.0,
            response_format=_SinglePassExtractionPayload,
            api_key=api_key,
        )
        parsed = _SinglePassExtractionPayload.model_validate_json(raw)
        edges = [
            ExtractedEdge(
                source=item.source,
                target=item.target,
                relation=item.relation,
                relationship_description=item.relationship_description,
                relationship_strength=item.relationship_strength,
                quote=item.quote,
            )
            for item in parsed.edges
            if item.source.strip().lower() != item.target.strip().lower()
        ]
        return parsed.nodes, edges, usage

    async def summarize_tables(
        self,
        parsed_tables: list[ParsedTable],
        api_key: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> list[tuple[ParsedTable, str, LLMUsage]]:
        """Generate concise clinical summaries for extracted tables concurrently."""
        if not parsed_tables:
            return []

        _emit(on_step, f"[ingest] Summarizing {len(parsed_tables)} tables...")
        t0 = time.perf_counter()
        completed_tables = 0

        async def _summarize_one(
            idx: int,
            table: ParsedTable,
        ) -> tuple[ParsedTable, str, LLMUsage]:
            nonlocal completed_tables
            tbl_t0 = time.perf_counter()
            sections: list[str] = []
            if table.page_number is not None:
                sections.append(f"Page: {table.page_number}")
            if table.context_before.strip():
                sections.append(f"Context before:\n{table.context_before.strip()}")
            sections.append(f"Markdown table:\n{table.markdown.strip()}")
            if table.context_after.strip():
                sections.append(f"Context after:\n{table.context_after.strip()}")
            user_prompt = "\n\n".join(sections)

            summary, usage = await complete_with_usage_compat(
                self.llm_client,
                model=self.node_model,
                messages=[
                    {"role": "system", "content": self._table_summary_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                api_key=api_key,
            )
            completed_tables += 1
            tbl_ms = (time.perf_counter() - tbl_t0) * 1000.0
            _emit(
                on_step,
                f"[ingest] Table {completed_tables}/{len(parsed_tables)} (page={table.page_number}) summarized ({tbl_ms:.2f} ms)",
            )
            return table, summary.strip(), usage

        results = list(
            await asyncio.gather(
                *(_summarize_one(idx, table) for idx, table in enumerate(parsed_tables))
            )
        )
        ms = (time.perf_counter() - t0) * 1000.0
        _emit(on_step, f"[ingest] Summarized {len(parsed_tables)} tables in {ms:.2f} ms")
        return results

    async def extract(
        self,
        text: str,
        api_key: str | None = None,
        max_chars: int = 1300,
        overlap_paragraphs: int = 1,
        doc_id: str | None = None,
        max_chunks: int | None = None,
        max_extract_chars: int = 5000,
        on_step: Callable[[str], None] | None = None,
    ) -> ExtractionResult:
        """Clean references/boilerplate, chunk ``text``, extract nodes/edges, and deduplicate."""
        if max_chunks is not None and max_chunks < 0:
            raise ValueError("max_chunks must be non-negative")

        total_t0 = time.perf_counter()

        t0 = time.perf_counter()
        cleaned_text, omissions = strip_references_and_boilerplate(text)
        cleaning_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        chunks_with_spans = chunk_text_with_spans(
            cleaned_text,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )
        if max_chunks is not None:
            chunks_with_spans = chunks_with_spans[:max_chunks]
        windows = group_chunks_for_extraction(
            chunks_with_spans,
            max_extract_chars=max_extract_chars,
        )
        chunking_ms = (time.perf_counter() - t0) * 1000.0

        prefix = doc_id or "doc"
        chunks = [chunk for chunk, _, _ in chunks_with_spans]
        _emit(
            on_step,
            f"[ingest] Cleaned & chunked text ({len(chunks)} retrieval chunks, {len(windows)} extraction windows) in {cleaning_ms + chunking_ms:.2f} ms",
        )
        if chunks:
            total_chars = sum(len(c) for c in chunks)
            _emit(
                on_step,
                f"[ingest] Prepared {len(chunks)} retrieval chunks: chunk 0..{len(chunks) - 1} ({total_chars:,} chars)",
            )
        chunk_ids = [f"{prefix}:chunk:{i}" for i in range(len(chunks))]
        chunk_spans = [
            (i, start_char, end_char)
            for i, (_, start_char, end_char) in enumerate(chunks_with_spans)
        ]
        if not chunks:
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return ExtractionResult(
                chunks=[],
                chunk_ids=[],
                chunk_spans=[],
                omissions=omissions,
                metrics=PipelineMetrics(
                    latencies_ms={
                        "cleaning": round(cleaning_ms, 3),
                        "chunking": round(chunking_ms, 3),
                        "graph_extraction": 0.0,
                        "total": round(total_ms, 3),
                    },
                    usage=LLMUsage(),
                ),
            )

        merged_nodes: dict[str, ExtractedNode] = {}
        alias_to_canonical: dict[str, str] = {}
        name_remap: dict[str, str] = {}
        merged_edges: dict[tuple[str, str, str], ExtractedEdge] = {}
        graph_extraction_ms = 0.0
        total_usage = LLMUsage()

        completed_windows = 0
        total_windows = len(windows)

        async def _extract_window(
            win_idx: int,
            window_text: str,
            chunk_indices: list[int],
        ) -> tuple[list[ExtractedNode], list[ExtractedEdge], float, LLMUsage]:
            nonlocal completed_windows
            win_t0 = time.perf_counter()
            t_extract = time.perf_counter()
            win_nodes, win_edges, win_usage = await self._extract_window_with_usage(
                window_text, api_key=api_key
            )
            extract_ms = (time.perf_counter() - t_extract) * 1000.0
            tagged_edges = [
                edge.model_copy(
                    update={
                        "chunk_index": _map_quote_to_chunk_index(
                            edge.quote, chunk_indices, chunks
                        )
                    }
                )
                for edge in win_edges
                if edge.source.strip().lower() != edge.target.strip().lower()
            ]
            completed_windows += 1
            win_ms = (time.perf_counter() - win_t0) * 1000.0
            _emit(
                on_step,
                f"[ingest] Window {completed_windows}/{total_windows} (chunks {chunk_indices[0]}..{chunk_indices[-1]}) done ({win_ms:.2f} ms, nodes={len(win_nodes)}, edges={len(tagged_edges)})",
            )
            return win_nodes, tagged_edges, extract_ms, win_usage

        _emit(
            on_step,
            f"[ingest] Running single-pass extraction across {len(windows)} windows...",
        )
        t_windows = time.perf_counter()
        window_results = await asyncio.gather(
            *(
                _extract_window(win_idx, window_text, chunk_indices)
                for win_idx, (window_text, chunk_indices) in enumerate(windows)
            )
        )
        all_edges: list[ExtractedEdge] = []
        for win_nodes, win_edges, extract_ms, win_usage in window_results:
            graph_extraction_ms += extract_ms
            total_usage = total_usage.add(win_usage)
            _merge_extracted_nodes(
                merged_nodes, win_nodes, alias_to_canonical, name_remap
            )
            all_edges.extend(win_edges)

        _merge_extracted_edges(merged_edges, all_edges, name_remap)
        nodes = list(merged_nodes.values())
        edges = list(merged_edges.values())
        extract_wall_ms = (time.perf_counter() - t_windows) * 1000.0
        _emit(
            on_step,
            f"[ingest] Extracted {len(nodes)} nodes, {len(edges)} edges in {extract_wall_ms:.2f} ms",
        )

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            chunks=chunks,
            chunk_ids=chunk_ids,
            chunk_spans=chunk_spans,
            omissions=omissions,
            metrics=PipelineMetrics(
                latencies_ms={
                    "cleaning": round(cleaning_ms, 3),
                    "chunking": round(chunking_ms, 3),
                    "graph_extraction": round(graph_extraction_ms, 3),
                    "total": round(total_ms, 3),
                },
                usage=total_usage,
            ),
        )

    async def ingest_with_extraction(
        self,
        manager: KnowledgeManager,
        payload: DocumentCreate,
        api_key: str | None = None,
        parsed_tables: list[ParsedTable] | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> tuple[DocumentRecord, ExtractionResult]:
        """Ingest ``payload`` into ``manager`` and populate extracted nodes, edges, chunks, and tables."""
        ingest_t0 = time.perf_counter()
        record = await asyncio.to_thread(manager.ingest_document, payload, api_key=api_key)
        result = await self.extract(
            record.content,
            api_key=api_key,
            doc_id=record.doc_id,
            on_step=on_step,
        )

        table_summaries: list[tuple[ParsedTable, str, LLMUsage]] = []
        table_summary_ms = 0.0
        if parsed_tables:
            t_table = time.perf_counter()
            table_summaries = await self.summarize_tables(
                parsed_tables, api_key=api_key, on_step=on_step
            )
            table_summary_ms = (time.perf_counter() - t_table) * 1000.0

        table_records: list[TableRecord] = []
        table_items: list[tuple[str, str]] = []
        for idx, (parsed_table, summary, table_usage) in enumerate(table_summaries):
            result.metrics.usage = result.metrics.usage.add(table_usage)
            table_id = f"{record.doc_id}:table:{idx}"
            table_records.append(
                TableRecord(
                    table_id=table_id,
                    doc_id=record.doc_id,
                    markdown=parsed_table.markdown,
                    summary=summary,
                    context_before=parsed_table.context_before,
                    context_after=parsed_table.context_after,
                    page_number=parsed_table.page_number,
                )
            )
            embed_text = summary.strip() or parsed_table.markdown.strip()
            if embed_text:
                table_items.append((table_id, embed_text))

        _emit(
            on_step,
            f"[ingest] Embedding & upserting vectors (nodes={len(result.nodes)}, chunks={len(result.chunks)}, tables={len(table_records)})...",
        )
        upsert_t0 = time.perf_counter()
        nodes_spec: list[dict[str, Any]] = []
        term_to_node_id: dict[str, str] = {}
        inferred_conditions: list[str] = []
        inferred_populations: list[str] = []
        for node in result.nodes:
            existing_id = manager.graph.resolve_existing_node_id(
                node.canonical_name,
                node.aliases,
            )
            if existing_id is not None:
                resolved_node_id = existing_id
                existing_node = manager.graph.get_node(existing_id)
            else:
                resolved_node_id = slugify(f"{node.node_type}-{node.canonical_name}")
                existing_node = None

            canonical_name = node.canonical_name
            node_type = node.node_type
            aliases = list(node.aliases)
            if existing_node is not None:
                canonical_name = existing_node.canonical_name
                node_type = existing_node.node_type or node.node_type
                incoming_norm = " ".join(node.canonical_name.strip().lower().split())
                existing_norm = " ".join(existing_node.canonical_name.strip().lower().split())
                if incoming_norm and incoming_norm != existing_norm:
                    aliases.append(node.canonical_name.strip())

            nodes_spec.append(
                {
                    "node_id": resolved_node_id,
                    "canonical_name": canonical_name,
                    "node_type": node_type,
                    "aliases": aliases,
                    "description": node.description,
                    "doc_id": record.doc_id,
                }
            )
            term_to_node_id[node.canonical_name.strip().lower()] = resolved_node_id
            term_to_node_id.setdefault(canonical_name.strip().lower(), resolved_node_id)
            for alias in aliases:
                norm = alias.strip().lower()
                if norm:
                    term_to_node_id.setdefault(norm, resolved_node_id)

            node_type_norm = node.node_type.strip().lower()
            if node_type_norm == "condition" and node.canonical_name.strip():
                inferred_conditions.append(node.canonical_name.strip())
            elif node_type_norm == "population" and node.canonical_name.strip():
                inferred_populations.append(node.canonical_name.strip())

        await asyncio.to_thread(
            manager.graph.upsert_nodes_batch,
            nodes_spec,
            doc_id=record.doc_id,
            api_key=api_key,
        )

        chunk_items = [
            (chunk_id, chunk_text)
            for chunk_id, chunk_text in zip(result.chunk_ids, result.chunks, strict=True)
            if chunk_text.strip()
        ]
        if chunk_items:
            await asyncio.to_thread(
                manager.graph.vector_store.upsert_many,
                chunk_items,
                api_key=api_key,
                entry_type="chunk",
                doc_id=record.doc_id,
            )

        if table_items:
            await asyncio.to_thread(
                manager.graph.vector_store.upsert_many,
                table_items,
                api_key=api_key,
                entry_type="table",
                doc_id=record.doc_id,
            )

        for edge in result.edges:
            source_id = term_to_node_id.get(
                edge.source.strip().lower(),
                slugify(edge.source),
            )
            target_id = term_to_node_id.get(
                edge.target.strip().lower(),
                slugify(edge.target),
            )
            if source_id == target_id:
                continue
            manager.graph.upsert_edge(
                source=source_id,
                target=target_id,
                relation=edge.relation,
                doc_id=record.doc_id,
                excerpt=edge.quote,
                chunk_id=(
                    f"{record.doc_id}:chunk:{edge.chunk_index}"
                    if edge.chunk_index is not None
                    else None
                ),
                weight=edge.relationship_strength,
                relationship_description=edge.relationship_description,
            )

        chunk_spans = [
            ChunkSpan(
                chunk_id=f"{record.doc_id}:chunk:{i}",
                start_char=start_char,
                end_char=end_char,
            )
            for i, start_char, end_char in result.chunk_spans
        ]
        updated_record = await asyncio.to_thread(
            manager.record_processing_metadata,
            doc_id=record.doc_id,
            chunk_ids=result.chunk_ids,
            omissions=result.omissions,
            chunk_spans=chunk_spans,
            tables=table_records or None,
        )
        if inferred_conditions or inferred_populations:
            inferred_record = await asyncio.to_thread(
                manager.update_inferred_metadata,
                record.doc_id,
                conditions=inferred_conditions or None,
                target_populations=inferred_populations or None,
            )
            if inferred_record is not None:
                updated_record = inferred_record

        graph_upsert_ms = (time.perf_counter() - upsert_t0) * 1000.0
        _emit(on_step, f"[ingest] Embedded & indexed vectors in {graph_upsert_ms:.2f} ms")
        total_ms = (time.perf_counter() - ingest_t0) * 1000.0
        if parsed_tables:
            result.metrics.latencies_ms["table_summary"] = round(table_summary_ms, 3)
        result.metrics.latencies_ms["graph_upsert"] = round(graph_upsert_ms, 3)
        result.metrics.latencies_ms["total"] = round(total_ms, 3)
        return (updated_record or record), result


def _emit(on_step: Callable[[str], None] | None, message: str) -> None:
    if on_step is not None:
        on_step(message)


def _merge_extracted_nodes(
    target: dict[str, ExtractedNode],
    incoming: list[ExtractedNode],
    alias_to_canonical: dict[str, str],
    name_remap: dict[str, str],
) -> None:
    """Merge and deduplicate extracted nodes across chunks using canonical names and aliases."""
    for node in incoming:
        incoming_name = node.canonical_name.strip()
        incoming_name_norm = incoming_name.lower()
        if not incoming_name_norm:
            continue

        matched_key = alias_to_canonical.get(incoming_name_norm)
        if matched_key is None:
            for alias in node.aliases:
                alias_norm = alias.strip().lower()
                if alias_norm and alias_norm in alias_to_canonical:
                    matched_key = alias_to_canonical[alias_norm]
                    break

        if matched_key is None:
            deduped_aliases = _dedup_aliases(node.aliases, exclude_norm=incoming_name_norm)
            stored = node.model_copy(
                update={
                    "canonical_name": incoming_name,
                    "aliases": deduped_aliases,
                }
            )
            target[incoming_name_norm] = stored
            alias_to_canonical.setdefault(incoming_name_norm, incoming_name_norm)
            for alias in deduped_aliases:
                alias_to_canonical.setdefault(alias.lower(), incoming_name_norm)
            name_remap[incoming_name_norm] = stored.canonical_name
            continue

        existing = target[matched_key]
        existing_name_norm = existing.canonical_name.strip().lower()
        extra_aliases = [*existing.aliases, *node.aliases]
        if incoming_name_norm != existing_name_norm:
            extra_aliases.append(incoming_name)
        existing.aliases = _dedup_aliases(extra_aliases, exclude_norm=existing_name_norm)

        if len(node.description.strip()) > len(existing.description.strip()):
            existing.description = node.description.strip()

        alias_to_canonical.setdefault(incoming_name_norm, matched_key)
        for alias in existing.aliases:
            alias_to_canonical.setdefault(alias.lower(), matched_key)
        name_remap[incoming_name_norm] = existing.canonical_name


def _map_quote_to_chunk_index(
    quote: str,
    chunk_indices: list[int],
    chunks: list[str],
) -> int:
    """Map an extracted edge's verbatim quote to the best matching retrieval chunk index."""
    if not chunk_indices:
        return 0
    if len(chunk_indices) == 1:
        return chunk_indices[0]

    def _norm(text: str) -> str:
        return " ".join(text.lower().split())

    norm_quote = _norm(quote)
    if norm_quote:
        for idx in chunk_indices:
            if norm_quote in _norm(chunks[idx]):
                return idx

        quote_tokens = set(re.findall(r"\w+", norm_quote))
        if quote_tokens:
            best_idx = chunk_indices[0]
            best_overlap = 0
            for idx in chunk_indices:
                chunk_tokens = set(re.findall(r"\w+", _norm(chunks[idx])))
                overlap = len(quote_tokens & chunk_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            if best_overlap > 0:
                return best_idx

    return chunk_indices[0]


def _merge_extracted_edges(
    target: dict[tuple[str, str, str], ExtractedEdge],
    incoming: list[ExtractedEdge],
    name_remap: dict[str, str],
) -> None:
    """Remap edge endpoints to merged canonical names and deduplicate by ``(source, target, relation)``."""
    for edge in incoming:
        remapped_source = name_remap.get(edge.source.strip().lower(), edge.source.strip())
        remapped_target = name_remap.get(edge.target.strip().lower(), edge.target.strip())
        if remapped_source.strip().lower() == remapped_target.strip().lower():
            continue
        if remapped_source != edge.source or remapped_target != edge.target:
            edge = edge.model_copy(
                update={"source": remapped_source, "target": remapped_target}
            )

        key = (
            edge.source.strip().lower(),
            edge.target.strip().lower(),
            edge.relation.strip().lower(),
        )
        existing = target.get(key)
        if existing is None:
            target[key] = edge
            continue
        if edge.relationship_strength > existing.relationship_strength:
            existing.relationship_strength = edge.relationship_strength
        if len(edge.relationship_description.strip()) > len(
            existing.relationship_description.strip()
        ):
            existing.relationship_description = edge.relationship_description.strip()


def _dedup_aliases(aliases: list[str], exclude_norm: str | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        norm = cleaned.lower()
        if cleaned and norm != exclude_norm and norm not in seen:
            seen.add(norm)
            result.append(cleaned)
    return result
