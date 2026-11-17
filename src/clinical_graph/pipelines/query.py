"""Framework-agnostic hybrid GraphRAG query pipeline."""

import asyncio
from collections import Counter
from collections.abc import Callable
import inspect
import math
import re
import time
from typing import Any

from clinical_graph.agents.answer_agent import AnswerAgent
from clinical_graph.agents.query_rewriter import QueryRewriter
from clinical_graph.graph.manager import KnowledgeManager
from clinical_graph.parser.cleaner import strip_references_and_boilerplate
from clinical_graph.schemas.document import ChunkSpan, DocumentFilter, DocumentRecord
from clinical_graph.schemas.metrics import PipelineMetrics
from clinical_graph.schemas.query import (
    Citation,
    QueryRequest,
    QueryResponse,
    ResolvedRetrievalMode,
    RetrievalTrace,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: float = 60.0,
) -> list[str]:
    """Fuse multiple ranked ID lists using Reciprocal Rank Fusion (``1 / (k + rank)``)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item_id in enumerate(ranked, start=1):
            if not item_id or item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _rank_traversed_edges(
    traversed_edges: list[dict[str, Any]],
    seed_nodes: list[tuple[str, float]],
    query: str,
    search_queries: list[str],
    *,
    graph: Any = None,
) -> list[dict[str, Any]]:
    """Score and sort non-self-loop edges by IDF-weighted BM25, bridge bonus, and hub damping."""
    non_self_loops = [
        edge
        for edge in traversed_edges
        if edge.get("source") != edge.get("target")
    ]
    if not non_self_loops:
        return []

    seed_scores = {nid: score for nid, score in seed_nodes}
    query_tokens = list(dict.fromkeys(_tokenize(" ".join([query, *search_queries]))))

    edge_docs: list[list[str]] = []
    for edge in non_self_loops:
        u = edge.get("source", "")
        v = edge.get("target", "")
        rel = edge.get("relation", "")
        rel_desc = edge.get("relationship_description") or ""
        evidence = edge.get("evidence") or []
        excerpts_text = " ".join(str(entry.get("excerpt") or "") for entry in evidence)
        edge_docs.append(_tokenize(f"{u} {rel} {v} {rel_desc} {excerpts_text}"))

    num_docs = len(non_self_loops)
    avg_dl = sum(len(toks) for toks in edge_docs) / num_docs
    df: Counter[str] = Counter()
    for toks in edge_docs:
        for tok in set(toks):
            df[tok] += 1

    k1 = 1.5
    b = 0.75

    def _edge_score(idx: int) -> float:
        edge = non_self_loops[idx]
        toks = edge_docs[idx]
        u = edge.get("source", "")
        v = edge.get("target", "")

        bm25 = 0.0
        if toks and avg_dl > 0.0 and query_tokens:
            tf = Counter(toks)
            dl = len(toks)
            for q_tok in query_tokens:
                freq = tf.get(q_tok, 0)
                if freq == 0:
                    continue
                idf = math.log(1.0 + (num_docs - df[q_tok] + 0.5) / (df[q_tok] + 0.5))
                denom = freq + k1 * (1.0 - b + b * (dl / avg_dl))
                bm25 += idf * ((freq * (k1 + 1.0)) / denom)

        if u in seed_scores and v in seed_scores:
            bm25 += 1.5

        if bm25 <= 0.0:
            return 0.0

        weight = float(edge.get("weight", 1.0) or 1.0)
        evidence = edge.get("evidence") or []
        evidence_factor = math.log1p(max(1, len(evidence)))
        deg_u = max(1, graph.degree(u)) if (graph is not None and u in graph) else 1
        deg_v = max(1, graph.degree(v)) if (graph is not None and v in graph) else 1
        hub_damping = 1.0 / math.sqrt(max(1, deg_u * deg_v))
        return bm25 * (0.5 + 0.05 * weight) * evidence_factor * hub_damping

    scored_indices = sorted(range(num_docs), key=_edge_score, reverse=True)
    return [non_self_loops[idx] for idx in scored_indices]


_rank_edges = _rank_traversed_edges


def _build_line_windows(
    text: str,
    target_chars: int = 400,
) -> list[tuple[int, int, list[str]]]:
    """Group single-newline lines into ~``target_chars`` windows with exact offsets."""
    windows: list[tuple[int, int, list[str]]] = []
    win_start: int | None = None
    win_end = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start = pos
        line_end = pos + len(line)
        pos = line_end
        if not line.strip():
            continue
        if win_start is None:
            win_start = line_start
            win_end = line_end
        elif (line_end - win_start) > target_chars:
            toks = _tokenize(text[win_start:win_end])
            if toks:
                windows.append((win_start, win_end, toks))
            win_start = line_start
            win_end = line_end
        else:
            win_end = line_end
    if win_start is not None:
        toks = _tokenize(text[win_start:win_end])
        if toks:
            windows.append((win_start, win_end, toks))
    return windows


def _locate_excerpt_or_bm25_paragraph(
    chunk_slice: str,
    excerpt: str,
    query_text: str,
) -> tuple[int, int]:
    """Locate ``excerpt`` in ``chunk_slice`` via normalized whitespace, else paragraph BM25."""
    cleaned_excerpt = " ".join(excerpt.split())
    if cleaned_excerpt:
        pattern = re.compile(
            r"(?:-\s+|\s+)".join(re.escape(tok) for tok in cleaned_excerpt.split()),
            re.IGNORECASE,
        )
        match = pattern.search(chunk_slice)
        if match is not None:
            return match.start(), match.end()

        # Fallback: handle PDF line-break hyphenation (e.g. "Aufklärungs-\ninitiative")
        if "-\n" in chunk_slice:
            norm_chars: list[str] = []
            orig_offsets: list[int] = []
            i = 0
            n = len(chunk_slice)
            while i < n:
                if chunk_slice[i : i + 2] == "-\n":
                    i += 2
                    while i < n and chunk_slice[i] in " \t":
                        i += 1
                    continue
                norm_chars.append(chunk_slice[i])
                orig_offsets.append(i)
                i += 1
            norm_text = "".join(norm_chars)
            match = pattern.search(norm_text)
            if match is not None and match.start() < len(orig_offsets):
                orig_start = orig_offsets[match.start()]
                orig_end = (
                    orig_offsets[match.end() - 1] + 1
                    if match.end() > 0 and match.end() - 1 < len(orig_offsets)
                    else n
                )
                return orig_start, orig_end

    # Paragraph-level BM25 / token-overlap fallback inside the chunk
    paragraphs: list[tuple[int, int, list[str]]] = []
    pos = 0
    for sep_match in _PARA_SPLIT_RE.finditer(chunk_slice):
        p_text = chunk_slice[pos : sep_match.start()]
        if p_text.strip():
            paragraphs.append((pos, sep_match.start(), _tokenize(p_text)))
        pos = sep_match.end()
    if pos < len(chunk_slice) and chunk_slice[pos:].strip():
        paragraphs.append((pos, len(chunk_slice), _tokenize(chunk_slice[pos:])))

    # Single-newline PDF chunks lack blank-line paragraph breaks; window lines for BM25
    if len(paragraphs) <= 2 and len(chunk_slice) > 800:
        line_windows = _build_line_windows(chunk_slice, target_chars=400)
        if line_windows:
            paragraphs = line_windows

    if not paragraphs:
        return 0, len(chunk_slice)

    query_tokens = _tokenize(f"{query_text} {excerpt}")
    if not query_tokens:
        return paragraphs[0][0], paragraphs[0][1]

    num_docs = len(paragraphs)
    avg_dl = sum(len(toks) for _, _, toks in paragraphs) / num_docs
    df: Counter[str] = Counter()
    for _, _, toks in paragraphs:
        for tok in set(toks):
            df[tok] += 1

    k1 = 1.5
    b = 0.75
    best_idx = 0
    best_score = -1.0
    for idx, (_, _, toks) in enumerate(paragraphs):
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for q_tok in query_tokens:
            freq = tf.get(q_tok, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (num_docs - df[q_tok] + 0.5) / (df[q_tok] + 0.5))
            denom = freq + k1 * (1.0 - b + b * (dl / avg_dl)) if avg_dl > 0 else freq
            score += idf * ((freq * (k1 + 1.0)) / denom)
        if score > best_score:
            best_score = score
            best_idx = idx

    return paragraphs[best_idx][0], paragraphs[best_idx][1]


def _expand_word_window(
    text: str,
    start: int,
    end: int,
    window_chars: int = 200,
) -> str:
    """Extend ``[start, end]`` by ``±window_chars`` snapped to word boundaries."""
    n = len(text)
    win_start = max(0, start - window_chars)
    win_end = min(n, end + window_chars)

    if win_start > 0 and not text[win_start].isspace() and not text[win_start - 1].isspace():
        cursor = win_start
        while cursor < start and not text[cursor].isspace():
            cursor += 1
        if cursor < start:
            win_start = cursor

    if win_end < n and not text[win_end - 1].isspace() and not text[win_end].isspace():
        cursor = win_end
        while cursor > end and not text[cursor - 1].isspace():
            cursor -= 1
        if cursor > end:
            win_end = cursor

    return text[win_start:win_end].strip()


def _expand_chunk_passages(
    ranked_edges: list[dict[str, Any]],
    matched_docs: list[DocumentRecord],
    query: str,
    search_queries: list[str],
    *,
    max_edges: int = 3,
    window_chars: int = 200,
) -> list[dict[str, str]]:
    """Extract deduplicated ``±200``-char chunk-scoped passages for top ranked edges."""
    if not ranked_edges or not matched_docs:
        return []

    docs_by_id = {doc.doc_id: doc for doc in matched_docs}
    span_by_doc_chunk: dict[tuple[str, str], ChunkSpan] = {}
    for doc in matched_docs:
        for span in doc.chunk_spans:
            span_by_doc_chunk[(doc.doc_id, span.chunk_id)] = span

    cleaned_cache: dict[str, str] = {}
    seen_doc_chunks: set[tuple[str, str]] = set()
    passages: list[dict[str, str]] = []
    query_text = " ".join([query, *search_queries])
    edges_used = 0

    for edge in ranked_edges:
        edge_added = False
        for entry in edge.get("evidence") or []:
            doc_id = entry.get("doc_id")
            if not doc_id:
                continue
            doc = docs_by_id.get(doc_id)
            if doc is None:
                continue
            raw_chunk_id = entry.get("chunk_id") or ""
            span = span_by_doc_chunk.get((doc_id, raw_chunk_id)) if raw_chunk_id else None
            key = (doc_id, raw_chunk_id) if span is not None else (doc_id, "__full_doc__")
            if key in seen_doc_chunks:
                continue

            cleaned = cleaned_cache.get(doc_id)
            if cleaned is None:
                cleaned = strip_references_and_boilerplate(doc.content)[0]
                cleaned_cache[doc_id] = cleaned

            if span is not None:
                if 0 <= span.start_char < span.end_char <= len(cleaned):
                    chunk_slice = cleaned[span.start_char : span.end_char]
                elif 0 <= span.start_char < span.end_char <= len(doc.content):
                    chunk_slice = doc.content[span.start_char : span.end_char]
                else:
                    chunk_slice = cleaned or doc.content
                chunk_label = raw_chunk_id
            else:
                # Fallback over doc.content when chunk_spans is empty or chunk_id is absent.
                chunk_slice = cleaned or doc.content
                chunk_label = raw_chunk_id or "doc"

            excerpt = entry.get("excerpt") or ""
            match_start, match_end = _locate_excerpt_or_bm25_paragraph(
                chunk_slice, excerpt, query_text
            )
            passage = _expand_word_window(
                chunk_slice, match_start, match_end, window_chars=window_chars
            )
            if not passage:
                continue

            seen_doc_chunks.add(key)
            passages.append(
                {
                    "doc_id": doc_id,
                    "slug": doc.slug,
                    "title": doc.metadata.title,
                    "chunk_id": chunk_label,
                    "passage": passage,
                }
            )
            edge_added = True
        if edge_added:
            edges_used += 1
            if edges_used >= max_edges:
                break

    return passages


def _extract_doc_evidence(
    doc: DocumentRecord,
    traversed_edges: list[dict],
    max_excerpts: int = 5,
) -> tuple[list[str], list[str]]:
    """Return up to ``max_excerpts`` unique evidence excerpts and chunk IDs for ``doc``."""
    excerpts: list[str] = []
    seen_excerpts: set[str] = set()
    chunk_ids: list[str] = []
    seen_chunks: set[str] = set()

    for edge in traversed_edges:
        for entry in edge.get("evidence") or []:
            if entry.get("doc_id") != doc.doc_id:
                continue
            excerpt = entry.get("excerpt")
            if isinstance(excerpt, str):
                excerpt = excerpt.strip()
            if excerpt and excerpt not in seen_excerpts:
                seen_excerpts.add(excerpt)
                excerpts.append(excerpt)
            chunk_id = entry.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                chunk_ids.append(chunk_id)
            if len(excerpts) >= max_excerpts:
                return excerpts, chunk_ids

    if not excerpts and doc.metadata.description:
        return [doc.metadata.description], []

    return excerpts, chunk_ids


def _emit(on_step: Callable[[str], None] | None, message: str) -> None:
    if on_step is not None:
        on_step(message)


def _search_vector_entries(
    vector_store: Any,
    query: str,
    *,
    entry_types: set[str],
    allowed_doc_ids: set[str] | None,
    top_k: int,
    api_key: str | None = None,
) -> list[tuple[str, float]]:
    """Search ``vector_store`` with ``entry_types`` and ``allowed_doc_ids`` when supported."""
    if vector_store is None or top_k <= 0:
        return []
    params = inspect.signature(vector_store.search).parameters
    if "entry_types" in params:
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "entry_types": entry_types,
            "api_key": api_key,
        }
        if "allowed_doc_ids" in params:
            kwargs["allowed_doc_ids"] = allowed_doc_ids
        return vector_store.search(query, **kwargs)
    if "node" in entry_types:
        return vector_store.search(query, top_k=top_k, api_key=api_key)
    return []


def _build_chunk_and_table_indexes(
    docs: list[DocumentRecord],
) -> tuple[
    dict[str, tuple[DocumentRecord, ChunkSpan]],
    dict[str, tuple[DocumentRecord, str, str]],
    dict[str, DocumentRecord],
]:
    """Build fast lookup maps for chunk spans, raw Markdown tables, and documents."""
    docs_by_id: dict[str, DocumentRecord] = {doc.doc_id: doc for doc in docs}
    chunk_index: dict[str, tuple[DocumentRecord, ChunkSpan]] = {}
    table_index: dict[str, tuple[DocumentRecord, str, str]] = {}

    for doc in docs:
        for span in doc.chunk_spans:
            chunk_index[span.chunk_id] = (doc, span)
            chunk_index[f"{doc.doc_id}:{span.chunk_id}"] = (doc, span)
            chunk_index[f"{doc.doc_id}::{span.chunk_id}"] = (doc, span)

        raw_tables = getattr(doc, "tables", None) or []
        for idx, tbl in enumerate(raw_tables):
            if isinstance(tbl, str):
                tid = f"{doc.doc_id}#table_{idx}"
                md = tbl
                table_index[tid] = (doc, tid, md)
                table_index[f"table_{idx}"] = (doc, tid, md)
            elif isinstance(tbl, dict):
                tid = str(tbl.get("table_id") or tbl.get("id") or f"{doc.doc_id}#table_{idx}")
                md = str(tbl.get("markdown") or tbl.get("content") or tbl.get("text") or "")
                table_index[tid] = (doc, tid, md)
                table_index[f"{doc.doc_id}:{tid}"] = (doc, tid, md)
                table_index[f"{doc.doc_id}::{tid}"] = (doc, tid, md)
            else:
                tid = str(getattr(tbl, "table_id", None) or getattr(tbl, "id", None) or f"{doc.doc_id}#table_{idx}")
                md = str(
                    getattr(tbl, "markdown", None)
                    or getattr(tbl, "content", None)
                    or getattr(tbl, "text", None)
                    or ""
                )
                table_index[tid] = (doc, tid, md)
                table_index[f"{doc.doc_id}:{tid}"] = (doc, tid, md)
                table_index[f"{doc.doc_id}::{tid}"] = (doc, tid, md)

    return chunk_index, table_index, docs_by_id


def _get_vector_entry_meta(vector_store: Any, entry_id: str) -> dict[str, Any]:
    """Best-effort metadata lookup for a vector store entry."""
    get_meta = getattr(vector_store, "get_metadata", None)
    if callable(get_meta):
        meta = get_meta(entry_id)
        if isinstance(meta, dict):
            return meta
    meta_map = getattr(vector_store, "_metadata", None)
    if isinstance(meta_map, dict):
        meta = meta_map.get(entry_id)
        if isinstance(meta, dict):
            return meta
    return {}


def _get_vector_entry_text(vector_store: Any, entry_id: str) -> str:
    """Best-effort text lookup for a vector store entry."""
    get_text = getattr(vector_store, "get_text", None)
    if callable(get_text):
        text = get_text(entry_id)
        if isinstance(text, str):
            return text
    texts_map = getattr(vector_store, "_texts", None)
    if isinstance(texts_map, dict):
        text = texts_map.get(entry_id)
        if isinstance(text, str):
            return text
    return ""


def _resolve_chunk_entry(
    entry_id: str,
    *,
    vector_store: Any,
    chunk_index: dict[str, tuple[DocumentRecord, ChunkSpan]],
    docs_by_id: dict[str, DocumentRecord],
    cleaned_cache: dict[str, str],
) -> dict[str, str] | None:
    """Resolve a chunk vector hit or edge ``chunk_id`` into a structured chunk dict."""
    meta = _get_vector_entry_meta(vector_store, entry_id)
    doc: DocumentRecord | None = None
    span: ChunkSpan | None = None
    chunk_id = str(meta.get("chunk_id") or entry_id)

    if entry_id in chunk_index:
        doc, span = chunk_index[entry_id]
        chunk_id = span.chunk_id
    elif chunk_id in chunk_index:
        doc, span = chunk_index[chunk_id]
        chunk_id = span.chunk_id
    elif meta.get("doc_id") and meta["doc_id"] in docs_by_id:
        doc = docs_by_id[meta["doc_id"]]

    text = _get_vector_entry_text(vector_store, entry_id).strip()
    if not text and doc is not None and span is not None:
        cleaned = cleaned_cache.get(doc.doc_id)
        if cleaned is None:
            cleaned = strip_references_and_boilerplate(doc.content)[0]
            cleaned_cache[doc.doc_id] = cleaned
        if 0 <= span.start_char < span.end_char <= len(cleaned):
            text = cleaned[span.start_char : span.end_char].strip()
        elif 0 <= span.start_char < span.end_char <= len(doc.content):
            text = doc.content[span.start_char : span.end_char].strip()

    if not text or doc is None:
        return None

    return {
        "doc_id": doc.doc_id,
        "slug": doc.slug,
        "title": doc.metadata.title,
        "chunk_id": chunk_id,
        "text": text,
    }


def _resolve_table_entry(
    entry_id: str,
    *,
    vector_store: Any,
    table_index: dict[str, tuple[DocumentRecord, str, str]],
    docs_by_id: dict[str, DocumentRecord],
) -> dict[str, str] | None:
    """Resolve a table vector hit into a structured raw Markdown table dict."""
    meta = _get_vector_entry_meta(vector_store, entry_id)
    doc: DocumentRecord | None = None
    table_id = str(meta.get("table_id") or entry_id)
    markdown = ""

    if entry_id in table_index:
        doc, table_id, markdown = table_index[entry_id]
    elif table_id in table_index:
        doc, table_id, markdown = table_index[table_id]
    elif meta.get("doc_id") and meta["doc_id"] in docs_by_id:
        doc = docs_by_id[meta["doc_id"]]

    if not markdown:
        markdown = _get_vector_entry_text(vector_store, entry_id).strip()

    if not markdown or doc is None:
        return None

    return {
        "doc_id": doc.doc_id,
        "slug": doc.slug,
        "title": doc.metadata.title,
        "table_id": table_id,
        "markdown": markdown,
    }


def _build_doc_citation_evidence(
    doc: DocumentRecord,
    ranked_edges: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, str]],
    retrieved_tables: list[dict[str, str]],
    query_text: str,
    *,
    max_excerpts: int = 5,
) -> tuple[list[str], list[str]]:
    """Build ``(excerpts, chunk_ids)`` from edge evidence, retrieved chunks, and tables."""
    excerpts, chunk_ids = _extract_doc_evidence(doc, ranked_edges, max_excerpts=max_excerpts)
    if excerpts == [doc.metadata.description] and not chunk_ids:
        excerpts = []

    seen_excerpts = set(excerpts)
    seen_chunks = set(chunk_ids)

    for chunk in retrieved_chunks:
        if chunk.get("doc_id") != doc.doc_id:
            continue
        cid = chunk.get("chunk_id")
        if cid and cid not in seen_chunks:
            seen_chunks.add(cid)
            chunk_ids.append(cid)
        if len(excerpts) < max_excerpts:
            text = chunk.get("text") or ""
            if text:
                start, end = _locate_excerpt_or_bm25_paragraph(text, "", query_text)
                snippet = _expand_word_window(text, start, end, window_chars=120)
                if snippet and snippet not in seen_excerpts:
                    seen_excerpts.add(snippet)
                    excerpts.append(snippet)

    for table in retrieved_tables:
        if table.get("doc_id") != doc.doc_id:
            continue
        tid = table.get("table_id")
        if tid and tid not in seen_chunks:
            seen_chunks.add(tid)
            chunk_ids.append(tid)
        if len(excerpts) < max_excerpts:
            md = (table.get("markdown") or "").strip()
            if md:
                preview = md[:300].strip()
                if preview and preview not in seen_excerpts:
                    seen_excerpts.add(preview)
                    excerpts.append(preview)

    if not excerpts and doc.metadata.description:
        excerpts = [doc.metadata.description]

    return excerpts, chunk_ids


class QueryPipeline:
    """Executes metadata pre-filtering, query rewriting, graph traversal, and synthesis."""

    def __init__(
        self,
        manager: KnowledgeManager,
        query_rewriter: QueryRewriter | None = None,
        answer_agent: AnswerAgent | None = None,
    ) -> None:
        self.manager = manager
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.answer_agent = answer_agent or AnswerAgent()

    async def run(
        self,
        payload: QueryRequest,
        *,
        api_key: str | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> QueryResponse:
        """Run the full hybrid GraphRAG query pipeline and return ``QueryResponse``."""
        total_t0 = time.perf_counter()
        latencies_ms: dict[str, float] = {}

        t0 = time.perf_counter()
        doc_filter = (
            DocumentFilter(
                doc_types=payload.filters.doc_types,
                local_restrictions=payload.filters.local_restrictions,
                conditions=payload.filters.conditions,
                target_populations=payload.filters.target_populations,
            )
            if payload.filters is not None
            else DocumentFilter()
        )
        candidate_docs = self.manager.db.filter_documents(doc_filter)
        if payload.filters is not None and payload.filters.document_ids is not None:
            allowed_ids = set(payload.filters.document_ids)
            candidate_docs = [doc for doc in candidate_docs if doc.doc_id in allowed_ids]

        allowed_doc_ids = {doc.doc_id for doc in candidate_docs}
        latencies_ms["filter_documents"] = round((time.perf_counter() - t0) * 1000.0, 3)
        _emit(
            on_step,
            f"[query] Filtering documents... done ({latencies_ms['filter_documents']:.2f} ms, {len(candidate_docs)} candidate docs)",
        )

        t0 = time.perf_counter()
        search_queries, rewriter_mode, rewrite_usage = (
            await self.query_rewriter.rewrite_with_routing_async(payload, api_key=api_key)
        )
        if payload.retrieval_mode == "rag":
            chosen_mode: ResolvedRetrievalMode = "rag"
        elif payload.retrieval_mode == "hybrid":
            chosen_mode = "hybrid"
        else:
            chosen_mode = "rag" if rewriter_mode in ("rag", "focused") else "hybrid"
        latencies_ms["query_rewrite"] = round((time.perf_counter() - t0) * 1000.0, 3)
        _emit(
            on_step,
            f"[query] Rewriting query... done ({latencies_ms['query_rewrite']:.2f} ms, mode={chosen_mode}, queries={len(search_queries)})",
        )

        vector_store = self.manager.graph.vector_store
        chunk_index, table_index, docs_by_id = _build_chunk_and_table_indexes(candidate_docs)
        cleaned_cache: dict[str, str] = {}
        query_text = " ".join([payload.query, *search_queries])

        seed_nodes: list[tuple[str, float]] = []
        traversed_edges: list[dict[str, Any]] = []
        ranked_edges: list[dict[str, Any]] = []
        matched_doc_ids: set[str] = set()
        seed_node_details: list[dict[str, Any]] = []
        retrieved_chunks: list[dict[str, str]] = []
        retrieved_tables: list[dict[str, str]] = []

        if chosen_mode == "rag":
            t0 = time.perf_counter()
            chunk_coros = [
                asyncio.to_thread(
                    _search_vector_entries,
                    vector_store,
                    sq,
                    entry_types={"chunk"},
                    allowed_doc_ids=allowed_doc_ids,
                    top_k=5,
                    api_key=api_key,
                )
                for sq in search_queries
            ]
            table_coros = [
                asyncio.to_thread(
                    _search_vector_entries,
                    vector_store,
                    sq,
                    entry_types={"table"},
                    allowed_doc_ids=allowed_doc_ids,
                    top_k=3,
                    api_key=api_key,
                )
                for sq in search_queries
            ]
            all_results = await asyncio.gather(*chunk_coros, *table_coros)
            n_sq = len(search_queries)
            chunk_lists = [[eid for eid, _ in res] for res in all_results[:n_sq]]
            table_lists = [[eid for eid, _ in res] for res in all_results[n_sq:]]

            fused_chunk_ids = _reciprocal_rank_fusion(chunk_lists)
            seen_chunk_ids: set[str] = set()
            for eid in fused_chunk_ids:
                resolved = _resolve_chunk_entry(
                    eid,
                    vector_store=vector_store,
                    chunk_index=chunk_index,
                    docs_by_id=docs_by_id,
                    cleaned_cache=cleaned_cache,
                )
                if resolved is None or resolved["chunk_id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(resolved["chunk_id"])
                retrieved_chunks.append(resolved)
                matched_doc_ids.add(resolved["doc_id"])
                if len(retrieved_chunks) >= 8:
                    break

            fused_table_ids = _reciprocal_rank_fusion(table_lists)
            seen_table_ids: set[str] = set()
            for eid in fused_table_ids:
                resolved = _resolve_table_entry(
                    eid,
                    vector_store=vector_store,
                    table_index=table_index,
                    docs_by_id=docs_by_id,
                )
                if resolved is None or resolved["table_id"] in seen_table_ids:
                    continue
                seen_table_ids.add(resolved["table_id"])
                retrieved_tables.append(resolved)
                matched_doc_ids.add(resolved["doc_id"])
                if len(retrieved_tables) >= 5:
                    break

            latencies_ms["rag_retrieval"] = round((time.perf_counter() - t0) * 1000.0, 3)
            _emit(
                on_step,
                f"[query] Searching is_chunk/is_table vectors... done ({latencies_ms['rag_retrieval']:.2f} ms, {len(retrieved_chunks)} chunks, {len(retrieved_tables)} tables)",
            )
        else:
            t0 = time.perf_counter()
            chunk_coros = [
                asyncio.to_thread(
                    _search_vector_entries,
                    vector_store,
                    sq,
                    entry_types={"chunk"},
                    allowed_doc_ids=allowed_doc_ids,
                    top_k=5,
                    api_key=api_key,
                )
                for sq in search_queries
            ]
            table_coros = [
                asyncio.to_thread(
                    _search_vector_entries,
                    vector_store,
                    sq,
                    entry_types={"table"},
                    allowed_doc_ids=allowed_doc_ids,
                    top_k=5,
                    api_key=api_key,
                )
                for sq in search_queries
            ]
            node_coros = [
                asyncio.to_thread(
                    self.manager.graph.find_seed_nodes,
                    sq,
                    top_k=max(payload.top_k_seeds, 5),
                    allowed_doc_ids=allowed_doc_ids,
                    api_key=api_key,
                )
                for sq in search_queries
            ]
            all_results = await asyncio.gather(*chunk_coros, *table_coros, *node_coros)
            n_sq = len(search_queries)
            chunk_lists = [[eid for eid, _ in res] for res in all_results[:n_sq]]
            table_lists = [[eid for eid, _ in res] for res in all_results[n_sq : 2 * n_sq]]
            node_results = all_results[2 * n_sq :]

            merged_seeds: dict[str, float] = {}
            for hits in node_results:
                for node_id, score in hits:
                    prev = merged_seeds.get(node_id)
                    if prev is None or score > prev:
                        merged_seeds[node_id] = score

            seed_nodes = sorted(
                merged_seeds.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: payload.top_k_seeds]
            latencies_ms["seed_search"] = round((time.perf_counter() - t0) * 1000.0, 3)

            t0 = time.perf_counter()
            traversed_edges, matched_doc_ids = self.manager.graph.traverse_k_hops(
                [nid for nid, _ in seed_nodes],
                k_hops=payload.k_hops,
                allowed_doc_ids=allowed_doc_ids,
            )
            ranked_edges = _rank_traversed_edges(
                traversed_edges,
                seed_nodes,
                payload.query,
                search_queries,
                graph=self.manager.graph.graph,
            )
            latencies_ms["k_hop_traversal"] = round((time.perf_counter() - t0) * 1000.0, 3)
            graph_ms = latencies_ms["seed_search"] + latencies_ms["k_hop_traversal"]
            _emit(
                on_step,
                f"[query] Searching is_node & traversing {payload.k_hops}-hop graph... done ({graph_ms:.2f} ms, {len(seed_nodes)} seeds, {len(traversed_edges)} edges)",
            )

            t_fuse = time.perf_counter()
            edge_chunk_ids: list[str] = []
            for edge in ranked_edges[:10]:
                for entry in edge.get("evidence") or []:
                    cid = entry.get("chunk_id")
                    if cid:
                        edge_chunk_ids.append(cid)

            chunk_rrf_inputs = [*chunk_lists]
            if edge_chunk_ids:
                chunk_rrf_inputs.append(edge_chunk_ids)

            fused_chunk_ids = _reciprocal_rank_fusion(chunk_rrf_inputs)
            seen_chunk_ids: set[str] = set()
            for eid in fused_chunk_ids:
                resolved = _resolve_chunk_entry(
                    eid,
                    vector_store=vector_store,
                    chunk_index=chunk_index,
                    docs_by_id=docs_by_id,
                    cleaned_cache=cleaned_cache,
                )
                if resolved is None or resolved["chunk_id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(resolved["chunk_id"])
                retrieved_chunks.append(resolved)
                matched_doc_ids.add(resolved["doc_id"])
                if len(retrieved_chunks) >= 8:
                    break

            fused_table_ids = _reciprocal_rank_fusion(table_lists)
            seen_table_ids: set[str] = set()
            for eid in fused_table_ids:
                resolved = _resolve_table_entry(
                    eid,
                    vector_store=vector_store,
                    table_index=table_index,
                    docs_by_id=docs_by_id,
                )
                if resolved is None or resolved["table_id"] in seen_table_ids:
                    continue
                seen_table_ids.add(resolved["table_id"])
                retrieved_tables.append(resolved)
                matched_doc_ids.add(resolved["doc_id"])
                if len(retrieved_tables) >= 5:
                    break

            fuse_ms = (time.perf_counter() - t_fuse) * 1000.0
            _emit(
                on_step,
                f"[query] Searching is_chunk/is_table vectors... done ({latencies_ms['seed_search'] + fuse_ms:.2f} ms, {len(retrieved_chunks)} chunks, {len(retrieved_tables)} tables)",
            )

            for nid, score in seed_nodes:
                node = self.manager.graph.get_node(nid)
                if node is None:
                    continue
                seed_node_details.append(
                    {
                        "node_id": node.node_id,
                        "canonical_name": node.canonical_name,
                        "node_type": node.node_type,
                        "aliases": list(node.aliases),
                        "description": node.description,
                        "score": score,
                    }
                )

        matched_docs: list[DocumentRecord] = []
        citations: list[Citation] = []
        for doc_id in sorted(matched_doc_ids):
            doc = docs_by_id.get(doc_id) or self.manager.db.get_by_id(doc_id)
            if doc is None:
                continue
            matched_docs.append(doc)
            excerpts, chunk_ids = _build_doc_citation_evidence(
                doc,
                ranked_edges,
                retrieved_chunks,
                retrieved_tables,
                query_text,
            )
            citations.append(
                Citation(
                    doc_id=doc.doc_id,
                    slug=doc.slug,
                    title=doc.metadata.title,
                    doc_type=doc.metadata.doc_type,
                    local_restriction=doc.metadata.local_restriction,
                    excerpt=excerpts[0] if excerpts else None,
                    excerpts=excerpts,
                    chunk_ids=chunk_ids,
                )
            )

        chunk_lookup: dict[str, str] = {}
        for doc in matched_docs:
            if not doc.chunk_spans:
                continue
            cleaned = cleaned_cache.get(doc.doc_id)
            if cleaned is None:
                cleaned = strip_references_and_boilerplate(doc.content)[0]
                cleaned_cache[doc.doc_id] = cleaned
            for span in doc.chunk_spans:
                if 0 <= span.start_char < span.end_char <= len(cleaned):
                    span_text = cleaned[span.start_char : span.end_char].strip()
                elif 0 <= span.start_char < span.end_char <= len(doc.content):
                    span_text = doc.content[span.start_char : span.end_char].strip()
                else:
                    span_text = ""
                if span_text:
                    chunk_lookup[span.chunk_id] = span_text
        for chunk in retrieved_chunks:
            cid = chunk.get("chunk_id")
            ctext = chunk.get("text") or chunk.get("passage") or chunk.get("content") or ""
            if cid and ctext and cid not in chunk_lookup:
                chunk_lookup[cid] = ctext.strip()

        assembled_context = (
            self.answer_agent.build_context_text(
                request=payload,
                seed_nodes=seed_nodes,
                traversed_edges=ranked_edges,
                documents=matched_docs,
                seed_node_details=seed_node_details,
                retrieved_chunks=retrieved_chunks,
                retrieved_tables=retrieved_tables,
            )
            if payload.include_trace
            else None
        )
        trace = (
            RetrievalTrace(
                search_queries=search_queries,
                retrieval_mode=rewriter_mode,
                requested_mode=payload.retrieval_mode,
                chosen_mode=chosen_mode,
                seed_nodes=[nid for nid, _ in seed_nodes],
                traversed_edges=ranked_edges,
                matched_doc_ids=sorted(matched_doc_ids),
                assembled_context=assembled_context,
            )
            if payload.include_trace
            else None
        )

        t0 = time.perf_counter()
        response, answer_usage = await self.answer_agent.synthesize_with_usage_async(
            request=payload,
            seed_nodes=seed_nodes,
            traversed_edges=ranked_edges,
            documents=matched_docs,
            citations=citations,
            trace=trace,
            api_key=api_key,
            seed_node_details=seed_node_details,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            chosen_mode=chosen_mode,
            chunk_lookup=chunk_lookup,
        )
        latencies_ms["answer_synthesis"] = round((time.perf_counter() - t0) * 1000.0, 3)
        latencies_ms["total"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
        _emit(
            on_step,
            f"[query] Synthesizing answer... done ({latencies_ms['answer_synthesis']:.2f} ms, total={latencies_ms['total']:.2f} ms)",
        )

        return response.model_copy(
            update={
                "metrics": PipelineMetrics(
                    latencies_ms=latencies_ms,
                    usage=rewrite_usage.add(answer_usage),
                )
            }
        )
