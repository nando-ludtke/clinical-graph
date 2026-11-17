"""Core Answer Agent synthesizing grounded clinical responses from graph and document context."""

from collections import Counter
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from clinical_graph.llm.client import LiteLLMClient, complete_with_usage_compat
from clinical_graph.prompts import load_prompt
from clinical_graph.schemas.document import DocumentRecord
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.schemas.query import (
    Citation,
    CitationPassage,
    QueryRequest,
    QueryResponse,
    ResolvedRetrievalMode,
    RetrievalTrace,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?:[^\n.!?]|(?<=\d)\.(?=\d))+(?:[.!?]+|$)")
_CHUNK_ID_ORDER_RE = re.compile(r":(chunk|table):(\d+)$")
_INLINE_REF_TAG_RE = re.compile(r"\[(?:[CTE]\d+)(?:\s*[,;]\s*[CTE]\d+)*\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")
_INLINE_MULTI_SPACE_RE = re.compile(r"(?<=\S)[ \t]{2,}")


def _strip_inline_ref_tags(text: str) -> str:
    """Remove stray inline [C*]/[T*]/[E*] reference tags while preserving formatting."""
    if not text:
        return text
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        cleaned = _INLINE_REF_TAG_RE.sub("", line)
        if cleaned != line:
            cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
            cleaned = _INLINE_MULTI_SPACE_RE.sub(" ", cleaned)
            cleaned = cleaned.rstrip(" \t")
            if line and not line[0].isspace():
                cleaned = cleaned.lstrip(" \t")
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def _chunk_sort_key(chunk_id: str | None, doc_id: str = "") -> tuple[str, int, int, str]:
    """Return a deterministic document-order sort key for a chunk or table ID."""
    cid = chunk_id or ""
    match = _CHUNK_ID_ORDER_RE.search(cid)
    if match:
        kind_rank = 0 if match.group(1) == "chunk" else 1
        return (doc_id, kind_rank, int(match.group(2)), cid)
    return (doc_id, 2, 0, cid)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _highlight_quote_in_text(
    source_text: str,
    quote: str,
) -> tuple[str, int | None, int | None]:
    """Locate ``quote`` in ``source_text`` and wrap the matched span in ``<mark>...</mark>``."""
    if not source_text:
        return source_text, None, None

    cleaned_quote = " ".join(quote.split())
    if not cleaned_quote:
        return source_text, None, None

    pattern = re.compile(
        r"(?:-\s+|\s+)".join(re.escape(tok) for tok in cleaned_quote.split()),
        re.IGNORECASE,
    )
    match = pattern.search(source_text)
    if match is not None:
        start, end = match.start(), match.end()
        highlighted = f"{source_text[:start]}<mark>{source_text[start:end]}</mark>{source_text[end:]}"
        return highlighted, start, end

    if "-\n" in source_text:
        norm_chars: list[str] = []
        orig_offsets: list[int] = []
        i = 0
        n = len(source_text)
        while i < n:
            if source_text[i : i + 2] == "-\n":
                i += 2
                while i < n and source_text[i] in " \t":
                    i += 1
                continue
            norm_chars.append(source_text[i])
            orig_offsets.append(i)
            i += 1
        norm_text = "".join(norm_chars)
        match = pattern.search(norm_text)
        if match is not None and match.start() < len(orig_offsets):
            start = orig_offsets[match.start()]
            end = (
                orig_offsets[match.end() - 1] + 1
                if match.end() > 0 and match.end() - 1 < len(orig_offsets)
                else n
            )
            highlighted = (
                f"{source_text[:start]}<mark>{source_text[start:end]}</mark>{source_text[end:]}"
            )
            return highlighted, start, end

    # Fallback: best BM25 sentence or line match inside source_text
    spans: list[tuple[int, int, list[str]]] = []
    for m in _SENTENCE_SPLIT_RE.finditer(source_text):
        seg = m.group(0)
        if seg.strip():
            toks = _tokenize(seg)
            if toks:
                spans.append((m.start(), m.end(), toks))

    if not spans:
        pos = 0
        for line in source_text.splitlines(keepends=True):
            line_start = pos
            line_end = pos + len(line)
            pos = line_end
            if line.strip():
                toks = _tokenize(line)
                if toks:
                    spans.append((line_start, line_end, toks))

    q_tokens = _tokenize(cleaned_quote)
    if not spans or not q_tokens:
        return source_text, None, None

    num_docs = len(spans)
    avg_dl = sum(len(toks) for _, _, toks in spans) / num_docs
    df: Counter[str] = Counter()
    for _, _, toks in spans:
        for tok in set(toks):
            df[tok] += 1

    k1 = 1.5
    b = 0.75
    q_token_set = set(q_tokens)
    best_idx = -1
    best_score = 0.0
    for idx, (_, _, toks) in enumerate(spans):
        if len(q_token_set & set(toks)) < 2:
            continue
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for q_tok in q_tokens:
            freq = tf.get(q_tok, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (num_docs - df[q_tok] + 0.5) / (df[q_tok] + 0.5))
            denom = freq + k1 * (1.0 - b + b * (dl / avg_dl)) if avg_dl > 0 else freq
            score += idf * ((freq * (k1 + 1.0)) / denom)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return source_text, None, None

    start, end, _ = spans[best_idx]
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    highlighted = f"{source_text[:start]}<mark>{source_text[start:end]}</mark>{source_text[end:]}"
    return highlighted, start, end


class CitationQuote(BaseModel):
    """Single grounded citation reference and verbatim quote from context."""

    ref_id: str
    quote: str


class SynthesizedAnswer(BaseModel):
    """Structured LLM output pairing a grounded clinical answer with verbatim citations."""

    answer: str
    citations: list[CitationQuote] = Field(default_factory=list)


class AnswerAgent:
    """Synthesizes structured ``QueryResponse`` payloads from retrieved graph context.

    Uses ``LiteLLMClient`` in ``synthesize_async`` when configured, and falls back to
    deterministic, context-grounded synthesis in ``synthesize``.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient | None = None,
        model: str = "gpt-4o",
        prompt_version: str | None = None,
    ) -> None:
        self.llm_client: LiteLLMClient | None = llm_client
        self.model = model
        template = load_prompt("answer_synthesis", version=prompt_version)
        self.prompt_version = prompt_version or template.version
        self._system_prompt = template.system_prompt

    def build_context_text(
        self,
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> str:
        """Build the exact context text passed to the answer synthesis LLM."""
        return "\n\n".join(
            self._build_context_sections(
                request,
                seed_nodes,
                traversed_edges,
                documents,
                seed_node_details=seed_node_details,
                expanded_passages=expanded_passages,
                retrieved_chunks=retrieved_chunks,
                retrieved_tables=retrieved_tables,
                chunk_lookup=chunk_lookup,
            )
        )

    def synthesize(
        self,
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        citations: list[Citation],
        trace: RetrievalTrace | None = None,
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        chosen_mode: Literal["rag", "hybrid"] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> QueryResponse:
        """Produce a deterministic, grounded ``QueryResponse`` from retrieved context."""
        if (
            not seed_nodes
            and not traversed_edges
            and not documents
            and not retrieved_chunks
            and not retrieved_tables
            and not expanded_passages
        ):
            return self._empty_response(request, trace, chosen_mode=chosen_mode)

        sections = self._build_context_sections(
            request,
            seed_nodes,
            traversed_edges,
            documents,
            seed_node_details=seed_node_details,
            expanded_passages=expanded_passages,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            chunk_lookup=chunk_lookup,
        )
        return QueryResponse(
            session_id=request.session_id,
            answer="\n\n".join(sections),
            chosen_mode=chosen_mode,
            citations=citations,
            used_web_fallback=False,
            audit_passed=None,
            trace=trace,
        )

    async def synthesize_with_usage_async(
        self,
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        citations: list[Citation],
        trace: RetrievalTrace | None = None,
        api_key: str | None = None,
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        chosen_mode: Literal["rag", "hybrid"] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> tuple[QueryResponse, LLMUsage]:
        """Produce a grounded ``QueryResponse`` and ``LLMUsage`` using ``LiteLLMClient``."""
        has_context = bool(
            seed_nodes
            or traversed_edges
            or documents
            or retrieved_chunks
            or retrieved_tables
            or expanded_passages
        )
        if self.llm_client is None or not has_context:
            return (
                self.synthesize(
                    request=request,
                    seed_nodes=seed_nodes,
                    traversed_edges=traversed_edges,
                    documents=documents,
                    citations=citations,
                    trace=trace,
                    seed_node_details=seed_node_details,
                    expanded_passages=expanded_passages,
                    chosen_mode=chosen_mode,
                    retrieved_chunks=retrieved_chunks,
                    retrieved_tables=retrieved_tables,
                    chunk_lookup=chunk_lookup,
                ),
                LLMUsage(),
            )

        context_sections, ref_map = self._build_context_sections_with_refs(
            request,
            seed_nodes,
            traversed_edges,
            documents,
            seed_node_details=seed_node_details,
            expanded_passages=expanded_passages,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            chunk_lookup=chunk_lookup,
        )
        raw_answer, usage = await complete_with_usage_compat(
            self.llm_client,
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": "\n\n".join(context_sections)},
            ],
            temperature=0.0,
            api_key=api_key,
            response_format=SynthesizedAnswer,
        )
        try:
            synthesized = SynthesizedAnswer.model_validate_json(raw_answer)
        except ValueError:
            synthesized = SynthesizedAnswer(answer=raw_answer, citations=[])

        resolved_citations = self._rebuild_citations(
            synthesized_citations=synthesized.citations,
            ref_map=ref_map,
            default_citations=citations,
            documents=documents,
        )
        clean_answer = _strip_inline_ref_tags(synthesized.answer)
        return (
            QueryResponse(
                session_id=request.session_id,
                answer=clean_answer,
                chosen_mode=chosen_mode,
                citations=resolved_citations,
                used_web_fallback=False,
                audit_passed=None,
                trace=trace,
            ),
            usage,
        )

    async def synthesize_async(
        self,
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        citations: list[Citation],
        trace: RetrievalTrace | None = None,
        api_key: str | None = None,
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        chosen_mode: Literal["rag", "hybrid"] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> QueryResponse:
        """Produce a grounded ``QueryResponse`` using ``LiteLLMClient`` when available."""
        response, _ = await self.synthesize_with_usage_async(
            request=request,
            seed_nodes=seed_nodes,
            traversed_edges=traversed_edges,
            documents=documents,
            citations=citations,
            trace=trace,
            api_key=api_key,
            seed_node_details=seed_node_details,
            expanded_passages=expanded_passages,
            chosen_mode=chosen_mode,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            chunk_lookup=chunk_lookup,
        )
        return response

    @staticmethod
    def _empty_response(
        request: QueryRequest,
        trace: RetrievalTrace | None,
        chosen_mode: Literal["rag", "hybrid"] | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            session_id=request.session_id,
            answer=(
                "No matching clinical evidence was found in the knowledge graph "
                "for the provided query and filters."
            ),
            chosen_mode=chosen_mode,
            citations=[],
            used_web_fallback=False,
            audit_passed=None,
            trace=trace,
        )

    @staticmethod
    def _rebuild_citations(
        synthesized_citations: list[CitationQuote],
        ref_map: dict[str, dict[str, str | None]],
        default_citations: list[Citation],
        documents: list[DocumentRecord],
    ) -> list[Citation]:
        if not synthesized_citations:
            return default_citations

        base_by_doc: dict[str, Citation] = {c.doc_id: c for c in default_citations}
        docs_by_id: dict[str, DocumentRecord] = {d.doc_id: d for d in documents}
        passages_by_doc: dict[str, list[CitationPassage]] = {}
        seen_by_doc: dict[str, set[tuple[str | None, str]]] = {}

        for cq in synthesized_citations:
            ref_id = cq.ref_id.strip().strip("[]")
            ref = ref_map.get(ref_id)
            if ref is None:
                continue
            doc_id = ref.get("doc_id")
            if not doc_id or (doc_id not in base_by_doc and doc_id not in docs_by_id):
                continue
            quote = cq.quote.strip()
            if not quote:
                continue

            chunk_id = ref.get("chunk_id")
            chunk_text = ref.get("text") or ""
            dedup_key = (chunk_id, quote)
            if dedup_key in seen_by_doc.get(doc_id, set()):
                continue
            seen_by_doc.setdefault(doc_id, set()).add(dedup_key)

            highlighted_chunk, match_start, match_end = _highlight_quote_in_text(
                chunk_text, quote
            )
            passages_by_doc.setdefault(doc_id, []).append(
                CitationPassage(
                    chunk_id=chunk_id,
                    quote=quote,
                    chunk_text=chunk_text,
                    highlighted_chunk=highlighted_chunk,
                    match_start=match_start,
                    match_end=match_end,
                )
            )

        if not passages_by_doc:
            return default_citations

        rebuilt: list[Citation] = []
        for doc_id, passages in passages_by_doc.items():
            passages.sort(key=lambda p: _chunk_sort_key(p.chunk_id, doc_id))
            excerpts = list(dict.fromkeys(p.quote for p in passages if p.quote))
            chunk_ids = list(dict.fromkeys(p.chunk_id for p in passages if p.chunk_id))
            base = base_by_doc.get(doc_id)
            if base is not None:
                rebuilt.append(
                    Citation(
                        doc_id=base.doc_id,
                        slug=base.slug,
                        title=base.title,
                        doc_type=base.doc_type,
                        local_restriction=base.local_restriction,
                        excerpt=excerpts[0] if excerpts else None,
                        excerpts=excerpts,
                        chunk_ids=chunk_ids,
                        passages=passages,
                    )
                )
            else:
                doc = docs_by_id[doc_id]
                rebuilt.append(
                    Citation(
                        doc_id=doc.doc_id,
                        slug=doc.slug,
                        title=doc.metadata.title,
                        doc_type=doc.metadata.doc_type,
                        local_restriction=doc.metadata.local_restriction,
                        excerpt=excerpts[0] if excerpts else None,
                        excerpts=excerpts,
                        chunk_ids=chunk_ids,
                        passages=passages,
                    )
                )
        return rebuilt

    @classmethod
    def _build_context_sections(
        cls,
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> list[str]:
        sections, _ = cls._build_context_sections_with_refs(
            request=request,
            seed_nodes=seed_nodes,
            traversed_edges=traversed_edges,
            documents=documents,
            seed_node_details=seed_node_details,
            expanded_passages=expanded_passages,
            retrieved_chunks=retrieved_chunks,
            retrieved_tables=retrieved_tables,
            chunk_lookup=chunk_lookup,
        )
        return sections

    @staticmethod
    def _build_context_sections_with_refs(
        request: QueryRequest,
        seed_nodes: list[tuple[str, float]],
        traversed_edges: list[dict[str, Any]],
        documents: list[DocumentRecord],
        seed_node_details: list[dict[str, Any]] | None = None,
        expanded_passages: list[dict[str, str]] | None = None,
        retrieved_chunks: list[dict[str, str]] | None = None,
        retrieved_tables: list[dict[str, str]] | None = None,
        chunk_lookup: dict[str, str] | None = None,
    ) -> tuple[list[str], dict[str, dict[str, str | None]]]:
        sections: list[str] = [f"Query: {request.query}"]
        ref_map: dict[str, dict[str, str | None]] = {}
        chunk_map = chunk_lookup or {}

        if request.patient_context is not None:
            ctx_parts: list[str] = []
            if request.patient_context.age_group:
                ctx_parts.append(f"age_group={request.patient_context.age_group}")
            if request.patient_context.pregnancy_status is not None:
                ctx_parts.append(f"pregnancy={request.patient_context.pregnancy_status}")
            if request.patient_context.renal_impairment is not None:
                ctx_parts.append(f"renal_impairment={request.patient_context.renal_impairment}")
            if request.patient_context.allergies:
                ctx_parts.append(f"allergies={', '.join(request.patient_context.allergies)}")
            if ctx_parts:
                sections.append(f"Patient context: {'; '.join(ctx_parts)}.")

        if seed_node_details:
            concept_lines: list[str] = []
            for item in seed_node_details:
                name = item.get("canonical_name") or item.get("node_id", "")
                ntype = item.get("node_type", "")
                score = float(item.get("score", 0.0))
                desc = (item.get("description") or "").strip()
                line = f"- {name} [{ntype}] ({score:.2f})" if ntype else f"- {name} ({score:.2f})"
                if desc:
                    line += f": {desc}"
                concept_lines.append(line)
            sections.append("Matched concepts:\n" + "\n".join(concept_lines))
        elif seed_nodes:
            seed_summary = ", ".join(f"{nid} ({score:.2f})" for nid, score in seed_nodes)
            sections.append(f"Matched concepts: {seed_summary}.")

        chunk_lines: list[str] = []
        chunk_id_to_ref: dict[str, str] = {}
        if retrieved_chunks:
            c_idx = 0
            for chunk in retrieved_chunks:
                text = (
                    chunk.get("text")
                    or chunk.get("passage")
                    or chunk.get("content")
                    or ""
                ).strip()
                if not text:
                    continue
                c_idx += 1
                ref_id = f"C{c_idx}"
                label = chunk.get("slug") or chunk.get("doc_id", "")
                cid = chunk.get("chunk_id", "")
                if cid:
                    chunk_id_to_ref[cid] = ref_id
                header = f"[{label}] ({cid})" if cid else f"[{label}]"
                chunk_lines.append(f"- [{ref_id}] {header}: {text}")
                ref_map[ref_id] = {
                    "doc_id": chunk.get("doc_id") or None,
                    "chunk_id": cid or None,
                    "text": text,
                }

        table_lines: list[str] = []
        if retrieved_tables:
            t_idx = 0
            for table in retrieved_tables:
                md = (
                    table.get("markdown")
                    or table.get("text")
                    or table.get("content")
                    or ""
                ).strip()
                if not md:
                    continue
                t_idx += 1
                ref_id = f"T{t_idx}"
                label = table.get("slug") or table.get("doc_id", "")
                tid = table.get("table_id") or table.get("chunk_id") or ""
                header = f"[{label}] ({tid})" if tid else f"[{label}]"
                table_lines.append(f"- [{ref_id}] {header}:\n{md}")
                ref_map[ref_id] = {
                    "doc_id": table.get("doc_id") or None,
                    "chunk_id": tid or None,
                    "text": md,
                }

        if traversed_edges:
            edge_lines: list[str] = []
            for idx, edge in enumerate(traversed_edges[:25], start=1):
                ref_id = f"E{idx}"
                line = f"- [{ref_id}] {edge['source']} --[{edge['relation']}]--> {edge['target']}"
                rel_desc = (edge.get("relationship_description") or "").strip()
                if rel_desc:
                    line += f" ({rel_desc})"
                evidence = edge.get("evidence") or []
                source_refs = list(
                    dict.fromkeys(
                        chunk_id_to_ref[str(e["chunk_id"])]
                        for e in evidence
                        if e.get("chunk_id") and str(e["chunk_id"]) in chunk_id_to_ref
                    )
                )
                if source_refs:
                    line += f" [source: {', '.join(source_refs)}]"
                else:
                    excerpts = list(
                        dict.fromkeys(
                            e["excerpt"].strip()
                            for e in evidence
                            if e.get("excerpt") and e["excerpt"].strip()
                        )
                    )
                    if excerpts:
                        line += f": {' | '.join(excerpts)}"
                edge_lines.append(line)
            sections.append("Graph evidence:\n" + "\n".join(edge_lines))

        if chunk_lines:
            sections.append("Retrieved chunks:\n" + "\n".join(chunk_lines))

        if table_lines:
            sections.append("Retrieved tables:\n" + "\n".join(table_lines))

        if expanded_passages:
            passage_lines = [
                f"- [{p.get('slug') or p.get('doc_id', '')}] ({p.get('chunk_id', '')}): {p.get('passage', '')}"
                for p in expanded_passages
                if p.get("passage")
            ]
            if passage_lines:
                sections.append("Supporting passages:\n" + "\n".join(passage_lines))

        if documents:
            doc_lines = [
                f"- [{doc.slug}] {doc.metadata.title} ({doc.metadata.doc_type.value})"
                for doc in documents
            ]
            sections.append("Supporting sources:\n" + "\n".join(doc_lines))

        return sections, ref_map
