"""Vector-similarity chunk selector, overlap reassembler, and regex pre-filter for web pages."""

import logging
from typing import Any

import numpy as np

from clinical_graph.graph.vector_store import DeterministicHashEmbedder, Embedder
from clinical_graph.guardrails.prompt_guard import GuardCheckResult, scan_text_for_injection
from clinical_graph.schemas.metrics import LLMUsage
from clinical_graph.search.allowlist import classify_domain_tier
from clinical_graph.search.fetcher import FetchedPage
from clinical_graph.search.schemas import AuditFinding, WebPageCacheEntry

logger = logging.getLogger(__name__)


def chunk_text_with_offsets(
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[tuple[int, int, str]]:
    """Split ``text`` into overlapping ``(start_char, end_char, chunk_text)`` windows."""
    if not text or not text.strip():
        return []
    n = len(text)
    if n <= chunk_chars:
        stripped = text.strip()
        if not stripped:
            return []
        start = text.find(stripped)
        return [(start, start + len(stripped), stripped)]

    overlap = max(0, min(overlap_chars, chunk_chars - 1))
    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_chars)
        if end < n:
            split_at = max(
                text.rfind("\n", start + chunk_chars // 2, end),
                text.rfind(" ", start + chunk_chars // 2, end),
            )
            if split_at > start:
                end = split_at
        raw_slice = text[start:end]
        stripped = raw_slice.strip()
        if stripped:
            lpad = len(raw_slice) - len(raw_slice.lstrip())
            s_idx = start + lpad
            e_idx = s_idx + len(stripped)
            windows.append((s_idx, e_idx, stripped))
        if end >= n:
            break
        next_start = max(start + 1, end - overlap)
        # Snap to the next word/line boundary so chunks never start mid-word
        if next_start < n and not text[next_start - 1].isspace():
            ws = next_start
            while ws < end and not text[ws].isspace():
                ws += 1
            while ws < end and text[ws].isspace():
                ws += 1
            if ws < end:
                next_start = ws
        start = next_start
    return windows


def merge_selected_chunks(full_text: str, spans: list[tuple[int, int]]) -> list[str]:
    """Sort ``spans`` in document order and reassemble overlapping/adjacent chunks."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s[0], s[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return [full_text[s:e].strip() for s, e in merged if full_text[s:e].strip()]


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class SourceAuditorAgent:
    """Filters fetched web pages with regex, selects relevant chunks via vector similarity, and merges overlaps."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        max_page_chars: int = 25000,
        min_similarity: float = 0.35,
        max_total_chunks: int = 6,
        max_chunks_per_doc: int = 3,
        # Accepted for backwards compatibility with callers/tests
        llm_client: Any = None,
        clinical_guard_client: Any = None,
        model: str = "",
        prompt_version: str | None = None,
        pdf_preview_chars: int = 3000,
        max_html_chars: int = 25000,
        max_pdf_chars: int = 25000,
        reasoning_effort: str | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or DeterministicHashEmbedder()
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_page_chars = max_page_chars
        self.min_similarity = min_similarity
        self.max_total_chunks = max_total_chunks
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pdf_preview_chars = pdf_preview_chars

    @staticmethod
    def _scan_text_local(text: str, *, extra_texts: list[str] | None = None) -> GuardCheckResult:
        if not text.strip():
            return GuardCheckResult(reachable=True, allowed=True, blocked=False)
        for candidate in (text, *(extra_texts or [])):
            hit, reason = scan_text_for_injection(candidate)
            if hit:
                return GuardCheckResult(reachable=True, allowed=False, blocked=True, reason=reason)
        return GuardCheckResult(reachable=True, allowed=True, blocked=False)

    def _embed_texts(self, texts: list[str], api_key: str | None = None) -> np.ndarray:
        embed_many_fn = getattr(self.embedder, "embed_many", None)
        if callable(embed_many_fn):
            vecs = embed_many_fn(texts, api_key=api_key)
        else:
            vecs = [self.embedder.embed(t, api_key=api_key) for t in texts]
        mat = np.stack([np.asarray(v, dtype=np.float32) for v in vecs])
        return _l2_normalize_rows(mat)

    async def audit_pages_batch(
        self,
        *,
        query: str,
        pages: list[tuple[str, FetchedPage]],
        query_vectors: list[Any] | None = None,
        api_key: str | None = None,
    ) -> tuple[list[AuditFinding], dict[str, WebPageCacheEntry], LLMUsage]:
        """Regex-scan ``pages``, rank overlapping chunks by vector similarity, and reassemble overlaps."""
        findings_by_id: dict[str, AuditFinding] = {}
        page_cache: dict[str, WebPageCacheEntry] = {}
        bounded_texts: dict[str, str] = {}

        # 1) Pre-filter unreadable or regex-flagged pages
        chunk_records: list[tuple[str, int, int, str]] = []
        for result_id, page in pages:
            tier: str = "regional" if classify_domain_tier(page.url) == "regional" else "dominant"
            if not page.readable:
                findings_by_id[result_id] = AuditFinding(
                    result_id=result_id,
                    url=page.url,
                    title=page.title,
                    domain=page.domain,
                    content_type=page.content_type,
                    tier=tier,  # type: ignore[arg-type]
                    guard_passed=True,
                    readable=False,
                    unreadable_reason=page.unreadable_reason,
                    relevant=False,
                    reason=f"Page unreadable ({page.unreadable_reason}).",
                )
                continue

            bounded = page.text[: self.max_page_chars]
            guard_res = self._scan_text_local(bounded, extra_texts=[page.title])
            if guard_res.blocked:
                findings_by_id[result_id] = AuditFinding(
                    result_id=result_id,
                    url=page.url,
                    title=page.title,
                    domain=page.domain,
                    content_type=page.content_type,
                    tier=tier,  # type: ignore[arg-type]
                    guard_passed=False,
                    guard_reason=guard_res.reason,
                    readable=True,
                    relevant=False,
                    reason=f"Blocked by local prompt guard: {guard_res.reason}",
                )
                continue

            bounded_texts[result_id] = bounded
            page_cache[result_id] = WebPageCacheEntry(
                result_id=result_id,
                url=page.url,
                title=page.title,
                domain=page.domain,
                full_text=bounded,
                content_type=page.content_type,
            )
            for s_idx, e_idx, c_text in chunk_text_with_offsets(
                bounded,
                chunk_chars=self.chunk_chars,
                overlap_chars=self.overlap_chars,
            ):
                chunk_records.append((result_id, s_idx, e_idx, c_text))

        # 2) Batch-embed all chunks (and query if not already provided)
        selected_spans_by_id: dict[str, list[tuple[int, int]]] = {}
        if chunk_records:
            chunk_texts = [rec[3] for rec in chunk_records]
            if query_vectors:
                q_mat = _l2_normalize_rows(
                    np.stack([np.asarray(qv, dtype=np.float32) for qv in query_vectors])
                )
                c_mat = self._embed_texts(chunk_texts, api_key=api_key)
            else:
                all_mat = self._embed_texts([query, *chunk_texts], api_key=api_key)
                q_mat = all_mat[:1]
                c_mat = all_mat[1:]

            sim_matrix = c_mat @ q_mat.T
            max_scores = sim_matrix.max(axis=1)

            scored_indices = sorted(
                range(len(chunk_records)),
                key=lambda i: float(max_scores[i]),
                reverse=True,
            )
            total_picked = 0
            per_doc_counts: dict[str, int] = {}
            for idx in scored_indices:
                score = float(max_scores[idx])
                if score < self.min_similarity:
                    break
                rid, s_idx, e_idx, _ = chunk_records[idx]
                if per_doc_counts.get(rid, 0) >= self.max_chunks_per_doc:
                    continue
                selected_spans_by_id.setdefault(rid, []).append((s_idx, e_idx))
                per_doc_counts[rid] = per_doc_counts.get(rid, 0) + 1
                total_picked += 1
                if total_picked >= self.max_total_chunks:
                    break

        # 3) Reassemble overlapping chunks in original document order
        ordered_findings: list[AuditFinding] = []
        for result_id, page in pages:
            if result_id in findings_by_id:
                ordered_findings.append(findings_by_id[result_id])
                continue
            tier = "regional" if classify_domain_tier(page.url) == "regional" else "dominant"
            spans = selected_spans_by_id.get(result_id, [])
            merged_chunks = merge_selected_chunks(bounded_texts[result_id], spans)
            ordered_findings.append(
                AuditFinding(
                    result_id=result_id,
                    url=page.url,
                    title=page.title,
                    domain=page.domain,
                    content_type=page.content_type,
                    tier=tier,  # type: ignore[arg-type]
                    guard_passed=True,
                    readable=True,
                    relevant=bool(merged_chunks),
                    reason=(
                        f"Selected {len(merged_chunks)} merged passage(s) via vector similarity."
                        if merged_chunks
                        else f"No chunks met similarity threshold ({self.min_similarity:.2f})."
                    ),
                    extracted_chunks=merged_chunks,
                )
            )
        return ordered_findings, page_cache, LLMUsage()

    async def audit_page(
        self,
        *,
        result_id: str,
        query: str,
        page: FetchedPage,
        session_id: str = "",
        client_ip: str | None = None,
        api_key: str | None = None,
        followed_from_id: str | None = None,
    ) -> tuple[AuditFinding, WebPageCacheEntry | None, GuardCheckResult | None, LLMUsage]:
        """Single-page wrapper around ``audit_pages_batch``."""
        findings, cache, usage = await self.audit_pages_batch(
            query=query,
            pages=[(result_id, page)],
            api_key=api_key,
        )
        finding = findings[0]
        guard_res = GuardCheckResult(
            reachable=True,
            allowed=finding.guard_passed,
            blocked=not finding.guard_passed,
            reason=finding.guard_reason,
        )
        return finding, cache.get(result_id), guard_res, usage
