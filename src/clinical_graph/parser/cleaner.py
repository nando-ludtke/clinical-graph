"""Deterministic reference, boilerplate, and repeated header/footer trimmer."""

from collections import Counter
import hashlib
import re
from typing import Literal

from clinical_graph.schemas.document import OmissionRecord

OmissionReason = Literal[
    "references",
    "boilerplate_section",
    "citation_density",
    "repeated_header_footer",
]

_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "bibliography",
        "literature cited",
        "works cited",
        "literaturverzeichnis",
        "literatur",
        "quellenverzeichnis",
        "quellen",
    }
)

_BOILERPLATE_HEADINGS = frozenset(
    {
        "acknowledgements",
        "acknowledgments",
        "funding",
        "conflicts of interest",
        "conflict of interest",
        "competing interests",
        "author contributions",
        "danksagung",
        "interessenkonflikte",
        "interessenkonflikt",
        "förderung",
        "autorenschaft",
    }
)

_PRESERVED_MAJOR_PREFIX_RE = re.compile(
    r"^(?:appendix|annex|supplement(?:ary)?|anhang|tabellenanhang)\b",
    re.IGNORECASE,
)
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_BOLD_HEADING_RE = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*$")
_LEADING_NUM_RE = re.compile(r"^(?:(?i:section\s+)?\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+")
_SEPARATOR_LINE_RE = re.compile(r"^[\s|:\-_=*]+$")
_OBVIOUS_WATERMARK_RE = re.compile(
    r"(?:©|\(c\)|\bcopyright\b|\ball rights reserved\b|\balle rechte vorbehalten\b"
    r"|\b(?:page|seite)\s+(?:\d+|x)(?:\s+(?:of|von)\s+(?:\d+|y))?\b"
    r"|\b(?:confidential|vertraulich|watermark)\b)",
    re.IGNORECASE,
)
_CITATION_LINE_RE = re.compile(
    r"(?:^\s*(?:\[\d+\]|\d+\.)\s+|doi:|PMID:|et al\.)",
    re.IGNORECASE,
)


def _normalize_heading_label(raw: str) -> str:
    """Normalize a heading string for case-insensitive matching."""
    text = raw.strip()
    md_match = _MD_HEADING_RE.match(text)
    if md_match:
        text = md_match.group(2).strip()
    else:
        bold_match = _BOLD_HEADING_RE.match(text)
        if bold_match:
            text = bold_match.group(1).strip()

    text = _LEADING_NUM_RE.sub("", text).strip()
    text = text.rstrip(":").strip()
    return re.sub(r"\s+", " ", text).lower()


def _is_standalone_heading(line: str) -> tuple[bool, bool, str]:
    """Return ``(is_heading, is_major_markdown, normalized_label)`` for ``line``."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False, False, ""
    if stripped.startswith("|"):
        return False, False, ""

    md_match = _MD_HEADING_RE.match(stripped)
    if md_match:
        level = len(md_match.group(1))
        return True, level <= 3, _normalize_heading_label(stripped)

    if _BOLD_HEADING_RE.match(stripped):
        return True, True, _normalize_heading_label(stripped)

    # Plain-text standalone heading: short, no trailing sentence period, not a citation line.
    if not stripped.endswith(".") and not re.match(r"^\s*\[\d+\]\s+", stripped):
        norm = _normalize_heading_label(stripped)
        if norm in _REFERENCE_HEADINGS or norm in _BOILERPLATE_HEADINGS:
            return True, True, norm
        if _PRESERVED_MAJOR_PREFIX_RE.match(norm):
            return True, True, norm
    return False, False, ""


def _build_omission(
    index: int,
    reason: OmissionReason,
    text: str,
    heading: str | None = None,
    char_count: int | None = None,
) -> OmissionRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return OmissionRecord(
        omission_id=f"omit_{index:03d}_{digest}",
        reason=reason,
        heading=heading,
        char_count=char_count if char_count is not None else len(text),
        preview=text[:120],
    )


def _is_header_footer_candidate(stripped: str) -> bool:
    return 1 <= len(stripped) <= 120 and not _SEPARATOR_LINE_RE.match(stripped)


def _strip_repeated_headers_footers(
    text: str,
) -> tuple[str, list[tuple[OmissionReason, str, str | None, int]]]:
    """Remove repeated page-boundary headers/footers (PDF) or obvious watermarks (.md/.txt)."""
    if "\f" in text:
        pages = text.split("\f")
        page_counts: Counter[str] = Counter()
        page_boundaries: list[tuple[list[str], set[int]]] = []

        for page in pages:
            page_lines = page.splitlines()
            non_empty = [i for i, ln in enumerate(page_lines) if ln.strip()]
            boundary_indices = set(non_empty[:2] + non_empty[-2:])
            page_boundaries.append((page_lines, boundary_indices))

            seen_on_page: set[str] = set()
            for idx in boundary_indices:
                stripped = page_lines[idx].strip()
                if _is_header_footer_candidate(stripped):
                    seen_on_page.add(stripped)
            for candidate in seen_on_page:
                page_counts[candidate] += 1

        repeated = {line for line, count in page_counts.items() if count >= 3}
        if not repeated:
            return "\n\n".join(page.strip("\n") for page in pages), []

        rebuilt_pages: list[str] = []
        removed_counts: Counter[str] = Counter()
        seen_order: list[str] = []

        for page_lines, boundary_indices in page_boundaries:
            kept_page_lines: list[str] = []
            for idx, line in enumerate(page_lines):
                stripped = line.strip()
                if idx in boundary_indices and stripped in repeated:
                    removed_counts[stripped] += 1
                    if stripped not in seen_order:
                        seen_order.append(stripped)
                    continue
                kept_page_lines.append(line)
            rebuilt_pages.append("\n".join(kept_page_lines).strip("\n"))

        omissions: list[tuple[OmissionReason, str, str | None, int]] = [
            ("repeated_header_footer", line, None, len(line) * removed_counts[line])
            for line in seen_order
        ]
        return "\n\n".join(rebuilt_pages), omissions

    lines = text.splitlines()
    counts: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if _is_header_footer_candidate(stripped) and _OBVIOUS_WATERMARK_RE.search(stripped):
            counts[stripped] += 1

    repeated = {line for line, count in counts.items() if count >= 3}
    if not repeated:
        return text, []

    kept_lines: list[str] = []
    seen_order = []
    for line in lines:
        stripped = line.strip()
        if stripped in repeated:
            if stripped not in seen_order:
                seen_order.append(stripped)
            continue
        kept_lines.append(line)

    omissions = [
        ("repeated_header_footer", line, None, len(line) * counts[line])
        for line in seen_order
    ]
    return "\n".join(kept_lines), omissions


def _strip_boilerplate_sections(
    text: str,
) -> tuple[str, list[tuple[OmissionReason, str, str | None, int]]]:
    """Skip reference and boilerplate sections until the next major heading or EOF."""
    lines = text.splitlines(keepends=True)
    total_chars = len(text)
    kept: list[str] = []
    omissions: list[tuple[OmissionReason, str, str | None, int]] = []

    skipping = False
    skip_reason: OmissionReason = "references"
    skip_heading: str | None = None
    skip_start_offset = 0
    skipped_chunks: list[str] = []
    offset = 0

    def flush_skipped() -> None:
        nonlocal skipping, skip_heading, skipped_chunks
        if not skipping:
            return
        omitted_text = "".join(skipped_chunks).strip()
        if omitted_text:
            omissions.append((skip_reason, omitted_text, skip_heading, len(omitted_text)))
        skipping = False
        skip_heading = None
        skipped_chunks = []

    for line in lines:
        is_heading, is_major, norm = _is_standalone_heading(line)
        section_reason: OmissionReason | None = None
        if is_heading:
            if norm in _REFERENCE_HEADINGS:
                section_reason = "references"
            elif norm in _BOILERPLATE_HEADINGS:
                section_reason = "boilerplate_section"

        if skipping:
            is_preserved = is_heading and bool(_PRESERVED_MAJOR_PREFIX_RE.match(norm))
            started_early = total_chars > 0 and skip_start_offset < 0.4 * total_chars
            stops_here = is_preserved or (
                started_early and is_heading and is_major and section_reason is None
            )
            if stops_here:
                flush_skipped()
                kept.append(line)
            elif section_reason is not None and section_reason != skip_reason:
                flush_skipped()
                skipping = True
                skip_reason = section_reason
                skip_heading = line.strip()
                skip_start_offset = offset
                skipped_chunks = [line]
            else:
                skipped_chunks.append(line)
        else:
            if section_reason is not None:
                skipping = True
                skip_reason = section_reason
                skip_heading = line.strip()
                skip_start_offset = offset
                skipped_chunks = [line]
            else:
                kept.append(line)

        offset += len(line)

    flush_skipped()
    return "".join(kept), omissions


def _strip_citation_dense_blocks(
    text: str,
) -> tuple[str, list[tuple[OmissionReason, str, str | None, int]]]:
    """In the trailing 40% of the document, skip blocks with >= 50% citation-like lines."""
    total_chars = len(text)
    if total_chars == 0:
        return text, []

    blocks = re.split(r"\n\s*\n", text)
    kept_blocks: list[str] = []
    omissions: list[tuple[OmissionReason, str, str | None, int]] = []
    search_pos = 0

    for block in blocks:
        idx = text.find(block, search_pos)
        block_offset = idx if idx >= 0 else search_pos
        search_pos = block_offset + len(block)

        stripped_block = block.strip()
        if not stripped_block:
            continue

        in_trailing_40 = block_offset >= 0.6 * total_chars
        non_empty_lines = [ln for ln in stripped_block.splitlines() if ln.strip()]
        if in_trailing_40 and len(non_empty_lines) >= 3:
            citation_matches = sum(
                1 for ln in non_empty_lines if _CITATION_LINE_RE.search(ln)
            )
            if citation_matches / len(non_empty_lines) >= 0.5:
                omissions.append(
                    ("citation_density", stripped_block, None, len(stripped_block))
                )
                continue

        kept_blocks.append(stripped_block)

    return "\n\n".join(kept_blocks), omissions


def strip_references_and_boilerplate(text: str) -> tuple[str, list[OmissionRecord]]:
    """Strip repeated headers/footers, reference/boilerplate sections, and trailing citation blocks."""
    if not text or not text.strip():
        return text.strip(), []

    working, header_omissions = _strip_repeated_headers_footers(text)
    working, section_omissions = _strip_boilerplate_sections(working)
    working, citation_omissions = _strip_citation_dense_blocks(working)

    cleaned = re.sub(r"\n{3,}", "\n\n", working).strip()
    raw_omissions = [*header_omissions, *section_omissions, *citation_omissions]
    records = [
        _build_omission(idx, reason, omitted_text, heading=heading, char_count=char_count)
        for idx, (reason, omitted_text, heading, char_count) in enumerate(raw_omissions)
    ]
    return cleaned, records
