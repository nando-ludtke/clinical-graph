"""Document parser supporting PDF (via PyMuPDF), Markdown, and plain text."""

from pathlib import Path
import re
import statistics
from typing import Any

from pydantic import BaseModel, Field

from clinical_graph.schemas.document import SourceType


class ParsedTable(BaseModel):
    """Structured table extracted from a document with surrounding page context."""

    markdown: str
    context_before: str = ""
    context_after: str = ""
    page_number: int | None = None


class ParsedDocument(BaseModel):
    """Normalized output produced by ``DocumentParser``."""

    text: str
    tables: list[str] = Field(default_factory=list)
    table_entries: list[ParsedTable] = Field(default_factory=list)
    page_count: int | None = None
    source_type: SourceType


def _format_markdown_table(rows: list[list[Any]]) -> str:
    """Convert extracted table rows into a GitHub-flavored Markdown table."""
    if not rows:
        return ""

    max_cols = max((len(row) for row in rows), default=0)
    if max_cols == 0:
        return ""

    def _normalize_row(row: list[Any]) -> list[str]:
        cells = [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ]
        return cells + [""] * (max_cols - len(cells))

    normalized = [_normalize_row(row) for row in rows]
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body])


_DOT_LEADER_RE = re.compile(r"\.{4,}|(?:\.\s*){4,}")
_MD_SEP_ROW_RE = re.compile(r"^\|\s*:?-{2,}:?(?:\s*\|\s*:?-{2,}:?)*\s*\|?$")
_TOC_PAGE_NUM_RE = re.compile(r"(?:\|\s*|\s)\d{1,4}\s*\|?\s*$")


def _is_toc_table(markdown: str) -> bool:
    """Return ``True`` if a Markdown table appears to be a Table of Contents."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    data_rows = lines[2:] if _MD_SEP_ROW_RE.match(lines[1]) else lines[1:]
    if not data_rows:
        return False

    dot_leader_count = sum(1 for row in data_rows if _DOT_LEADER_RE.search(row))
    if dot_leader_count * 2 >= len(data_rows):
        return True

    lower_md = markdown.lower()
    if "inhaltsverzeichnis" in lower_md or "table of contents" in lower_md:
        page_num_count = sum(1 for row in data_rows if _TOC_PAGE_NUM_RE.search(row))
        if page_num_count * 2 >= len(data_rows):
            return True

    return False


def _extract_table_context(
    page: Any,
    table: Any,
    page_text: str,
    max_chars: int = 350,
) -> tuple[str, str]:
    """Capture up to ``max_chars`` of text before and after a detected PDF table."""
    if not page_text:
        return "", ""

    bbox = getattr(table, "bbox", None)
    if bbox is not None and len(bbox) == 4 and hasattr(page, "get_text"):
        try:
            blocks = page.get_text("blocks")
        except Exception:
            blocks = None
        if isinstance(blocks, (list, tuple)) and blocks:
            table_top = float(bbox[1])
            table_bottom = float(bbox[3])
            before_parts: list[str] = []
            after_parts: list[str] = []
            for block in blocks:
                if not isinstance(block, (list, tuple)) or len(block) < 5:
                    continue
                if len(block) >= 7 and block[6] != 0:
                    continue
                text_val = block[4]
                if not isinstance(text_val, str) or not text_val.strip():
                    continue
                if float(block[3]) <= table_top:
                    before_parts.append(text_val.strip())
                elif float(block[1]) >= table_bottom:
                    after_parts.append(text_val.strip())
            if before_parts or after_parts:
                context_before = "\n".join(before_parts)[-max_chars:].strip()
                context_after = "\n".join(after_parts)[:max_chars].strip()
                return context_before, context_after

    return page_text[:max_chars].strip(), page_text[-max_chars:].strip()


_BULLET_GLYPHS = ("•", "", "▪", "◦", "‣")
_BOLD_FLAG = 16
_ITALIC_FLAG = 2


def _extract_valid_page_dict(page: Any) -> dict[str, Any] | None:
    """Return ``page.get_text('dict')`` if it is a valid block/line/span structure."""
    try:
        raw = page.get_text("dict")
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    blocks = raw.get("blocks")
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, dict):
            return None
        if block.get("type", 0) != 0:
            continue
        lines = block.get("lines")
        if lines is not None and not isinstance(lines, list):
            return None
        for line in lines or []:
            if not isinstance(line, dict):
                return None
            spans = line.get("spans")
            if spans is not None and not isinstance(spans, list):
                return None
            for span in spans or []:
                if not isinstance(span, dict):
                    return None
    return raw


def _collect_span_sizes(page_dicts: list[dict[str, Any] | None]) -> list[float]:
    """Collect font sizes from all non-empty text spans across parsed page dicts."""
    sizes: list[float] = []
    for page_dict in page_dicts:
        if page_dict is None:
            continue
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines") or []:
                for span in line.get("spans") or []:
                    text = span.get("text")
                    size = span.get("size")
                    if (
                        isinstance(text, str)
                        and text.strip()
                        and isinstance(size, (int, float))
                        and float(size) > 0
                    ):
                        sizes.append(float(size))
    return sizes


def _format_span_markdown(text: str, flags: int) -> str:
    """Wrap a span's non-whitespace content in Markdown bold/italic markers."""
    core = text.strip()
    if not core:
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    is_bold = bool(flags & _BOLD_FLAG)
    is_italic = bool(flags & _ITALIC_FLAG)
    if is_bold and is_italic:
        formatted = f"***{core}***"
    elif is_bold:
        formatted = f"**{core}**"
    elif is_italic:
        formatted = f"*{core}*"
    else:
        formatted = core
    return f"{leading}{formatted}{trailing}"


def _strip_leading_bullet(spans: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """Strip a leading bullet glyph and following whitespace from a line's spans."""
    plain = "".join(str(s.get("text", "")) for s in spans).lstrip()
    if not plain or plain[0] not in _BULLET_GLYPHS:
        return False, spans

    adjusted: list[dict[str, Any]] = []
    removed_glyph = False
    stripping_ws = False
    for span in spans:
        text = str(span.get("text", ""))
        if not removed_glyph:
            stripped = text.lstrip()
            if not stripped:
                continue
            if stripped[0] in _BULLET_GLYPHS:
                removed_glyph = True
                rest = stripped[1:]
                if rest:
                    rest_no_ws = rest.lstrip()
                    if rest_no_ws:
                        adjusted.append({**span, "text": rest_no_ws})
                    else:
                        stripping_ws = True
                else:
                    stripping_ws = True
                continue
        if stripping_ws:
            rest_no_ws = text.lstrip()
            if not rest_no_ws:
                continue
            stripping_ws = False
            adjusted.append({**span, "text": rest_no_ws})
            continue
        adjusted.append(span)
    return True, adjusted


def _format_line_markdown(spans: list[dict[str, Any]], body_size: float) -> str:
    """Format a single PDF line dict's spans into Markdown."""
    non_empty = [
        s for s in spans if isinstance(s.get("text"), str) and s["text"].strip()
    ]
    if not non_empty:
        return ""

    plain_line = "".join(str(s.get("text", "")) for s in spans).strip()
    sized_spans = [
        (float(s["size"]), len(s["text"].strip()))
        for s in non_empty
        if isinstance(s.get("size"), (int, float)) and float(s["size"]) > 0
    ]
    if sized_spans:
        total_chars = sum(length for _, length in sized_spans)
        line_size = (
            sum(size * length for size, length in sized_spans) / total_chars
            if total_chars > 0
            else body_size
        )
    else:
        line_size = body_size

    all_bold = all(bool(int(s.get("flags", 0) or 0) & _BOLD_FLAG) for s in non_empty)
    if line_size >= body_size * 1.45:
        return f"# {plain_line}"
    if line_size >= body_size * 1.25:
        return f"## {plain_line}"
    if line_size >= body_size * 1.05 and len(plain_line) <= 120 and all_bold:
        return f"### {plain_line}"

    is_bullet, body_spans = _strip_leading_bullet(spans)
    formatted_parts = [
        _format_span_markdown(str(s.get("text", "")), int(s.get("flags", 0) or 0))
        for s in body_spans
        if isinstance(s.get("text"), str)
    ]
    body_text = "".join(formatted_parts).strip()
    if not body_text:
        return ""
    if is_bullet:
        return f"- {body_text}"
    return body_text


def _render_page_markdown(page_dict: dict[str, Any], body_size: float) -> str:
    """Render a validated PyMuPDF page dict into Markdown."""
    block_texts: list[str] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        line_texts: list[str] = []
        for line in block.get("lines") or []:
            formatted = _format_line_markdown(line.get("spans") or [], body_size)
            if formatted:
                line_texts.append(formatted)
        if line_texts:
            block_texts.append("\n".join(line_texts))
    return "\n\n".join(block_texts).strip()


class DocumentParser:
    """Parse ``.pdf``, ``.md``, and ``.txt`` clinical documents into text and tables."""

    _TEXT_SOURCE_TYPES: dict[str, SourceType] = {
        ".md": SourceType.MARKDOWN,
        ".txt": SourceType.RAW_TEXT,
    }

    def parse_bytes(self, data: bytes, filename: str) -> ParsedDocument:
        """Parse raw file bytes based on the extension of ``filename``."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(data)
        if ext in self._TEXT_SOURCE_TYPES:
            return self._parse_text(data, self._TEXT_SOURCE_TYPES[ext])
        raise ValueError(
            f"Unsupported file extension '{ext or filename}'. "
            "Supported extensions are: .pdf, .md, .txt"
        )

    def parse_file(self, path: str | Path) -> ParsedDocument:
        """Read and parse a document from disk."""
        file_path = Path(path)
        return self.parse_bytes(file_path.read_bytes(), file_path.name)

    def _parse_text(self, data: bytes, source_type: SourceType) -> ParsedDocument:
        """Decode UTF-8 text or Markdown content."""
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("File content is not valid UTF-8 text.") from exc
        return ParsedDocument(text=text, tables=[], page_count=None, source_type=source_type)

    def _parse_pdf(self, data: bytes) -> ParsedDocument:
        """Extract font-aware Markdown text and tables from PDF bytes using PyMuPDF.

        Images and figures are intentionally omitted.
        """
        import pymupdf

        page_texts: list[str] = []
        tables: list[str] = []
        table_entries: list[ParsedTable] = []

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            pages = list(doc)
            page_dicts = [_extract_valid_page_dict(page) for page in pages]
            span_sizes = _collect_span_sizes(page_dicts)
            body_size = statistics.median(span_sizes) if span_sizes else 12.0

            for page_number, (page, page_dict) in enumerate(
                zip(pages, page_dicts), start=1
            ):
                if page_dict is not None:
                    page_text = _render_page_markdown(page_dict, body_size)
                else:
                    raw_text = page.get_text("text")
                    page_text = raw_text.strip() if isinstance(raw_text, str) else ""

                page_tables: list[str] = []
                if hasattr(page, "find_tables"):
                    table_finder = page.find_tables()
                    found_tables = getattr(table_finder, "tables", table_finder) or []
                    for table in found_tables:
                        md_table = _format_markdown_table(table.extract())
                        if md_table and not _is_toc_table(md_table):
                            tables.append(md_table)
                            page_tables.append(md_table)
                            ctx_before, ctx_after = _extract_table_context(
                                page, table, page_text
                            )
                            table_entries.append(
                                ParsedTable(
                                    markdown=md_table,
                                    context_before=ctx_before,
                                    context_after=ctx_after,
                                    page_number=page_number,
                                )
                            )

                for md_table in page_tables:
                    if md_table not in page_text:
                        page_text = (
                            f"{page_text}\n\n{md_table}" if page_text else md_table
                        )

                if page_text:
                    page_texts.append(page_text)

        return ParsedDocument(
            text="\n\f\n".join(page_texts),
            tables=tables,
            table_entries=table_entries,
            page_count=page_count,
            source_type=SourceType.UPLOADED_PDF,
        )
