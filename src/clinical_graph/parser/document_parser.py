"""Document parser supporting PDF (via PyMuPDF), Markdown, and plain text."""

from pathlib import Path
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
        """Extract text and Markdown tables from PDF bytes using PyMuPDF.

        Images and figures are intentionally omitted.
        """
        import pymupdf

        page_texts: list[str] = []
        tables: list[str] = []
        table_entries: list[ParsedTable] = []

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            for page_number, page in enumerate(doc, start=1):
                # Extract plain text only; images and figures are omitted.
                page_text = page.get_text("text").strip()
                if page_text:
                    page_texts.append(page_text)

                if hasattr(page, "find_tables"):
                    table_finder = page.find_tables()
                    found_tables = getattr(table_finder, "tables", table_finder) or []
                    for table in found_tables:
                        md_table = _format_markdown_table(table.extract())
                        if md_table:
                            tables.append(md_table)
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

        return ParsedDocument(
            text="\n\f\n".join(page_texts),
            tables=tables,
            table_entries=table_entries,
            page_count=page_count,
            source_type=SourceType.UPLOADED_PDF,
        )
