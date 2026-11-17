"""Lean section- and paragraph-aware chunker preserving Markdown tables."""

import re

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Split an oversized non-table block along line or word boundaries."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_at = max(window.rfind("\n"), window.rfind(". "), window.rfind(" "))
        if split_at <= 0:
            split_at = max_chars
        elif remaining[split_at : split_at + 2] == ". ":
            split_at += 1
        piece = remaining[:split_at].strip()
        if piece:
            parts.append(piece)
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _parse_blocks_with_spans(
    text: str,
    max_chars: int,
) -> list[tuple[str, str, int, int]]:
    """Parse ``text`` into ``(kind, content, start_char, end_char)`` blocks."""
    blocks: list[tuple[str, str, int, int]] = []
    para_lines: list[tuple[str, int, int]] = []
    table_lines: list[tuple[str, int, int]] = []

    def flush_para() -> None:
        if not para_lines:
            return
        content = "\n".join(line for line, _, _ in para_lines).strip()
        block_start = para_lines[0][1]
        block_end = para_lines[-1][2]
        para_lines.clear()
        if content:
            pieces = _split_oversized_text(content, max_chars)
            if len(pieces) == 1:
                blocks.append(("paragraph", pieces[0], block_start, block_end))
            else:
                search_from = block_start
                for piece in pieces:
                    idx = text.find(piece, search_from, block_end)
                    if idx != -1:
                        p_start = idx
                        p_end = idx + len(piece)
                    else:
                        p_start = search_from
                        p_end = min(block_end, search_from + len(piece))
                    search_from = p_end
                    blocks.append(("paragraph", piece, p_start, p_end))

    def flush_table() -> None:
        if not table_lines:
            return
        content = "\n".join(line for line, _, _ in table_lines).strip()
        block_start = table_lines[0][1]
        block_end = table_lines[-1][2]
        table_lines.clear()
        if content:
            blocks.append(("table", content, block_start, block_end))

    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line_no_nl = raw_line.rstrip("\r\n")
        line_len = len(raw_line)
        if not line_no_nl.strip():
            flush_para()
            flush_table()
        elif _is_table_line(line_no_nl):
            flush_para()
            stripped_r = line_no_nl.rstrip()
            l_start = (
                offset + (len(stripped_r) - len(stripped_r.lstrip()))
                if not table_lines
                else offset
            )
            table_lines.append((stripped_r, l_start, offset + len(stripped_r)))
        elif _HEADING_RE.match(line_no_nl):
            flush_para()
            flush_table()
            stripped = line_no_nl.strip()
            l_start = offset + (len(line_no_nl) - len(line_no_nl.lstrip()))
            blocks.append(("heading", stripped, l_start, l_start + len(stripped)))
        else:
            flush_table()
            stripped_r = line_no_nl.rstrip()
            l_start = (
                offset + (len(stripped_r) - len(stripped_r.lstrip()))
                if not para_lines
                else offset
            )
            para_lines.append((stripped_r, l_start, offset + len(stripped_r)))
        offset += line_len

    flush_para()
    flush_table()
    return blocks


def _joined_length(blocks: list[tuple[str, str, int, int]]) -> int:
    if not blocks:
        return 0
    return sum(len(content) for _, content, _, _ in blocks) + 2 * (len(blocks) - 1)


def chunk_text_with_spans(
    text: str,
    max_chars: int = 5000,
    overlap_paragraphs: int = 1,
) -> list[tuple[str, int, int]]:
    """Split ``text`` into chunks and return ``(chunk_str, start_char, end_char)`` offsets."""
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if overlap_paragraphs < 0:
        raise ValueError("overlap_paragraphs must be non-negative")

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        leading_ws = len(text) - len(text.lstrip())
        return [(cleaned, leading_ws, leading_ws + len(cleaned))]

    blocks = _parse_blocks_with_spans(text, max_chars=max_chars)
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, str, int, int]] = []

    def flush_current() -> list[tuple[str, str, int, int]]:
        if not current:
            return []
        chunks.append(
            (
                "\n\n".join(content for _, content, _, _ in current),
                current[0][2],
                current[-1][3],
            )
        )
        if overlap_paragraphs == 0:
            return []
        trailing = [blk for blk in current if blk[0] != "heading"][-overlap_paragraphs:]
        return trailing

    for block in blocks:
        kind = block[0]
        candidate = [*current, block]
        candidate_len = _joined_length(candidate)

        should_split_at_heading = (
            kind == "heading"
            and bool(current)
            and _joined_length(current) >= max_chars // 2
        )

        if current and (candidate_len > max_chars or should_split_at_heading):
            overlap = flush_current()
            # Keep overlap only if it fits with the new block inside max_chars
            if overlap and _joined_length([*overlap, block]) <= max_chars:
                current = [*overlap, block]
            else:
                current = [block]
        else:
            current = candidate

    if current:
        chunks.append(
            (
                "\n\n".join(content for _, content, _, _ in current),
                current[0][2],
                current[-1][3],
            )
        )

    return chunks


def chunk_text(
    text: str,
    max_chars: int = 5000,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """Split ``text`` on Markdown/section headings and paragraphs, keeping tables intact."""
    return [
        chunk
        for chunk, _, _ in chunk_text_with_spans(
            text,
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )
    ]


def group_chunks_for_extraction(
    retrieval_chunks: list[tuple[str, int, int]],
    max_extract_chars: int = 5000,
) -> list[tuple[str, list[int]]]:
    """Group consecutive retrieval chunks into larger extraction windows.

    Returns ``(joined_window_text, [chunk_indices])`` tuples where each window
    stays within ``max_extract_chars`` whenever possible.
    """
    if max_extract_chars <= 0:
        raise ValueError("max_extract_chars must be a positive integer")
    if not retrieval_chunks:
        return []

    windows: list[tuple[str, list[int]]] = []
    current_texts: list[str] = []
    current_indices: list[int] = []
    current_len = 0

    for idx, (chunk_text_item, _, _) in enumerate(retrieval_chunks):
        addition = len(chunk_text_item) if not current_texts else 2 + len(chunk_text_item)
        if current_texts and current_len + addition > max_extract_chars:
            windows.append(("\n\n".join(current_texts), current_indices))
            current_texts = [chunk_text_item]
            current_indices = [idx]
            current_len = len(chunk_text_item)
        else:
            current_texts.append(chunk_text_item)
            current_indices.append(idx)
            current_len += addition

    if current_texts:
        windows.append(("\n\n".join(current_texts), current_indices))

    return windows
