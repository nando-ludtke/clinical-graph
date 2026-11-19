"""Allowlist-enforced HTML and PDF fetcher with boilerplate stripping and JS-shell detection."""

import re
import asyncio
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urljoin

import httpx
import pymupdf

from clinical_graph.search.allowlist import extract_hostname, is_url_allowed
from clinical_graph.search.schemas import CandidateLink

_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "nav", "header", "footer", "aside"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br"}
)
_HIDDEN_STYLE_RE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_JS_SHELL_MARKERS = (
    "enable javascript",
    "javascript is required",
    "please enable js",
    "checking your browser",
    "just a moment...",
)


class _SafeHTMLTextExtractor(HTMLParser):
    """Strip scripts, styles, nav/chrome, and hidden elements while extracting text & links."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=False)
        self.base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._current_href: str | None = None
        self._current_link_text: list[str] = []
        self.links: list[CandidateLink] = []
        self._seen_links: set[str] = set()

    def _is_hidden(self, attrs_dict: dict[str, str | None]) -> bool:
        if "hidden" in attrs_dict:
            return True
        if (attrs_dict.get("aria-hidden") or "").strip().lower() == "true":
            return True
        style = attrs_dict.get("style") or ""
        return bool(_HIDDEN_STYLE_RE.search(style))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attrs_dict = {k.lower(): v for k, v in attrs}

        if self._skip_depth > 0:
            self._skip_depth += 1
            return

        if tag_l in _SKIP_TAGS or self._is_hidden(attrs_dict):
            self._skip_depth = 1
            return

        if tag_l == "title":
            self._in_title = True
        elif tag_l in _BLOCK_TAGS:
            self._text_parts.append("\n")
        elif tag_l == "a":
            href = (attrs_dict.get("href") or "").strip()
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                self._current_href = urljoin(self.base_url, href)
                self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag_l == "title":
            self._in_title = False
        elif tag_l in _BLOCK_TAGS:
            self._text_parts.append("\n")
        elif tag_l == "a" and self._current_href:
            href = self._current_href
            anchor = _MULTI_SPACE_RE.sub(" ", "".join(self._current_link_text)).strip()
            self._current_href = None
            self._current_link_text = []
            if (
                href not in self._seen_links
                and is_url_allowed(href)
                and (href.lower().endswith(".pdf") or len(anchor) >= 4)
            ):
                self._seen_links.add(href)
                self.links.append(
                    CandidateLink(
                        url=href,
                        anchor_text=anchor[:140],
                        is_pdf=href.lower().split("?", 1)[0].endswith(".pdf"),
                    )
                )

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = unescape(data)
        if self._in_title:
            self._title_parts.append(text)
        else:
            self._text_parts.append(text)
            if self._current_href is not None:
                self._current_link_text.append(text)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")

    def get_result(self) -> tuple[str, str | None, list[CandidateLink]]:
        raw_text = "".join(self._text_parts)
        lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in raw_text.splitlines()]
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()
        title = _MULTI_SPACE_RE.sub(" ", "".join(self._title_parts)).strip() or None
        return cleaned, title, self.links


def strip_html_to_text(
    html: str,
    *,
    base_url: str = "",
) -> tuple[str, str | None, list[CandidateLink]]:
    """Strip scripts, styles, nav chrome, and hidden elements from HTML."""
    extractor = _SafeHTMLTextExtractor(base_url=base_url)
    extractor.feed(html)
    extractor.close()
    return extractor.get_result()


def is_js_gated_text(text: str) -> bool:
    """Return ``True`` when stripped HTML looks like an empty JS-rendered shell."""
    stripped = text.strip()
    if len(stripped) < 200:
        return True
    lowered = stripped[:1500].lower()
    if any(marker in lowered for marker in _JS_SHELL_MARKERS) and len(stripped) < 800:
        return True
    return False


def extract_pdf_text(pdf_bytes: bytes, *, max_pages: int = 40) -> str:
    """Extract clean plain text from PDF bytes using PyMuPDF up to ``max_pages``."""
    if not pdf_bytes:
        return ""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[str] = []
        limit = min(len(doc), max(1, max_pages))
        for page_idx in range(limit):
            page_text = (doc[page_idx].get_text("text") or "").strip()
            if page_text:
                pages.append(page_text)
        return _MULTI_NEWLINE_RE.sub("\n\n", "\n\n".join(pages)).strip()
    finally:
        doc.close()


@dataclass(slots=True)
class FetchedPage:
    """Normalized result of fetching an allowlisted HTML page or PDF."""

    url: str
    domain: str
    title: str
    content_type: Literal["html", "pdf"]
    text: str
    readable: bool = True
    unreadable_reason: str | None = None
    candidate_links: list[CandidateLink] = field(default_factory=list)


async def fetch_allowlisted_url(
    url: str,
    *,
    fallback_title: str = "",
    timeout: float = 10.0,
    max_bytes: int = 15 * 1024 * 1024,
    max_pdf_pages: int = 40,
) -> FetchedPage:
    """Fetch an allowlisted URL (HTML or PDF), enforcing allowlist across redirects."""
    if not is_url_allowed(url):
        return FetchedPage(
            url=url,
            domain=extract_hostname(url),
            title=fallback_title or url,
            content_type="html",
            text="",
            readable=False,
            unreadable_reason="disallowed_domain",
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ClinicalGraphBot/0.1; +https://github.com/clinical-graph)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = await client.get(url)
    except Exception as exc:
        return FetchedPage(
            url=url,
            domain=extract_hostname(url),
            title=fallback_title or url,
            content_type="html",
            text="",
            readable=False,
            unreadable_reason=f"fetch_error: {type(exc).__name__}",
        )

    final_url = str(response.url)
    if not is_url_allowed(final_url):
        return FetchedPage(
            url=final_url,
            domain=extract_hostname(final_url),
            title=fallback_title or final_url,
            content_type="html",
            text="",
            readable=False,
            unreadable_reason="redirect_to_disallowed_domain",
        )

    if response.status_code >= 400:
        return FetchedPage(
            url=final_url,
            domain=extract_hostname(final_url),
            title=fallback_title or final_url,
            content_type="html",
            text="",
            readable=False,
            unreadable_reason=f"http_{response.status_code}",
        )

    if len(response.content) > max_bytes:
        return FetchedPage(
            url=final_url,
            domain=extract_hostname(final_url),
            title=fallback_title or final_url,
            content_type="html",
            text="",
            readable=False,
            unreadable_reason="payload_too_large",
        )

    content_type_header = (response.headers.get("content-type") or "").lower()
    is_pdf = "application/pdf" in content_type_header or (
        final_url.lower().split("?", 1)[0].endswith(".pdf")
        and response.content[:5] == b"%PDF-"
    )

    if is_pdf:
        try:
            pdf_text = await asyncio.to_thread(
                extract_pdf_text,
                response.content,
                max_pages=max_pdf_pages,
            )
        except Exception as exc:
            return FetchedPage(
                url=final_url,
                domain=extract_hostname(final_url),
                title=fallback_title or final_url,
                content_type="pdf",
                text="",
                readable=False,
                unreadable_reason=f"pdf_parse_error: {type(exc).__name__}",
            )
        if not pdf_text:
            return FetchedPage(
                url=final_url,
                domain=extract_hostname(final_url),
                title=fallback_title or final_url,
                content_type="pdf",
                text="",
                readable=False,
                unreadable_reason="empty_pdf_text",
            )
        return FetchedPage(
            url=final_url,
            domain=extract_hostname(final_url),
            title=fallback_title or final_url,
            content_type="pdf",
            text=pdf_text,
            readable=True,
        )

    plain_text, parsed_title, links = strip_html_to_text(response.text, base_url=final_url)
    title = parsed_title or fallback_title or final_url
    if is_js_gated_text(plain_text):
        return FetchedPage(
            url=final_url,
            domain=extract_hostname(final_url),
            title=title,
            content_type="html",
            text=plain_text,
            readable=False,
            unreadable_reason="js_required_or_empty",
            candidate_links=links,
        )

    return FetchedPage(
        url=final_url,
        domain=extract_hostname(final_url),
        title=title,
        content_type="html",
        text=plain_text,
        readable=True,
        candidate_links=links,
    )
