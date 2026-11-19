"""Tavily Search API async client with parallel dominant and regional allowlist calls."""

import asyncio
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import httpx

from clinical_graph.search.allowlist import (
    DOMINANT_DOMAINS,
    REGIONAL_DOMAINS,
    classify_domain_tier,
    extract_hostname,
    is_url_allowed,
)
from clinical_graph.search.schemas import WebSearchResult

_TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"


def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def _looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


class TavilySearchClient:
    """Async client running parallel Tavily searches across dominant and regional allowlists."""

    def __init__(
        self,
        api_key: str | None,
        *,
        dominant_domains: list[str] | None = None,
        regional_domains: list[str] | None = None,
        search_depth: str = "basic",
        timeout: float = 10.0,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.dominant_domains = dominant_domains or sorted(DOMINANT_DOMAINS)
        self.regional_domains = regional_domains or sorted(REGIONAL_DOMAINS)
        self.search_depth = search_depth
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when a Tavily API key is configured."""
        return bool(self.api_key)

    async def _search_tier(
        self,
        client: httpx.AsyncClient,
        *,
        query: str,
        include_domains: list[str],
        tier_hint: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max(1, min(20, max_results)),
            "include_domains": include_domains,
            "include_domains_mode": "restrict",
            "include_raw_content": "text",
        }
        try:
            response = await client.post(_TAVILY_SEARCH_ENDPOINT, json=payload)
            if response.status_code != 200:
                return []
            data = response.json()
        except Exception:
            return []

        results = data.get("results") or []
        tagged: list[dict[str, Any]] = []
        for idx, item in enumerate(results):
            if isinstance(item, dict):
                tagged.append({**item, "_tier_hint": tier_hint, "_rank": idx})
        return tagged

    async def search_parallel(
        self,
        query: str,
        *,
        tier: Literal["dominant", "regional", "both"] = "both",
        allow_pdf: bool = False,
        results_per_tier: int = 3,
        result_id_prefix: str = "w",
        start_index: int = 1,
    ) -> list[WebSearchResult]:
        """Run Tavily search(es) across ``dominant``, ``regional``, or ``both`` tiers."""
        if not self.is_configured or not query.strip():
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks: list[Any] = []
            if tier in ("dominant", "both"):
                tasks.append(
                    self._search_tier(
                        client,
                        query=query,
                        include_domains=self.dominant_domains,
                        tier_hint="dominant",
                        max_results=results_per_tier,
                    )
                )
            if tier in ("regional", "both"):
                tasks.append(
                    self._search_tier(
                        client,
                        query=query,
                        include_domains=self.regional_domains,
                        tier_hint="regional",
                        max_results=results_per_tier,
                    )
                )
            batches = await asyncio.gather(*tasks)

        raw_items = [item for batch in batches for item in batch]
        seen_urls: set[str] = set()
        filtered: list[tuple[int, int, WebSearchResult]] = []
        counter = start_index

        for raw in raw_items:
            url = str(raw.get("url") or "").strip()
            if not url or not is_url_allowed(url):
                continue
            is_pdf = _looks_like_pdf(url)
            if is_pdf and not allow_pdf:
                continue
            canon = _canonicalize_url(url)
            if canon in seen_urls:
                continue
            seen_urls.add(canon)

            classified = classify_domain_tier(url)
            resolved_tier = "regional" if classified == "regional" else "dominant"
            raw_content = raw.get("raw_content")
            result = WebSearchResult(
                result_id=f"{result_id_prefix}{counter}",
                url=url,
                title=str(raw.get("title") or extract_hostname(url) or url).strip(),
                snippet=str(raw.get("content") or "").strip(),
                raw_content=str(raw_content).strip() if isinstance(raw_content, str) and raw_content.strip() else None,
                domain=extract_hostname(url),
                tier=resolved_tier,
                is_pdf=is_pdf,
            )
            counter += 1
            # Prefer HTML/structured drug pages ahead of PDFs at equal search rank.
            filtered.append((1 if is_pdf else 0, int(raw.get("_rank", 0)), result))

        filtered.sort(key=lambda entry: (entry[0], entry[1]))
        return [entry[2] for entry in filtered]
