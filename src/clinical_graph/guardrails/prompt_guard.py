"""Prompt-injection detection client and deterministic local fallback scanner."""

import asyncio
from dataclasses import dataclass
import math
import re
from typing import Any

import httpx

_BLOCK_VERDICTS = frozenset(
    {
        "block",
        "blocked",
        "hard_block",
        "reject",
        "deny",
        "unsafe",
        "injection",
    }
)

# Zero-width and bidirectional override characters often used to evade text filters.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|bypass|override)\s+"
            r"(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|above|preceding|earlier|system|safety|developer)\s+"
            r"(?:instructions|prompts|rules|directives|constraints)\b",
            re.IGNORECASE,
        ),
        "instruction override pattern detected",
    ),
    (
        re.compile(
            r"\b(?:reveal|show|print|output|repeat|leak|dump|expose|display)\s+"
            r"(?:me\s+)?(?:the\s+|your\s+)?(?:(?:full|entire|exact|complete|verbatim|original|internal)\s+)*"
            r"(?:system\s+prompt|system\s+instructions|hidden\s+instructions|developer\s+prompt|"
            r"developer\s+instructions|initial\s+instructions)\b",
            re.IGNORECASE,
        ),
        "system prompt extraction pattern detected",
    ),
    (
        re.compile(
            r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|endoftext\|>|"
            r"<<SYS>>|<</SYS>>|\[INST\]|\[/INST\]|"
            r"(?:^|\n)\s*###\s*(?:SYSTEM|INSTRUCTION|DEVELOPER)\s*:)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "chat delimiter smuggling detected",
    ),
    (
        re.compile(
            r"\byou\s+are\s+now\s+(?:in\s+)?(?:developer|jailbreak|dan|unrestricted)\s+mode\b",
            re.IGNORECASE,
        ),
        "jailbreak role override detected",
    ),
)


@dataclass(frozen=True, slots=True)
class GuardCheckResult:
    """Result of evaluating a prompt against ``clinical-guard`` or local fallback."""

    reachable: bool
    allowed: bool
    blocked: bool
    reason: str | None = None


def scan_text_for_injection(text: str) -> tuple[bool, str | None]:
    """Deterministically scan ``text`` for prompt-injection and delimiter smuggling."""
    if not text:
        return False, None

    # Strip a single leading UTF-8 BOM if present, then flag remaining invisible chars.
    body = text[1:] if text.startswith("\ufeff") else text
    if _ZERO_WIDTH_RE.search(body):
        return True, "zero-width or bidirectional control character smuggling detected"

    normalized = _ZERO_WIDTH_RE.sub("", body)
    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True, reason

    return False, None


def _split_guard_windows(
    text: str,
    window_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[str]:
    """Split ``text`` into overlapping windows of at most ``window_chars`` characters."""
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    if not text or not text.strip():
        return []

    overlap = max(0, min(overlap_chars, window_chars - 1))
    n = len(text)
    if n <= window_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    windows: list[str] = []
    start = 0
    while start < n:
        end = min(n, start + window_chars)
        if end < n:
            split_at = max(
                text.rfind("\n", start + window_chars // 2, end),
                text.rfind(" ", start + window_chars // 2, end),
            )
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)

    return windows


def _parse_retry_after(response: httpx.Response, default: float = 1.0) -> float:
    """Parse ``Retry-After`` seconds header, falling back to ``default`` and clamping to ``[0.2, 10.0]``."""
    raw = response.headers.get("Retry-After")
    delay = default
    if raw is not None:
        try:
            delay = float(raw.strip())
        except ValueError:
            delay = default
    if not math.isfinite(delay):
        delay = default
    return max(0.2, min(10.0, delay))


class ClinicalGuardClient:
    """Async HTTP client for the ``clinical-guard`` prompt-injection service."""

    def __init__(self, base_url: str, timeout: float = 2.5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def check_health(self) -> bool:
        """Return ``True`` if ``GET {base_url}/health`` responds with 2xx within 1.0s."""
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return 200 <= response.status_code < 300
        except Exception:
            return False

    async def _post_prompt(
        self,
        client: httpx.AsyncClient,
        *,
        query: str,
        session_id: str,
        client_ip: str | None = None,
    ) -> httpx.Response | None:
        """POST ``/v1/guard/prompt`` with ``X-Forwarded-For`` and 429/503 ``Retry-After`` retries."""
        url = f"{self.base_url}/v1/guard/prompt"
        payload = {"query": query, "session_id": session_id}
        stripped_ip = client_ip.strip() if client_ip else ""
        headers = {"X-Forwarded-For": stripped_ip} if stripped_ip else None

        for attempt in range(3):
            kwargs: dict[str, Any] = {"json": payload}
            if headers is not None:
                kwargs["headers"] = headers
            response = await client.post(url, **kwargs)
            if response.status_code not in (429, 503):
                return response
            if attempt == 2:
                return None
            await asyncio.sleep(_parse_retry_after(response))
        return None

    async def check_prompt(
        self,
        query: str,
        session_id: str = "sess-prompt-001",
        client_ip: str | None = None,
    ) -> GuardCheckResult:
        """Check ``query`` via local regex rules first, then ``POST {base_url}/v1/guard/prompt``."""
        flagged, reason = scan_text_for_injection(query)
        if flagged:
            return GuardCheckResult(
                reachable=True,
                allowed=False,
                blocked=True,
                reason=reason,
            )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._post_prompt(
                    client,
                    query=query,
                    session_id=session_id,
                    client_ip=client_ip,
                )
        except Exception:
            return self._local_fallback(query)

        if response is None:
            return self._local_fallback(query)

        parsed = self._interpret_guard_response(response)
        if parsed is None:
            return self._local_fallback(query)
        return parsed

    async def scan_document(
        self,
        text: str,
        *,
        extra_texts: list[str] | None = None,
        session_id: str = "upload-scan",
        client_ip: str | None = None,
    ) -> GuardCheckResult:
        """Scan a document and its metadata/tables via local regex and concurrent guard windows."""
        extras = [item.strip() for item in (extra_texts or []) if item and item.strip()]
        for candidate in (text, *extras):
            hit, reason = scan_text_for_injection(candidate)
            if hit:
                return GuardCheckResult(
                    reachable=True,
                    allowed=False,
                    blocked=True,
                    reason=reason,
                )

        windows = _split_guard_windows(text)
        for extra in extras:
            windows.extend(_split_guard_windows(extra))

        if not windows:
            return GuardCheckResult(reachable=True, allowed=True, blocked=False)

        semaphore = asyncio.Semaphore(2)
        async with httpx.AsyncClient(timeout=self.timeout) as client:

            async def _scan_window(window: str) -> GuardCheckResult:
                async with semaphore:
                    try:
                        response = await self._post_prompt(
                            client,
                            query=window,
                            session_id=session_id,
                            client_ip=client_ip,
                        )
                    except Exception:
                        return self._local_fallback(window)
                    if response is None:
                        return self._local_fallback(window)
                    parsed = self._interpret_guard_response(response)
                    if parsed is None:
                        return self._local_fallback(window)
                    return parsed

            pending: set[asyncio.Task[GuardCheckResult]] = {
                asyncio.create_task(_scan_window(win)) for win in windows
            }
            try:
                while pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        try:
                            res = task.result()
                        except Exception:
                            await self._cancel_tasks(pending)
                            return GuardCheckResult(reachable=False, allowed=True, blocked=False)

                        if not res.reachable:
                            await self._cancel_tasks(pending)
                            return GuardCheckResult(reachable=False, allowed=True, blocked=False)

                        if res.blocked:
                            await self._cancel_tasks(pending)
                            return GuardCheckResult(
                                reachable=True,
                                allowed=False,
                                blocked=True,
                                reason=res.reason,
                            )
            except BaseException:
                await self._cancel_tasks(pending)
                raise

        return GuardCheckResult(reachable=True, allowed=True, blocked=False)

    @staticmethod
    async def _cancel_tasks(tasks: set[asyncio.Task[GuardCheckResult]]) -> None:
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @classmethod
    def _interpret_guard_response(cls, response: httpx.Response) -> GuardCheckResult | None:
        if response.status_code == 403:
            reason = cls._extract_reason(response) or "Blocked by clinical-guard (HTTP 403)"
            return GuardCheckResult(
                reachable=True,
                allowed=False,
                blocked=True,
                reason=reason,
            )

        if response.status_code >= 500 or not (200 <= response.status_code < 300):
            return None

        try:
            data = response.json()
        except Exception:
            return GuardCheckResult(reachable=True, allowed=True, blocked=False)

        if not isinstance(data, dict):
            return GuardCheckResult(reachable=True, allowed=True, blocked=False)

        is_blocked = (
            data.get("blocked") is True
            or data.get("is_injection") is True
            or data.get("safe") is False
            or data.get("is_safe") is False
            or data.get("allowed") is False
            or any(
                str(data.get(key) or "").strip().lower() in _BLOCK_VERDICTS
                for key in ("action", "decision", "verdict")
            )
        )
        reason = data.get("reason") or data.get("detail") or data.get("message")
        reason_str = str(reason) if reason else None
        return GuardCheckResult(
            reachable=True,
            allowed=not is_blocked,
            blocked=is_blocked,
            reason=reason_str if is_blocked else None,
        )

    @staticmethod
    def _extract_reason(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except Exception:
            return None
        if isinstance(payload, dict):
            raw = payload.get("reason") or payload.get("detail") or payload.get("message")
            return str(raw) if raw else None
        return None

    @staticmethod
    def _local_fallback(query: str) -> GuardCheckResult:
        hit, reason = scan_text_for_injection(query)
        return GuardCheckResult(
            reachable=False,
            allowed=not hit,
            blocked=hit,
            reason=reason,
        )


async def filter_web_snippets(
    snippets: list[str],
    guard_client: ClinicalGuardClient | None,
    *,
    session_id: str = "web-fallback-scan",
    client_ip: str | None = None,
) -> list[str]:
    """Filter web search snippets through local and ``clinical-guard`` injection checks.

    Fails closed: returns ``[]`` when ``guard_client`` is ``None`` or unreachable.
    """
    if not snippets or guard_client is None:
        return []

    non_empty = [snippet for snippet in snippets if snippet and snippet.strip()]
    if not non_empty:
        return []

    results = await asyncio.gather(
        *(
            guard_client.scan_document(
                snippet,
                session_id=session_id,
                client_ip=client_ip,
            )
            for snippet in non_empty
        )
    )
    if any(not res.reachable for res in results):
        return []

    return [
        snippet
        for snippet, res in zip(non_empty, results, strict=True)
        if res.allowed and not res.blocked
    ]
