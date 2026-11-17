"""Per-client sliding-window rate limiter and FastAPI dependency."""

from collections import defaultdict, deque
import math
from threading import RLock
import time

from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by client identifier."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Record a request for ``key`` if within quota.

        Returns ``(allowed, retry_after_seconds)``.
        """
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._max_requests:
                retry_after = max(1, math.ceil(bucket[0] + self._window_seconds - now))
                return False, retry_after
            bucket.append(now)
            return True, 0


def _extract_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing per-client sliding-window rate limits."""
    settings = getattr(request.app.state, "settings", None)
    if settings is not None and not settings.rate_limit_enabled:
        return
    limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    client_ip = _extract_client_ip(request)
    allowed, retry_after = limiter.allow(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


__all__ = ["SlidingWindowRateLimiter", "check_rate_limit"]
