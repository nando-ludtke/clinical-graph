"""Async LiteLLM client wrapper supporting structured Pydantic outputs and usage tracking."""

import asyncio
import inspect
import logging
import random
import time
from typing import Any
from unittest.mock import AsyncMock, NonCallableMock
import weakref

import litellm
from pydantic import BaseModel

from clinical_graph.schemas.metrics import LLMUsage

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})
_NON_RETRYABLE_QUOTA_MARKERS: tuple[str, ...] = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if ``exc`` represents an HTTP 429 / rate-limit error."""
    return isinstance(exc, litellm.RateLimitError) or getattr(exc, "status_code", None) == 429


def _is_transient_error(exc: Exception) -> bool:
    """Return True if ``exc`` is a retryable transient LLM provider error."""
    exc_text = f"{exc} {getattr(exc, 'code', '') or ''} {getattr(exc, 'type', '') or ''}".lower()
    if any(marker in exc_text for marker in _NON_RETRYABLE_QUOTA_MARKERS):
        return False
    if isinstance(
        exc,
        (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            litellm.Timeout,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS_CODES


_is_retryable_error = _is_transient_error


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract numeric ``Retry-After`` seconds from an exception if present."""
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        headers = getattr(exc, "headers", None)
        if headers is None:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


def _extract_usage(response: Any) -> LLMUsage:
    """Extract token usage and estimated USD cost from a LiteLLM completion response."""
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
    elif isinstance(raw_usage, dict):
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or 0)
    else:
        prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(raw_usage, "total_tokens", 0) or 0)

    try:
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        cost = 0.0

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
    )


class LiteLLMClient:
    """Thin async wrapper around ``litellm.acompletion`` with concurrency limits and backoff."""

    def __init__(
        self,
        max_concurrent_calls: int = 5,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        timeout: float = 45.0,
    ) -> None:
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if base_retry_delay < 0:
            raise ValueError("base_retry_delay must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.max_concurrent_calls = max_concurrent_calls
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.timeout = timeout
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
        self._cooldown_until: float = 0.0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return an ``asyncio.Semaphore`` bound to the currently running event loop."""
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrent_calls)
            self._semaphores[loop] = semaphore
        return semaphore

    async def _wait_for_cooldown(self) -> None:
        remaining = self._cooldown_until - time.monotonic()
        if remaining > 0.0:
            await asyncio.sleep(remaining)

    def _compute_retry_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = _extract_retry_after(exc)
        if retry_after is not None:
            return retry_after
        return self.base_retry_delay * (2**attempt) + random.uniform(0.0, 0.5)

    async def complete_with_usage(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, LLMUsage]:
        """Execute a chat completion via LiteLLM and return ``(text, LLMUsage)``."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if api_key is not None:
            kwargs["api_key"] = api_key
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        semaphore = self._get_semaphore()
        response: Any = None
        for attempt in range(self.max_retries + 1):
            await self._wait_for_cooldown()
            async with semaphore:
                await self._wait_for_cooldown()
                try:
                    response = await asyncio.wait_for(
                        litellm.acompletion(**kwargs),
                        timeout=self.timeout + 5.0,
                    )
                    break
                except Exception as exc:
                    if not _is_transient_error(exc):
                        raise
                    delay = self._compute_retry_delay(exc, attempt)
                    if _is_rate_limit_error(exc):
                        self._cooldown_until = max(
                            self._cooldown_until,
                            time.monotonic() + delay,
                        )
                    if attempt >= self.max_retries:
                        raise
                    logger.warning(
                        "LiteLLM call transient error (%s); retrying in %.2fs (attempt %d/%d)",
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
            await asyncio.sleep(delay)

        content: str | None = response.choices[0].message.content
        return (content or ""), _extract_usage(response)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: type[BaseModel] | None = None,
        api_key: str | None = None,
    ) -> str:
        """Execute a chat completion via LiteLLM and return the response text."""
        text, _ = await self.complete_with_usage(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            api_key=api_key,
        )
        return text


_ORIG_COMPLETE = LiteLLMClient.complete
_ORIG_COMPLETE_WITH_USAGE = LiteLLMClient.complete_with_usage


async def complete_with_usage_compat(
    llm_client: LiteLLMClient,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    response_format: type[BaseModel] | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, LLMUsage]:
    """Call ``complete_with_usage`` when available, falling back to ``complete`` for legacy mocks."""
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "api_key": api_key,
    }
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        call_kwargs["reasoning_effort"] = reasoning_effort

    if isinstance(llm_client, NonCallableMock):
        method = getattr(llm_client, "complete_with_usage", None)
        is_configured = (
            isinstance(method, AsyncMock)
            and (
                method.side_effect is not None
                or not isinstance(method.return_value, NonCallableMock)
            )
        )
        if is_configured:
            return await method(**call_kwargs)
        text = await llm_client.complete(**call_kwargs)
        return text, LLMUsage()

    client_cls = type(llm_client)
    if issubclass(client_cls, LiteLLMClient):
        complete_patched = (
            "complete" in getattr(llm_client, "__dict__", {})
            or client_cls.complete is not _ORIG_COMPLETE
        )
        with_usage_patched = (
            "complete_with_usage" in getattr(llm_client, "__dict__", {})
            or client_cls.complete_with_usage is not _ORIG_COMPLETE_WITH_USAGE
        )
        if complete_patched and not with_usage_patched:
            text = await llm_client.complete(**call_kwargs)
            return text, LLMUsage()
        return await llm_client.complete_with_usage(**call_kwargs)

    method = getattr(llm_client, "complete_with_usage", None)
    if inspect.iscoroutinefunction(method):
        return await method(**call_kwargs)

    text = await llm_client.complete(**call_kwargs)
    return text, LLMUsage()
