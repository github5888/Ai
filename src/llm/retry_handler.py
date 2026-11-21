"""Resilient retry & rate-limit primitives (Phase III + V).

* exponential backoff (2s, 4s, 8s, 16s ...) with full jitter
* ``Retry-After`` header support for HTTP 429
* per-host minimum interval ("polite crawling")
* global concurrency semaphore (bounded parallelism)
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional

log = logging.getLogger(__name__)

RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


class RetryExhausted(RuntimeError):
    """Raised when every attempt of a retry loop failed."""


@dataclass
class RetryPolicy:
    max_attempts: int = 5          # total attempts, including the first
    base_delay: float = 2.0        # seconds; attempt n waits base * factor**(n-1)
    factor: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.25           # +-25% full jitter around the nominal delay
    respect_retry_after: bool = True

    def nominal_delay(self, attempt: int) -> float:
        """attempt is 1-based (1 = first retry)."""
        delay = self.base_delay * (self.factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def compute_delay(self, attempt: int, retry_after: float | None = None) -> float:
        if self.respect_retry_after and retry_after and retry_after > 0:
            # honour the server's guidance with a little extra jitter
            return min(retry_after * random.uniform(1.0, 1.25), 300.0)
        nominal = self.nominal_delay(attempt)
        if self.jitter <= 0:
            return nominal
        spread = nominal * self.jitter
        return max(0.05, nominal + random.uniform(-spread, spread))


DEFAULT_HTTP_POLICY = RetryPolicy()


async def retry_async(
    factory: Callable[[], Awaitable[Any]],
    policy: RetryPolicy = DEFAULT_HTTP_POLICY,
    *,
    classify: Callable[[Any], tuple[bool, float | None]] | None = None,
    on_retry: Callable[[int, float, str], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Run ``factory()`` with retries.

    ``factory`` must be an async callable invoked fresh on every attempt.
    ``classify(result_or_exception)`` decides ``(retry?, retry_after_seconds)``;
    the default treats :class:`Exception` as retryable (except when the
    exception explicitly opts out via ``.no_retry = True``).

    Raises :class:`RetryExhausted` after ``max_attempts`` failures.
    """
    last_error: Any = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = await factory()
        except Exception as exc:  # noqa: BLE001 - classified below
            result = exc

        if classify is None:
            should_retry, retry_after = _default_classify(result)
        else:
            should_retry, retry_after = classify(result)

        if not should_retry:
            if isinstance(result, Exception):
                raise result
            return result

        last_error = result
        if attempt >= policy.max_attempts:
            break

        delay = policy.compute_delay(attempt, retry_after)
        if on_retry:
            on_retry(attempt, delay, repr(last_error)[:200])
        log.warning(
            "retry %s/%s after %.2fs due to %s",
            attempt, policy.max_attempts, delay, repr(last_error)[:160],
        )
        await sleep(delay)

    raise RetryExhausted(f"all {policy.max_attempts} attempts failed; last error: {last_error!r}")


class NoRetry(Exception):
    """Attach as ``exc.no_retry = True`` (or raise this) to stop retrying."""

    no_retry = True


def _default_classify(result: Any) -> tuple[bool, float | None]:
    if isinstance(result, Exception):
        if getattr(result, "no_retry", False):
            return False, None
        retry_after = getattr(result, "retry_after", None)
        return True, float(retry_after) if retry_after else None
    status = getattr(result, "status", None)
    if status is None:
        return False, None
    if status in RETRYABLE_HTTP_STATUS:
        retry_after = None
        headers = getattr(result, "headers", None)
        if headers:
            try:
                retry_after = float(headers.get("Retry-After") or 0) or None
            except (TypeError, ValueError):
                retry_after = None
        return True, retry_after
    return False, None


@dataclass
class HostRateLimiter:
    """Per-host sliding minimum-interval limiter + optional global lock."""

    min_interval: float = 0.25
    _last: dict = field(default_factory=dict)

    async def acquire(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        last = self._last.get(host, 0.0)
        wait = self.min_interval - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last[host] = time.monotonic()
