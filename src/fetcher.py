"""Asynchronous HTTP fetch layer (Phase V).

Design goals
------------
* **asyncio + aiohttp** with a global semaphore and per-host rate limiting,
  so 100 sources can be crawled concurrently without hammering any single host.
* **Robust error handling**: every request goes through the shared
  exponential-backoff retry helper (429/5xx aware, Retry-After aware).
* **JavaScript-rendered pages**: ``render()`` transparently upgrades to
  Playwright when it is installed; otherwise a documented
  :class:`BrowserUnavailable` error is raised so callers can gracefully
  fall back to official APIs / feeds (compliant strategy, no CAPTCHA bypass).
* **Robots-awareness**: ``robots_allowed()`` checks robots.txt before HTML
  crawls (APIs and RSS feeds are exempt by convention).
"""
from __future__ import annotations

import asyncio
import logging
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Optional
from urllib.parse import urlsplit

import aiohttp

from .config import Settings
from .llm.retry_handler import DEFAULT_HTTP_POLICY, HostRateLimiter, retry_async
from .utils.text import normalize_url

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=45, connect=15)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class BrowserUnavailable(RuntimeError):
    """Playwright/Chromium is not installed for JS-rendered sources."""


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status
        self.url = url
        self.body = body


@dataclass
class FetchResponse:
    status: int
    url: str
    text: str = ""
    headers: dict = field(default_factory=dict)
    raw: bytes = b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class Fetcher:
    """One Fetcher per pipeline run; safe to share across async tasks."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._session: aiohttp.ClientSession | None = None
        self._sem = asyncio.Semaphore(max(1, self.settings.concurrency))
        self._limiter = HostRateLimiter(min_interval=self.settings.request_min_interval)
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.stats = {"requests": 0, "retries": 0, "failures": 0, "rendered": 0}

    # -- session management -------------------------------------------------
    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=DEFAULT_TIMEOUT,
                headers={"User-Agent": self.settings.user_agent, "Accept": "*/*"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- core fetch ----------------------------------------------------------
    async def _raw_get(self, url: str, headers: dict | None = None) -> FetchResponse:
        session = await self.session()
        host = urlsplit(url).netloc
        async with self._sem:
            await self._limiter.acquire(host)
            self.stats["requests"] += 1
            try:
                async with session.get(url, headers=headers or {}, allow_redirects=True) as resp:
                    raw = await resp.read()
                    text = raw.decode("utf-8", errors="replace")
                    if resp.status in RETRYABLE_STATUS:
                        self.stats["retries"] += 1
                    return FetchResponse(resp.status, str(resp.url), text, dict(resp.headers), raw)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                raise HttpError(599, url, repr(exc)) from exc

    async def get(self, url: str, *, headers: dict | None = None,
                  max_attempts: int = 4) -> FetchResponse:
        """GET with shared retry policy.  4xx client errors (except 408/429)
        are not retried -- they are deterministic."""
        def classify(resp: FetchResponse | Exception):
            if isinstance(resp, Exception):
                return True, None  # network/timeout errors are retryable
            if resp.status in RETRYABLE_STATUS:
                retry_after = None
                try:
                    retry_after = float(resp.headers.get("Retry-After") or 0) or None
                except (TypeError, ValueError):
                    pass
                return True, retry_after
            return False, None

        policy = DEFAULT_HTTP_POLICY.__class__(
            max_attempts=max_attempts,
            base_delay=2.0, factor=2.0, max_delay=30.0, jitter=0.25,
        )

        def _on_retry(attempt: int, delay: float, err: str) -> None:
            self.stats["retries"] += 1

        resp = await retry_async(
            lambda: self._raw_get(url, headers),
            policy=policy,
            classify=classify,
            on_retry=_on_retry,
        )
        if not resp.ok:
            self.stats["failures"] += 1
        return resp

    async def get_json(self, url: str, *, headers: dict | None = None) -> Any:
        import json

        resp = await self.get(url, headers=headers)
        if not resp.ok:
            raise HttpError(resp.status, url, resp.text[:300])
        return json.loads(resp.text)

    async def get_many(self, urls: Iterable[str], handler: Callable[[FetchResponse], Awaitable[Any]],
                       *, headers: dict | None = None) -> list[Any]:
        """Concurrent fetch+handle over many URLs (bounded by the shared semaphore)."""
        async def _one(u: str) -> Any:
            try:
                resp = await self.get(u, headers=headers)
                if resp.ok:
                    return await handler(resp)
                log.warning("fetch failed %s -> HTTP %s", u, resp.status)
                return None
            except Exception as exc:  # noqa: BLE001
                log.warning("fetch error %s: %s", u, exc)
                return None

        return await asyncio.gather(*[_one(u) for u in urls])

    # -- JS-rendered sources (documented compliant strategy) ------------------
    async def render(self, url: str, *, wait_ms: int = 1200) -> str:
        """Render a JS-heavy page with Playwright (headless Chromium).

        Strategy documented in architecture.pdf:
          1. try plain HTTP first (cheap, respectful);
          2. only for high-value sources lacking static fallbacks, upgrade
             to a headless browser with realistic pacing;
          3. never touch CAPTCHA/anti-bot bypass services -- if a source
             aggressively blocks us, we drop it and log a graceful fallback.
        """
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError as exc:
            raise BrowserUnavailable(
                "playwright is not installed; skipping JS-rendered source "
                f"{url} (documented fallback: use official API/feed instead)"
            ) from exc
        self.stats["rendered"] += 1
        async with self._sem:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=self.settings.user_agent)
                try:
                    await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(wait_ms)
                    return await page.content()
                finally:
                    await browser.close()

    # -- robots.txt ------------------------------------------------------------
    def robots_allowed(self, url: str) -> bool:
        """Best-effort robots.txt check (HTML pages; APIs/feeds are exempt).
        Network failure => assume allowed (same as most crawlers' default)."""
        try:
            parts = urlsplit(url)
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            rp = self._robots.get(robots_url)
            if rp is None:
                import urllib.request

                rp = urllib.robotparser.RobotFileParser()
                try:
                    req = urllib.request.Request(robots_url, headers={"User-Agent": self.settings.user_agent})
                    with urllib.request.urlopen(req, timeout=5) as fh:  # noqa: S310
                        rp.parse(fh.read(64_000).decode("utf-8", "ignore").splitlines())
                except Exception:  # noqa: BLE001
                    self._robots[robots_url] = rp
                    return True  # unreachable robots.txt -> default allow
                self._robots[robots_url] = rp
            return rp.can_fetch(self.settings.user_agent, url)
        except Exception:  # noqa: BLE001
            return True

    # -- sync helper for scripts -----------------------------------------------
    @staticmethod
    def run(coro: Awaitable[Any]) -> Any:
        return asyncio.run(coro)


async def fetch_all_text(fetcher: Fetcher, urls: list[str]) -> list[str]:
    out: list[str] = []
    for r in await fetcher.get_many(urls, lambda resp: _text(resp)):
        out.append(r or "")
    return out


async def _text(resp: FetchResponse) -> str:
    return resp.text
