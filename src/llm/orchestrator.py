"""Multi-tier LLM extraction orchestrator (Phase III).

Responsibilities
----------------
1. Build the provider chain from configuration: Gemini Flash -> Groq Llama 3
   -> DeepSeek.  Providers without API keys are skipped.
2. For every extraction: chunk oversized payloads *semantically*, run chunks
   through the provider with the shared retry policy (429 -> exponential
   backoff with jitter, honouring Retry-After), merge chunk results, and
   validate against the canonical pydantic schemas.
3. Handle 413 by *shrinking and re-chunking* (never blind truncation), and
   fail over to the next provider when a tier is exhausted.
4. When NO provider is configured (or all fail), fall back to the
   deterministic extractor and record that honestly in the record metadata
   (``extractionMode``) -- fabricated data is an automatic disqualification,
   so unstructured fields we cannot derive stay null.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp

from ..config import RECORD_STARTUP, Settings
from .chunking import merge_chunk_results, semantic_chunks
from .providers import (
    BaseLLMProvider,
    DeepSeekProvider,
    GeminiFlashProvider,
    GroqLlamaProvider,
    PayloadTooLarge,
    ProviderAuthError,
    ProviderError,
    RateLimitError,
)
from .retry_handler import RetryExhausted, RetryPolicy, retry_async
from .schemas import build_envelope
from .json_utils import as_list

log = logging.getLogger(__name__)

LIST_FIELDS_FOR = {
    "RESEARCH_PAPER": ("authors",),
    "STARTUP": ("products", "founders", "aliases"),
    "PRODUCT": ("tags",),
    "JOB": (),
    "NEWS": (),
}

SYSTEM_PROMPT = """You are a precise data-extraction engine for an AI-industry
intelligence graph. Extract ONLY facts explicitly present in the provided text.
Never invent companies, products, numbers, dates or URLs. If a field is not
stated in the text, return null for it. Output a single JSON object, no prose."""


@dataclass
class OrchestratorStats:
    extractions: int = 0
    provider_successes: dict = field(default_factory=dict)
    provider_failures: dict = field(default_factory=dict)
    rate_limit_retries: int = 0
    chunked_payloads: int = 0
    deterministic_fallbacks: int = 0


class ExtractionOrchestrator:
    """Chain-of-responsibility over LLM providers with graceful degradation."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: aiohttp.ClientSession | None = None,
        providers: list[BaseLLMProvider] | None = None,
    ):
        self.settings = settings or Settings()
        self._own_session = session is None
        self.session = session
        self.stats = OrchestratorStats()
        self.providers = providers if providers is not None else self._build_chain()

    def _build_chain(self) -> list[BaseLLMProvider]:
        s = self.settings
        chain: list[BaseLLMProvider] = []
        if s.gemini_api_key:
            chain.append(GeminiFlashProvider(self._sess(), s.gemini_api_key, s.gemini_model))
        if s.groq_api_key:
            chain.append(GroqLlamaProvider(self._sess(), s.groq_api_key, s.groq_model))
        if s.deepseek_api_key:
            chain.append(DeepSeekProvider(self._sess(), s.deepseek_api_key, s.deepseek_model))
        log.info("LLM fallback chain: %s", [p.name for p in chain] or "DETERMINISTIC-ONLY")
        return chain

    def _sess(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("orchestrator session accessed before initialisation")
        return self.session

    async def open(self) -> None:
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                headers={"User-Agent": self.settings.user_agent},
            )

    async def close(self) -> None:
        if self._own_session and self.session and not self.session.closed:
            await self.session.close()

    # -- public API ------------------------------------------------------------
    async def extract(
        self,
        text: str,
        record_type: str,
        source_name: str,
        source_url: str,
        *,
        user_hint: str = "",
        collected_at: str | None = None,
    ) -> dict | None:
        """Extract one record from raw text, returning a validated envelope."""
        self.stats.extractions += 1
        prompt = self._prompt_for(record_type, user_hint)

        for provider in self.providers:
            try:
                merged = await self._extract_via(provider, prompt, text, record_type)
            except PayloadTooLarge as exc:
                log.warning("%s 413/context overflow (%s); shrinking chunks", provider.name, exc)
                merged = await self._extract_chunked_shrinking(provider, prompt, text, record_type)
                if merged is None:
                    self.stats.provider_failures[provider.name] = (
                        self.stats.provider_failures.get(provider.name, 0) + 1
                    )
                    continue
            except RateLimitError as exc:
                log.warning("%s rate limit exhausted after retries (%s)", provider.name, exc)
                self.stats.provider_failures[provider.name] = (
                    self.stats.provider_failures.get(provider.name, 0) + 1
                )
                continue
            except RetryExhausted as exc:
                # retries for 429/5xx ran out on this tier -> fail over gracefully
                log.warning("%s retries exhausted (%s); failing over", provider.name, exc)
                self.stats.provider_failures[provider.name] = (
                    self.stats.provider_failures.get(provider.name, 0) + 1
                )
                continue
            except (ProviderAuthError, ProviderError) as exc:
                log.warning("%s failed (%s); failing over", provider.name, exc)
                self.stats.provider_failures[provider.name] = (
                    self.stats.provider_failures.get(provider.name, 0) + 1
                )
                continue

            self.stats.provider_successes[provider.name] = (
                self.stats.provider_successes.get(provider.name, 0) + 1
            )
            envelope = build_envelope(record_type, source_name, source_url, merged, collected_at)
            envelope["extractionMode"] = f"llm:{provider.name}"
            return envelope

        # No provider succeeded (or none configured) -> deterministic fallback.
        self.stats.deterministic_fallbacks += 1
        deterministic = deterministic_extract(text, record_type)
        if deterministic is None:
            log.warning("deterministic extractor found nothing for %s", source_url)
            return None
        envelope = build_envelope(record_type, source_name, source_url, deterministic, collected_at)
        envelope["extractionMode"] = "deterministic"
        return envelope

    # -- internals ---------------------------------------------------------------
    async def _extract_via(self, provider: BaseLLMProvider, prompt: str, text: str,
                           record_type: str) -> dict:
        chunks = semantic_chunks(text, self.settings.llm_chunk_chars)
        policy = RetryPolicy(
            max_attempts=self.settings.llm_max_attempts,
            base_delay=self.settings.llm_backoff_base,
            max_delay=self.settings.llm_backoff_max_delay,
            jitter=self.settings.llm_backoff_jitter,
        )

        async def attempt(chunk_text: str) -> dict:
            def on_retry(attempt: int, delay: float, err: str) -> None:
                self.stats.rate_limit_retries += 1
                log.info("retry %s attempt=%d wait=%.1fs (%s)", provider.name, attempt, delay, err[:120])

            return await retry_async(
                lambda: provider.complete_json(prompt, chunk_text),
                policy=policy,
                classify=_classify_provider_error,
                on_retry=on_retry,
            )

        if len(chunks) <= 1:
            self.stats.chunked_payloads += len(chunks) - 1
            return await attempt(text)

        self.stats.chunked_payloads += len(chunks)
        results = await asyncio.gather(*[attempt(c.text) for c in chunks])
        return merge_chunk_results(
            [r for r in results if isinstance(r, dict)],
            list_fields=LIST_FIELDS_FOR.get(record_type, ()),
            max_fields=("github_stars", "employeeCount"),
        )

    async def _extract_chunked_shrinking(self, provider: BaseLLMProvider, prompt: str,
                                         text: str, record_type: str) -> dict | None:
        """413 escalation path: halve the chunk size up to 3 times."""
        chunk_chars = self.settings.llm_chunk_chars // 2
        for _ in range(3):
            try:
                chunks = semantic_chunks(text, chunk_chars)
                results = await asyncio.gather(
                    *[provider.complete_json(prompt, c.text) for c in chunks]
                )
                return merge_chunk_results(
                    [r for r in results if isinstance(r, dict)],
                    list_fields=LIST_FIELDS_FOR.get(record_type, ()),
                    max_fields=("github_stars", "employeeCount"),
                )
            except PayloadTooLarge:
                chunk_chars //= 2
            except (RateLimitError, ProviderError):
                return None
        return None

    @staticmethod
    def _prompt_for(record_type: str, user_hint: str) -> str:
        base = {
            "STARTUP": (
                'Extract the startup/company described by the text. JSON keys: '
                '{"entityName": str, "data": {"employeeCount": int|null}, "description": str|null}.'
            ),
            "PRODUCT": (
                'Extract the AI product described by the text. JSON keys: '
                '{"productName": str, "startupName": str|null, "pricingModel": '
                '"FREE"|"FREEMIUM"|"PAID"|"ENTERPRISE"|null}.'
            ),
            "RESEARCH_PAPER": (
                'Extract the research paper. JSON keys: {"title": str, "authors": [str], '
                '"paper_url": str|null, "github_url": str|null, "github_stars": int|null, '
                '"published_date": "ISO-8601"|null}.'
            ),
            "JOB": (
                'Extract the job posting. JSON keys: {"company": str, "jobTitle": str, '
                '"date": "ISO-8601"|null, "is_remote": bool, "role_family": str}.'
            ),
            "NEWS": (
                'Extract the news article metadata. JSON keys: {"headline": str, '
                '"publishedAt": "ISO-8601"|null, "content": str}.'
            ),
        }.get(record_type, "")
        return SYSTEM_PROMPT + "\n\n" + base + ("\n\n" + user_hint if user_hint else "")


def _classify_provider_error(result_or_exc):
    """Shared classification for provider retries.

    Retryable: RateLimitError (429, with Retry-After) and transient network
    errors.  Not retryable: 413 (handled by chunking escalation), auth errors,
    bad request shapes."""
    if isinstance(result_or_exc, RateLimitError):
        return True, result_or_exc.retry_after
    if isinstance(result_or_exc, (PayloadTooLarge, ProviderAuthError)):
        return False, None
    if isinstance(result_or_exc, ProviderError):
        return True, None  # transient 5xx/timeouts worth retrying
    if isinstance(result_or_exc, Exception):
        return True, None
    return False, None


# --------------------------------------------------------------------------
# Deterministic extractor (no-LLM fallback; regex/keyword based)
# --------------------------------------------------------------------------
_EMPLOYEES_RE = re.compile(
    r"(?:~|about|over|more than|approximately)?\s*"
    r"(\d{1,3}(?:[,\s]\d{3})*|\d{2,6})\s*(?:\+)?\s*"
    r"(?:full[- ]time )?(?:employees|people|team members|staff members|staff)\b",
    re.IGNORECASE,
)
_PRICING_RULES = [
    (re.compile(r"\benterprise[- ](grade|plan|only|ready)\b|\bfor enterprise\b", re.I), "ENTERPRISE"),
    (re.compile(r"\bfreemium\b|\bfree trial\b", re.I), "FREEMIUM"),
    (re.compile(r"\bfree\b[^.\n]{0,40}\b(paid|premium|pro plan)s?\b|\b(paid|premium)\b[^.\n]{0,40}\bfree\b", re.I), "FREEMIUM"),
    (re.compile(r"\bpaid\b|\bsubscription\b|\bpricing starts\b|\$\d|\bper month\b|\bper user\b|\b/\s?month\b", re.I), "PAID"),
    (re.compile(r"\bopen[- ]?source\b|\bfree\b|\bno cost\b|\bfree to try\b", re.I), "FREE"),
]


def classify_pricing(text: str | None) -> str | None:
    """Keyword-based pricing classification; None when the source says nothing."""
    if not text:
        return None
    for regex, label in _PRICING_RULES:
        if regex.search(text):
            return label
    return None


def deterministic_extract(text: str, record_type: str) -> dict | None:
    """Rule-based extraction used when no LLM is configured.

    Honest by design: only patterns literally present in the text are
    extracted; everything else stays null."""
    if not text:
        return None
    snippet = text.strip()

    if record_type == RECORD_STARTUP:
        employees = None
        m = _EMPLOYEES_RE.search(text)
        if m:
            digits = re.sub(r"[^\d]", "", m.group(1))
            if digits:
                employees = int(digits)
        name_match = re.match(r"\s*([A-Z][\w&.'\- ]{1,60})", snippet)
        return {
            "entityName": (name_match.group(1).strip() if name_match else snippet[:60].split(".")[0]),
            "data": {"employeeCount": employees},
            "description": snippet[:400] or None,
        }
    return None
