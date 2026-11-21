"""LLM provider adapters for the multi-tier fallback chain (Phase III).

Chain: Gemini Flash (primary) -> Groq Llama-3 (fallback 1) -> DeepSeek (fallback 2).

Every provider speaks its own REST API directly over aiohttp (no SDK needed),
and maps transport failures onto precise exception types so the orchestrator
can make the right recovery decision:

    429  -> RateLimitError          (retry with exponential backoff + jitter)
    413 or context-overflow body -> PayloadTooLarge  (intelligent chunking)
    401/403 -> ProviderAuthError    (no retry; fail over to next provider)
    other -> ProviderError          (fail over)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from .json_utils import extract_json

log = logging.getLogger(__name__)

# Provider payloads that indicate a context-window overflow even when the
# HTTP status is 400/500 rather than a literal 413.
_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    "payload too large",
    "input too large",
    "too many tokens",
    "exceeds the maximum",
    "token limit",
)


class ProviderError(RuntimeError):
    """Generic provider failure -> fail over to the next tier."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimitError(ProviderError):
    """HTTP 429 - Too Many Requests. Carries Retry-After when present."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, status=429)
        self.retry_after = retry_after


class PayloadTooLarge(ProviderError):
    """HTTP 413 (or context overflow) - triggers intelligent chunking."""

    def __init__(self, message: str):
        super().__init__(message, status=413)


class ProviderAuthError(ProviderError):
    """Bad/missing credentials - never retried."""


class BaseLLMProvider(ABC):
    name: str = "base"

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_text: str) -> dict:
        """Return the parsed JSON object produced by the model."""

    # -- shared plumbing ------------------------------------------------------
    async def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        async with self.session.post(url, json=payload, headers=headers) as resp:
            body = await resp.text()
            if resp.status == 429:
                retry_after = None
                try:
                    retry_after = float(resp.headers.get("Retry-After") or 0) or None
                except (TypeError, ValueError):
                    pass
                raise RateLimitError(f"{self.name}: 429 rate limited", retry_after=retry_after)
            if resp.status == 413:
                raise PayloadTooLarge(f"{self.name}: 413 payload too large")
            if resp.status in (401, 403):
                raise ProviderAuthError(f"{self.name}: auth failed ({resp.status})")
            if resp.status >= 400:
                lower = body.lower()
                if any(marker in lower for marker in _CONTEXT_OVERFLOW_MARKERS):
                    raise PayloadTooLarge(f"{self.name}: context overflow on {resp.status}")
                raise ProviderError(f"{self.name}: HTTP {resp.status}: {body[:200]}", status=resp.status)
            try:
                return await resp.json(content_type=None)
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"{self.name}: invalid JSON response") from exc

    @staticmethod
    def _looks_like_overflow(status: int, body: str) -> bool:
        if status == 413:
            return True
        lower = body.lower()
        return any(marker in lower for marker in _CONTEXT_OVERFLOW_MARKERS)


class GeminiFlashProvider(BaseLLMProvider):
    """Google Gemini 2.0 Flash via Generative Language API (json mime mode)."""

    name = "gemini-flash"

    def __init__(self, session: aiohttp.ClientSession, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__(session)
        self.api_key = api_key
        self.model = model

    async def complete_json(self, system_prompt: str, user_text: str) -> dict:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_text}"}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
        }
        data = await self._post_json(url, payload, {"x-goog-api-key": self.api_key})
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        return extract_json(text)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Shared implementation for Groq and DeepSeek (OpenAI-style chat API)."""

    name = "openai-compatible"
    endpoint = ""
    api_key = ""
    model = ""

    async def complete_json(self, system_prompt: str, user_text: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        data = await self._post_json(
            self.endpoint, payload, {"Authorization": f"Bearer {self.api_key}"}
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        return extract_json(text)


class GroqLlamaProvider(OpenAICompatibleProvider):
    """Groq-hosted Llama 3.3 70B (fallback tier 1)."""

    name = "groq-llama3"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, session: aiohttp.ClientSession, api_key: str,
                 model: str = "llama-3.3-70b-versatile"):
        super().__init__(session)
        self.api_key = api_key
        self.model = model


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek chat (fallback tier 2)."""

    name = "deepseek"
    endpoint = "https://api.deepseek.com/chat/completions"

    def __init__(self, session: aiohttp.ClientSession, api_key: str,
                 model: str = "deepseek-chat"):
        super().__init__(session)
        self.api_key = api_key
        self.model = model
