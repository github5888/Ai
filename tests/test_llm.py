"""Phase III tests: retry/backoff/jitter, chunking, JSON recovery, fallback chain."""
from __future__ import annotations

import asyncio

import pytest

from src.config import RECORD_STARTUP, Settings
from src.llm.chunking import merge_chunk_results, semantic_chunks
from src.llm.json_utils import extract_json
from src.llm.orchestrator import ExtractionOrchestrator, deterministic_extract
from src.llm.providers import PayloadTooLarge, ProviderError, RateLimitError
from src.llm.retry_handler import RetryPolicy, retry_async
from src.llm.schemas import build_envelope


# ---------------------------------------------------------------------------
# Retry policy: exponential backoff + jitter (429 handling)
# ---------------------------------------------------------------------------
def test_exponential_backoff_growth():
    policy = RetryPolicy(base_delay=2.0, factor=2.0, max_delay=60.0, jitter=0.0)
    assert policy.nominal_delay(1) == 2.0   # 1st retry -> 2s
    assert policy.nominal_delay(2) == 4.0   # 2nd retry -> 4s
    assert policy.nominal_delay(3) == 8.0
    assert policy.nominal_delay(4) == 16.0
    assert policy.nominal_delay(7) == 60.0  # capped


def test_jitter_stays_in_band():
    policy = RetryPolicy(base_delay=8.0, jitter=0.25)
    for _ in range(50):
        delay = policy.compute_delay(1)
        assert 8.0 * 0.75 <= delay <= 8.0 * 1.25


def test_retry_after_header_wins():
    policy = RetryPolicy(base_delay=2.0, jitter=0.0, respect_retry_after=True)
    delay = policy.compute_delay(1, retry_after=17.0)
    assert 17.0 <= delay <= 17.0 * 1.25 + 0.01


def test_retry_async_eventually_succeeds():
    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429", retry_after=0.01)
        return {"ok": True}

    policy = RetryPolicy(max_attempts=5, base_delay=0.01, jitter=0.0)
    result = asyncio.run(retry_async(flaky, policy=policy))
    assert result == {"ok": True}
    assert attempts["n"] == 3


def test_retry_async_exhausts_and_raises():
    async def always_429():
        raise RateLimitError("429", retry_after=0.01)

    policy = RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(retry_async(always_429, policy=policy))
    assert "all 3 attempts" in str(excinfo.value)


def test_413_is_not_retried_but_raised_immediately():
    calls = {"n": 0}

    async def too_big():
        calls["n"] += 1
        raise PayloadTooLarge("413")

    policy = RetryPolicy(max_attempts=5, base_delay=0.01)

    def classify(exc):
        if isinstance(exc, PayloadTooLarge):
            return False, None  # chunking escalation handles this, not retries
        return True, None

    with pytest.raises(PayloadTooLarge):
        asyncio.run(retry_async(too_big, policy=policy, classify=classify))
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Intelligent chunking (413 prevention)
# ---------------------------------------------------------------------------
LONG_TEXT = "\n\n".join(
    f"## Section {i}\nParagraph {i} " + ("content about AI systems. " * 12)
    for i in range(30)
)


def test_semantic_chunks_respect_max_size():
    chunks = semantic_chunks(LONG_TEXT, max_chars=2000, overlap_chars=0)
    assert len(chunks) > 1
    assert all(chunk.chars <= 2200 for chunk in chunks)  # small tolerance
    joined = "\n".join(c.text for c in chunks)
    assert "Section 0" in joined and "Section 29" in joined  # no content loss


def test_chunks_are_semantically_aligned():
    chunks = semantic_chunks(LONG_TEXT, max_chars=1500, overlap_chars=200)
    # headings start chunks, never split mid-word
    for chunk in chunks:
        assert "Section" in chunk.text or chunk.index == 0


def test_small_text_single_chunk():
    assert len(semantic_chunks("short", max_chars=12000)) == 1


def test_merge_chunk_results_unions_lists_and_maxes_numbers():
    results = [
        {"authors": ["A", "B"], "github_stars": 12, "title": "Paper"},
        {"authors": ["B", "C"], "github_stars": 40, "title": None},
    ]
    merged = merge_chunk_results(results, list_fields=("authors",), max_fields=("github_stars",))
    assert merged["authors"] == ["A", "B", "C"]
    assert merged["github_stars"] == 40
    assert merged["title"] == "Paper"


# ---------------------------------------------------------------------------
# JSON recovery from model output
# ---------------------------------------------------------------------------
def test_extract_json_from_fenced_block():
    text = "Here you go:\n```json\n{\"entityName\": \"OpenAI\"}\n```"
    assert extract_json(text) == {"entityName": "OpenAI"}


def test_extract_json_from_prose():
    text = 'The result is {"a": 1, "b": [2, 3]} as requested.'
    assert extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


# ---------------------------------------------------------------------------
# Orchestrator: fallback chain, 413 shrink-chunking, deterministic fallback
# ---------------------------------------------------------------------------
class FakeProvider:
    def __init__(self, name, behaviour):
        self.name = name
        self.behaviour = behaviour  # callable(text) -> dict | raises
        self.calls = []

    async def complete_json(self, system_prompt, text):
        self.calls.append(text)
        return self.behaviour(text)


def _test_settings(**overrides) -> Settings:
    defaults = dict(
        llm_max_attempts=2, llm_backoff_base=0.01, llm_backoff_max_delay=0.02,
        llm_backoff_jitter=0.0, llm_chunk_chars=1200,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_fallback_chain_skips_rate_limited_primary():
    primary = FakeProvider("gemini-flash", lambda t: (_ for _ in ()).throw(RateLimitError("429")))
    secondary = FakeProvider("groq-llama3", lambda t: {"entityName": "OpenAI", "data": {"employeeCount": None}})

    orchestrator = ExtractionOrchestrator(
        settings=_test_settings(), providers=[primary, secondary]
    )
    envelope = asyncio.run(orchestrator.extract(
        "OpenAI builds safe AI.", RECORD_STARTUP, "Test", "https://example.com/o"
    ))
    assert envelope["content"]["entityName"] == "OpenAI"
    assert envelope["extractionMode"] == "llm:groq-llama3"
    assert envelope["recordType"] == "STARTUP"
    assert envelope["schemaVersion"] == "1.0"
    assert envelope["source"]["url"] == "https://example.com/o"


def test_fallback_chain_after_provider_error():
    primary = FakeProvider("gemini-flash", lambda t: (_ for _ in ()).throw(ProviderError("boom")))
    secondary = FakeProvider("groq-llama3", lambda t: {"entityName": "Anthropic"})

    orchestrator = ExtractionOrchestrator(
        settings=_test_settings(), providers=[primary, secondary]
    )
    envelope = asyncio.run(orchestrator.extract(
        "Anthropic text", RECORD_STARTUP, "Test", "https://example.com/a"
    ))
    assert envelope["content"]["entityName"] == "Anthropic"


def test_413_triggers_intelligent_chunking():
    big_text = "\n\n".join(f"Chunk block number {i} with padding text. " * 3 for i in range(120))

    def behaviour(text):
        if len(text) > 400:
            raise PayloadTooLarge("413")
        return {"entityName": "ChunkCorp", "data": {"employeeCount": None}}

    provider = FakeProvider("gemini-flash", behaviour)
    orchestrator = ExtractionOrchestrator(
        settings=_test_settings(llm_chunk_chars=1000), providers=[provider]
    )
    envelope = asyncio.run(orchestrator.extract(
        big_text, RECORD_STARTUP, "Test", "https://example.com/c"
    ))
    assert envelope["content"]["entityName"] == "ChunkCorp"
    assert orchestrator.stats.chunked_payloads > 0


def test_deterministic_fallback_when_all_providers_fail():
    provider = FakeProvider("gemini-flash", lambda t: (_ for _ in ()).throw(ProviderError("down")))
    orchestrator = ExtractionOrchestrator(settings=_test_settings(), providers=[provider])
    envelope = asyncio.run(orchestrator.extract(
        "Anthropic develops Claude.\nAround 1,500 employees work here.",
        RECORD_STARTUP, "Test", "https://example.com/d",
    ))
    assert envelope["extractionMode"] == "deterministic"
    assert envelope["content"]["data"]["employeeCount"] == 1500


def test_deterministic_extractor_employees_regex():
    result = deterministic_extract("Company has 1,200 employees worldwide.", RECORD_STARTUP)
    assert result["data"]["employeeCount"] == 1200
    result2 = deterministic_extract("No numbers here.", RECORD_STARTUP)
    assert result2["data"]["employeeCount"] is None


def test_envelope_shape_matches_assignment_schema():
    envelope = build_envelope(
        "STARTUP", "Example Source", "https://example.com/openai",
        {"entityName": "OpenAI", "data": {"employeeCount": None}},
        collected_at="2026-11-21T10:00:00Z",
    )
    assert envelope["schemaVersion"] == "1.0"
    assert envelope["recordType"] == "STARTUP"
    assert envelope["source"] == {"name": "Example Source", "url": "https://example.com/openai"}
    assert envelope["content"] == {"entityName": "OpenAI", "data": {"employeeCount": None}}
    assert envelope["collectedAt"] == "2026-11-21T10:00:00Z"
