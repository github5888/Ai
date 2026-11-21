"""Phase IV tests: deterministic entity resolution."""
from __future__ import annotations

from src.resolver.entity_resolution import (
    AMBIGUOUS,
    MATCHED,
    UNMATCHED,
    EntityResolver,
    normalize_for_match,
)


def test_normalization_strips_legal_suffixes_and_punctuation():
    assert normalize_for_match("OpenAI, Inc.") == "openai"
    assert normalize_for_match("Open AI") == "open ai"   # joining handled via seed alias/fuzzy
    assert normalize_for_match("OpenAI Inc") == "openai"
    assert normalize_for_match("Anthropic PBC") == "anthropic"
    assert normalize_for_match("Weights & Biases") == "weights and biases"


def test_exact_seed_matches():
    resolver = EntityResolver()
    assert resolver.resolve("OpenAI").canonical_name == "OpenAI"
    assert resolver.resolve("Open AI").canonical_name == "OpenAI"
    assert resolver.resolve("OpenAI, Inc.").canonical_name == "OpenAI"
    assert resolver.resolve("OpenAI Inc.").match_status == MATCHED
    assert resolver.resolve("Anthropic Inc.").canonical_name == "Anthropic"


def test_alias_matches():
    resolver = EntityResolver()
    result = resolver.resolve("HuggingFace")
    assert result.canonical_name == "Hugging Face"
    assert result.method == "seed-exact"


def test_distinct_companies_never_merge():
    resolver = EntityResolver()
    # similar-looking but different entities must NOT collapse together
    cohere = resolver.resolve("Cohere")
    cohere_health = resolver.resolve("Cohere Health")
    assert cohere.canonical_name == "Cohere"
    assert cohere_health.canonical_name != "Cohere"       # guardrail: distinctive extra token
    assert cohere_health.match_status == UNMATCHED

    openai = resolver.resolve("OpenAI")
    open_webui = resolver.resolve("Open WebUI")
    assert openai.canonical_name == "OpenAI"
    assert open_webui.canonical_name != "OpenAI"


def test_fuzzy_variant_matches_seed():
    resolver = EntityResolver()
    result = resolver.resolve("OpenAI Labs")
    assert result.canonical_name == "OpenAI"
    assert result.method == "seed-fuzzy"
    assert result.match_status == MATCHED


def test_new_entities_learned_for_consistency():
    resolver = EntityResolver()
    first = resolver.resolve("Zeta Robotics")
    assert first.match_status == UNMATCHED
    assert first.canonical_name == "Zeta Robotics"

    second = resolver.resolve("Zeta Robotics Inc.")
    assert second.match_status == MATCHED
    assert second.method == "learned-exact"
    assert second.canonical_name == "Zeta Robotics"


def test_ambiguous_cases_flagged_not_merged():
    resolver = EntityResolver()
    # "DeepMind Health" has a distinctive extra token -> must not merge into DeepMind
    result = resolver.resolve("DeepMind Health")
    assert result.canonical_name != "Google DeepMind"
    assert result.match_status in (UNMATCHED, AMBIGUOUS)


def test_mapping_log_has_required_fields():
    resolver = EntityResolver()
    resolver.resolve("Open AI", source_url="https://source.example/openai", source_name="Src")
    entry = resolver.log_dicts()[0]
    assert entry["raw_name"] == "Open AI"
    assert entry["canonical_name"] == "OpenAI"
    assert entry["match_status"] == MATCHED
    assert entry["score"] == 100.0
    assert entry["resolved_at"].endswith("Z")


def test_stats_counts():
    resolver = EntityResolver()
    resolver.resolve("OpenAI")
    resolver.resolve("Anthropic")
    resolver.resolve("Totally New Corp")
    stats = resolver.stats()
    assert stats["total"] == 3
    assert stats["MATCHED"] == 2
    assert stats["UNMATCHED"] == 1
    assert stats["seedEntities"] == 50
