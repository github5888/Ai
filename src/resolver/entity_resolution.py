"""Deterministic entity resolution (Phase IV).

Problem: the same organisation surfaces as "OpenAI", "Open AI", "OpenAI, Inc.",
"OpenAI Inc." across sources.  Strategy:

1. **Normalisation** - lowercase, strip punctuation and legal suffixes
   (Inc/LLC/Ltd/GmbH/Corp/...), collapse whitespace.
2. **Seed matching** - 50 known canonical AI startups with documented aliases.
3. **Guarded fuzzy matching** - RapidFuzz ratios with two safety rails so
   distinct companies never merge:
   a. at least one shared distinctive token, and
   b. any extra words on either side must be generic (labs/research/ai/...)
      - "Cohere Health" can NOT merge into "Cohere".
4. **Learned entities** - once a raw name is canonicalised, later records reuse
   the mapping, so a company keeps one canonical form across the dataset.
5. **Conservative by design** - scores in the review band become AMBIGUOUS
   (manual review) instead of merged; the assignment explicitly prefers
   unmatched-over-wrong.  Distinct names are never merged on spelling alone.

Every decision is logged to the Entity Mapping Log (Sheet tab 06):
``{raw_name, canonical_name, match_status, score, method}``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from rapidfuzz import fuzz

log = logging.getLogger(__name__)

MATCHED = "MATCHED"
UNMATCHED = "UNMATCHED"
AMBIGUOUS = "AMBIGUOUS"

FUZZY_MATCH_THRESHOLD = 93   # >= this + guardrails -> MATCHED
REVIEW_THRESHOLD = 85        # [85, 93) -> AMBIGUOUS (manual review)

_LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "gmbh", "corp",
    "corporation", "co", "company", "pvt", "private", "pty", "plc", "sa",
    "ag", "nv", "bv", "oy", "aps", "pbc", "plc",
}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")

# Words that may appear alongside a canonical name without making it a
# different company ("OpenAI Labs" ~ "OpenAI").  Anything else means the raw
# name is (probably) a DIFFERENT entity and must not merge.
GENERIC_EXTRA_TOKENS = {
    "ai", "labs", "lab", "research", "group", "intelligence", "tech",
    "technologies", "technology", "systems", "software", "solutions",
}


def normalize_for_match(name: str | None) -> str:
    """Aggressive but deterministic normalisation for *matching keys*."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name).lower()
    text = text.replace("&", "and")
    text = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _WS_RE.split(text) if t]
    # drop legal-suffix tokens anywhere (OpenAI Inc. == OpenAI)
    tokens = [t for t in tokens if t not in _LEGAL_SUFFIXES]
    return " ".join(tokens).strip()


def display_name(name: str | None) -> str:
    """Cleaned raw name for display/canonical use (keeps inner casing)."""
    if not name:
        return ""
    cleaned = re.sub(r"\s+", " ", name.strip().strip(",;.-\u2013\u2014"))
    cleaned = re.sub(r"[,;]+$", "", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Seed database: 50 known canonical AI startups + documented aliases
# ---------------------------------------------------------------------------
SEED_STARTUPS: list[dict] = [
    {"canonical": "OpenAI", "aliases": ["Open AI"]},
    {"canonical": "Anthropic", "aliases": ["Anthropic PBC"]},
    {"canonical": "Google DeepMind", "aliases": ["DeepMind", "Google AI", "Google Brain"]},
    {"canonical": "Meta AI", "aliases": ["Meta FAIR", "Facebook AI Research"]},
    {"canonical": "Microsoft AI", "aliases": ["Microsoft Research AI"]},
    {"canonical": "NVIDIA", "aliases": ["Nvidia"]},
    {"canonical": "Hugging Face", "aliases": ["HuggingFace"]},
    {"canonical": "Perplexity AI", "aliases": ["Perplexity"]},
    {"canonical": "xAI", "aliases": ["X.AI"]},
    {"canonical": "Mistral AI", "aliases": ["Mistral"]},
    {"canonical": "Cohere", "aliases": ["Cohere AI"]},
    {"canonical": "AI21 Labs", "aliases": ["AI21"]},
    {"canonical": "Stability AI", "aliases": ["StabilityAI", "Stability"]},
    {"canonical": "Midjourney", "aliases": ["Mid Journey"]},
    {"canonical": "Runway", "aliases": ["RunwayML"]},
    {"canonical": "ElevenLabs", "aliases": ["Eleven Labs"]},
    {"canonical": "Character.AI", "aliases": ["Character AI"]},
    {"canonical": "Inflection AI", "aliases": ["Inflection"]},
    {"canonical": "Synthesia", "aliases": []},
    {"canonical": "Jasper", "aliases": ["Jasper AI"]},
    {"canonical": "Writer", "aliases": ["Writer AI"]},
    {"canonical": "Grammarly", "aliases": []},
    {"canonical": "DeepL", "aliases": []},
    {"canonical": "AssemblyAI", "aliases": ["Assembly AI"]},
    {"canonical": "Deepgram", "aliases": []},
    {"canonical": "Speechmatics", "aliases": []},
    {"canonical": "Replicate", "aliases": []},
    {"canonical": "Together AI", "aliases": ["Together Computer"]},
    {"canonical": "Fireworks AI", "aliases": ["Fireworks"]},
    {"canonical": "Anyscale", "aliases": []},
    {"canonical": "Modal", "aliases": ["Modal Labs"]},
    {"canonical": "Groq", "aliases": []},
    {"canonical": "Cerebras", "aliases": ["Cerebras Systems"]},
    {"canonical": "SambaNova", "aliases": ["SambaNova Systems"]},
    {"canonical": "Lambda Labs", "aliases": ["LambdaGPU"]},
    {"canonical": "CoreWeave", "aliases": []},
    {"canonical": "Scale AI", "aliases": ["ScaleAI"]},
    {"canonical": "Labelbox", "aliases": []},
    {"canonical": "Snorkel AI", "aliases": ["Snorkel"]},
    {"canonical": "Weights & Biases", "aliases": ["Weights and Biases", "WandB"]},
    {"canonical": "LangChain", "aliases": ["LangChain AI"]},
    {"canonical": "LlamaIndex", "aliases": ["GPT Index"]},
    {"canonical": "Pinecone", "aliases": ["Pinecone AI"]},
    {"canonical": "Weaviate", "aliases": ["SeMI Technologies"]},
    {"canonical": "Qdrant", "aliases": []},
    {"canonical": "Chroma", "aliases": ["ChromaDB"]},
    {"canonical": "MosaicML", "aliases": []},
    {"canonical": "Copy.ai", "aliases": ["Copy AI"]},
    {"canonical": "HeyGen", "aliases": ["Movio"]},
    {"canonical": "Descript", "aliases": []},
]

SEED_ALIASES: dict[str, str] = {}
for _seed in SEED_STARTUPS:
    for _alias in [_seed["canonical"], *_seed["aliases"]]:
        SEED_ALIASES[normalize_for_match(_alias)] = _seed["canonical"]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
@dataclass
class ResolutionResult:
    raw_name: str
    canonical_name: str
    match_status: str
    score: float
    method: str
    source_url: str | None = None
    source_name: str | None = None
    entity_type: str = "STARTUP"
    resolved_at: str = ""


class EntityResolver:
    """Seed + learned index + guardrailed fuzzy matching."""

    def __init__(self, learned_entities: Iterable[str] = ()):
        self.seed_index = dict(SEED_ALIASES)
        self.learned_index: dict[str, str] = {}       # normalised -> canonical
        self.learned_canonical_keys: set[str] = set()
        self.log: list[ResolutionResult] = []
        for entity in learned_entities:
            self.register_entity(entity)
        self._candidates: list[str] = sorted(set(self.seed_index.values()) | self.learned_canonical_keys)

    # -- index maintenance ----------------------------------------------------
    def register_entity(self, canonical: str) -> None:
        if not canonical:
            return
        key = normalize_for_match(canonical)
        if key and key not in self.learned_index:
            self.learned_index[key] = canonical
            self.learned_canonical_keys.add(canonical)
            self._candidates.append(canonical)

    # -- matching ---------------------------------------------------------------
    def resolve(self, raw_name: str, source_url: str | None = None,
                source_name: str | None = None,
                entity_type: str = "STARTUP") -> ResolutionResult:
        """Resolve one raw entity name to a canonical form."""
        raw = display_name(raw_name)
        key = normalize_for_match(raw)

        if not key:
            return self._log_result(ResolutionResult(
                raw_name=raw, canonical_name=raw, match_status=UNMATCHED,
                score=0.0, method="empty", source_url=source_url,
                source_name=source_name, entity_type=entity_type,
            ))

        # 1) exact seed / alias
        canonical = self.seed_index.get(key)
        if canonical:
            return self._log_result(ResolutionResult(
                raw_name=raw, canonical_name=canonical, match_status=MATCHED,
                score=100.0, method="seed-exact", source_url=source_url,
                source_name=source_name, entity_type=entity_type,
            ))

        # 2) exact learned entity
        canonical = self.learned_index.get(key)
        if canonical:
            return self._log_result(ResolutionResult(
                raw_name=raw, canonical_name=canonical, match_status=MATCHED,
                score=100.0, method="learned-exact", source_url=source_url,
                source_name=source_name, entity_type=entity_type,
            ))

        # 3) guardrailed fuzzy match against seed + learned canonicals
        best_name, best_score = None, 0.0
        raw_tokens = set(key.split())
        for candidate in self._candidates:
            c_key = normalize_for_match(candidate)
            score = max(fuzz.token_sort_ratio(key, c_key), fuzz.token_set_ratio(key, c_key))
            raw_tokens_set, c_tokens_set = set(key.split()), set(c_key.split())
            if not (raw_tokens_set & c_tokens_set):
                score = min(score, REVIEW_THRESHOLD - 1)          # no shared token
            else:
                extra_raw = raw_tokens_set - c_tokens_set
                extra_cand = c_tokens_set - raw_tokens_set
                if any(t not in GENERIC_EXTRA_TOKENS for t in extra_raw) or \
                        any(t not in GENERIC_EXTRA_TOKENS for t in extra_cand):
                    score = min(score, REVIEW_THRESHOLD - 1)      # distinctive extra word
            if score > best_score:
                best_name, best_score = candidate, score

        if best_score >= FUZZY_MATCH_THRESHOLD and best_name:
            self.learned_index[key] = best_name
            return self._log_result(ResolutionResult(
                raw_name=raw, canonical_name=best_name, match_status=MATCHED,
                score=best_score, method="seed-fuzzy", source_url=source_url,
                source_name=source_name, entity_type=entity_type,
            ))

        if best_score >= REVIEW_THRESHOLD and best_name:
            return self._log_result(ResolutionResult(
                raw_name=raw, canonical_name=raw, match_status=AMBIGUOUS,
                score=best_score, method=f"fuzzy-review:{best_name}", source_url=source_url,
                source_name=source_name, entity_type=entity_type,
            ))

        # 4) unmatched -> register as a new entity (consistent future matches)
        canonical = raw
        self.register_entity(canonical)
        return self._log_result(ResolutionResult(
            raw_name=raw, canonical_name=canonical, match_status=UNMATCHED,
            score=best_score, method="new-entity", source_url=source_url,
            source_name=source_name, entity_type=entity_type,
        ))

    # -- logging ------------------------------------------------------------------
    def _log_result(self, result: ResolutionResult) -> ResolutionResult:
        result.resolved_at = datetime.now(timezone.utc).replace(microsecond=0) \
            .isoformat().replace("+00:00", "Z")
        self.log.append(result)
        return result

    def log_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.log]

    def stats(self) -> dict:
        counts = {MATCHED: 0, UNMATCHED: 0, AMBIGUOUS: 0}
        for result in self.log:
            counts[result.match_status] = counts.get(result.match_status, 0) + 1
        return {"total": len(self.log), **counts,
                "learnedEntities": len(self.learned_canonical_keys),
                "seedEntities": len({self.seed_index[k] for k in self.seed_index})}
