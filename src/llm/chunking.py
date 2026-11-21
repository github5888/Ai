"""Semantic chunking for LLM payloads (Phase III, 413-prevention).

Random cutting would destroy meaning, so text is split on *semantic
boundaries*: markdown/HTML headings first, then blank-line paragraphs, then
sentence endings.  Chunks carry a small paragraph overlap so entities
spanning a boundary are still visible to both sides.

Merging uses field-aware union semantics:
    list fields   -> union across chunks (authors, products, ...)
    scalar fields -> first non-null wins
    numeric       -> max (e.g. star counts seen in any chunk)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

_HEADING_RE = re.compile(r"^(#{1,6}\s|\s*\*\*\w+.*\*\*$)", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    index: int
    text: str
    chars: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token for English) used *only* to
    decide when to chunk; real limits are enforced by providers catching 413."""
    return max(1, len(text) // 4)


def _split_units(text: str) -> list[str]:
    """Break text into atomic units, preferring the largest semantic units."""
    if _HEADING_RE.search(text):
        blocks = _HEADING_RE.split(text)
        units: list[str] = []
        for piece in blocks:
            piece = piece.strip()
            if piece:
                units.extend(_paragraph_units(piece))
        return units
    return _paragraph_units(text)


def _paragraph_units(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para) <= 1400 or "\n" in para:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return units


def semantic_chunks(text: str, max_chars: int = 12000, overlap_chars: int = 900) -> list[Chunk]:
    """Split ``text`` into semantically coherent chunks of at most ``max_chars``.

    The first unit that alone exceeds ``max_chars`` is hard-split at sentence
    boundaries (never mid-word), keeping the "meaningful chunks" guarantee.
    """
    text = text or ""
    if len(text) <= max_chars:
        return [Chunk(0, text, len(text))] if text else []

    units = _split_units(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            body = "\n\n".join(current)
            if chunks and overlap_chars > 0:
                tail = chunks[-1].text[-overlap_chars:]
                cut = tail.find("\n\n")
                tail = tail[cut + 2 :] if cut != -1 else tail
                body = f"{tail}\n\n{body}" if tail else body
            chunks.append(Chunk(len(chunks), body, len(body)))
            current, current_len = [], 0

    for unit in units:
        if len(unit) > max_chars:
            # hard-split oversized unit at sentence boundaries
            for piece in _hard_split(unit, max_chars):
                if current_len + len(piece) + 2 > max_chars:
                    flush()
                current.append(piece)
                current_len += len(piece) + 2
            continue
        if current_len + len(unit) + 2 > max_chars:
            flush()
        current.append(unit)
        current_len += len(unit) + 2

    flush()
    return chunks


def _hard_split(unit: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for sentence in _SENTENCE_RE.split(unit):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:  # pathological: split on words
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not pieces or len(pieces[-1]) + len(sentence) + 1 > max_chars:
            pieces.append(sentence)
        else:
            pieces[-1] += " " + sentence
    return pieces or [unit]


def merge_chunk_results(
    results: Sequence[dict],
    list_fields: Iterable[str] = (),
    max_fields: Iterable[str] = (),
) -> dict:
    """Merge per-chunk extraction results into one record.

    list_fields  -> unioned (order-preserving, de-duplicated)
    max_fields   -> maximum non-null value (stars, counts)
    everything else -> first non-null/scalar wins
    """
    merged: dict = {}
    list_fields = set(list_fields or ())
    max_fields = set(max_fields or ())
    for result in results:
        if not isinstance(result, dict):
            continue
        for key, value in result.items():
            if value is None or value == "":
                continue
            if key in list_fields:
                bucket = merged.setdefault(key, [])
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if item and item not in bucket:
                        bucket.append(item)
            elif key in max_fields:
                try:
                    if merged.get(key) is None or float(value) > float(merged[key]):
                        merged[key] = value
                except (TypeError, ValueError):
                    if merged.get(key) is None:
                        merged[key] = value
            elif merged.get(key) is None:
                merged[key] = value
    return merged
