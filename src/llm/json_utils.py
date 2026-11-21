"""Robust JSON recovery from LLM output.

Models love wrapping JSON in prose or code fences; a strict json.loads
would fail the whole pipeline.  This scanner finds the first balanced
JSON object/array and decodes it, tolerating fences and leading text.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Return the first JSON value found in ``text`` (object or array)."""
    if text is None:
        raise ValueError("empty model output")

    # Prefer fenced blocks: usually the cleanest candidate.
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced { ... } or [ ... ] using raw_decode.
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON object found in model output")


def as_list(value: Any) -> list:
    """Normalise scalar-or-list model output to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return [value]
