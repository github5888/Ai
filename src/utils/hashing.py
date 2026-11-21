"""Content hashing for idempotent ingestion.

Two crawler workers collecting the same article/job must produce the same
content hash so that only one survives database insertion (Phase VI, Q3).
Hash inputs are *canonical* forms (normalised URL, normalised text) so
cosmetic differences across nodes do not create phantom duplicates.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .text import normalize_url


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def url_hash(url: str | None) -> str:
    normalized = normalize_url(url) or (url or "")
    return sha1_hex(normalized.lower())


def _canonical(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def content_hash(*fields: Any) -> str:
    """Stable hash over the identity fields of a record.

    For news: (headline, url).  For jobs: (company, title, url).
    Callers decide the identity fields; this function guarantees that the
    same logical record hashes identically on every worker.
    """
    payload = "\x1f".join(_canonical(f) for f in fields)
    return sha1_hex(payload)
