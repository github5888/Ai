"""Shared scraper infrastructure: raw snapshot persistence + source registry.

``collect-raw`` writes per-source snapshot files under data/raw/.  Downstream
stages (ingest -> llm -> resolve -> export) read those snapshots, so:

* a network failure never corrupts already-collected data,
* every exported record traces back to the raw payload + source URL
  (anti-hallucination provenance), and
* crawling can run on worker nodes while enrichment runs elsewhere.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import OUT_DIR, RAW_DIR

log = logging.getLogger(__name__)

Snapshot = dict | list


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_snapshot(name: str, payload: Snapshot, *, raw_dir: Path = RAW_DIR) -> Path:
    """Persist a raw scrape payload as JSON (or JSONL for lists of dicts)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    if name.endswith(".jsonl") and isinstance(payload, list):
        with path.open("w", encoding="utf-8") as fh:
            for record in payload:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    else:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    size_kb = path.stat().st_size / 1024
    log.info("snapshot %s saved (%.1f KB, %s records)", name, size_kb,
             len(payload) if isinstance(payload, list) else "n/a")
    return path


def save_meta(name: str, meta: dict, *, raw_dir: Path = RAW_DIR) -> Path:
    meta = {"savedAt": datetime.now(timezone.utc).isoformat(), **meta}
    return save_snapshot(name, meta, raw_dir=raw_dir)


def load_snapshot(name: str, *, raw_dir: Path = RAW_DIR):
    path = raw_dir / name
    if not path.exists():
        raise FileNotFoundError(f"snapshot not found: {path}")
    if name.endswith(".jsonl"):
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    return json.loads(path.read_text(encoding="utf-8"))


def write_out_jsonl(name: str, records: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def source_record(source_name: str, source_url: str) -> dict:
    return {"sourceName": source_name, "sourceUrl": source_url}
