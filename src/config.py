"""Central configuration.

Everything is environment-driven with safe defaults so the pipeline runs
out-of-the-box (SQLite + deterministic extraction) and can be turned into
a production deployment purely via .env changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root = two levels above this file (src/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "out"
LOGS_DIR = DATA_DIR / "logs"
TMP_DIR = DATA_DIR / "tmp"

SCHEMA_VERSION = "1.0"

RECORD_STARTUP = "STARTUP"
RECORD_PRODUCT = "PRODUCT"
RECORD_PAPER = "RESEARCH_PAPER"
RECORD_JOB = "JOB"
RECORD_NEWS = "NEWS"

VALID_PRICING_MODELS = ("FREE", "FREEMIUM", "PAID", "ENTERPRISE")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings resolved from environment variables."""

    database_url: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'graphone.db'}")
    )

    # Crawler behaviour (Phase V)
    user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "USER_AGENT", "GraphOneBot/1.0 (+research data pipeline)"
        )
    )
    concurrency: int = field(default_factory=lambda: _env_int("CONCURRENCY", 8))
    request_min_interval: float = field(
        default_factory=lambda: _env_float("REQUEST_MIN_INTERVAL", 0.25)
    )
    freshness_hours: int = field(default_factory=lambda: _env_int("FRESHNESS_HOURS", 24))

    # GitHub metrics
    github_token: str = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN", ""))

    # LLM fallback chain (Phase III)
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    groq_model: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    deepseek_model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    llm_max_attempts: int = field(default_factory=lambda: _env_int("LLM_MAX_ATTEMPTS", 5))
    llm_backoff_base: float = field(default_factory=lambda: _env_float("LLM_BACKOFF_BASE", 2.0))
    llm_backoff_max_delay: float = field(default_factory=lambda: _env_float("LLM_BACKOFF_MAX_DELAY", 60.0))
    llm_backoff_jitter: float = field(default_factory=lambda: _env_float("LLM_BACKOFF_JITTER", 0.25))
    llm_chunk_chars: int = field(default_factory=lambda: _env_int("LLM_CHUNK_CHARS", 12000))

    # Google Sheets
    sheets_credentials_file: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_SHEETS_CREDENTIALS_FILE", "")
    )
    sheets_spreadsheet_id: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_SHEETS_ID", "")
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key or self.groq_api_key or self.deepseek_api_key)


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, OUT_DIR, LOGS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
