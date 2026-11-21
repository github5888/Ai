"""Canonical record schemas (pydantic) - exactly the assignment envelopes.

Every record exported by the pipeline looks like::

    {
      "schemaVersion": "1.0",
      "recordType": "STARTUP",
      "source": {"name": "...", "url": "https://..."},
      "content": {...entity-specific fields...},
      "collectedAt": "2026-11-21T10:00:00Z"
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..config import (
    RECORD_JOB,
    RECORD_NEWS,
    RECORD_PAPER,
    RECORD_PRODUCT,
    RECORD_STARTUP,
    SCHEMA_VERSION,
    VALID_PRICING_MODELS,
)


class SourceRef(BaseModel):
    name: str
    url: str


class StartupData(BaseModel):
    employeeCount: Optional[int] = None


class StartupContent(BaseModel):
    entityName: str
    data: StartupData = Field(default_factory=StartupData)
    description: Optional[str] = None


class ProductContent(BaseModel):
    productName: str
    startupName: Optional[str] = None
    pricingModel: Optional[Literal["FREE", "FREEMIUM", "PAID", "ENTERPRISE"]] = None

    @field_validator("pricingModel")
    @classmethod
    def _validate_pricing(cls, v):
        if v is not None and v not in VALID_PRICING_MODELS:
            return None
        return v


class PaperContent(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    paper_url: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: Optional[str] = None  # ISO-8601


class JobContent(BaseModel):
    company: str
    jobTitle: str
    date: Optional[str] = None       # ISO-8601
    is_remote: bool = False
    role_family: str = "Other"
    dateSource: Optional[str] = None  # "stated" | "heuristic:first_seen"


class NewsContent(BaseModel):
    headline: str
    content: str = ""
    publishedAt: Optional[str] = None
    dateSource: Optional[str] = None


class RecordEnvelope(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: str
    source: SourceRef
    content: dict
    collectedAt: str = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_envelope(record_type: str, source_name: str, source_url: str,
                   content: dict, collected_at: str | None = None) -> dict:
    """Single choke point for envelope construction (all scrapers funnel here)."""
    env = {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": record_type,
        "source": {"name": source_name, "url": source_url},
        "content": content,
    }
    if collected_at:
        env["collectedAt"] = collected_at
    validated = RecordEnvelope(**env)
    return validated.model_dump()
