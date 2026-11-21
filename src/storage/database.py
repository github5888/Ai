"""Relational storage layer (SQLAlchemy 2.0).

Default engine is SQLite (zero-config development store); the exact same
models run on PostgreSQL for production scale - only DATABASE_URL changes.
Scale-relevant choices documented in architecture.pdf:

* ``raw_record`` keeps every collected payload with a ``content_hash`` -
  the (record_type, content_hash) UNIQUE constraint makes ingestion
  **idempotent** across distributed crawler nodes (no double processing).
* ``url_hash`` (normalised URL) is unique for news/jobs/papers so the same
  article found by 3 workers is stored exactly once (atomic upsert).
* lookup indexes cover every export/filter path (published_at, canonical
  names, arxiv_id, github_url).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import Session, declarative_base

from ..config import RECORD_JOB, RECORD_NEWS, RECORD_PAPER, RECORD_PRODUCT, RECORD_STARTUP, TMP_DIR

log = logging.getLogger(__name__)

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RawRecord(Base):
    """Every collected payload, stored once (idempotent ingestion)."""

    __tablename__ = "raw_records"
    __table_args__ = (UniqueConstraint("record_type", "content_hash", name="uq_raw_content"),)

    id = Column(Integer, primary_key=True)
    record_type = Column(String(32), index=True, nullable=False)
    source_name = Column(String(120))
    source_url = Column(Text)
    content_hash = Column(String(64), index=True, nullable=False)
    payload = Column(Text, nullable=False)  # canonical JSON envelope
    collected_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    processed_at = Column(DateTime(timezone=True))


class Startup(Base):
    __tablename__ = "startups"
    __table_args__ = (UniqueConstraint("entity_name", "source_url", name="uq_startup_source"),)

    id = Column(Integer, primary_key=True)
    entity_name = Column(String(200), nullable=False, index=True)
    canonical_name = Column(String(200), index=True)
    employee_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    source_name = Column(String(120))
    source_url = Column(Text)
    collected_at = Column(DateTime(timezone=True), default=_utcnow)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("product_name", "source_url", name="uq_product_source"),)

    id = Column(Integer, primary_key=True)
    product_name = Column(String(200), nullable=False, index=True)
    startup_name = Column(String(200), index=True, nullable=True)
    canonical_startup = Column(String(200), index=True, nullable=True)
    pricing_model = Column(String(20), index=True, nullable=True)
    source_name = Column(String(120))
    source_url = Column(Text)
    collected_at = Column(DateTime(timezone=True), default=_utcnow)


class ResearchPaper(Base):
    __tablename__ = "research_papers"
    __table_args__ = (
        UniqueConstraint("url_hash", name="uq_paper_url"),
        UniqueConstraint("arxiv_id", name="uq_paper_arxiv"),
    )

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False, index=True)
    authors_json = Column(Text, default="[]")
    paper_url = Column(Text, nullable=False)
    url_hash = Column(String(64), nullable=False, index=True)
    arxiv_id = Column(String(32), index=True, nullable=True)
    github_url = Column(Text, nullable=True, index=True)
    github_stars = Column(Integer, nullable=True)
    published_date = Column(String(32), index=True, nullable=True)
    source_name = Column(String(120))
    source_url = Column(Text)
    collected_at = Column(DateTime(timezone=True), default=_utcnow)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (UniqueConstraint("url_hash", name="uq_news_url"),)

    id = Column(Integer, primary_key=True)
    headline = Column(String(400), nullable=False, index=True)
    content = Column(Text, default="")
    url = Column(Text, nullable=False)
    url_hash = Column(String(64), nullable=False, index=True)
    published_at = Column(String(32), index=True, nullable=True)
    date_source = Column(String(40), nullable=True)
    source_name = Column(String(120))
    source_url = Column(Text)
    collected_at = Column(DateTime(timezone=True), default=_utcnow)


class JobPost(Base):
    __tablename__ = "job_posts"
    __table_args__ = (UniqueConstraint("url_hash", name="uq_job_url"),)

    id = Column(Integer, primary_key=True)
    company = Column(String(200), nullable=False, index=True)
    canonical_company = Column(String(200), index=True, nullable=True)
    title = Column(String(300), nullable=False, index=True)
    published_at = Column(String(32), index=True, nullable=True)
    date_source = Column(String(40), nullable=True)
    is_remote = Column(Boolean, default=False, index=True)
    role_family = Column(String(40), index=True)
    location = Column(String(200), nullable=True)
    url = Column(Text, nullable=False)
    url_hash = Column(String(64), nullable=False, index=True)
    source_name = Column(String(120))
    source_url = Column(Text)
    collected_at = Column(DateTime(timezone=True), default=_utcnow)


class EntityMapping(Base):
    """Raw vs canonical names - the audit log behind Sheet tab 06."""

    __tablename__ = "entity_mappings"

    id = Column(Integer, primary_key=True)
    raw_name = Column(String(300), nullable=False, index=True)
    canonical_name = Column(String(300), nullable=False, index=True)
    match_status = Column(String(20), index=True)
    score = Column(Float)
    method = Column(String(60))
    entity_type = Column(String(20), default="STARTUP")
    source_url = Column(Text, nullable=True)
    resolved_at = Column(String(32))


class Storage:
    """Thin repository over the models; one instance per pipeline run."""

    def __init__(self, database_url: str):
        Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
        kwargs = {}
        if database_url.startswith("sqlite"):
            kwargs = {"connect_args": {"timeout": 30}}
        self.engine = create_engine(database_url, future=True, **kwargs)
        Base.metadata.create_all(self.engine)

    # -- raw ingestion (idempotent) ---------------------------------------------
    def insert_raw(self, session: Session, record_type: str, envelope: dict,
                   content_hash: str, source_name: str, source_url: str) -> bool:
        """Insert raw envelope; returns False if it already exists (dedup)."""
        exists = session.execute(
            select(RawRecord.id).where(
                RawRecord.record_type == record_type, RawRecord.content_hash == content_hash
            )
        ).first()
        if exists:
            return False
        session.add(RawRecord(
            record_type=record_type,
            source_name=source_name,
            source_url=source_url,
            content_hash=content_hash,
            payload=json.dumps(envelope, ensure_ascii=False, default=str),
        ))
        return True

    # -- upserts ------------------------------------------------------------------
    def upsert_startup(self, session: Session, **fields) -> bool:
        entity_name = fields.get("entity_name")
        source_url = fields.get("source_url")
        if not entity_name or not source_url:
            return False
        exists = session.execute(
            select(Startup.id).where(
                Startup.entity_name == entity_name, Startup.source_url == source_url
            )
        ).first()
        if exists:
            return False
        session.add(Startup(**fields))
        return True

    def upsert_product(self, session: Session, **fields) -> bool:
        product_name = fields.get("product_name")
        source_url = fields.get("source_url")
        if not product_name or not source_url:
            return False
        exists = session.execute(
            select(Product.id).where(
                Product.product_name == product_name, Product.source_url == source_url
            )
        ).first()
        if exists:
            return False
        session.add(Product(**fields))
        return True

    def upsert_paper(self, session: Session, **fields) -> bool:
        url_hash = fields.get("url_hash")
        arxiv_id = fields.get("arxiv_id")
        if not url_hash:
            return False
        conditions = [ResearchPaper.url_hash == url_hash]
        if arxiv_id:
            conditions.append(ResearchPaper.arxiv_id == arxiv_id)
        existing = session.execute(
            select(ResearchPaper).where(
                ResearchPaper.url_hash == url_hash if arxiv_id is None else
                (ResearchPaper.arxiv_id == arxiv_id)
            )
        ).scalar_one_or_none()
        if existing:
            changed = False
            if fields.get("github_url") and not existing.github_url:
                existing.github_url = fields["github_url"]
                changed = True
            if fields.get("github_stars") is not None:
                existing.github_stars = fields["github_stars"]
                changed = True
            if fields.get("title") and not existing.title:
                existing.title = fields["title"]
                changed = True
            return changed
        session.add(ResearchPaper(**fields))
        return True

    def upsert_news(self, session: Session, **fields) -> bool:
        if session.execute(select(NewsArticle.id).where(
                NewsArticle.url_hash == fields["url_hash"])).first():
            return False
        session.add(NewsArticle(**fields))
        return True

    def upsert_job(self, session: Session, **fields) -> bool:
        if session.execute(select(JobPost.id).where(
                JobPost.url_hash == fields["url_hash"])).first():
            return False
        session.add(JobPost(**fields))
        return True

    def add_entity_mappings(self, session: Session, mappings: list[dict]) -> int:
        added = 0
        for m in mappings:
            session.add(EntityMapping(
                raw_name=m.get("raw_name", ""),
                canonical_name=m.get("canonical_name", ""),
                match_status=m.get("match_status"),
                score=m.get("score"),
                method=m.get("method"),
                entity_type=m.get("entity_type", "STARTUP"),
                source_url=m.get("source_url"),
                resolved_at=m.get("resolved_at"),
            ))
            added += 1
        return added

    # -- readers --------------------------------------------------------------------
    def load_table(self, model) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.execute(select(model)).scalars().all()
            return [row_to_dict(row) for row in rows]

    def update_paper_stars(self, star_map: dict[str, dict]) -> int:
        """star_map: 'owner/repo'.lower() -> {stars, url}; updates matched papers."""
        import re as _re

        updated = 0
        with Session(self.engine) as session:
            papers = session.execute(select(ResearchPaper).where(
                ResearchPaper.github_url.is_not(None))).scalars().all()
            for paper in papers:
                m = _re.match(r"https?://github\.com/([^/]+)/([^/#?]+)", paper.github_url or "")
                if not m:
                    continue
                key = f"{m.group(1)}/{m.group(2)}".lower()
                payload = star_map.get(key)
                if payload and payload.get("stars") is not None and payload["stars"] != paper.github_stars:
                    paper.github_stars = payload["stars"]
                    updated += 1
            session.commit()
        return updated

    def stats(self) -> dict:
        counts = {}
        with Session(self.engine) as session:
            for name, model in (
                ("startups", Startup), ("products", Product), ("papers", ResearchPaper),
                ("news", NewsArticle), ("jobs", JobPost), ("entityMappings", EntityMapping),
                ("rawRecords", RawRecord),
            ):
                counts[name] = session.execute(
                    select(func_count(model))).scalar() or 0
        return counts


def func_count(model):
    from sqlalchemy import func

    return func.count(model.id)


def row_to_dict(row) -> dict:
    return {c.key: getattr(row, c.key) for c in row.__table__.columns}
