"""Pipeline entrypoint.

Sub-commands (all log structured progress):

    collect-raw      Phase I+II crawling -> data/raw snapshots (network)
    ingest           snapshots -> canonical envelopes -> database (dedup)
    enrich-stars     GitHub GraphQL current star counts -> papers table
    resolve          Phase IV entity resolution + mapping log
    export-sheets    Deliverable 1: 6 CSVs + 6-tab XLSX (+ optional upload)
    pipeline         collect-raw -> ingest -> enrich-stars -> resolve -> export
    stats            current database counts

Examples:
    python -m src.main pipeline
    python -m src.main collect-raw --arxiv-limit 1200
    python -m src.main ingest && python -m src.main resolve && python -m src.main export-sheets
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import urllib.parse
from typing import Any

from .config import LOGS_DIR, Settings, ensure_dirs
from .llm.orchestrator import ExtractionOrchestrator
from .llm.schemas import build_envelope
from .scrapers.base import load_snapshot
from .storage.database import Storage
from .utils.hashing import content_hash, url_hash
from .utils.text import normalize_url

logger = logging.getLogger("graphone")


def setup_logging(verbose: bool = False) -> None:
    ensure_dirs()
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        log_path = LOGS_DIR / "pipeline.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Phase I+II: raw collection
# ---------------------------------------------------------------------------
async def _run_collect_raw(settings: Settings, args) -> dict[str, Any]:
    from .fetcher import Fetcher
    from .scrapers.jobs_scraper import collect_jobs
    from .scrapers.news_scraper import collect_news
    from .scrapers.product_scraper import collect_products
    from .scrapers.research_scraper import collect_papers
    from .scrapers.startup_scraper import collect_startups

    fetcher = Fetcher(settings)
    summary: dict[str, Any] = {}
    try:
        async def safe(label: str, coro):
            try:
                summary[label] = await coro
            except Exception as exc:  # noqa: BLE001
                logger.exception("collect stage failed: %s", label)
                summary[label] = {"error": str(exc), "records": []}

        await asyncio.gather(
            safe("startups", collect_startups(fetcher, settings)),
            safe("papers", collect_papers(fetcher, settings,
                                          arxiv_limit=args.arxiv_limit, pwc_limit=args.pwc_limit)),
            safe("news", collect_news(fetcher, settings.freshness_hours)),
            safe("jobs", collect_jobs(fetcher, settings.freshness_hours)),
        )
        # products need the startup universe for company inference
        startup_names = [r["entityName"] for r in summary.get("startups", {}).get("records", [])]
        await safe("products", collect_products(fetcher, settings, known_startups=startup_names))
    finally:
        await fetcher.close()
        logger.info("fetcher stats: %s", fetcher.stats)
    return summary


def cmd_collect_raw(args) -> int:
    settings = Settings()
    summary = asyncio.run(_run_collect_raw(settings, args))
    logger.info("COLLECT-RAW SUMMARY: %s", json.dumps(
        {k: (v.get("counts") if isinstance(v, dict) and "counts" in v else v) for k, v in summary.items()},
        default=str)[:2000])
    return 0


# ---------------------------------------------------------------------------
# Ingestion: snapshots -> envelopes -> DB
# ---------------------------------------------------------------------------
def _startup_envelopes(records: list[dict]) -> list[dict]:
    envelopes = []
    for record in records:
        content = {
            "entityName": record["entityName"],
            "data": {"employeeCount": record.get("employeeCount")},
            "description": record.get("description"),
        }
        envelopes.append(build_envelope(
            "STARTUP", record.get("sourceName", "Unknown"), record.get("sourceUrl", ""),
            content, record.get("collectedAt"),
        ))
    return envelopes


def _product_envelopes(records: list[dict]) -> list[dict]:
    envelopes = []
    for record in records:
        content = {
            "productName": record["productName"],
            "startupName": record.get("startupName"),
            "pricingModel": record.get("pricingModel"),
        }
        envelopes.append(build_envelope(
            "PRODUCT", record.get("sourceName", "Unknown"), record.get("sourceUrl", ""),
            content, record.get("collectedAt"),
        ))
    return envelopes


def _paper_envelopes(records: list[dict]) -> list[dict]:
    import re as _re

    envelopes = []
    for record in records:
        paper_url = _re.sub(r"(arxiv\.org/abs/[\w.\-]+)v\d+$", r"\1", record.get("paper_url", "") or "")
        content = {
            "title": record.get("title", ""),
            "authors": record.get("authors", []),
            "paper_url": paper_url,
            "github_url": record.get("githubUrl"),
            "github_stars": record.get("githubStars"),
            "published_date": record.get("publishedDate"),
        }
        if not content["title"] or not content["paper_url"]:
            continue
        envelopes.append(build_envelope(
            "RESEARCH_PAPER", record.get("sourceName", "Unknown"), record.get("sourceUrl", ""),
            content, record.get("collectedAt"),
        ))
    return envelopes


def _news_envelopes(records: list[dict]) -> list[dict]:
    envelopes = []
    for record in records:
        content = {
            "headline": record.get("headline", ""),
            "content": record.get("content", ""),
            "publishedAt": record.get("publishedAt"),
            "dateSource": record.get("dateSource"),
        }
        envelopes.append(build_envelope(
            "NEWS", record.get("sourceName", "Unknown"), record.get("sourceUrl", ""),
            content, record.get("collectedAt"),
        ))
    return envelopes


def _job_envelopes(records: list[dict]) -> list[dict]:
    envelopes = []
    for record in records:
        content = {
            "company": record.get("company", ""),
            "jobTitle": record.get("jobTitle", ""),
            "date": record.get("publishedAt"),
            "is_remote": bool(record.get("isRemote")),
            "role_family": record.get("roleFamily", "Other"),
            "dateSource": record.get("dateSource"),
            "location": record.get("location"),
        }
        envelopes.append(build_envelope(
            "JOB", record.get("sourceName", "Unknown"), record.get("sourceUrl", ""),
            content, record.get("collectedAt"),
        ))
    return envelopes


def cmd_ingest(args) -> int:
    from .storage.database import (
        JobPost,
        NewsArticle,
        Product,
        RawRecord,
        ResearchPaper,
        Startup,
    )
    from sqlalchemy.orm import Session

    from .utils.hashing import sha1_hex

    settings = Settings()
    storage = Storage(settings.database_url)
    inserted = {"startups": 0, "products": 0, "papers": 0, "news": 0, "jobs": 0, "raw": 0}

    def feed(kind: str, snapshots: list[str], builder, table) -> None:
        records: list[dict] = []
        for snapshot in snapshots:
            try:
                records.extend(load_snapshot(snapshot))
            except FileNotFoundError:
                logger.warning("snapshot missing: %s", snapshot)
        envelopes = builder(records)
        logger.info("ingest %s: %d envelopes from %d raw records", kind, len(envelopes), len(records))
        with Session(storage.engine) as session:
            for envelope in envelopes:
                content = envelope["content"]
                source_url = envelope["source"]["url"] or ""
                if kind == "news":
                    identity_hash = url_hash(content.get("sourceUrl") or source_url)
                elif kind == "jobs":
                    identity_hash = content_hash(
                        content.get("company"), content.get("jobTitle"),
                        normalize_url(content.get("sourceUrl") or source_url),
                    )
                elif kind == "papers":
                    identity_hash = url_hash(content.get("paper_url") or source_url)
                elif kind == "products":
                    identity_hash = content_hash(
                        content.get("productName"), normalize_url(source_url))
                else:
                    identity_hash = content_hash(
                        content.get("entityName"), normalize_url(source_url))

                # raw provenance layer (idempotent)
                if storage.insert_raw(session, envelope["recordType"], envelope,
                                      identity_hash, envelope["source"]["name"], source_url):
                    inserted["raw"] += 1

                url_key = normalize_url(source_url) or source_url
                common = {"source_name": envelope["source"]["name"], "source_url": source_url}
                if kind == "startups":
                    ok = storage.upsert_startup(session,
                        entity_name=content["entityName"], canonical_name=None,
                        employee_count=content.get("data", {}).get("employeeCount"),
                        description=content.get("description"), **common)
                elif kind == "products":
                    ok = storage.upsert_product(session,
                        product_name=content["productName"], startup_name=content.get("startupName"),
                        pricing_model=content.get("pricingModel"), **common)
                elif kind == "papers":
                    import re as _re

                    arxiv_match = _re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)", content["paper_url"])
                    arxiv_id = _re.sub(r"v\d+$", "", arxiv_match.group(1)) if arxiv_match else None
                    ok = storage.upsert_paper(session,
                        title=content["title"][:500],
                        authors_json=json.dumps(content.get("authors", []), ensure_ascii=False),
                        paper_url=content["paper_url"],
                        url_hash=url_hash(content["paper_url"]),
                        arxiv_id=arxiv_id,
                        github_url=content.get("github_url"),
                        github_stars=content.get("github_stars"),
                        published_date=content.get("published_date"), **common)
                elif kind == "news":
                    ok = storage.upsert_news(session,
                        headline=content["headline"][:400], content=content.get("content", ""),
                        url=content.get("sourceUrl") or source_url,
                        url_hash=identity_hash, published_at=content.get("publishedAt"),
                        date_source=content.get("dateSource"), **common)
                elif kind == "jobs":
                    ok = storage.upsert_job(session,
                        company=content["company"][:200],
                        title=content.get("jobTitle", "")[:300],
                        published_at=content.get("date"),
                        date_source=content.get("dateSource"),
                        is_remote=bool(content.get("is_remote")),
                        role_family=content.get("role_family", "Other"),
                        location=content.get("location"),
                        url=source_url, url_hash=identity_hash, **common)
                else:
                    ok = False
                if ok:
                    inserted[kind] += 1
            session.commit()

    import json as _json  # noqa: F401  (kept for parity with other commands)

    feed("startups", ["startups_raw.json"], _startup_envelopes, Startup)
    feed("products", ["products_raw.json"], _product_envelopes, Product)
    feed("papers", ["papers_raw.jsonl"], _paper_envelopes, ResearchPaper)
    feed("news", ["news_raw.json"], _news_envelopes, NewsArticle)
    feed("jobs", ["jobs_raw.json"], _job_envelopes, JobPost)

    # backfill arxiv ids on papers (from paper_url) so dedup stays exact
    import re as _re
    from sqlalchemy import select

    with Session(storage.engine) as session:
        papers = session.execute(select(ResearchPaper).where(ResearchPaper.arxiv_id.is_(None))).scalars().all()
        for paper in papers:
            match = _re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)", paper.paper_url or "")
            if match:
                paper.arxiv_id = _re.sub(r"v\d+$", "", match.group(1))
        session.commit()

    logger.info("INGEST RESULT: %s", inserted)
    logger.info("DB stats: %s", storage.stats())
    return 0


# ---------------------------------------------------------------------------
# GitHub star enrichment (current values via GraphQL)
# ---------------------------------------------------------------------------
def cmd_enrich_stars(args) -> int:
    import asyncio as _asyncio

    from sqlalchemy import select
    from sqlalchemy.orm import Session as _Session

    from .scrapers.research_scraper import enrich_github_stars
    from .storage.database import ResearchPaper

    settings = Settings()
    storage = Storage(settings.database_url)
    if not settings.github_token:
        logger.warning("GITHUB_TOKEN not set; unauthenticated enrichment is heavily "
                       "rate-limited (60/hr). Proceeding anyway.")

    with _Session(storage.engine) as session:
        papers = session.execute(
            select(ResearchPaper).where(ResearchPaper.github_url.is_not(None))
        ).scalars().all()
        github_urls = [p.github_url for p in papers if p.github_url]
    logger.info("enriching stars for %d papers with GitHub repos", len(github_urls))

    star_map = _asyncio.run(enrich_github_stars(github_urls, settings.github_token))
    updated = storage.update_paper_stars(star_map)
    logger.info("stars updated for %d papers; %d repos resolved", updated, len(star_map))
    return 0


# ---------------------------------------------------------------------------
# Phase IV: entity resolution
# ---------------------------------------------------------------------------
def cmd_resolve(args) -> int:
    from sqlalchemy import select
    from sqlalchemy.orm import Session as _Session

    from .resolver.entity_resolution import EntityResolver
    from .scrapers.base import save_snapshot, write_out_jsonl
    from .storage.database import EntityMapping, Product, Startup

    settings = Settings()
    storage = Storage(settings.database_url)

    startups = storage.load_table(Startup)
    products = storage.load_table(Product)
    logger.info("resolving %d startups, %d product->startup names", len(startups), len(products))

    resolver = EntityResolver()

    with _Session(storage.engine) as session:
        rows = session.execute(select(Startup)).scalars().all()
        for row in rows:
            result = resolver.resolve(row.entity_name, source_url=row.source_url,
                                      source_name=row.source_name)
            row.canonical_name = result.canonical_name

        product_rows = session.execute(select(Product)).scalars().all()
        for row in product_rows:
            if row.startup_name:
                result = resolver.resolve(row.startup_name)
                row.canonical_startup = result.canonical_name

        session.commit()

    log_dicts = resolver.log_dicts()
    write_out_jsonl("entity_mapping_log.jsonl", log_dicts)
    save_snapshot("entity_resolution_stats.json", resolver.stats())
    with _Session(storage.engine) as session:
        added = storage.add_entity_mappings(session, log_dicts)
        session.commit()
    logger.info("ENTITY RESOLUTION: %s (%d mapping rows persisted)", resolver.stats(), added)
    return 0


# ---------------------------------------------------------------------------
# Deliverable 1: sheets export
# ---------------------------------------------------------------------------
def cmd_export_sheets(args) -> int:
    from .storage.google_sheets import export_csv, export_xlsx, upload_sheets

    settings = Settings()
    storage = Storage(settings.database_url)
    stats = storage.stats()
    logger.info("exporting sheets from: %s", stats)

    csv_paths = export_csv(storage)
    xlsx_path = export_xlsx(storage)
    logger.info("CSV tabs: %s", [p.name for p in csv_paths])
    logger.info("XLSX workbook: %s", xlsx_path)

    if args.upload:
        ok = upload_sheets(storage, settings.sheets_credentials_file,
                           settings.sheets_spreadsheet_id)
        logger.info("Google Sheets upload: %s", "OK" if ok else "skipped/failed")
    return 0


def cmd_stats(args) -> int:
    settings = Settings()
    storage = Storage(settings.database_url)
    print(json.dumps(storage.stats(), indent=2))
    return 0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def cmd_pipeline(args) -> int:
    logger.info("=== STAGE 1/5: collect-raw ===")
    cmd_collect_raw(args)
    logger.info("=== STAGE 2/5: ingest ===")
    cmd_ingest(args)
    logger.info("=== STAGE 3/5: enrich-stars ===")
    cmd_enrich_stars(args)
    logger.info("=== STAGE 4/5: resolve ===")
    cmd_resolve(args)
    logger.info("=== STAGE 5/5: export-sheets ===")
    cmd_export_sheets(args)
    logger.info("PIPELINE COMPLETE")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="GraphOne Intelligence Pipeline (FrontierAtlas assignment)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arxiv-limit", type=int, default=1200)
    common.add_argument("--pwc-limit", type=int, default=2500)

    for name, help_text in (
        ("collect-raw", "Phase I+II: crawl every source into data/raw snapshots"),
        ("pipeline", "full pipeline: collect -> ingest -> stars -> resolve -> export"),
    ):
        cmd = sub.add_parser(name, parents=[common], help=help_text)
        cmd.set_defaults(func=cmd_collect_raw if name == "collect-raw" else cmd_pipeline)

    sub.add_parser("ingest", help="snapshots -> canonical envelopes -> DB (idempotent)") \
        .set_defaults(func=cmd_ingest)
    sub.add_parser("enrich-stars", help="fetch current GitHub stars for paper repos") \
        .set_defaults(func=cmd_enrich_stars)
    sub.add_parser("resolve", help="Phase IV entity resolution + mapping log") \
        .set_defaults(func=cmd_resolve)
    export = sub.add_parser("export-sheets", help="export 6 CSVs + XLSX (+ optional upload)")
    export.add_argument("--upload", action="store_true",
                        help="also push tabs via Google Sheets API (needs credentials)")
    export.set_defaults(func=cmd_export_sheets)
    sub.add_parser("stats", help="database record counts").set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
