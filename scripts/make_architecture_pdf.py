"""Generates architecture.pdf (max 3 pages) with reportlab.

Run:  python scripts/make_architecture_pdf.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "architecture.pdf"

ACCENT = colors.HexColor("#1F3864")
LIGHT = colors.HexColor("#EAF0FA")
GREY = colors.HexColor("#5A5A5A")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=styles["Title"], fontSize=17, leading=20,
                    textColor=ACCENT, alignment=TA_LEFT, spaceAfter=2)
SUB = ParagraphStyle("SUBx", parent=styles["Normal"], fontSize=8.5, leading=11,
                     textColor=GREY, spaceAfter=6)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=11.5, leading=14,
                    textColor=ACCENT, spaceBefore=7, spaceAfter=3)
H3 = ParagraphStyle("H3x", parent=styles["Heading3"], fontSize=9.5, leading=12,
                    textColor=colors.HexColor("#2E4E7E"), spaceBefore=4, spaceAfter=2)
BODY = ParagraphStyle("BODYx", parent=styles["Normal"], fontSize=8.6, leading=11.6)
SMALL = ParagraphStyle("SMALLx", parent=BODY, fontSize=7.6, leading=9.8, textColor=GREY)
CELL = ParagraphStyle("CELLx", parent=BODY, fontSize=7.9, leading=9.6)
CELLB = ParagraphStyle("CELLBx", parent=CELL, fontName="Helvetica-Bold")


def table(data: list[list[str]], widths: list[float] | None = None, header: bool = True) -> Table:
    rows = [[Paragraph(f"<b>{c}</b>", CELL) if header and i == 0 else Paragraph(c, CELL)
             for c in row] for i, row in enumerate(data)]
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C4D6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        rows[0] = [Paragraph(f"<b>{c}</b>", CELL) for c in data[0]]
    if widths:
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]))
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    return t


def pipeline_diagram() -> Table:
    stages = [
        ("SOURCES", "10+ directories<br/>8 news feeds<br/>6 job boards<br/>Arxiv + PwC dump"),
        ("ASYNC CRAWLERS", "asyncio + aiohttp<br/>semaphore + per-host<br/>rate limit + retries<br/>Playwright fallback"),
        ("RAW SNAPSHOTS", "data/raw/*.json<br/>content-hash dedup<br/>full provenance<br/>(worker-safe commits)"),
        ("LLM EXTRACTION", "Gemini Flash -><br/>Groq Llama 3 -><br/>DeepSeek<br/>(429 backoff / 413 chunk)"),
        ("ENTITY RESOLUTION", "normalize -> seed -><br/>guarded fuzzy -><br/>canonical + map log"),
        ("STORAGE", "PostgreSQL / SQLite<br/>unique constraints<br/>idempotent upserts"),
        ("EXPORT", "Google Sheets<br/>6 tabs (+CSV/XLSX)"),
    ]
    cells = [[Paragraph(f"<b>{a}</b><br/><font size=7 color='#5A5A5A'>{b}</font>", CELL) for a, b in stages]]
    row = Table(cells, colWidths=[44 * mm] * 7)
    row.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C4D6")),
        ("BACKGROUND", (0, 0), (0, 0), LIGHT),
        ("BACKGROUND", (4, 0), (4, 0), LIGHT),
        ("BACKGROUND", (6, 0), (6, 0), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return row


story = []

# ============================ PAGE 1 =======================================
story.append(Paragraph("GraphOne Intelligence Pipeline - Technical Architecture", H1))
story.append(Paragraph(
    "FrontierAtlas / GraphOne assignment - an automated, fault-tolerant ingestion pipeline for the global "
    "AI Intelligence Graph: startups, products, research papers, and 24-hour-fresh signals (news, jobs). "
    "Every exported record traces back to a legitimate source URL; nothing is fabricated.", SUB))
story.append(pipeline_diagram())
story.append(Spacer(1, 4))

story.append(Paragraph("1. Component Map", H2))
story.append(table([
    ["Layer", "Module", "Technology", "Responsibility"],
    ["Crawl", "src/fetcher.py, src/scrapers/", "asyncio, aiohttp, BeautifulSoup, feedparser, Playwright (optional)",
     "Concurrent, rate-limited crawling of 25+ sources; raw snapshots with full provenance."],
    ["Extract", "src/llm/", "Gemini Flash -> Groq Llama 3 -> DeepSeek, pydantic",
     "Raw text/HTML -> canonical JSON envelopes; 429 backoff + jitter; 413 semantic chunking; deterministic fallback."],
    ["Resolve", "src/resolver/", "RapidFuzz + normalisation",
     "Deterministic canonicalisation of org names against a 50-company seed; ambiguity escalated, never guessed."],
    ["Store", "src/storage/database.py", "SQLAlchemy 2.0 (SQLite dev / PostgreSQL prod)",
     "Idempotent ingestion via content/url hashes + unique constraints; indexed for export and freshness queries."],
    ["Deliver", "src/storage/google_sheets.py", "Google Sheets API (gspread), openpyxl, CSV",
     "The 6 required tabs; live API upload or ready-made XLSX import."],
    ["Operate", ".github/workflows/collect.yml, src/main.py", "GitHub Actions worker, CLI",
     "Crawler worker nodes commit snapshots; ingest/enrich/resolve/export stages run anywhere."],
], [16 * mm, 38 * mm, 44 * mm, 82 * mm]))

story.append(Paragraph("2. Source Strategy (compliance-first)", H2))
story.append(table([
    ["Vertical", "Primary sources (all official/public channels)", "Fallback path"],
    ["Startups (1,000+)", "Wikipedia category tree via MediaWiki API; curated GitHub lists (applied-ml, "
     "awesome-european-ai, ...); builtin.com directory; Wikidata P1128 for employee counts",
     "If a directory blocks crawlers (Cloudflare), drop it and widen API/feeds - never bypass CAPTCHAs."],
    ["Products (1,000+)", "Curated AI-tool lists on GitHub (awesome-generative-ai, ai-collection, ...)",
     "Pricing classified only from the source's own words; else null."],
    ["Papers (1,000+)", "Arxiv Atom API (cs.AI/CL/LG, newest) + Papers-with-Code open-data dump "
     "(576k papers, CC-BY-SA) with paper<->code links",
     "GitHub GraphQL gives CURRENT stars at run time (stars are dynamic, never hard-coded)."],
    ["News (24h)", "8 official RSS/Atom feeds (TechCrunch AI, The Verge, VentureBeat, Ars, Wired, MIT TR, "
     "Marktechpost, HN)", "Full-text fetch for fresh articles, bounded + robots-aware."],
    ["Jobs (24h)", "RemoteOK API, Remotive API, Arbeitnow API, Jobicy RSS, We Work Remotely RSS, "
     "Hacker News hiring (Algolia API)", "First-seen heuristic when a board publishes no date."],
], [22 * mm, 100 * mm, 58 * mm]))

story.append(Paragraph(
    "Anti-bot posture: official APIs and feeds first; robots.txt checked before HTML crawls; per-host pacing "
    "(0.25 s) behind a global semaphore; realistic browser rendering (Playwright) only for high-value JS sources; "
    "aggressively blocking sources are skipped with a logged graceful fallback - no CAPTCHA/Cloudflare bypassing.", BODY))

story.append(PageBreak())

# ============================ PAGE 2 =======================================
story.append(Paragraph("3. Scale Strategy - 500,000+ records without code changes", H2))
story.append(Paragraph(
    "The pipeline already runs as <b>stateless workers + raw snapshots + idempotent upserts</b>; scaling to lakhs "
    "of records is an infrastructure exercise, not a rewrite:", BODY))
story.append(table([
    ["Lever", "Design"],
    ["Horizontal crawler fleet", "N worker nodes run the identical collect-raw container. A Redis (or SQS/RQ/Celery) "
     "queue hands out (source, cursor) tasks; each node streams snapshots to shared object storage. "
     "The GitHub Actions workflow is the reference worker."],
    ["Partitioned sources", "Sources are partitioned by vertical and cursor (category/country/page). arXiv alone "
     "yields ~1.5M ML papers via category+date cursors; PwC dump 576k. No global state is needed per worker."],
    ["Bounded concurrency", "Each worker: semaphore (8) + per-host min interval (250 ms) + exponential backoff. "
     "Politeness, not thread count, is the true throughput ceiling - hence more hosts, not more threads per host."],
    ["Throughput math", "1 worker ~= 2 records/s sustained (polite). 10 workers x 2/s x 8h/day ~= 576k records/day "
     "for API-grade sources; LLM extraction is decoupled and scales independently (see below)."],
    ["Bulk ingest", "COPY/batch inserts of 10k-row batches; UNIQUE (record_type, content_hash) makes replay safe. "
     "Postgres partitioning by collected_at keeps indexes small at 10M+ rows."],
    ["DB capacity", "500k papers ~= 1.5 GB rows + 6 GB JSONB payloads: single well-indexed Postgres node; "
     "read replicas for the export stage."],
], [34 * mm, 146 * mm]))

story.append(Paragraph("4. 429 and 413 Handling (exact mechanics)", H2))
story.append(table([
    ["Failure", "Detection", "Recovery"],
    ["HTTP 429 (rate limit)", "status 429 on any provider/crawler call",
     "Retry loop: 2s -> 4s -> 8s -> 16s (cap 60s) with +-25% full jitter; Retry-After header overrides the "
     "computed delay (+up to 25%). After LLM_MAX_ATTEMPTS the tier fails over to the next provider."],
    ["Provider outage / 5xx", "non-retryable statuses classified per provider",
     "Chain-of-responsibility failover: Gemini Flash -> Groq Llama 3 -> DeepSeek -> deterministic extractor; "
     "the record is never dropped silently and every fallback is logged with the reason."],
    ["HTTP 413 / context overflow", "status 413 or provider error body markers "
     '("maximum context length", "token limit", ...)',
     "Intelligent chunking: split on headings -> paragraphs -> sentences (never mid-word), ~12k chars "
     "(~3k tokens) per chunk with paragraph overlap; results merged field-aware (lists union, numerics max). "
     "If still 413: chunk size halves up to 3 times, then the tier fails over."],
    ["Threading the needle", "concurrency limits + token-bucket pacing per host",
     "Bursts never coalesce into a second 429: per-provider concurrency = 1..3, crawler per-host interval "
     "enforced even inside asyncio.gather."],
], [30 * mm, 52 * mm, 98 * mm]))

story.append(Paragraph(
    "Chunking illustration: a 50k-word paper -> [chunk 1: abstract+intro] [chunk 2: methods] ... each independently "
    "extracted -> merge (authors unioned across chunks, star counts maxed) -> one validated pydantic record.", SMALL))

story.append(PageBreak())

# ============================ PAGE 3 =======================================
story.append(Paragraph("5. Freshness & Exactly-Once Processing across distributed nodes", H2))
story.append(table([
    ["Mechanism", "Implementation"],
    ["Canonical identity", "URLs normalised (https, lowercase host, tracking params & fragments stripped, trailing "
     "slash removed) then SHA-1 -> url_hash. Content identity for jobs = hash(company, title, url); papers = "
     "hash(arxiv_id, url); news = url_hash."],
    ["Atomic dedup", "DB unique constraints: raw_records(record_type, content_hash), news/jobs/papers(url_hash), "
     "papers(arxiv_id). Workers INSERT ... ON CONFLICT DO NOTHING - three workers finding the same article yield "
     "exactly one row without any lock coordination."],
    ["Distributed lock (optional fast-path)", "Redis SETNX lock per url_hash short-circuits duplicate work "
     "before the DB round-trip; the DB constraint remains the correctness backstop."],
    ["Processed tracking", "raw_records.processed_at marks envelope consumption; re-runs skip already-processed "
     "payloads (idempotent replay after crashes)."],
    ["24h freshness", "Every date ('2 hours ago', 'Yesterday', RFC-822, ISO-8601, unix s/ms) -> ISO-8601 UTC; "
     "is_fresh(ts, now, 24h) filters news/jobs; future-dated junk rejected."],
    ["Missing dates", "No fabricated timestamps. Un-dated items get dateSource=heuristic:first_seen (first time "
     "this pipeline observed the item) and expire from the fresh window naturally on later runs."],
], [40 * mm, 140 * mm]))

story.append(Paragraph("6. Storage Choices (with justification)", H2))
story.append(table([
    ["Technology", "Role", "Why this one"],
    ["PostgreSQL", "Primary structured store", "ACID + unique constraints give exactly-once semantics under "
     "concurrent workers; JSONB stores raw payloads; partitions + indexes (published_at, url_hash, canonical "
     "names) serve both freshness scans and exports; mature ops story. SQLite runs the identical schema in dev."],
    ["Redis", "Task queue, dedup locks, rate-limit counters", "Sub-ms atomic ops (SETNX/INCR) match the "
     "crawler's hot path; queues hand out source cursors to N workers."],
    ["Neo4j", "Intelligence graph", "The product IS a graph: startup -[founded_by]-> person, startup -[makes]-> "
     "product, paper -[implements]-> method, paper -[cites]-> paper. Cypher models multi-hop queries (e.g. 'which "
     "company's alumni founded startups using which models') that are painful in SQL."],
    ["pgvector", "Semantic layer", "Embeddings of papers/articles inside Postgres: near-duplicate papers, "
     "'find papers like X', and LLM grounding via vector similarity - no second database to operate."],
    ["Google Sheets API", "Final delivery", "The assignment's required human-readable surface; the 6 tabs are "
     "generated deterministically from the DB (also CSV/XLSX for portability)."],
], [26 * mm, 42 * mm, 112 * mm]))

story.append(Paragraph("7. Engineering Rigor & Reproducibility", H2))
story.append(Paragraph(
    "Async everywhere (asyncio + aiohttp, bounded); retries with jitter shared by crawler and LLM tiers; "
    "pydantic-validated canonical envelopes; structured logging to data/logs/pipeline.log; 50 unit tests covering "
    "date parsing, freshness, hash identity, chunking, JSON recovery, fallback failover, entity resolution and all "
    "parsers (pytest). collect-raw -> ingest -> enrich-stars -> resolve -> export-sheets are separate, replayable "
    "stages; snapshots keep full provenance so any sheet row can be traced to its raw payload and source URL.", BODY))

story.append(Paragraph("8. Honest Limitations", H2))
story.append(Paragraph(
    "Papers-with-Code data dump is frozen at its July-2025 snapshot (Meta sunset); current stars are re-fetched "
    "live from GitHub to compensate. Cloudflare-protected directories are skipped, not bypassed, so startup "
    "employeeCount is null wherever no source states it - guessing would violate the data-fidelity rule. "
    "News/jobs tabs reflect the freshness window at the last crawl; re-running collect-raw refreshes them.", SMALL))

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
    title="GraphOne Intelligence Pipeline - Technical Architecture",
    author="GraphOne assignment",
)
doc.build(story)
print(f"architecture.pdf written ({OUT.stat().st_size / 1024:.0f} KB)")
