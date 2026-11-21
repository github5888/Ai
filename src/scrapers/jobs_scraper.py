"""AI job scraper (Phase II-B).

Six official job-board endpoints (assignment minimum: 5):

======================  =====================================================
Board                   Ingestion channel (all official/public)
======================  =====================================================
RemoteOK                https://remoteok.com/api  (public JSON)
Remotive                https://remotive.com/api/remote-jobs  (public JSON)
Arbeitnow               https://www.arbeitnow.com/api/job-board-api  (public JSON)
Jobicy                  https://jobicy.com/feed  (official RSS)
We Work Remotely        category RSS feeds (official)
Hacker News (hiring)    hn.algolia.com API (public JSON)
======================  =====================================================

Every job record gets: company, jobTitle, publication date (normalised to
ISO-8601; relative dates handled), is_remote, role_family.  Jobs without a
stated date get the documented ``heuristic:first_seen`` freshness treatment.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import timedelta
from typing import Any

from ..fetcher import Fetcher
from ..utils.dates import (
    DATE_SOURCE_HEURISTIC,
    is_fresh,
    now_utc,
    parse_timestamp,
    to_iso,
)
from ..utils.text import norm_space, normalize_url, strip_html, truncate
from .base import save_snapshot

log = logging.getLogger(__name__)

WWR_FEEDS = [
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-all-others-jobs.rss",
]

AI_RELEVANT_RE = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|\bml\b|deep learning|"
    r"\bllm|large language model|nlp|natural language|computer vision|"
    r"data scien|generative|prompt engineer|neural network|transformers?|"
    r"\brl\b|reinforcement learning|agent engineer)\b",
    re.IGNORECASE,
)

ROLE_FAMILIES = [
    ("Research", re.compile(r"\b(research|scientist|phd|postdoc|researcher)\b", re.I)),
    ("Data & AI", re.compile(
        r"\b(machine learning|ml engineer|ai engineer|data scien|nlp|llm|deep learning|"
        r"mlops|data engineer|computer vision|applied scientist|prompt)\b", re.I)),
    ("Engineering", re.compile(
        r"\b(software engineer|backend|back-end|frontend|front-end|full[- ]?stack|developer|"
        r"devops|sre|platform|mobile|ios|android|ruby|python engineer|golang|java\b|"
        r"engineering manager|cto|architect|web developer)\b", re.I)),
    ("Product", re.compile(r"\b(product manager|product owner|product design|pm\b)\b", re.I)),
    ("Design", re.compile(r"\b(designer|ux|ui\b|creative)\b", re.I)),
    ("Sales", re.compile(r"\b(sales|account executive|business development|partnerships)\b", re.I)),
    ("Marketing", re.compile(r"\b(marketing|growth|content writer|copywriter|seo|communications)\b", re.I)),
    ("Operations", re.compile(r"\b(operations|recruiter|hr\b|people ops|finance|legal|support)\b", re.I)),
]

REMOTE_HINT_RE = re.compile(r"\b(remote|work from home|wfh|anywhere|distributed)\b", re.IGNORECASE)


def classify_role_family(title: str, tags: list[str] | None = None) -> str:
    text = f"{title} {' '.join(tags or [])}"
    for family, regex in ROLE_FAMILIES:
        if regex.search(text):
            return family
    return "Other"


def infer_remote(title: str, text: str, board_default: bool = False) -> bool:
    if board_default:
        return True
    return bool(REMOTE_HINT_RE.search(f"{title} {text}"))


def is_ai_relevant(title: str, description: str | None = None, tags: list[str] | None = None) -> bool:
    blob = f"{title} {description or ''} {' '.join(tags or [])}"
    return bool(AI_RELEVANT_RE.search(blob))


# ---------------------------------------------------------------------------
# Pure per-source parsers
# ---------------------------------------------------------------------------
def parse_remoteok(payload: list | dict) -> list[dict]:
    records: list[dict] = []
    if isinstance(payload, dict):
        payload = [payload]
    for item in payload:
        if not isinstance(item, dict) or not item.get("position"):
            continue  # first element is a legal notice object
        title = norm_space(item.get("position", ""))
        company = norm_space(item.get("company", ""))
        url = normalize_url(item.get("url"))
        if not title or not company or not url:
            continue
        published_iso, _ = parse_timestamp(item.get("date"))
        description = strip_html(str(item.get("description", "")))
        records.append({
            "company": company,
            "jobTitle": title,
            "publishedAt": published_iso,
            "dateSource": "stated" if published_iso else None,
            "isRemote": infer_remote(title, item.get("location", ""), board_default=True),
            "roleFamily": classify_role_family(title, item.get("tags")),
            "location": norm_space(str(item.get("location", ""))) or None,
            "url": url,
            "sourceName": "RemoteOK",
            "sourceUrl": url,
            "aiRelevant": is_ai_relevant(title, description, item.get("tags")),
        })
    return records


def parse_remotive(payload: dict) -> list[dict]:
    records: list[dict] = []
    for item in payload.get("jobs", []):
        title = norm_space(item.get("title", ""))
        company = norm_space(item.get("company_name", ""))
        url = normalize_url(item.get("url"))
        if not title or not company or not url:
            continue
        published_iso, _ = parse_timestamp(item.get("publication_date"))
        description = strip_html(str(item.get("description", "")))
        records.append({
            "company": company,
            "jobTitle": title,
            "publishedAt": published_iso,
            "dateSource": "stated" if published_iso else None,
            "isRemote": infer_remote(title, item.get("candidate_required_location", ""), board_default=True),
            "roleFamily": classify_role_family(title, item.get("tags")),
            "location": norm_space(str(item.get("candidate_required_location", ""))) or None,
            "url": url,
            "sourceName": "Remotive",
            "sourceUrl": url,
            "aiRelevant": is_ai_relevant(title, description, item.get("tags")),
        })
    return records


def parse_arbeitnow(payload: dict) -> list[dict]:
    records: list[dict] = []
    for item in payload.get("data", []):
        title = norm_space(item.get("title", ""))
        company = norm_space(item.get("company_name", ""))
        url = normalize_url(item.get("url"))
        if not title or not url:
            continue
        company = company or "Unknown company"
        published_iso, _ = parse_timestamp(item.get("created_at"))
        description = strip_html(str(item.get("description", "")))
        records.append({
            "company": company,
            "jobTitle": title,
            "publishedAt": published_iso,
            "dateSource": "stated" if published_iso else None,
            "isRemote": bool(item.get("remote", False)) or infer_remote(
                title, f"{item.get('location', '')} {description[:300]}"
            ),
            "roleFamily": classify_role_family(title, item.get("tags")),
            "location": norm_space(str(item.get("location", ""))) or None,
            "url": url,
            "sourceName": "Arbeitnow",
            "sourceUrl": url,
            "aiRelevant": is_ai_relevant(title, description, item.get("tags")),
        })
    return records


def parse_rss_jobs(xml_text: str, source_name: str) -> list[dict]:
    """Jobicy / WeWorkRemotely RSS: title convention 'Company: Role'."""
    import feedparser

    feed = feedparser.parse(xml_text)
    records: list[dict] = []
    for entry in feed.entries:
        title = norm_space(entry.get("title", ""))
        url = normalize_url(entry.get("link"))
        if not title or not url:
            continue
        if ":" in title:
            company, role = title.split(":", 1)
        else:
            company, role = "", title
        company = norm_space(company) or "Unknown company"
        role = norm_space(role) or title
        published_iso, _ = parse_timestamp(entry.get("published") or entry.get("updated"))
        description = strip_html(str(entry.get("summary", "")))
        tags = [norm_space(t.get("term", "")) for t in entry.get("tags", []) if isinstance(t, dict)]
        records.append({
            "company": company,
            "jobTitle": role,
            "publishedAt": published_iso,
            "dateSource": "stated" if published_iso else None,
            "isRemote": infer_remote(role, f"{description[:300]} {title}", board_default=True),
            "roleFamily": classify_role_family(role, tags),
            "location": None,
            "url": url,
            "sourceName": source_name,
            "sourceUrl": url,
            "aiRelevant": is_ai_relevant(role, description, tags),
        })
    return records


def parse_hn_hiring(story_json: dict, max_jobs: int = 60) -> list[dict]:
    """Parse 'Ask HN: Who is hiring?' top-level comments into job records."""
    records: list[dict] = []
    for child in story_json.get("children", []):
        text_html = child.get("text") or ""
        text = strip_html(text_html)
        first_line = norm_space(text.split("\n", 1)[0])
        # "Company X | Senior ML Engineer | San Francisco | Remote"
        parts = [norm_space(p) for p in first_line.split("|") if norm_space(p)]
        if not parts:
            continue
        company = parts[0][:60] or "Unknown company"
        role = parts[1][:80] if len(parts) > 1 else first_line[:80]
        published_iso, _ = parse_timestamp(child.get("created_at"))
        if not is_ai_relevant(first_line, text[:600]):
            continue
        records.append({
            "company": company,
            "jobTitle": role,
            "publishedAt": published_iso,
            "dateSource": "stated" if published_iso else None,
            "isRemote": infer_remote(role, text[:600], board_default=False),
            "roleFamily": classify_role_family(role),
            "location": parts[2][:60] if len(parts) > 2 else None,
            "url": f"https://news.ycombinator.com/item?id={child.get('id')}",
            "sourceName": "Hacker News (Who is hiring?)",
            "sourceUrl": f"https://news.ycombinator.com/item?id={child.get('id')}",
            "aiRelevant": True,
        })
        if len(records) >= max_jobs:
            break
    return records


# ---------------------------------------------------------------------------
# Async collector
# ---------------------------------------------------------------------------
HN_STORIES_URL = "https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring&hitsPerPage=3"
HN_ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"


async def collect_jobs(fetcher: Fetcher, window_hours: int = 24) -> dict:
    per_source: dict[str, int] = {}
    candidates: dict[str, dict] = {}

    async def add(records: list[dict], source: str) -> None:
        per_source[source] = len(records)
        for record in records:
            key = record["url"]
            if key and key not in candidates:
                candidates[key] = record

    async def safe(name: str, coro_factory) -> None:
        try:
            await coro_factory()
        except Exception as exc:  # noqa: BLE001
            log.warning("job source failed: %s (%s)", name, exc)
            per_source.setdefault(name, 0)

    async def remoteok() -> None:
        payload = await fetcher.get_json("https://remoteok.com/api")
        await add(parse_remoteok(payload), "RemoteOK")

    async def remotive() -> None:
        payload = await fetcher.get_json("https://remotive.com/api/remote-jobs?limit=100")
        await add(parse_remotive(payload), "Remotive")

    async def arbeitnow() -> None:
        payload = await fetcher.get_json("https://www.arbeitnow.com/api/job-board-api")
        await add(parse_arbeitnow(payload), "Arbeitnow")

    async def jobicy() -> None:
        resp = await fetcher.get("https://jobicy.com/feed")
        if resp.ok:
            await add(parse_rss_jobs(resp.text, "Jobicy"), "Jobicy")

    async def wwr() -> None:
        collected: list[dict] = []
        for feed_url in WWR_FEEDS:
            resp = await fetcher.get(feed_url)
            if resp.ok:
                collected.extend(parse_rss_jobs(resp.text, "We Work Remotely"))
        await add(collected, "We Work Remotely")

    async def hackernews() -> None:
        stories = await fetcher.get_json(HN_STORIES_URL)
        hits = stories.get("hits", []) if isinstance(stories, dict) else []
        if not hits:
            return
        story = await fetcher.get_json(HN_ITEM_URL.format(item_id=hits[0]["objectID"]))
        await add(parse_hn_hiring(story), "Hacker News (Who is hiring?)")

    await asyncio.gather(
        safe("RemoteOK", remoteok), safe("Remotive", remotive), safe("Arbeitnow", arbeitnow),
        safe("Jobicy", jobicy), safe("We Work Remotely", wwr), safe("Hacker News", hackernews),
    )
    log.info("job candidates: %s (total %d)", per_source, len(candidates))

    now = now_utc()
    fresh: list[dict] = []
    for record in candidates.values():
        if not record.get("aiRelevant"):
            continue
        if record.get("publishedAt") and is_fresh(record["publishedAt"], now, window_hours):
            fresh.append(record)
        elif not record.get("publishedAt"):
            # documented freshness heuristic: first time this run saw the item
            record["publishedAt"], record["dateSource"] = (
                to_iso(now), DATE_SOURCE_HEURISTIC
            )
            fresh.append(record)

    log.info("fresh AI jobs (<=%dh): %d", window_hours, len(fresh))
    snapshot = fresh
    save_snapshot("jobs_raw.json", snapshot)
    save_snapshot("jobs_sources_meta.json", {
        "perSource": per_source, "candidates": len(candidates), "fresh": len(fresh),
    })
    return {"records": snapshot, "counts": {"perSource": per_source, "fresh": len(fresh)}}
