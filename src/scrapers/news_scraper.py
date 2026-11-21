"""AI news scraper (Phase II-A).

Monitors 8 configured AI news feeds (assignment minimum: 5).  Every feed is
an official RSS/Atom endpoint - a compliant, permitted ingestion channel.
For items published within the freshness window the crawler additionally
fetches the article page and extracts full-text content.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..fetcher import Fetcher
from ..utils.dates import is_fresh, now_utc, parse_timestamp
from ..utils.text import norm_space, normalize_url, strip_html, truncate
from .base import save_snapshot

log = logging.getLogger(__name__)

NEWS_FEEDS = [
    {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai/index.xml"},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
    {"name": "Ars Technica AI", "url": "https://arstechnica.com/ai/feed/"},
    {"name": "Wired AI", "url": "https://www.wired.com/feed/tag/ai/latest/rss"},
    {"name": "MIT Technology Review AI",
     "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    {"name": "Marktechpost AI", "url": "https://www.marktechpost.com/feed/"},
    {"name": "Hacker News (AI)", "url": "https://hnrss.org/newest?q=AI+OR+%22machine+learning%22"},
]


def parse_feed(xml_text: str, source_name: str) -> list[dict]:
    """Pure RSS/Atom parser (unit-testable)."""
    import feedparser

    feed = feedparser.parse(xml_text)
    records: list[dict] = []
    for entry in feed.entries:
        link = entry.get("link") or ""
        headline = norm_space(entry.get("title", ""))
        if not headline or not link:
            continue
        published_iso, date_source = parse_timestamp(
            entry.get("published") or entry.get("updated")
        )
        content = ""
        content_entries = entry.get("content") or []
        if content_entries:
            content = strip_html(content_entries[0].get("value", ""))
        if not content:
            content = strip_html(entry.get("summary", ""))
        records.append(
            {
                "headline": headline,
                "url": normalize_url(link) or link,
                "publishedAt": published_iso,
                "dateSource": date_source if published_iso else None,
                "content": truncate(content, 8000),
                "sourceName": source_name,
                "sourceUrl": normalize_url(link) or link,
            }
        )
    return records


def extract_article_text(html: str) -> str:
    """Full-text extraction heuristic: prefer <article>, else the <div> with
    the most paragraph text.  Works across WordPress/Next.js publishers."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        paragraphs = [norm_space(p.get_text()) for p in article.find_all("p")]
    else:
        best_div, best_len = None, 0
        for div in soup.find_all(["div", "main", "section"]):
            text_len = sum(len(norm_space(p.get_text())) for p in div.find_all("p"))
            if text_len > best_len:
                best_div, best_len = div, text_len
        paragraphs = [norm_space(p.get_text()) for p in best_div.find_all("p")] if best_div else []

    paragraphs = [p for p in paragraphs if len(p) > 40]
    return "\n\n".join(paragraphs)


async def collect_news(fetcher: Fetcher, window_hours: int = 24,
                       max_full_text: int = 30) -> dict:
    """Fetch all feeds, apply the 24-hour freshness filter, then pull full
    text for the freshest articles (bounded so the run stays polite)."""
    feed_results = await asyncio.gather(
        *[fetcher.get(src["url"], max_attempts=3) for src in NEWS_FEEDS],
        return_exceptions=True,
    )

    candidates: dict[str, dict] = {}
    per_source: dict[str, int] = {}
    for src, resp in zip(NEWS_FEEDS, feed_results):
        if isinstance(resp, Exception):
            log.warning("news feed failed: %s (%s)", src["name"], resp)
            per_source[src["name"]] = 0
            continue
        if not resp.ok:
            log.warning("news feed failed: %s -> HTTP %s", src["name"], resp.status)
            per_source[src["name"]] = 0
            continue
        parsed = parse_feed(resp.text, src["name"])
        per_source[src["name"]] = len(parsed)
        for record in parsed:
            key = record["url"]
            if key not in candidates or (
                record.get("publishedAt") and not candidates[key].get("publishedAt")
            ):
                candidates[key] = record
    log.info("news candidates: %s (total %d)", per_source, len(candidates))

    now = now_utc()
    fresh = [
        r for r in candidates.values()
        if r["publishedAt"] and is_fresh(r["publishedAt"], now, window_hours)
    ]
    log.info("fresh (<=%dh) news articles: %d", window_hours, len(fresh))

    # Full-text fetch for the freshest items (bounded, robots-aware).
    fresh.sort(key=lambda r: r.get("publishedAt") or "", reverse=True)
    full_text_targets = fresh[:max_full_text]
    for record in full_text_targets:
        url = record["url"]
        try:
            if not fetcher.robots_allowed(url):
                continue
            resp = await fetcher.get(url, max_attempts=2)
            if resp.ok:
                body = extract_article_text(resp.text)
                if len(body) > 400:
                    record["content"] = truncate(body, 20000)
        except Exception as exc:  # noqa: BLE001
            log.warning("article full-text failed %s (%s)", url, exc)
        await asyncio.sleep(0.2)

    snapshot = fresh
    save_snapshot("news_raw.json", snapshot)
    save_snapshot("news_sources_meta.json", {
        "perSource": per_source, "candidates": len(candidates), "fresh": len(fresh),
    })
    return {"records": snapshot, "counts": {"perSource": per_source, "fresh": len(fresh)}}
