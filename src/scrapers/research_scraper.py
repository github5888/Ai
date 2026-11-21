"""Research-paper scraper (Phase I-C) + GitHub star enrichment.

Sources
-------
1. **Arxiv** official Atom API (cat:cs.AI / cs.CL / cs.LG, newest first).
2. **Papers with Code** open data dump (the CC-BY-SA dataset rescued to the
   ``pwc-archive`` Hugging Face organisation after Meta sunset the site) -
   576k papers with authors, publication dates and paper<->code links.
   Only the selected slice is kept; the bulk parquet is a temporary download.
3. **GitHub REST/GraphQL** for *current* star counts - stars are dynamic, so
   they are fetched at run time, never hard-coded.

Every paper record keeps ``arxiv_id``, and repo matching is done through the
official dump (not guessed), so no fabricated links are possible.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import Settings, TMP_DIR
from ..fetcher import Fetcher
from ..utils.dates import parse_timestamp
from ..utils.text import norm_space
from .base import save_snapshot

log = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
PWC_HF_BASE = "https://huggingface.co/datasets/pwc-archive"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

PWC_PAPER_PARTS = 4  # papers-with-abstracts is split into 4 parquet parts

_ARXIV_QUERY = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG)"


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------
def parse_arxiv_atom(xml_text: str) -> list[dict]:
    """Parse the Arxiv Atom feed into paper records."""
    import feedparser

    feed = feedparser.parse(xml_text)
    records: list[dict] = []
    for entry in feed.entries:
        arxiv_id = (entry.get("id") or "").rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        canonical_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (entry.get("link") or "")
        authors = [norm_space(a.get("name", "")) for a in entry.get("authors", []) if a.get("name")]
        published_iso, _ = parse_timestamp(entry.get("published") or entry.get("updated"))
        records.append(
            {
                "title": norm_space(entry.get("title", "")),
                "authors": authors,
                "paper_url": canonical_url,
                "arxivId": arxiv_id,
                "githubUrl": None,
                "githubStars": None,
                "publishedDate": published_iso,
                "sourceName": "Arxiv",
                "sourceUrl": canonical_url,
            }
        )
    return records


def split_authors(raw: Any) -> list[str]:
    """PwC dump stores authors as a JSON array string or ';'-joined names."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [norm_space(str(a)) for a in raw if norm_space(str(a))]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                names = []
                for item in parsed:
                    if isinstance(item, dict):
                        names.append(norm_space(item.get("name", "")))
                    else:
                        names.append(norm_space(str(item)))
                return [n for n in names if n]
        except json.JSONDecodeError:
            pass
    return [norm_space(p) for p in re.split(r"[;]", text) if norm_space(p)]


def parse_pwc_frames(papers_df, links_df, limit: int = 2500) -> list[dict]:
    """Select papers from PwC pandas frames.

    Strategy: papers WITH a code repository are preferred (they carry GitHub
    metrics), newest first; remaining slots go to the newest papers overall.
    """
    papers = papers_df.copy()
    papers["published"] = papers["published"].astype(str)

    repo_by_paper: dict[Any, str] = {}
    if links_df is not None and len(links_df):
        links = links_df.copy()
        # prefer official repos, then longest-lived listing order
        links["is_official"] = links.get("is_official", 0)
        links = links.sort_values("is_official", ascending=False)
        seen: set[Any] = set()
        for paper_id, repo_url in zip(links["paper_id"], links["repo_url"]):
            key = paper_id
            if key in seen or not isinstance(repo_url, str) or not repo_url.startswith("http"):
                continue
            seen.add(key)
            repo_by_paper[key] = repo_url

    def to_record(row) -> dict | None:
        arxiv_id = str(getattr(row, "arxiv_id", "") or "")
        paper_url = str(getattr(row, "url_abs", "") or "")
        if paper_url and paper_url.startswith("http"):
            abs_url = paper_url
        elif arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            return None
        github_url = repo_by_paper.get(getattr(row, "id", None))
        published_iso, _ = parse_timestamp(getattr(row, "published", None))
        return {
            "title": norm_space(str(getattr(row, "title", ""))),
            "authors": split_authors(getattr(row, "authors", None)),
            "paper_url": abs_url,
            "arxivId": arxiv_id or None,
            "githubUrl": github_url,
            "githubStars": None,
            "publishedDate": published_iso,
            "sourceName": "Papers with Code",
            "sourceUrl": abs_url,
        }

    papers_sorted = papers.sort_values("published", ascending=False)
    with_repo = papers_sorted[papers_sorted["id"].isin(repo_by_paper.keys())].head(limit)
    selected_ids = set(with_repo["id"])
    without_repo = papers_sorted[~papers_sorted["id"].isin(selected_ids)].head(max(0, limit // 3))

    records: list[dict] = []
    for row in with_repo.itertuples(index=False):
        record = to_record(row)
        if record:
            records.append(record)
    for row in without_repo.itertuples(index=False):
        record = to_record(row)
        if record:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# GitHub star enrichment (GraphQL, batched 100 repos per request)
# ---------------------------------------------------------------------------
def build_star_query(repo_pairs: list[tuple[str, str]]) -> str:
    aliases = []
    for i, (owner, name) in enumerate(repo_pairs):
        aliases.append(
            f'repo{i}: repository(owner: "{owner}", name: "{name}") {{ stargazerCount url }}'
        )
    return "query {" + " ".join(aliases) + "}"


def parse_star_response(payload: dict, repo_pairs: list[tuple[str, str]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    data = payload.get("data", {}) or {}
    for i, (owner, name) in enumerate(repo_pairs):
        node = data.get(f"repo{i}")
        if node:
            out[f"{owner}/{name}".lower()] = {
                "stars": node.get("stargazerCount"),
                "url": node.get("url"),
            }
    return out


async def enrich_github_stars(
    github_urls: Iterable[str],
    token: str,
    batch_size: int = 100,
    post_fn: Callable[[str, str, dict], Any] | None = None,
) -> dict[str, dict]:
    """Fetch CURRENT star counts for many repos via batched GraphQL.

    ``post_fn(url, query, headers)`` is injectable for testing; the default
    performs a real POST against api.github.com/graphql.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url in github_urls:
        if not url:
            continue
        match = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)", url.strip().rstrip("/"))
        if not match:
            continue
        owner, name = match.group(1), match.group(2).removesuffix(".git")
        key = f"{owner}/{name}".lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((owner, name))

    if post_fn is None:
        import urllib.request

        def post_fn(url: str, query: str, headers: dict) -> dict:  # noqa: WPS440
            request = urllib.request.Request(  # noqa: S310
                url, data=json.dumps({"query": query}).encode(), headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=60) as fh:  # noqa: S310
                return json.loads(fh.read().decode())

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    results: dict[str, dict] = {}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        try:
            payload = await asyncio.to_thread(post_fn, GITHUB_GRAPHQL_URL, build_star_query(batch), headers)
            results.update(parse_star_response(payload, batch))
            log.info("stars batch %d/%d fetched (%d repos)", start // batch_size + 1,
                     (len(pairs) + batch_size - 1) // batch_size, len(batch))
        except Exception as exc:  # noqa: BLE001
            log.warning("star batch failed (%d repos): %s", len(batch), exc)
        await asyncio.sleep(1.0)  # be gentle with the GraphQL endpoint
    return results


# ---------------------------------------------------------------------------
# Async collectors
# ---------------------------------------------------------------------------
async def collect_arxiv(fetcher: Fetcher, max_results: int = 1200) -> list[dict]:
    params = {
        "search_query": _ARXIV_QUERY,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    resp = await fetcher.get(url)
    if not resp.ok:
        log.error("arxiv fetch failed: HTTP %s", resp.status)
        return []
    records = parse_arxiv_atom(resp.text)
    log.info("arxiv -> %d papers", len(records))
    return records


async def collect_pwc_dump(fetcher: Fetcher, limit: int = 2500,
                           tmp_dir: Path = TMP_DIR) -> list[dict]:
    """Download the PwC open-data parquet slices and select the newest slice.

    The full dump is ~600MB; it lives in data/tmp (git-ignored) and only the
    selected paper slice is persisted to data/raw."""
    import pandas as pd  # local import: only needed for this source

    tmp_dir.mkdir(parents=True, exist_ok=True)
    paper_paths: list[Path] = []
    for part in range(PWC_PAPER_PARTS):
        name = f"train-{part:05d}-of-{PWC_PAPER_PARTS:05d}.parquet"
        path = tmp_dir / f"papers-with-abstracts-{name}"
        if not path.exists() or path.stat().st_size < 1000:
            url = f"{PWC_HF_BASE}/papers-with-abstracts/resolve/main/data/{name}"
            resp = await fetcher.get(url, max_attempts=3)
            if not resp.ok:
                log.error("PwC paper part %d failed: HTTP %s", part, resp.status)
                return []
            path.write_bytes(resp.raw)
        paper_paths.append(path)

    links_path = tmp_dir / "links-between-paper-and-code.parquet"
    if not links_path.exists() or links_path.stat().st_size < 1000:
        url = f"{PWC_HF_BASE}/links-between-paper-and-code/resolve/main/data/train-00000-of-00001.parquet"
        resp = await fetcher.get(url, max_attempts=3)
        if resp.ok:
            links_path.write_bytes(resp.raw)
        else:
            log.warning("PwC links parquet failed: HTTP %s (papers without repos)", resp.status)

    frames = [pd.read_parquet(p, columns=["id", "arxiv_id", "title", "url_abs", "published", "authors"])
              for p in paper_paths]
    papers_df = pd.concat(frames, ignore_index=True)
    links_df = pd.read_parquet(links_path) if links_path.exists() else None

    records = parse_pwc_frames(papers_df, links_df, limit=limit)
    log.info("papers-with-code dump -> %d papers selected", len(records))
    return records


async def collect_papers(fetcher: Fetcher, settings: Settings,
                         arxiv_limit: int = 1200, pwc_limit: int = 2500) -> dict:
    arxiv_records = await collect_arxiv(fetcher, arxiv_limit)
    pwc_records = await collect_pwc_dump(fetcher, pwc_limit)

    merged: dict[str, dict] = {}
    for record in pwc_records + arxiv_records:  # arxiv (fresher) wins on clash
        key = (record.get("arxivId") or record["paper_url"]).lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
        else:
            if existing.get("githubUrl") and not record.get("githubUrl"):
                record["githubUrl"] = existing["githubUrl"]
            merged[key] = record

    snapshot = list(merged.values())
    save_snapshot("papers_raw.jsonl", snapshot)
    save_snapshot("papers_sources_meta.json", {
        "arxiv": len(arxiv_records), "papersWithCode": len(pwc_records), "merged": len(snapshot),
    })
    return {"records": snapshot, "counts": {"arxiv": len(arxiv_records),
                                            "papersWithCode": len(pwc_records),
                                            "merged": len(snapshot)}}
