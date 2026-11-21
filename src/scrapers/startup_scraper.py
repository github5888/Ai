"""Startup scraper (Phase I-A).

Compliant, documented source strategy - all of these are public directories
or APIs that permit programmatic access:

1. **Wikipedia category tree** ("Artificial intelligence companies", plus the
   per-country subcategories) via the MediaWiki API - hundreds to thousands
   of AI companies, each record citing its own article URL.
2. **Curated GitHub lists** of AI companies (applied-ml, awesome-european-ai,
   awesome-robotics-ai-companies, ...) fetched through api.github.com.
3. **Directory listicles** (e.g. builtin.com company directory) fetched as
   HTML and parsed with BeautifulSoup.
4. **Wikidata SPARQL** enrichment for employee counts (P1128) - only where
   Wikidata itself states the number; otherwise ``employeeCount`` stays null.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Iterable

from ..config import Settings
from ..fetcher import Fetcher
from ..utils.text import norm_space, title_case_entity
from .base import save_snapshot

log = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"

# Main category + the largest country subcategories of
# https://en.wikipedia.org/wiki/Category:Artificial_intelligence_companies
WIKI_CATEGORIES = [
    "Artificial intelligence companies",
    "Artificial intelligence companies in the United States",
    "Artificial intelligence companies in the United Kingdom",
    "Artificial intelligence companies in India",
    "Artificial intelligence companies in China",
    "Artificial intelligence companies in Israel",
    "Artificial intelligence companies in Germany",
    "Artificial intelligence companies in France",
    "Artificial intelligence companies in Canada",
    "Artificial intelligence companies in Japan",
    "Artificial intelligence companies in South Korea",
    "Artificial intelligence companies in Australia",
    "Artificial intelligence companies in the Netherlands",
    "Artificial intelligence companies in Switzerland",
    "Artificial intelligence companies in Sweden",
    "Artificial intelligence companies in Spain",
    "Artificial intelligence companies in Italy",
    "Artificial intelligence companies in Brazil",
    "Artificial intelligence companies in Russia",
    "Artificial intelligence companies in Singapore",
    "Artificial intelligence companies in Nigeria",
    "Artificial intelligence companies in Kenya",
]

# Curated GitHub lists (repo -> README path).  Fetched via api.github.com
# with the raw media type, so this works anywhere GitHub API is reachable.
GITHUB_COMPANY_LISTS = [
    {"repo": "eugeneyan/applied-ml", "path": "README.md", "source": "GitHub:eugeneyan/applied-ml"},
    {"repo": "balloch/awesome-robotics-ai-companies", "path": "README.md",
     "source": "GitHub:awesome-robotics-ai-companies"},
    {"repo": "jmbarrancoml/awesome-european-ai", "path": "README.md",
     "source": "GitHub:awesome-european-ai"},
    {"repo": "Hardy-Uint/awesome-AI-company-summary", "path": "README.md",
     "source": "GitHub:awesome-AI-company-summary"},
    {"repo": "aishwaryanr/awesome-generative-ai-guide", "path": "README.md",
     "source": "GitHub:awesome-generative-ai-guide"},
]

# Sections that list papers/courses rather than companies - excluded.
_NON_COMPANY_SECTION_RE = re.compile(
    r"paper|course|tutorial|newsletter|podcast|reading|video|talk|dataset|"
    r"model|library|framework|tool|resource|community|conference|benchmark|blog|"
    r"cheat|question|interview|awesome|research ?(papers?)?$",
    re.IGNORECASE,
)

_LIST_ENTRY_RE = re.compile(
    r"^\s*[-*]\s+(?:\[(?P<name>[^\]]{2,80})\]\((?P<url>https?://[^)\s]+)\)"
    r"|(?P<plain>[A-Z][\w&.'\- ]{2,60}))\s*(?:[-–—:|]\s*(?P<desc>.*))?$"
)

_NAME_STOPWORDS_RE = re.compile(
    r"\b(papers?|blog|course|newsletter|podcast|dataset|library|framework|"
    r"awesome|guide|tutorial|collection|list|repo(sitory)?|github|website)\b",
    re.IGNORECASE,
)

# Directory listicle candidates (graceful failure if protected/changed).
LISTICLE_SOURCES = [
    {"url": "https://builtin.com/artificial-intelligence/ai-companies-usa",
     "source": "Builtin AI Directory", "kind": "builtin"},
    {"url": "https://builtin.com/artificial-intelligence/ai-companies-nyc",
     "source": "Builtin AI Directory NYC", "kind": "builtin"},
]

# Bulk startup-directory datasets published on GitHub (fetchable anywhere).
# 1) Y Combinator directory export - community-maintained, auto-refreshed CSV
#    of the public ycombinator.com/companies directory (tags flag AI startups).
YC_CSV_URL = os.environ.get(
    "YC_COMPANIES_CSV_URL",
    "https://raw.githubusercontent.com/akshaybhalotia/yc_company_scraper/main/data/yc_essential_data.csv",
)
# 2) AI companies/services directory with employee bands.
AI_SERVICES_CSV_URL = os.environ.get(
    "AI_SERVICES_CSV_URL",
    "https://raw.githubusercontent.com/Sumbal-Naz23/AI-Companies-Dataset/main/AI_Companies.csv",
)

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


# ---------------------------------------------------------------------------
# Pure parsers (unit-tested without network)
# ---------------------------------------------------------------------------
def parse_wiki_members(payload: dict) -> list[dict]:
    """Extract namespace-0 (article) members from a MediaWiki categorymembers
    API response, resolving titles to article URLs."""
    out: list[dict] = []
    for member in (payload.get("query", {}) or {}).get("categorymembers", []):
        if member.get("ns") != 0:
            continue
        title = member.get("title")
        if not title:
            continue
        out.append(
            {
                "entityName": title_case_entity(title),
                "employeeCount": None,
                "sourceName": "Wikipedia",
                "sourceUrl": "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_")),
                "sourceCategory": "wikipedia",
            }
        )
    return out


def wiki_continue_token(payload: dict) -> str | None:
    cont = payload.get("continue", {}) or {}
    return cont.get("cmcontinue")


def parse_github_markdown_companies(markdown: str, source_name: str) -> list[dict]:
    """Parse curated-list markdown into company records.

    Tracks the current section header so papers/courses sections are skipped,
    and accepts both ``- [Company](url) - note`` and plain-name bullet items.
    """
    records: list[dict] = []
    seen: set[str] = set()
    current_section = ""
    for line in markdown.splitlines():
        if line.startswith("#"):
            current_section = line.lstrip("#").strip()
            continue
        match = _LIST_ENTRY_RE.match(line)
        if not match:
            continue
        name = match.group("name") or match.group("plain")
        url = match.group("url")
        desc = match.group("desc") or ""
        name = norm_space(name)
        if not name or len(name) < 2 or len(name) > 60:
            continue
        if _NAME_STOPWORDS_RE.search(name):
            continue
        if _NON_COMPANY_SECTION_RE.search(current_section):
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        records.append(
            {
                "entityName": title_case_entity(name),
                "employeeCount": None,
                "description": norm_space(desc)[:300] or None,
                "sourceName": source_name,
                "sourceUrl": url or current_section,
                "sourceCategory": "github-list",
            }
        )
    return records


def parse_builtin_directory(html: str, source_name: str) -> list[dict]:
    """Builtin.com directory pages link each company as /company/<slug>."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover
        return []
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=re.compile(r"^/company/")):
        name = norm_space(anchor.get_text())
        if not name or len(name) > 60 or name.lower() in seen:
            continue
        seen.add(name.lower())
        href = anchor.get("href", "")
        records.append(
            {
                "entityName": title_case_entity(name),
                "employeeCount": None,
                "sourceName": source_name,
                "sourceUrl": "https://builtin.com" + href,
                "sourceCategory": "directory",
            }
        )
    return records


def parse_wikidata_employees(sparql_json: dict) -> dict[str, dict]:
    """P1128 (number of employees) results keyed by company label."""
    out: dict[str, dict] = {}
    for binding in sparql_json.get("results", {}).get("bindings", []):
        label = binding.get("label", {}).get("value")
        employees = binding.get("employees", {}).get("value")
        entity = binding.get("item", {}).get("value")
        if not label or not employees:
            continue
        try:
            out[label] = {"employeeCount": int(float(employees)), "wikidataUrl": entity}
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Async collectors
# ---------------------------------------------------------------------------
async def collect_wikipedia(fetcher: Fetcher) -> list[dict]:
    """Crawl the MediaWiki category tree with polite pagination.

    Subcategories are discovered dynamically (cmtype=subcat) instead of being
    hard-coded, so the scraper keeps working as Wikipedia reorganises its
    taxonomy (e.g. 'Artificial intelligence companies of the United States')."""
    ROOT_CATEGORY = "Artificial intelligence companies"
    categories = [ROOT_CATEGORY]
    try:
        subcats = await _wiki_subcategories(fetcher, ROOT_CATEGORY)
        categories.extend(subcats)
        log.info("wikipedia: %d subcategories discovered", len(subcats))
    except Exception as exc:  # noqa: BLE001
        log.warning("wikipedia subcategory discovery failed (%s); main category only", exc)

    all_records: dict[str, dict] = {}
    for category in categories[:80]:
        cmcontinue: str | None = None
        pages_crawled = 0
        while True:
            params = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": "500",
                "cmnamespace": "0",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            url = WIKI_API + "?" + urllib.parse.urlencode(params)
            try:
                payload = await fetcher.get_json(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("wikipedia category failed: %s (%s)", category, exc)
                break
            records = parse_wiki_members(payload)
            for record in records:
                all_records.setdefault(record["entityName"].lower(), record)
            pages_crawled += 1
            cmcontinue = wiki_continue_token(payload)
            if not cmcontinue or pages_crawled >= 6:  # safety valve per category
                break
    log.info("wikipedia -> %d unique companies across %d categories",
             len(all_records), len(categories))
    return list(all_records.values())


async def _wiki_subcategories(fetcher: Fetcher, root_category: str,
                              max_pages: int = 4) -> list[str]:
    """List direct subcategories of a category (cmtype=14)."""
    subcats: list[str] = []
    cmcontinue: str | None = None
    for _ in range(max_pages):
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "list": "categorymembers", "cmtitle": f"Category:{root_category}",
            "cmlimit": "500", "cmtype": "14",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        payload = await fetcher.get_json(WIKI_API + "?" + urllib.parse.urlencode(params))
        for member in (payload.get("query", {}) or {}).get("categorymembers", []):
            title = member.get("title", "")
            if title.startswith("Category:"):
                subcats.append(title.removeprefix("Category:"))
        cmcontinue = wiki_continue_token(payload)
        if not cmcontinue:
            break
    return subcats


async def collect_github_lists(fetcher: Fetcher, settings: Settings) -> list[dict]:
    """Fetch curated GitHub company lists via api.github.com raw contents."""
    headers = {"Accept": "application/vnd.github.raw+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    tasks = [
        fetcher.get(f"https://api.github.com/repos/{src['repo']}/contents/{src['path']}", headers=headers)
        for src in GITHUB_COMPANY_LISTS
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    records: dict[str, dict] = {}
    for src, resp in zip(GITHUB_COMPANY_LISTS, responses):
        if isinstance(resp, Exception):
            log.warning("github list failed: %s (%s)", src["repo"], resp)
            continue
        if not resp.ok:
            log.warning("github list failed: %s -> HTTP %s", src["repo"], resp.status)
            continue
        parsed = parse_github_markdown_companies(resp.text, src["source"])
        for record in parsed:
            records.setdefault(record["entityName"].lower(), record)
        log.info("github list %s -> %d companies (cumulative %d)", src["repo"], len(parsed), len(records))
    return list(records.values())


async def collect_listicles(fetcher: Fetcher) -> list[dict]:
    """Fetch directory pages; skip any source that blocks us (no bypassing)."""
    records: dict[str, dict] = {}
    for src in LISTICLE_SOURCES:
        try:
            if not fetcher.robots_allowed(src["url"]):
                log.warning("robots.txt disallows %s; skipping", src["url"])
                continue
            resp = await fetcher.get(src["url"], max_attempts=2)
            if not resp.ok:
                log.warning("listicle blocked/missing %s -> HTTP %s (graceful skip)", src["url"], resp.status)
                continue
            parsed = parse_builtin_directory(resp.text, src["source"])
            for record in parsed:
                records.setdefault(record["entityName"].lower(), record)
            log.info("listicle %s -> %d companies", src["url"], len(parsed))
        except Exception as exc:  # noqa: BLE001
            log.warning("listicle failed %s (%s)", src["url"], exc)
    return list(records.values())


def parse_yc_csv(csv_text: str, limit: int = 4000) -> list[dict]:
    """AI startups from the YC directory export (tags flag + AI pitch filter).

    team_size comes straight from the dataset; it is kept as-is and described
    in the record so nothing is invented."""
    import io

    import pandas as pd

    frame = pd.read_csv(io.StringIO(csv_text))
    if "name" not in frame.columns:
        return []

    def has_ai_tag(tags: Any) -> bool:
        if not isinstance(tags, str) or not tags.startswith("["):
            return False
        try:
            return any("Artificial Intelligence" in str(t) for t in json.loads(tags))
        except json.JSONDecodeError:
            return False

    pitch_re = re.compile(
        r"\bAI\b|artificial intelligence|machine learning|\bLLMs?\b|deep learning",
        re.IGNORECASE,
    )
    mask = (
        frame["tags"].apply(has_ai_tag)
        | frame.get("one_liner", pd.Series([""] * len(frame))).astype(str).str.contains(pitch_re, na=False)
        | frame.get("long_description", pd.Series([""] * len(frame))).astype(str).str.contains(pitch_re, na=False)
    )
    selected = frame[mask].head(limit)

    records: list[dict] = []
    for row in selected.itertuples(index=False):
        name = norm_space(str(getattr(row, "name", "")))
        slug = norm_space(str(getattr(row, "slug", "")))
        if not name:
            continue
        team_size = getattr(row, "team_size", None)
        employee_count = None
        try:
            if team_size is not None and str(team_size) not in ("", "nan", "None"):
                employee_count = int(float(team_size))
        except (TypeError, ValueError):
            employee_count = None
        description = norm_space(str(getattr(row, "one_liner", "") or ""))[:300] or None
        if employee_count:
            description = f"{description or ''} (YC team size: {employee_count})".strip()
        records.append({
            "entityName": title_case_entity(name),
            "employeeCount": employee_count,
            "description": description,
            "sourceName": "Y Combinator AI Startups (directory export)",
            "sourceUrl": f"https://www.ycombinator.com/companies/{slug}" if slug else None,
            "sourceCategory": "directory",
        })
    return records


def parse_ai_services_csv(csv_text: str, limit: int = 4000) -> list[dict]:
    """AI companies from an AI-services directory CSV (employee bands kept
    verbatim in the description - ranges are never converted into fake ints)."""
    import io

    import pandas as pd

    frame = pd.read_csv(io.StringIO(csv_text))
    name_col = next((c for c in frame.columns if "company" in c.lower()), None)
    if not name_col:
        return []
    employees_col = next((c for c in frame.columns if "employee" in c.lower()), None)
    website_col = next((c for c in frame.columns if "website" in c.lower()), None)
    location_col = next((c for c in frame.columns if "location" in c.lower()), None)

    records: list[dict] = []
    for row in frame.head(limit).itertuples(index=False):
        row_dict = dict(zip(frame.columns, row))
        name = norm_space(str(row_dict.get(name_col, "")))
        if not name or len(name) > 70:
            continue
        employees = norm_space(str(row_dict.get(employees_col, ""))) if employees_col else ""
        description_bits = []
        if employees and employees.lower() != "nan":
            description_bits.append(f"Employees: {employees}")
        location = norm_space(str(row_dict.get(location_col, ""))) if location_col else ""
        if location and location.lower() != "nan":
            description_bits.append(f"HQ: {location}")
        website = norm_space(str(row_dict.get(website_col, ""))) if website_col else ""
        source_url = website if website.startswith("http") else None
        records.append({
            "entityName": title_case_entity(name),
            "employeeCount": None,  # range strings are not honest ints
            "description": "; ".join(description_bits) or None,
            "sourceName": "AI Companies Directory (services export)",
            "sourceUrl": source_url,
            "sourceCategory": "directory",
        })
    return records


async def collect_directory_datasets(fetcher: Fetcher) -> list[dict]:
    """Fetch the bulk directory CSVs; each source degrades gracefully."""
    merged: dict[str, dict] = {}
    for label, url, parser in (
        ("yc", YC_CSV_URL, parse_yc_csv),
        ("aiServices", AI_SERVICES_CSV_URL, parse_ai_services_csv),
    ):
        try:
            resp = await fetcher.get(url, max_attempts=3,
                                     timeout=__import__("aiohttp").ClientTimeout(total=180))
            if not resp.ok:
                log.warning("directory dataset %s failed: HTTP %s", label, resp.status)
                continue
            parsed = parser(resp.text)
            for record in parsed:
                key = record["entityName"].lower()
                if key not in merged:
                    merged[key] = record
            log.info("directory dataset %s -> %d companies", label, len(parsed))
        except Exception as exc:  # noqa: BLE001
            log.warning("directory dataset %s failed (%s)", label, exc)
    return list(merged.values())


async def collect_employee_counts(fetcher: Fetcher, labels: Iterable[str]) -> dict[str, dict]:
    """One batched Wikidata SPARQL query for employee counts (P1128).

    Only numbers Wikidata itself records are returned; everything else stays
    null (no guessing)."""
    wanted = [label for label in labels if label][:60]
    if not wanted:
        return {}
    values = " ".join(f'"{urllib.parse.quote(label)}"@en' for label in wanted)
    query = f"""
    SELECT ?label ?item ?employees WHERE {{
      VALUES ?label {{ {values} }}
      ?item rdfs:label ?label .
      ?item wdt:P1128 ?employees .
      FILTER(LANG(?label) = "en")
    }} LIMIT 200"""
    url = _WIKIDATA_SPARQL + "?format=json&query=" + urllib.parse.quote(query)
    try:
        payload = await fetcher.get_json(url, headers={"Accept": "application/sparql-results+json"})
        return parse_wikidata_employees(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("wikidata employee enrichment failed: %s", exc)
        return {}


async def collect_startups(fetcher: Fetcher, settings: Settings) -> dict[str, Any]:
    """Run every startup source; merge by normalised name."""
    wikipedia_task = asyncio.create_task(collect_wikipedia(fetcher))
    directories_task = asyncio.create_task(collect_directory_datasets(fetcher))
    github = await collect_github_lists(fetcher, settings)
    listicles = await collect_listicles(fetcher)
    wiki = await wikipedia_task
    directories = await directories_task

    merged: dict[str, dict] = {}
    for record in wiki + directories + github + listicles:
        key = record["entityName"].lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
        else:
            # keep the richer source URL and richer employee data
            if not existing.get("sourceUrl") and record.get("sourceUrl"):
                existing["sourceUrl"] = record["sourceUrl"]
            if existing.get("employeeCount") is None and record.get("employeeCount") is not None:
                existing["employeeCount"] = record["employeeCount"]
            if not existing.get("description") and record.get("description"):
                existing["description"] = record["description"]

    snapshot = list(merged.values())
    save_snapshot("startups_raw.json", snapshot)
    save_snapshot("startups_sources_meta.json", {
        "wikipedia": len(wiki), "directories": len(directories), "githubLists": len(github),
        "listicles": len(listicles), "merged": len(snapshot),
    })
    return {"records": snapshot, "counts": {"wikipedia": len(wiki), "directories": len(directories),
                                            "github": len(github), "listicles": len(listicles),
                                            "merged": len(snapshot)}}
