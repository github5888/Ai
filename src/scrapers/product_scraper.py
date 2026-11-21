"""Product scraper (Phase I-B).

Sources: curated AI-tool lists hosted on GitHub (awesome-generative-ai,
ai-collection, awesome-ai-tools), fetched via api.github.com.  Each list
entry yields productName + source URL; pricingModel is classified ONLY from
what the entry's own description says (FREE / FREEMIUM / PAID / ENTERPRISE,
else null - no guessing), and startupName is assigned only when the product
URL domain or description explicitly matches a known company.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Iterable

from ..config import Settings
from ..fetcher import Fetcher
from ..llm.orchestrator import classify_pricing
from ..utils.text import domain_of, norm_space, title_case_entity
from .base import save_snapshot

log = logging.getLogger(__name__)

PRODUCT_GITHUB_LISTS = [
    {"repo": "steven2358/awesome-generative-ai", "path": "README.md",
     "source": "GitHub:awesome-generative-ai"},
    {"repo": "ai-collection/ai-collection", "path": "README.md",
     "source": "GitHub:ai-collection"},
    {"repo": "filipecalegario/awesome-generative-ai", "path": "README.md",
     "source": "GitHub:awesome-generative-ai (filipecalegario)"},
    {"repo": "AI-Tools-Inc/awesome-ai-tools", "path": "README.md",
     "source": "GitHub:awesome-ai-tools"},
]

# Product-URL domain -> owning company (explicit, source-implied mapping).
DOMAIN_COMPANY_MAP = {
    "openai.com": "OpenAI",
    "chatgpt.com": "OpenAI",
    "anthropic.com": "Anthropic",
    "claude.ai": "Anthropic",
    "deepmind.google": "Google DeepMind",
    "gemini.google.com": "Google DeepMind",
    "ai.meta.com": "Meta AI",
    "mistral.ai": "Mistral AI",
    "cohere.com": "Cohere",
    "cohere.ai": "Cohere",
    "huggingface.co": "Hugging Face",
    "perplexity.ai": "Perplexity AI",
    "x.ai": "xAI",
    "midjourney.com": "Midjourney",
    "stability.ai": "Stability AI",
    "runwayml.com": "Runway",
    "elevenlabs.io": "ElevenLabs",
    "character.ai": "Character.AI",
    "synthesia.io": "Synthesia",
    "jasper.ai": "Jasper",
    "writer.com": "Writer",
    "grammarly.com": "Grammarly",
    "deepl.com": "DeepL",
    "deepgram.com": "Deepgram",
    "assemblyai.com": "AssemblyAI",
    "replicate.com": "Replicate",
    "together.ai": "Together AI",
    "fireworks.ai": "Fireworks AI",
    "langchain.com": "LangChain",
    "llamaindex.ai": "LlamaIndex",
    "pinecone.io": "Pinecone",
    "weaviate.io": "Weaviate",
    "qdrant.tech": "Qdrant",
    "trychroma.com": "Chroma",
    "modal.com": "Modal",
    "anyscale.com": "Anyscale",
    "scale.com": "Scale AI",
    "groq.com": "Groq",
    "cerebras.ai": "Cerebras",
    "lambdalabs.com": "Lambda Labs",
    "inflection.ai": "Inflection AI",
    "notion.so": "Notion",
    "codeium.com": "Codeium",
    "tabnine.com": "Tabnine",
    "sourcegraph.com": "Sourcegraph",
    "descript.com": "Descript",
    "suno.com": "Suno",
    "udio.com": "Udio",
    "pika.art": "Pika",
    "heygen.com": "HeyGen",
}

_ENTRY_RE = re.compile(
    r"^\s*[-*]\s+\[(?P<name>[^\]]{2,80})\]\((?P<url>https?://[^)\s]+)\)\s*(?:[-–—:|]\s*(?P<desc>.*))?$"
)
_BADGE_NOISE_RE = re.compile(
    r"\b(badge|shield|stars?|license|npm|pypi|downloads|awesome|repo|github\.com/[^\s]+/?)\b"
    r"|!\[", re.IGNORECASE,
)
_NAME_NOISE_RE = re.compile(
    r"\b(paper|blog|course|newsletter|podcast|dataset|awesome|tutorial|guide|"
    r"chapter|lecture|video|book|community|forum|newsletters?)\b", re.IGNORECASE,
)


def parse_markdown_tools(markdown: str, source_name: str) -> list[dict]:
    """Parse ``- [Tool](url) - description`` entries from a curated list."""
    records: list[dict] = []
    seen: set[str] = set()
    for line in markdown.splitlines():
        match = _ENTRY_RE.match(line)
        if not match:
            continue
        name = norm_space(match.group("name"))
        url = match.group("url").rstrip(")").strip()
        desc = norm_space(match.group("desc") or "")
        if not name or len(name) > 70:
            continue
        if "github.com" in (url or "") and "github.com/topics" not in url:
            continue  # skip repo entries; products live on product sites
        if _BADGE_NOISE_RE.search(name) or _NAME_NOISE_RE.search(name):
            continue
        key = (name.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "productName": title_case_entity(name),
                "pricingHint": desc[:400],
                "sourceName": source_name,
                "sourceUrl": url,
            }
        )
    return records


def infer_startup_name(product_name: str, url: str | None, description: str | None,
                       known_startups: Iterable[str] = ()) -> str | None:
    """Assign startupName ONLY when the source itself implies it:
    the product's own domain maps to a known company, a known company name
    appears verbatim in the description, or the product name *is* the company."""
    url = url or ""
    domain = domain_of(url)
    if domain and domain in DOMAIN_COMPANY_MAP:
        return DOMAIN_COMPANY_MAP[domain]
    text = f"{product_name} {description or ''}"
    for known in known_startups:
        if len(known) >= 4 and re.search(re.escape(known), text, re.IGNORECASE):
            return known
    # product IS the company (e.g. Midjourney.com -> "Midjourney")
    if domain and product_name and domain.split(".")[0] == re.sub(r"[^a-z0-9]", "", product_name.lower()):
        return product_name
    return None


async def collect_products(fetcher: Fetcher, settings: Settings,
                           known_startups: Iterable[str] = ()) -> dict:
    headers = {"Accept": "application/vnd.github.raw+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    tasks = [
        fetcher.get(f"https://api.github.com/repos/{src['repo']}/contents/{src['path']}", headers=headers)
        for src in PRODUCT_GITHUB_LISTS
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    merged: dict[tuple, dict] = {}
    per_source = {}
    for src, resp in zip(PRODUCT_GITHUB_LISTS, responses):
        if isinstance(resp, Exception):
            log.warning("product list failed: %s (%s)", src["repo"], resp)
            continue
        if not resp.ok:
            log.warning("product list failed: %s -> HTTP %s", src["repo"], resp.status)
            continue
        parsed = parse_markdown_tools(resp.text, src["source"])
        per_source[src["repo"]] = len(parsed)
        for record in parsed:
            pricing = classify_pricing(record.pop("pricingHint", None))
            startup = infer_startup_name(
                record["productName"], record["sourceUrl"], None, known_startups
            )
            key = (record["productName"].lower(), record["sourceUrl"])
            if key in merged:
                existing = merged[key]
                if existing.get("pricingModel") is None and pricing:
                    existing["pricingModel"] = pricing
                if existing.get("startupName") is None and startup:
                    existing["startupName"] = startup
                continue
            record["pricingModel"] = pricing
            record["startupName"] = startup
            merged[key] = record

    snapshot = list(merged.values())
    save_snapshot("products_raw.json", snapshot)
    save_snapshot("products_sources_meta.json", {"perSource": per_source, "merged": len(snapshot)})
    return {"records": snapshot, "counts": {"perSource": per_source, "merged": len(snapshot)}}
