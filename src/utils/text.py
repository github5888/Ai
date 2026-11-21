"""Text & URL normalisation helpers shared by every scraper."""
from __future__ import annotations

import html as html_mod
import re
from urllib.parse import urlsplit, urlunsplit

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")

# Tracking params stripped before hashing URLs so the same article seen on
# three crawler nodes collapses to one canonical key (Phase VI, Q3).
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "ref_src", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid",
    "cmpid", "share", "sh", "s", "sp_ab_test", "s_campaign", "referrer",
}


def norm_space(text: str | None) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def strip_html(raw: str | None) -> str:
    """Cheap HTML tag strip + entity unescape + whitespace collapse."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = html_mod.unescape(text)
    return norm_space(text)


def truncate(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "\u2026"


def normalize_url(url: str | None) -> str | None:
    """Canonical form of a URL: https, lowercase host, no tracking params,
    no fragment, no trailing slash.  Returns None for non-http junk."""
    if not url:
        return None
    url = url.strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    scheme = "https"
    netloc = parts.netloc.lower()
    # strip default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443"):
        netloc = netloc[:-4]
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = parts.query
    if query:
        kept = []
        for kv in query.split("&"):
            key = kv.split("=", 1)[0].lower()
            if key not in _TRACKING_PARAMS:
                kept.append(kv)
        query = "&".join(kept)
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return None
    host = host.removeprefix("www.")
    return host or None


def title_case_entity(name: str | None) -> str:
    """Cosmetic cleanup for entity names, preserving known inner casing
    (OpenAI, xAI, Hugging Face) rather than blindly lowercasing."""
    if not name:
        return ""
    name = norm_space(name).strip(" -,;\u2013\u2014")
    return name
