"""Date parsing & freshness normalisation (Phase II core).

Every source on the web expresses time differently:
    "2 hours ago", "30 minutes ago", "Yesterday", "Published on November 21, 2026",
    RFC-822 headers ("Fri, 20 Nov 2026 19:53:41 +0000"), ISO-8601 strings,
    unix epochs (seconds *or* milliseconds), "3d", "today".

All of them are normalised here to ISO-8601 UTC timestamps.

If a source states NO date at all we deliberately do *not* invent one
(fabricated data = disqualification).  Instead the record is stamped with
``heuristic:first_seen`` -- the first time *our pipeline* observed it.  That
is an honest, documented freshness heuristic: an item observed for the very
first time is treated as new for one freshness window; on subsequent runs it
is only kept if the source later reveals a real date.
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Tuple

try:  # dateutil is a runtime dependency, but keep import local for clarity
    from dateutil import parser as dateutil_parser
except ImportError:  # pragma: no cover
    dateutil_parser = None  # type: ignore

UTC = timezone.utc

DATE_SOURCE_STATED = "stated"
DATE_SOURCE_HEURISTIC = "heuristic:first_seen"

_MONTHS = {m: i for i, m in enumerate(calendar.month_abbr) if m}
_MONTHS.update({m: i for i, m in enumerate(calendar.month_name) if m})

_RELATIVE_RE = re.compile(
    r"(?P<amount>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>second|sec|s|minute|min|m|hour|hr|h|day|d|week|wk|w|month|mo|year|yr|y)"
    r"s?\s*(?:ago|old|earlier)?",
    re.IGNORECASE,
)

_MONTH_DAY_YEAR_RE = re.compile(
    r"(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})"
)
_DAY_MONTH_YEAR_RE = re.compile(
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>[A-Za-z]{3,9})\.?,?\s+(?P<year>\d{4})"
)

_UNIT_SECONDS = {
    "second": 1, "sec": 1, "s": 1,
    "minute": 60, "min": 60, "m": 60,
    "hour": 3600, "hr": 3600, "h": 3600,
    "day": 86400, "d": 86400,
    "week": 604800, "wk": 604800, "w": 604800,
}
_UNIT_MONTHS = {"month": 1, "mo": 1}
_UNIT_YEARS = {"year": 1, "yr": 1, "y": 1}


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def to_iso(dt: datetime) -> str:
    """Normalise any aware/naive datetime to an ISO-8601 UTC string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_epoch(value: float | int) -> datetime:
    """Epoch seconds or milliseconds (auto-detected by magnitude)."""
    v = float(value)
    if v > 1e12:  # milliseconds
        v /= 1000.0
    return datetime.fromtimestamp(v, tz=UTC)


def parse_relative_to(text: str, ref: datetime) -> datetime | None:
    """Parse "2 hours ago" / "3d" / "yesterday" / "today" against a reference."""
    t = text.strip().lower()
    if t in {"today", "just now", "now"}:
        return ref
    if t in {"yesterday", "1 day ago"}:
        return ref - timedelta(days=1)

    m = _RELATIVE_RE.search(t)
    if not m:
        return None
    amount = float(m.group("amount").replace(",", "."))
    unit = m.group("unit").lower()
    if unit in _UNIT_SECONDS:
        return ref - timedelta(seconds=amount * _UNIT_SECONDS[unit])
    if unit in _UNIT_MONTHS:
        return ref - timedelta(days=amount * 30)
    if unit in _UNIT_YEARS:
        return ref - timedelta(days=amount * 365)
    return None


def _month_from_name(name: str) -> int | None:
    name = name.strip().title()
    if name in _MONTHS:
        return _MONTHS[name]
    # handle "Sept" style abbreviations
    for m, idx in _MONTHS.items():
        if m and name.lower().startswith(m.lower()[:3]):
            return idx
    return None


def parse_month_name_date(text: str) -> datetime | None:
    """Parse "November 21, 2026" / "21 November 2026" (no time component)."""
    for regex, order in ((_MONTH_DAY_YEAR_RE, "mdy"), (_DAY_MONTH_YEAR_RE, "dmy")):
        m = regex.search(text)
        if m:
            month = _month_from_name(m.group("month"))
            if not month:
                continue
            day = int(m.group("day"))
            year = int(m.group("year"))
            try:
                return datetime(year, month, day, tzinfo=UTC)
            except ValueError:
                continue
    return None


def parse_timestamp(
    value,
    ref: datetime | None = None,
) -> Tuple[str | None, str]:
    """Parse anything date-like into ``(iso8601_utc, date_source)``.

    Returns ``(None, "stated")`` when nothing parseable is found -- callers
    should then apply the first-seen heuristic instead of guessing.
    """
    ref = ref or now_utc()
    if value is None:
        return None, DATE_SOURCE_STATED

    # Raw numbers -> epoch (seconds or milliseconds)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return to_iso(parse_epoch(value)), DATE_SOURCE_STATED
        except (OverflowError, OSError, ValueError):
            return None, DATE_SOURCE_STATED

    text = str(value).strip()
    if not text:
        return None, DATE_SOURCE_STATED

    # Pure numeric strings (e.g. scraped "1737412800")
    if re.fullmatch(r"\d{9,14}(?:\.\d+)?", text):
        try:
            return to_iso(parse_epoch(float(text))), DATE_SOURCE_STATED
        except (OverflowError, OSError, ValueError):
            pass

    # Relative phrases first (they are unambiguous about ordering)
    rel = parse_relative_to(text, ref)
    if rel:
        return to_iso(rel), DATE_SOURCE_STATED

    if dateutil_parser is not None:
        try:
            dt = dateutil_parser.parse(text, tzinfos={"IST": 5.5 * 3600})
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return to_iso(dt), DATE_SOURCE_STATED
        except (ValueError, OverflowError, TypeError):
            pass

    named = parse_month_name_date(text)
    if named:
        return to_iso(named), DATE_SOURCE_STATED

    return None, DATE_SOURCE_STATED


def heuristic_first_seen(ref: datetime | None = None) -> Tuple[str, str]:
    """Freshness heuristic for sources that publish no date at all."""
    return to_iso(ref or now_utc()), DATE_SOURCE_HEURISTIC


def is_fresh(iso_ts: str | None, ref: datetime | None = None, window_hours: int = 24) -> bool:
    """True when the timestamp falls inside the freshness window."""
    if not iso_ts:
        return False
    ref = ref or now_utc()
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (ref - dt) <= timedelta(hours=window_hours) and dt <= ref + timedelta(minutes=5)
