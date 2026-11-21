"""Deliverable 1 exporters: Google Sheets (6 tabs) + local CSV/XLSX fallback.

* ``export_csv``   - one CSV per tab (importable into Google Sheets).
* ``export_xlsx``  - a single workbook with exactly the 6 required tabs,
  ready to upload to Google Drive ("File > Import" preserves all tabs).
* ``upload_sheets``- writes directly into an existing Google Sheet via the
  official Sheets API (gspread + service account) when credentials exist.

Tab layout (exactly as the assignment specifies):
    01 Startups | 02 Products | 03 Research Papers | 04 Jobs | 05 News |
    06 Entity Mapping Log

Each tab is declared as ``(pretty header, model field)`` pairs; canonical
deduplication collapses multi-source rows to one row per canonical entity.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from ..config import OUT_DIR
from .database import JobPost, NewsArticle, Product, ResearchPaper, Startup, row_to_dict

log = logging.getLogger(__name__)

TAB_STARTUPS = "01 Startups"
TAB_PRODUCTS = "02 Products"
TAB_PAPERS = "03 Research Papers"
TAB_JOBS = "04 Jobs"
TAB_NEWS = "05 News"
TAB_MAPPING = "06 Entity Mapping Log"

# (pretty header, model field name)
TAB_SCHEMA = {
    TAB_STARTUPS: [
        ("entityName", "entity_name"),
        ("employeeCount", "employee_count"),
        ("canonicalName", "canonical_name"),
        ("sourceName", "source_name"),
        ("sourceUrl", "source_url"),
        ("collectedAt", "collected_at"),
    ],
    TAB_PRODUCTS: [
        ("productName", "product_name"),
        ("startupName", "startup_name"),
        ("canonicalStartup", "canonical_startup"),
        ("pricingModel", "pricing_model"),
        ("sourceName", "source_name"),
        ("sourceUrl", "source_url"),
        ("collectedAt", "collected_at"),
    ],
    TAB_PAPERS: [
        ("title", "title"),
        ("authors", "authors_json"),
        ("paper_url", "paper_url"),
        ("github_url", "github_url"),
        ("github_stars", "github_stars"),
        ("published_date", "published_date"),
        ("sourceName", "source_name"),
        ("sourceUrl", "source_url"),
        ("collectedAt", "collected_at"),
    ],
    TAB_JOBS: [
        ("company", "company"),
        ("jobTitle", "title"),
        ("date", "published_at"),
        ("is_remote", "is_remote"),
        ("role_family", "role_family"),
        ("location", "location"),
        ("date_source", "date_source"),
        ("sourceName", "source_name"),
        ("sourceUrl", "source_url"),
    ],
    TAB_NEWS: [
        ("headline", "headline"),
        ("publishedAt", "published_at"),
        ("date_source", "date_source"),
        ("content", "content"),
        ("sourceName", "source_name"),
        ("sourceUrl", "source_url"),
    ],
    TAB_MAPPING: [
        ("raw_name", "raw_name"),
        ("canonical_name", "canonical_name"),
        ("match_status", "match_status"),
        ("score", "score"),
        ("method", "method"),
        ("entity_type", "entity_type"),
        ("source_url", "source_url"),
        ("resolved_at", "resolved_at"),
    ],
}

TAB_ORDER = [TAB_STARTUPS, TAB_PRODUCTS, TAB_PAPERS, TAB_JOBS, TAB_NEWS, TAB_MAPPING]


def _cell(model_field: str, record: dict):
    value = record.get(model_field)
    if model_field == "authors_json":
        import json

        try:
            value = ", ".join(json.loads(record.get("authors_json") or "[]"))
        except (ValueError, TypeError):
            value = ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:49000]  # Google Sheets cell-limit safety
    return value


def _dedupe(rows: list[list], key_indices: list[int]) -> list[list]:
    """Collapse rows sharing an identity key; keep the most-filled row so
    richer records (employee counts, stars) win over bare ones."""
    best: dict[tuple, list] = {}
    for row in rows[1:]:
        key = tuple(str(row[i] or "").strip().lower() for i in key_indices)
        if not any(key):
            continue
        filled = sum(1 for cell in row if cell not in ("", None))
        current = best.get(key)
        if current is None or filled > sum(1 for c in current if c not in ("", None)):
            best[key] = row
    return [rows[0]] + [best[k] for k in sorted(best)]


def _tab_rows(tab: str, storage) -> list[list]:
    """Rows (with pretty headers) for one tab, canonical-deduped where needed:
    startups -> unique canonical entities; products -> unique (product,
    startup) pairs; papers -> unique canonical URLs."""
    schema = TAB_SCHEMA[tab]
    model_map = {
        TAB_STARTUPS: Startup,
        TAB_PRODUCTS: Product,
        TAB_PAPERS: ResearchPaper,
        TAB_JOBS: JobPost,
        TAB_NEWS: NewsArticle,
        TAB_MAPPING: None,
    }
    model = model_map[tab]
    if model is None:
        from .database import EntityMapping

        records = storage.load_table(EntityMapping)
        rows = [[pretty for pretty, _ in schema]]
        rows += [[_cell(field, record) for _, field in schema] for record in records]
        return rows

    records = storage.load_table(model)
    rows = [[pretty for pretty, _ in schema]]
    rows += [[_cell(field, record) for _, field in schema] for record in records]

    if tab == TAB_STARTUPS:
        rows = _dedupe(rows, [schema.index(("canonicalName", "canonical_name")),
                              schema.index(("entityName", "entity_name"))])
    elif tab == TAB_PRODUCTS:
        rows = _dedupe(rows, [schema.index(("productName", "product_name")),
                              schema.index(("startupName", "startup_name")),
                              schema.index(("canonicalStartup", "canonical_startup"))])
    elif tab == TAB_PAPERS:
        rows = _dedupe(rows, [schema.index(("paper_url", "paper_url"))])
    return rows


def export_csv(storage, out_dir: Path = OUT_DIR) -> list[Path]:
    """Write the 6 tab CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, tab in [
        ("01_startups.csv", TAB_STARTUPS),
        ("02_products.csv", TAB_PRODUCTS),
        ("03_research_papers.csv", TAB_PAPERS),
        ("04_jobs.csv", TAB_JOBS),
        ("05_news.csv", TAB_NEWS),
        ("06_entity_mapping_log.csv", TAB_MAPPING),
    ]:
        rows = _tab_rows(tab, storage)
        path = out_dir / filename
        with path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(rows)
        paths.append(path)
        log.info("exported %s (%d logical rows incl. header)", path.name, len(rows))
    return paths


def export_xlsx(storage, out_path: Path | None = None) -> Path:
    """Single 6-tab workbook - the fastest way to populate the public
    Google Sheet (Drive upload keeps tabs, values and order)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    out_path = out_path or (OUT_DIR / "GraphOne_Output.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(color="FFFFFF", bold=True)

    for tab in TAB_ORDER:
        rows = _tab_rows(tab, storage)
        columns = [pretty for pretty, _ in TAB_SCHEMA[tab]]
        sheet = workbook.create_sheet(tab)
        for row in rows:
            sheet.append(row)
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(vertical="center")
        for idx, column in enumerate(columns, start=1):
            width = min(60, max(12, len(column) + 6))
            if column in ("content", "authors", "description", "sourceUrl", "paper_url", "github_url"):
                width = 80
            sheet.column_dimensions[get_column_letter(idx)].width = width
        sheet.freeze_panes = "A2"

    workbook.save(out_path)
    log.info("workbook saved: %s", out_path)
    return out_path


def upload_sheets(storage, credentials_file: str, spreadsheet_id: str) -> bool:
    """Push all 6 tabs into the live Google Sheet via the official API."""
    if not credentials_file or not spreadsheet_id:
        log.warning("Google Sheets credentials not configured; local export only")
        return False
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:  # pragma: no cover
        log.error("gspread not installed: %s", exc)
        return False

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_file(credentials_file, scopes=scopes)
    client = gspread.authorize(credentials)
    document = client.open_by_key(spreadsheet_id)

    existing = {ws.title for ws in document.worksheets()}
    for tab in TAB_ORDER:
        columns = [pretty for pretty, _ in TAB_SCHEMA[tab]]
        rows = _tab_rows(tab, storage)
        if tab in existing:
            worksheet = document.worksheet(tab)
            worksheet.clear()
        else:
            worksheet = document.add_worksheet(title=tab, rows=max(len(rows) + 10, 100),
                                               cols=len(columns) + 4)
        worksheet.update(rows, "A1")
        log.info("uploaded tab %s (%d rows)", tab, len(rows))
    return True
