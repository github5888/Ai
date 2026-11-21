"""Deliverable 1 exporters: Google Sheets (6 tabs) + local CSV/XLSX fallback.

* ``export_csv``   - one CSV per tab (importable into Google Sheets).
* ``export_xlsx``  - a single workbook with exactly the 6 required tabs,
  ready to upload to Google Drive ("File > Import" preserves all tabs).
* ``upload_sheets``- writes directly into an existing Google Sheet via the
  official Sheets API (gspread + service account) when credentials exist.

Tab layout (exactly as the assignment specifies):
    01 Startups | 02 Products | 03 Research Papers | 04 Jobs | 05 News |
    06 Entity Mapping Log
"""
from __future__ import annotations

import csv
import logging
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

SHEET_COLUMNS = {
    TAB_STARTUPS: ["entityName", "employeeCount", "canonicalName", "sourceName", "sourceUrl", "collectedAt"],
    TAB_PRODUCTS: ["productName", "startupName", "canonicalStartup", "pricingModel", "sourceName", "sourceUrl", "collectedAt"],
    TAB_PAPERS: ["title", "authors", "paper_url", "github_url", "github_stars", "published_date", "sourceName", "sourceUrl", "collectedAt"],
    TAB_JOBS: ["company", "title", "published_at", "is_remote", "role_family", "location", "date_source", "sourceName", "sourceUrl"],
    TAB_NEWS: ["headline", "published_at", "date_source", "content", "sourceName", "sourceUrl"],
    TAB_MAPPING: ["raw_name", "canonical_name", "match_status", "score", "method", "entity_type", "source_url", "resolved_at"],
}


def _rows_for(model, columns: list[str], storage) -> list[list]:
    records = storage.load_table(model)
    rows: list[list] = [columns]
    for record in records:
        row = []
        for column in columns:
            value = record.get(column)
            if column == "authors":
                import json

                try:
                    value = ", ".join(json.loads(record.get("authors_json") or "[]"))
                except (ValueError, TypeError):
                    value = ""
            if isinstance(value, bool):
                value = "TRUE" if value else "FALSE"
            if value is None:
                value = ""
            if isinstance(value, str):
                value = value[:49000]  # Google Sheets cell limit safety
            row.append(value)
        rows.append(row)
    return rows


def export_csv(storage, out_dir: Path = OUT_DIR) -> list[Path]:
    """Write the 6 tab CSVs (numbers preserved for Startups/Products/Papers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tab_map = [
        ("01_startups.csv", TAB_STARTUPS, Startup),
        ("02_products.csv", TAB_PRODUCTS, Product),
        ("03_research_papers.csv", TAB_PAPERS, ResearchPaper),
        ("04_jobs.csv", TAB_JOBS, JobPost),
        ("05_news.csv", TAB_NEWS, NewsArticle),
        ("06_entity_mapping_log.csv", TAB_MAPPING, None),
    ]
    for filename, tab, model in tab_map:
        columns = SHEET_COLUMNS[tab]
        if model is None:
            from .database import EntityMapping

            records = storage.load_table(EntityMapping)
            rows = [columns]
            for record in records:
                rows.append([record.get(c, "") or "" for c in columns])
        else:
            rows = _rows_for(model, columns, storage)
        path = out_dir / filename
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        paths.append(path)
        log.info("exported %s (%d rows incl. header)", path.name, len(rows))
    return paths


def export_xlsx(storage, out_path: Path | None = None) -> Path:
    """Single 6-tab workbook - the fastest way to populate the public
    Google Sheet (Drive upload keeps tabs, formatting and values)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    out_path = out_path or (OUT_DIR / "GraphOne_Output.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(color="FFFFFF", bold=True)

    tab_map = [
        (TAB_STARTUPS, Startup),
        (TAB_PRODUCTS, Product),
        (TAB_PAPERS, ResearchPaper),
        (TAB_JOBS, JobPost),
        (TAB_NEWS, NewsArticle),
        (TAB_MAPPING, None),
    ]
    for tab, model in tab_map:
        columns = SHEET_COLUMNS[tab]
        if model is None:
            from .database import EntityMapping

            records = storage.load_table(EntityMapping)
            rows = [columns] + [[record.get(c, "") or "" for c in columns] for record in records]
        else:
            rows = _rows_for(model, columns, storage)
        sheet = workbook.create_sheet(tab)
        for row in rows:
            sheet.append(row)
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(vertical="center")
        for idx, column in enumerate(columns, start=1):
            width = min(60, max(12, len(column) + 6))
            sheet.column_dimensions[get_column_letter(idx)].width = width
            if column in ("content", "authors", "description"):
                sheet.column_dimensions[get_column_letter(idx)].width = 80
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
    tab_map = [
        (TAB_STARTUPS, Startup),
        (TAB_PRODUCTS, Product),
        (TAB_PAPERS, ResearchPaper),
        (TAB_JOBS, JobPost),
        (TAB_NEWS, NewsArticle),
        (TAB_MAPPING, None),
    ]
    for tab, model in tab_map:
        columns = SHEET_COLUMNS[tab]
        if model is None:
            from .database import EntityMapping

            records = storage.load_table(EntityMapping)
            rows = [columns] + [[record.get(c, "") or "" for c in columns] for record in records]
        else:
            rows = _rows_for(model, columns, storage)
        if tab in existing:
            worksheet = document.worksheet(tab)
            worksheet.clear()
        else:
            worksheet = document.add_worksheet(title=tab, rows=max(len(rows) + 10, 100), cols=len(columns) + 4)
        worksheet.update(rows, "A1")
        log.info("uploaded tab %s (%d rows)", tab, len(rows))
    return True
