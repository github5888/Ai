# GraphOne Intelligence Pipeline

An automated, fault-tolerant **data pipeline for the global AI Intelligence Graph** (FrontierAtlas / GraphOne assignment).
It collects AI-industry entities and real-time signals, structures them through a multi-tier LLM extraction engine,
resolves duplicate organisation names deterministically, and exports a **six-tab intelligence dataset** for Google Sheets.

```
collect-raw ──► ingest ──► enrich-stars ──► resolve ──► export-sheets
 (crawl)        (LLM/env)   (GitHub API)    (Phase IV)   (6 tabs)
```

| Deliverable | Where |
|---|---|
| Six-tab output (Startups, Products, Research Papers, Jobs, News, Entity Mapping Log) | `data/out/*.csv`, `data/out/GraphOne_Output.xlsx` |
| Architecture document (max 3 pages) | `architecture.pdf` |
| Source code | `src/` |
| Tests (50, all passing) | `tests/` |
| Distributed crawler worker | `.github/workflows/collect.yml` |

---

## 1. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add API keys

# Full pipeline (crawl -> ingest -> stars -> resolve -> export):
python -m src.main pipeline

# Or stage by stage:
python -m src.main collect-raw --arxiv-limit 1200 --pwc-limit 2500
python -m src.main ingest
python -m src.main enrich-stars
python -m src.main resolve
python -m src.main export-sheets --upload        # --upload needs Google credentials in .env
python -m src.main stats
```

Everything runs **without any API keys** (deterministic extraction + SQLite). Add keys to unlock
LLM extraction, faster GitHub rates, and live Google Sheets upload.

### Making the public Google Sheet (Deliverable 1)
1. Google Drive → *New → File upload* → `data/out/GraphOne_Output.xlsx` (contains exactly the 6 required tabs).
2. Open in Google Sheets → *File → Share → Anyone with the link – Viewer* → copy link → submit.
3. (Optional, automatic path) create a service account with the **Sheets API** enabled, share the sheet with its
   email as **Editor**, then set `GOOGLE_SHEETS_CREDENTIALS_FILE` + `GOOGLE_SHEETS_ID` in `.env` and run
   `python -m src.main export-sheets --upload`.

---

## 2. What each phase does (assignment mapping)

### Phase I — Massive one-time data acquisition (`src/scrapers/`)
* **Startups** (`startup_scraper.py`): the **Y Combinator directory export** (community-maintained
  mirror of the official ycombinator.com/companies directory — ~1.5k AI-tagged startups with team
  sizes), an **AI-companies directory CSV**, Wikipedia's AI-company category tree (MediaWiki API),
  curated GitHub lists, builtin.com directory, and **Wikidata SPARQL** (P1128) for employee counts.
  Employee numbers are copied verbatim from the dataset (real counts kept as ints, bands like
  "50 - 249" never faked into integers). Every record cites its own source URL.
* **Products** (`product_scraper.py`): curated AI-tool lists on GitHub (`awesome-generative-ai`,
  `ai-collection`, ...). `pricingModel` is classified **only from the source's own words**
  (`FREE / FREEMIUM / PAID / ENTERPRISE`, else null); when a tagline carries no pricing signal, one
  compliant request to the **vendor's homepage** (`probe-pricing` stage, robots.txt aware) supplies
  evidence from the vendor's own page — otherwise pricing stays null, never guessed.
  `startupName` is assigned only when the product domain or description explicitly names a known company.
* **Research papers** (`research_scraper.py`): **Hugging Face daily-papers API** (curated, ~40–80
  papers/weekday, with `githubRepo`/`githubStars`), the **Arxiv Atom API** (cs.AI/cs.CL/cs.LG,
  newest first) plus the **Papers with Code open-data dump** (576k papers, CC-BY-SA, optional
  `PWC_DUMP_ENABLED=1`). Stars are **fetched live** via GitHub GraphQL at run time — never
  hard-coded, because they are dynamic metrics.
* Designed for lakhs of records: cursors + snapshots + idempotent upserts (see `architecture.pdf` §3).

### Phase II — High-fidelity signal ingestion (`news_scraper.py`, `jobs_scraper.py`)
* 8 AI news feeds (≥5 required) and 6 job boards (≥5 required) — all official APIs/RSS channels.
* **24-hour freshness**: every timestamp is normalised to ISO-8601 UTC (`src/utils/dates.py`) —
  `"2 hours ago"`, `"Yesterday"`, `"November 21, 2026"`, RFC-822 headers, unix epochs (s **and** ms).
* Sources that publish no date get the documented `heuristic:first_seen` treatment (never a fabricated date).
* Full article text is extracted for fresh news (bounded, robots-aware).

### Phase III — Multi-tier LLM extraction engine (`src/llm/`)
* **Fallback chain**: `Gemini Flash → Groq Llama 3 → DeepSeek → deterministic extractor`.
* **429 handling**: exponential backoff `2s → 4s → 8s → 16s` (cap 60s) **with ±25% jitter**, `Retry-After`
  honoured, per-tier attempt budget, then graceful failover.
* **413 handling**: *semantic* chunking (headings → paragraphs → sentences, ~3k tokens per chunk, paragraph
  overlap) with field-aware merging (lists union, numerics max); chunk size halves up to 3× on repeated 413s.
* Every envelope is pydantic-validated against the assignment schemas (`schemaVersion`, `recordType`,
  `source{name,url}`, `content{...}`, `collectedAt`).
* **No keys? No problem:** the deterministic extractor runs instead and records are tagged
  `extractionMode: deterministic`. Nothing is invented — unstructured fields stay null.

### Phase IV — Deterministic entity resolution (`src/resolver/entity_resolution.py`)
* 50 canonical seed AI startups + aliases; normalisation strips punctuation/legal suffixes (`Inc`, `LLC`, ...).
* Guardrailed RapidFuzz matching: high similarity **and** shared tokens **and** no distinctive extra words —
  `"OpenAI"`, `"Open AI"`, `"OpenAI, Inc."` → `OpenAI`; but `"Cohere Health"` stays distinct from `"Cohere"`.
* Ambiguous scores are flagged `AMBIGUOUS` for review instead of merged; new names are registered and reused.
* Full audit log (raw vs canonical, score, method) → Sheet tab **06 Entity Mapping Log**.

### Phase V — Anti-bot & scale thinking (`src/fetcher.py`)
* asyncio + aiohttp crawler: global semaphore, per-host minimum interval, exponential-backoff retries,
  robots.txt awareness, structured failure logging.
* Playwright upgrade path for JS-rendered high-value sources (graceful error when unavailable).
* Compliant strategy: official APIs/feeds first; blockers are skipped with logged fallbacks — **no CAPTCHA bypassing**.
* The GitHub Actions workflow (`collect.yml`) is a live example of a crawler worker node.

### Phase VI — Architecture & production design
See **`architecture.pdf`** (3 pages): 500k-record scale-out, exact 429/413 mechanics, exactly-once processing
across distributed nodes, and justified storage choices (PostgreSQL, Redis, Neo4j, pgvector, Sheets API).

---

## 3. Repository layout

```
├── README.md
├── architecture.pdf
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py                  # env-driven settings
│   ├── fetcher.py                 # async crawler + robots + Playwright path
│   ├── main.py                    # CLI stages
│   ├── scrapers/                  # startup / product / research / news / jobs
│   ├── llm/                       # orchestrator, providers, chunking, retry, schemas
│   ├── resolver/entity_resolution.py
│   ├── storage/                   # SQLAlchemy models + Google Sheets / CSV / XLSX export
│   └── utils/                     # dates, text/URL normalisation, hashing
├── scripts/make_architecture_pdf.py
├── tests/                         # 50 unit tests (no network needed)
├── data/raw/                      # raw crawl snapshots (provenance for every record)
├── data/out/                      # final 6 tabs + mapping log + run stats
└── .github/workflows/collect.yml  # crawler worker on GitHub Actions
```

## 4. Configuration (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/graphone.db` or PostgreSQL URL | SQLite |
| `GITHUB_TOKEN` | GitHub API for star metrics (60→5000 req/hr) | unauthenticated |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `DEEPSEEK_API_KEY` | LLM fallback chain | deterministic mode |
| `GOOGLE_SHEETS_CREDENTIALS_FILE` / `GOOGLE_SHEETS_ID` | live Sheets upload | CSV/XLSX only |
| `CONCURRENCY`, `REQUEST_MIN_INTERVAL` | crawler politeness | 8 / 0.25 s |
| `FRESHNESS_HOURS` | news/jobs window | 24 |
| `LLM_MAX_ATTEMPTS`, `LLM_BACKOFF_*`, `LLM_CHUNK_CHARS` | retry/chunk policy | 5 / 2.0× / 12000 |

## 5. Data fidelity policy (non-negotiable)

* Every record carries `source.name` + `source.url` and a raw snapshot under `data/raw/`.
* **Nothing is invented**: unknown employee counts, unknown pricing, unknown repos and unknown dates stay
  `null`/empty and are labelled. GitHub stars are always read live from the API at run time.
* Idempotent ingestion: normalised-URL + content hashes + DB unique constraints — the same article collected
  by three workers is stored once.

## 6. Tests

```bash
pytest -q          # 50 passed: dates, freshness, hashes, parsers, pricing, role families,
                   # chunking, JSON recovery, retry/backoff/jitter, fallback chain, entity resolution
```

## 7. Known limitations (honest engineering)

* The Papers-with-Code dataset mirror is frozen at its 2025-07 snapshot (Meta sunset the site); the pipeline
  compensates by fetching **current** stars from GitHub at run time and mixing in fresh Arxiv API results.
* Cloudflare/Datadome-protected directories are skipped (documented graceful fallback), which is why some
  `employeeCount` values are null — guessing is worse than absence.
* Sandbox/dev runs collect the target-scale slice (3,000+ core records); the architecture doc describes the
  queue+workers path to 500k without code changes.
