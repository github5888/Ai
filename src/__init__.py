"""GraphOne Intelligence Pipeline.

An automated, fault-tolerant data pipeline that collects AI-industry
entities (startups, products, research papers) and real-time signals
(news, jobs), normalizes them through a multi-tier LLM extraction engine,
resolves duplicate entity names deterministically, and exports a
six-tab intelligence dataset for Google Sheets.

Phases:
    I    Massive one-time data acquisition      (scrapers)
    II   High-fidelity signal ingestion         (news + jobs, 24h freshness)
    III  Multi-tier LLM extraction engine       (llm/)
    IV   Deterministic entity resolution        (resolver/)
    V    Anti-bot & asynchronous scale design   (fetcher.py)
    VI   Architecture & production design       (architecture.pdf)
"""

__version__ = "1.0.0"
