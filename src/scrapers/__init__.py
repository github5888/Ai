"""Scraper package: one module per entity vertical (Phase I + II).

Every scraper separates *fetching* (async, network) from *parsing* (pure
functions), so parsers are unit-testable without network access and the
same code path runs locally, in GitHub Actions workers, or on any worker fleet.
"""
