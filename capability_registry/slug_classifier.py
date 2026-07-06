"""Deterministic capability slug inference for registry entries."""

from __future__ import annotations

from capability_registry.models import CapabilityEntry


KEYWORD_TO_SLUG: dict[str, str] = {
    "bot": "bot-block-hardening",
    "block": "bot-block-hardening",
    "captcha": "bot-block-hardening",
    "escalat": "crawler-escalation",
    "crawl": "crawler",
    "scrape": "scraper",
    "spider": "scraper",
    "retry": "retry-logic",
    "backoff": "retry-logic",
    "rate_limit": "rate-limiting",
    "ratelimit": "rate-limiting",
    "auth": "authentication",
    "oauth": "oauth",
    "token": "token-management",
    "cache": "caching",
    "digest": "digest-cache",
    "embed": "embedding",
    "vector": "vector-search",
    "recall": "memory-recall",
    "memory": "memory-recall",
    "migration": "db-migration",
    "schema": "db-schema",
    "cli": "cli-tool",
    "argparse": "cli-tool",
}


def infer_slugs(entry_name: str, docstring: str | None, imports: list[str]) -> list[str]:
    """Infer sorted capability slugs from entry text using substring rules."""
    text_blob = " ".join(
        [
            entry_name,
            docstring or "",
            " ".join(imports or []),
        ]
    ).lower()
    slugs = {
        slug
        for keyword, slug in KEYWORD_TO_SLUG.items()
        if keyword in text_blob
    }
    return sorted(slugs)


def classify_entry(entry: CapabilityEntry) -> CapabilityEntry:
    """Populate capability slugs on ``entry`` when it has no manual slugs."""
    if entry.capability_slugs:
        return entry

    entry_name = entry.symbol_name or entry.file_path
    entry.capability_slugs = infer_slugs(
        entry_name,
        entry.docstring,
        entry.imports or [],
    )
    return entry


def classify_entries(entries: list[CapabilityEntry]) -> list[CapabilityEntry]:
    """Classify each entry and return the same list."""
    for entry in entries:
        classify_entry(entry)
    return entries
