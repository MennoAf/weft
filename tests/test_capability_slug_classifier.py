"""Tests for Capability Registry slug classification."""

from __future__ import annotations

from capability_registry.models import CapabilityEntry
from capability_registry.slug_classifier import (
    KEYWORD_TO_SLUG,
    classify_entries,
    classify_entry,
    infer_slugs,
)


def test_keyword_map_contains_required_entries() -> None:
    assert KEYWORD_TO_SLUG["bot"] == "bot-block-hardening"
    assert KEYWORD_TO_SLUG["argparse"] == "cli-tool"


def test_infer_slugs_uses_name_docstring_and_imports() -> None:
    slugs = infer_slugs(
        "LazyEscalationPolicy",
        "Bot block detection with retry backoff.",
        ["requests", "argparse"],
    )

    assert slugs == [
        "bot-block-hardening",
        "cli-tool",
        "crawler-escalation",
        "retry-logic",
    ]


def test_infer_slugs_is_case_insensitive_deduped_and_sorted() -> None:
    slugs = infer_slugs(
        "CaptchaBlocker",
        "CAPTCHA block bot handling.",
        ["tokenizer", "oauth"],
    )

    assert slugs == [
        "authentication",
        "bot-block-hardening",
        "oauth",
        "token-management",
    ]


def test_infer_slugs_handles_none_docstring_and_empty_imports() -> None:
    assert infer_slugs("schema_migration", None, []) == [
        "db-migration",
        "db-schema",
    ]


def test_classify_entry_uses_file_path_for_file_level_entries() -> None:
    entry = CapabilityEntry(
        repo_slug="weft",
        file_path="tools/cache_digest.py",
    )

    classified = classify_entry(entry)

    assert classified is entry
    assert classified.capability_slugs == ["caching", "digest-cache"]


def test_classify_entry_preserves_manual_capability_slugs() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/escalation.py",
        symbol_name="LazyEscalationPolicy",
        docstring="bot block retry logic",
        capability_slugs=["manual-tag"],
    )

    assert classify_entry(entry).capability_slugs == ["manual-tag"]


def test_classify_entries_classifies_each_entry() -> None:
    entries = [
        CapabilityEntry(repo_slug="weft", file_path="recall.py"),
        CapabilityEntry(repo_slug="weft", file_path="cli.py", imports=["argparse"]),
    ]

    classified = classify_entries(entries)

    assert classified == entries
    assert classified[0].capability_slugs == ["memory-recall"]
    assert classified[1].capability_slugs == ["cli-tool"]
