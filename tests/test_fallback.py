"""Tests for weft.fallback — graceful degradation when DB is unavailable."""

from __future__ import annotations

from pathlib import Path

import pytest

from weft.fallback import _parse_sections, read_fallback, search_fallback

# ---------------------------------------------------------------------------
# Sample exported markdown matching the format produced by weft.exporter
# ---------------------------------------------------------------------------

SAMPLE_EXPORT = """\
# Weft Memory Export

## Topic: infrastructure

### [fact] Redis is used for caching (confidence: 0.9, accessed: 3 times)
Redis is used for caching in the Weft memory system.

### [fact] Postgres stores all memory records (confidence: 0.85, accessed: 5 times)
Postgres with pgvector stores all memory records and embeddings.

## Topic: preferences

### [preference] Always use dark mode (confidence: 1.0, accessed: 1 times)
Always use dark mode in the editor.

## Topic: testing

### [solution] Testcontainers reaper fix (confidence: 0.8, accessed: 2 times)
Clean up stale Ryuk reaper containers in conftest before starting new ones.

## Uncategorized

### [fact] Orphan memory without topics (confidence: 0.5, accessed: 0 times)
Orphan memory without topics that has no topic tags.
"""


# ---------------------------------------------------------------------------
# read_fallback
# ---------------------------------------------------------------------------


def test_read_fallback_missing_file(tmp_path: Path):
    """read_fallback returns empty string for a non-existent path."""
    result = read_fallback(tmp_path / "does_not_exist.md")
    assert result == ""


def test_read_fallback_existing_file(tmp_path: Path):
    """read_fallback returns file contents when the file exists."""
    p = tmp_path / "export.md"
    p.write_text(SAMPLE_EXPORT, encoding="utf-8")

    result = read_fallback(p)
    assert result == SAMPLE_EXPORT


def test_read_fallback_empty_file(tmp_path: Path):
    """read_fallback returns empty string for an empty file."""
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")

    result = read_fallback(p)
    assert result == ""


# ---------------------------------------------------------------------------
# search_fallback
# ---------------------------------------------------------------------------


def test_search_fallback_no_file(tmp_path: Path):
    """search_fallback returns empty list when the file does not exist."""
    result = search_fallback("redis", path=tmp_path / "missing.md")
    assert result == []


def test_search_fallback_keyword_match(tmp_path: Path):
    """search_fallback finds entries matching a keyword."""
    p = tmp_path / "export.md"
    p.write_text(SAMPLE_EXPORT, encoding="utf-8")

    results = search_fallback("reaper", path=p)
    assert len(results) >= 1
    # The reaper fix entry should be in the results
    titles = [r["title"] for r in results]
    assert any("reaper" in t.lower() or "Testcontainers" in t for t in titles)


def test_search_fallback_no_match(tmp_path: Path):
    """search_fallback returns empty list when no entries match."""
    p = tmp_path / "export.md"
    p.write_text(SAMPLE_EXPORT, encoding="utf-8")

    results = search_fallback("zyxwvutsrqp", path=p)
    assert results == []


def test_search_fallback_limit(tmp_path: Path):
    """search_fallback respects the limit parameter."""
    p = tmp_path / "export.md"
    p.write_text(SAMPLE_EXPORT, encoding="utf-8")

    # "memory" appears in several entries — limit to 1
    results = search_fallback("memory", path=p, limit=1)
    assert len(results) == 1


def test_search_fallback_ranked_by_hits(tmp_path: Path):
    """search_fallback ranks results by keyword hit count descending."""
    # Build markdown with entries that have different numbers of keyword matches.
    md = """\
# Weft Memory Export

## Topic: animals

### [fact] Cats are great (confidence: 0.9, accessed: 1 times)
Cats are wonderful pets. Cats love naps.

### [fact] Dogs are loyal (confidence: 0.8, accessed: 2 times)
Dogs are loyal animals. Dogs cats dogs.

### [fact] Fish swim in water (confidence: 0.7, accessed: 0 times)
Fish have fins and gills.
"""
    p = tmp_path / "export.md"
    p.write_text(md, encoding="utf-8")

    # Search for "cats" — second entry has "cats" once in content + "dogs" entries
    # but first entry has "Cats" twice in content + once in title
    results = search_fallback("cats", path=p)

    # Filter to entries that matched
    assert len(results) >= 2

    # The entry with MORE occurrences of "cats" should be first
    assert results[0]["hits"] >= results[1]["hits"]

    # Verify first result is the one with more hits (Cats entry has more "cats")
    assert "Cats" in results[0]["title"] or "cats" in results[0]["content"].lower()


# ---------------------------------------------------------------------------
# _parse_sections — verify it matches exporter format
# ---------------------------------------------------------------------------


def test_parse_sections_matches_export_format():
    """_parse_sections correctly extracts type, confidence, topic, content
    from markdown produced by weft.exporter._format_memory_md."""
    sections = _parse_sections(SAMPLE_EXPORT)

    assert len(sections) == 5

    # Check the first section (Redis caching fact)
    redis_section = sections[0]
    assert redis_section["type"] == "fact"
    assert "Redis" in redis_section["title"]
    assert redis_section["confidence"] == 0.9
    assert redis_section["access_count"] == 3
    assert redis_section["topic"] == "infrastructure"
    assert "caching" in redis_section["content"].lower()

    # Check the preference section
    pref_sections = [s for s in sections if s["type"] == "preference"]
    assert len(pref_sections) == 1
    assert pref_sections[0]["confidence"] == 1.0
    assert pref_sections[0]["topic"] == "preferences"
    assert "dark mode" in pref_sections[0]["content"].lower()

    # Check the solution section
    sol_sections = [s for s in sections if s["type"] == "solution"]
    assert len(sol_sections) == 1
    assert sol_sections[0]["confidence"] == 0.8
    assert sol_sections[0]["topic"] == "testing"

    # Check uncategorized section (topic should be empty string)
    orphan = [s for s in sections if "Orphan" in s.get("title", "")]
    assert len(orphan) == 1
    assert orphan[0]["topic"] == ""
    assert orphan[0]["confidence"] == 0.5
    assert orphan[0]["access_count"] == 0
