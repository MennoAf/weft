"""Tests for the MEMORY.md importer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from weft.importer import (
    ParseResult,
    _clean_content,
    _extract_topics,
    _infer_type,
    parse_memory_md,
    parse_memory_md_text,
)
from weft.models import MemoryCreate, MemorySource, MemoryType


# ---------------------------------------------------------------------------
# Sample markdown fixtures
# ---------------------------------------------------------------------------

SAMPLE_MEMORY_MD = """\
## Preferences
- Always use bun instead of npm for package management
- Prefer sonnet for subagent tasks to minimize cost

## Architecture
- Loom uses a DAG-based task dependency system with topological ordering
- The MCP server runs on stdio transport and is registered in .mcp.json

## Known Patterns
- Three-layer config (global YAML, project YAML, env vars) provides good flexibility
- MCP tools should be thin coordinators with business logic in separate modules

## Solutions and Fixes
- The Ryuk reaper container from testcontainers can become stale — clean it up in conftest.py
- Float precision issues with Postgres REAL columns handled by pytest.approx

## Relationships and Ownership
- Jason Bauman owns Weft, Loom, and Muttr projects

## Facts
- Loom has 760 tests across unit, integration, and e2e suites
- The Loom project was started in January 2026

## Empty Section
"""

MINIMAL_MD = """\
## Debug Notes
Fixed the connection pool leak by closing cursors explicitly.
"""

NO_HEADERS_MD = """\
This is just plain text with no markdown headers at all.
It should still be parsed as a single fact.
"""


# ---------------------------------------------------------------------------
# Type inference tests
# ---------------------------------------------------------------------------


class TestInferType:
    def test_preference_keywords(self):
        assert _infer_type("Preferences") == MemoryType.preference
        assert _infer_type("Things I Always Do") == MemoryType.preference
        assert _infer_type("Never do this") == MemoryType.preference
        assert _infer_type("I Prefer This") == MemoryType.preference

    def test_pattern_keywords(self):
        assert _infer_type("Known Patterns") == MemoryType.pattern
        assert _infer_type("Lessons Learned") == MemoryType.pattern

    def test_architecture_keywords(self):
        assert _infer_type("Architecture") == MemoryType.architecture
        assert _infer_type("Project Structure") == MemoryType.architecture
        assert _infer_type("Arch Notes") == MemoryType.architecture

    def test_solution_keywords(self):
        assert _infer_type("Solutions and Fixes") == MemoryType.solution
        assert _infer_type("Debug Notes") == MemoryType.solution
        assert _infer_type("Workaround for Bug") == MemoryType.solution

    def test_relationship_keywords(self):
        assert _infer_type("Relationships and Ownership") == MemoryType.relationship
        assert _infer_type("Who Does What") == MemoryType.relationship
        assert _infer_type("Owner Info") == MemoryType.relationship

    def test_default_to_fact(self):
        assert _infer_type("Facts") == MemoryType.fact
        assert _infer_type("Random Notes") == MemoryType.fact
        assert _infer_type("Miscellaneous") == MemoryType.fact


# ---------------------------------------------------------------------------
# Topic extraction tests
# ---------------------------------------------------------------------------


class TestExtractTopics:
    def test_basic_extraction(self):
        topics = _extract_topics("Architecture Notes")
        assert "architecture" in topics
        assert "notes" in topics

    def test_stop_words_removed(self):
        topics = _extract_topics("The Best Patterns for the Project")
        assert "the" not in topics
        assert "for" not in topics
        assert "best" in topics
        assert "patterns" in topics
        assert "project" in topics

    def test_short_words_removed(self):
        topics = _extract_topics("A Quick Fix")
        assert "a" not in topics  # stop word
        # single-char words removed

    def test_punctuation_removed(self):
        topics = _extract_topics("Solutions & Fixes!")
        assert "solutions" in topics
        assert "fixes" in topics
        assert "&" not in topics

    def test_empty_header(self):
        assert _extract_topics("") == []


# ---------------------------------------------------------------------------
# Content cleaning tests
# ---------------------------------------------------------------------------


class TestCleanContent:
    def test_header_and_body(self):
        result = _clean_content("My Header", "Some body text\nwith lines")
        assert result == "My Header\n\nSome body text\nwith lines"

    def test_empty_body(self):
        result = _clean_content("Header Only", "")
        assert result == "Header Only"

    def test_excessive_whitespace_collapsed(self):
        body = "Line one\n\n\n\nLine two\n\n\nLine three"
        result = _clean_content("Title", body)
        assert "\n\n\n" not in result
        assert "Line one\n\nLine two\n\nLine three" in result

    def test_trailing_whitespace_stripped(self):
        body = "Content here   \nMore content   "
        result = _clean_content("Title", body)
        assert "   " not in result


# ---------------------------------------------------------------------------
# Full parser tests (from text)
# ---------------------------------------------------------------------------


class TestParseMemoryMdText:
    def test_sample_parse(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        assert isinstance(result, ParseResult)
        # 6 sections with content + 1 empty section
        assert len(result.memories) == 6
        assert result.skipped == 1  # "Empty Section"

    def test_all_memories_are_memorycreate(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        for m in result.memories:
            assert isinstance(m, MemoryCreate)

    def test_source_is_documentation(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        for m in result.memories:
            assert m.source == MemorySource.documentation

    def test_type_inference_correct(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        types = [m.type for m in result.memories]
        assert types[0] == MemoryType.preference  # "Preferences"
        assert types[1] == MemoryType.architecture  # "Architecture"
        assert types[2] == MemoryType.pattern  # "Known Patterns"
        assert types[3] == MemoryType.solution  # "Solutions and Fixes"
        assert types[4] == MemoryType.relationship  # "Relationships and Ownership"
        assert types[5] == MemoryType.fact  # "Facts"

    def test_confidence_by_type(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        confidence_map = {m.type: m.confidence for m in result.memories}
        assert confidence_map[MemoryType.preference] == 1.0
        assert confidence_map[MemoryType.architecture] == 0.8
        assert confidence_map[MemoryType.pattern] == 0.7
        assert confidence_map[MemoryType.solution] == 0.8
        assert confidence_map[MemoryType.relationship] == 0.9
        assert confidence_map[MemoryType.fact] == 0.7

    def test_topics_extracted(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        # "Preferences" → ["preferences"]
        assert "preferences" in result.memories[0].topic
        # "Architecture" → ["architecture"]
        assert "architecture" in result.memories[1].topic
        # "Known Patterns" → ["known", "patterns"]
        assert "known" in result.memories[2].topic
        assert "patterns" in result.memories[2].topic

    def test_content_includes_header_and_body(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        pref = result.memories[0]
        assert pref.content.startswith("Preferences")
        assert "Always use bun" in pref.content
        assert "Prefer sonnet" in pref.content

    def test_empty_sections_skipped(self):
        result = parse_memory_md_text(SAMPLE_MEMORY_MD)
        # The "Empty Section" should be skipped
        assert result.skipped == 1
        contents = [m.content for m in result.memories]
        assert not any("Empty Section" in c for c in contents)

    def test_minimal_markdown(self):
        result = parse_memory_md_text(MINIMAL_MD)
        assert len(result.memories) == 1
        assert result.skipped == 0
        m = result.memories[0]
        assert m.type == MemoryType.solution  # "Debug" keyword
        assert "connection pool leak" in m.content

    def test_no_headers(self):
        result = parse_memory_md_text(NO_HEADERS_MD)
        assert len(result.memories) == 1
        assert result.skipped == 0
        m = result.memories[0]
        assert m.type == MemoryType.fact
        assert "plain text" in m.content

    def test_empty_text(self):
        result = parse_memory_md_text("")
        assert len(result.memories) == 0
        assert result.skipped == 0

    def test_whitespace_only(self):
        result = parse_memory_md_text("   \n\n   \n")
        assert len(result.memories) == 0
        assert result.skipped == 0

    def test_h3_headers_supported(self):
        md = "### Subsection Preferences\nAlways prefer type hints.\n"
        result = parse_memory_md_text(md)
        assert len(result.memories) == 1
        assert result.memories[0].type == MemoryType.preference

    def test_mixed_h2_h3(self):
        md = """\
## Architecture
Top-level arch notes.

### Structure Details
Detailed structure info here.
"""
        result = parse_memory_md_text(md)
        assert len(result.memories) == 2
        # Both should be architecture type
        assert result.memories[0].type == MemoryType.architecture
        assert result.memories[1].type == MemoryType.architecture


# ---------------------------------------------------------------------------
# File-based parser tests
# ---------------------------------------------------------------------------


class TestParseMemoryMdFile:
    def test_parse_from_file(self, tmp_path: Path):
        md_file = tmp_path / "MEMORY.md"
        md_file.write_text(SAMPLE_MEMORY_MD, encoding="utf-8")

        result = parse_memory_md(md_file)
        assert len(result.memories) == 6
        assert result.skipped == 1

    def test_parse_from_string_path(self, tmp_path: Path):
        md_file = tmp_path / "test_memory.md"
        md_file.write_text(MINIMAL_MD, encoding="utf-8")

        result = parse_memory_md(str(md_file))
        assert len(result.memories) == 1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_memory_md("/nonexistent/path/MEMORY.md")


# ---------------------------------------------------------------------------
# Smoke test (matches the task description)
# ---------------------------------------------------------------------------


def test_smoke_importer(tmp_path: Path):
    """Smoke test matching the task's inline verification script."""
    md_content = """\
## Preferences
- Always use uv instead of pip
- Never commit .env files

## Architecture Overview
- The system uses a three-tier architecture: MCP server, storage layer, embedding provider

## Debug Workaround
- Restart the dev server if hot-reload fails after changing __init__.py

## Project Owner
- Jason Bauman is the owner and primary developer

## Empty Notes
"""
    md_file = tmp_path / "MEMORY.md"
    md_file.write_text(md_content, encoding="utf-8")

    result = parse_memory_md(md_file)

    assert len(result.memories) == 4
    assert result.skipped == 1

    # Verify types
    types = [m.type for m in result.memories]
    assert types[0] == MemoryType.preference
    assert types[1] == MemoryType.architecture
    assert types[2] == MemoryType.solution  # "workaround" and "debug"
    assert types[3] == MemoryType.relationship  # "owner"

    # Verify confidences
    assert result.memories[0].confidence == 1.0
    assert result.memories[1].confidence == 0.8
    assert result.memories[2].confidence == 0.8
    assert result.memories[3].confidence == 0.9

    # Verify source
    for m in result.memories:
        assert m.source == MemorySource.documentation

    # Verify topics are non-empty
    for m in result.memories:
        assert len(m.topic) > 0

    # Print for visual verification
    print(f"Parsed {len(result.memories)} memories, skipped {result.skipped}")
    for m in result.memories:
        print(f"  [{m.type.value}] conf={m.confidence} topics={m.topic}")
