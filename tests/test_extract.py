"""Tests for weft.extract — memory candidate extraction from text."""

from __future__ import annotations

from weft.extract import extract_candidates


def test_extract_preference():
    """Detects user preference statements."""
    text = "I prefer dark mode for all editors"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "preference"
    assert results[0]["confidence"] >= 0.7


def test_extract_fact():
    """Detects factual statements about project."""
    text = "The project uses PostgreSQL 16 with pgvector"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "fact"
    assert results[0]["confidence"] >= 0.6


def test_extract_pattern():
    """Detects pattern/convention statements."""
    text = "We typically run integration tests before deploying"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "pattern"


def test_extract_architecture():
    """Detects architecture statements."""
    text = "The architecture follows a microservices pattern with event sourcing"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "architecture"


def test_extract_multiple_lines():
    """Extracts candidates from multiple lines."""
    text = """I prefer concise responses
The project uses FastAPI for the backend
We typically deploy on Fridays
The architecture follows hexagonal design"""
    results = extract_candidates(text)
    types = {r["type"] for r in results}
    assert "preference" in types
    assert "fact" in types


def test_extract_min_confidence_filter():
    """min_confidence filters out low-confidence candidates."""
    text = "I prefer dark mode\nWe typically use logging"
    all_results = extract_candidates(text, min_confidence=0.0)
    high_results = extract_candidates(text, min_confidence=0.9)
    assert len(high_results) <= len(all_results)


def test_extract_empty_input():
    """Empty or whitespace input returns empty list."""
    assert extract_candidates("") == []
    assert extract_candidates("   ") == []
    assert extract_candidates("\n\n") == []


def test_extract_short_lines_ignored():
    """Lines shorter than 10 characters are skipped."""
    text = "hi\nok\nyes"
    assert extract_candidates(text) == []


def test_extract_no_match():
    """Text with no recognizable patterns returns empty."""
    text = "The quick brown fox jumps over the lazy dog"
    results = extract_candidates(text)
    assert results == []


def test_extract_deduplicates():
    """Same content appearing twice only yields one candidate."""
    text = "I prefer dark mode\nI prefer dark mode"
    results = extract_candidates(text)
    assert len(results) == 1


def test_extract_topics_populated():
    """Extracted candidates include topic tags from known tech terms."""
    text = "The project uses PostgreSQL and Redis for caching"
    results = extract_candidates(text)
    assert len(results) >= 1
    topics = results[0].get("topic", [])
    assert any(t in topics for t in ["postgres", "postgresql", "redis", "cache"])


def test_extract_has_source_line():
    """Each candidate includes the original source_line."""
    text = "I always use type hints in Python code"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert "source_line" in results[0]
    assert results[0]["source_line"] == text
