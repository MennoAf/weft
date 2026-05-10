#!/usr/bin/env python3
"""
test_turn_recall.py — Unit tests for the _answer_text_match helper.

Tests the heuristic gold-answer text-match used for turn-level recall@k
instrumentation (loom-f658cd55). The helper lives in adapter.py and is a
pure function — no DB, no embedder, no network.

Author:  Jason Bauman
Python:  >= 3.12
"""

from __future__ import annotations

import pytest

from benchmarks.longmemeval.adapter import _answer_text_match


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_positive_exact_match() -> None:
    """Gold answer appears verbatim as a substring of the turn content."""
    assert _answer_text_match("Paris", "I went to Paris last week") is True


def test_positive_case_insensitive() -> None:
    """Match is case-insensitive on both sides."""
    assert _answer_text_match("paris", "I went to PARIS last week") is True


def test_positive_gold_uppercased() -> None:
    """Gold answer can be uppercased — both sides are lowercased before comparison."""
    assert _answer_text_match("PARIS", "I went to Paris last week") is True


def test_positive_short_word_boundary_match() -> None:
    """Short gold answers (≤3 chars) use word-boundary matching.

    'no' should match a turn that contains the word 'no' as a standalone token.
    """
    assert _answer_text_match("no", "No, I haven't been there") is True


def test_positive_whitespace_stripped_from_gold() -> None:
    """Leading/trailing whitespace in the gold answer is stripped before matching."""
    assert _answer_text_match("  Paris  ", "I went to Paris") is True


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_negative_no_match() -> None:
    """Gold answer does not appear in the turn content."""
    assert _answer_text_match("Tokyo", "I went to Paris last week") is False


def test_negative_short_word_boundary_blocks_substring() -> None:
    """Short gold 'no' must NOT match 'north' — word-boundary guard required."""
    assert _answer_text_match("no", "heading north on the highway") is False


def test_negative_empty_gold() -> None:
    """An empty gold answer (after stripping) always returns False."""
    assert _answer_text_match("", "some content here") is False
    assert _answer_text_match("   ", "some content here") is False


def test_negative_gold_longer_than_content() -> None:
    """Gold answer longer than the turn content cannot be a substring."""
    assert _answer_text_match("a very long gold answer string", "short") is False


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


def test_boundary_exactly_three_chars_uses_word_boundary() -> None:
    """A gold answer exactly 3 characters long uses word-boundary matching."""
    # 'yes' as a standalone word — should match
    assert _answer_text_match("yes", "Yes, that's correct") is True
    # 'yes' inside 'yesterday' — should NOT match (word boundary blocks it)
    assert _answer_text_match("yes", "yesterday was great") is False


def test_boundary_four_chars_uses_substring() -> None:
    """A gold answer of 4 characters uses plain substring matching (no boundary)."""
    # 'arts' inside 'arts and crafts' — should match
    assert _answer_text_match("arts", "he studies arts and crafts") is True


def test_positive_match_mid_sentence() -> None:
    """Gold answer embedded mid-sentence is found via substring match."""
    assert _answer_text_match("rock climbing", "Rock climbing is going great.") is True
