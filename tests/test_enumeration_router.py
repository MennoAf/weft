"""Unit tests for the enumeration-intent classifier (Phase 1, V7).

detect_enumeration_intent(query) -> (is_enumeration, noun). Pure regex, no I/O.
"""

from __future__ import annotations

import pytest

from weft.enumeration_router import detect_enumeration_intent


@pytest.mark.parametrize(
    "query,expected_noun",
    [
        ("list all the plants", "plants"),
        ("list all plants", "plants"),
        ("list my medications", "medications"),
        ("enumerate the open issues", "open issues"),
        ("how many plants do I have", "plants"),
        ("how many medications", "medications"),
        ("every project I'm working on", "project"),
        ("what are all the projects that are open", "projects"),
        ("give me all my trackers", "trackers"),
        ("show me all the decisions we made", "decisions"),
    ],
)
def test_enumeration_intent_extracts_noun(query, expected_noun):
    is_enum, noun = detect_enumeration_intent(query)
    assert is_enum is True, f"{query!r} should be classified as enumeration"
    assert noun == expected_noun, f"{query!r} → expected noun {expected_noun!r}, got {noun!r}"


@pytest.mark.parametrize(
    "query",
    [
        "what did I decide about the database",
        "tell me about the launch plan",
        "redis architecture",
        "how long since the move",  # temporal, not enumeration
        "when did I start the project",
        "",
        "   ",
    ],
)
def test_non_enumeration_queries_pass_through(query):
    is_enum, noun = detect_enumeration_intent(query)
    assert is_enum is False, f"{query!r} should NOT be classified as enumeration"
    assert noun is None


def test_boundary_trimming_stops_at_clause_words():
    # The noun run is trimmed at the first clause-boundary token.
    _, noun = detect_enumeration_intent("list all the plants that I water weekly")
    assert noun == "plants"


def test_enumeration_shaped_but_no_noun_returns_true_none():
    # "every" with nothing resolvable after it → intent yes, noun no. Caller
    # falls back to an explicit topic arg.
    is_enum, noun = detect_enumeration_intent("list all")
    assert is_enum is True
    assert noun is None


def test_case_insensitive():
    is_enum, noun = detect_enumeration_intent("LIST ALL THE PLANTS")
    assert is_enum is True
    assert noun == "plants"
