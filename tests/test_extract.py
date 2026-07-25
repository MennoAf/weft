"""Tests for weft.extract — memory candidate extraction from text."""

from __future__ import annotations

from weft.extract import extract_behaviors, extract_candidates, validate_memory_content


def test_extract_preference():
    """Detects user preference statements."""
    text = "I prefer dark mode for all editors"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "preference"
    assert results[0]["confidence"] >= 0.7


def test_extract_preference_metadata_polarity_and_strength():
    positive = extract_candidates("I prefer history podcasts on my commute")[0]
    avoidance = extract_candidates("I never want true crime podcasts")[0]
    constraint = extract_candidates("I always use a written checklist")[0]

    assert positive["preference_metadata"] == {
        "polarity": "positive", "strength": "soft",
        "value": "history podcasts on my commute",
    }
    assert avoidance["preference_metadata"]["polarity"] == "avoidance"
    assert avoidance["preference_metadata"]["value"] == "true crime podcasts"
    assert constraint["preference_metadata"] == {
        "polarity": "constraint", "strength": "hard",
        "value": "use a written checklist",
    }


def test_extracts_classifier_normalized_preference_summaries():
    preferred = extract_candidates("Prefers history podcasts during commute")[0]
    avoided = extract_candidates("Avoids true crime content")[0]
    required = extract_candidates("Requires audio-compatible activities while driving")[0]
    assert preferred["preference_metadata"]["polarity"] == "positive"
    assert avoided["preference_metadata"]["polarity"] == "avoidance"
    assert required["preference_metadata"] == {
        "polarity": "constraint", "strength": "hard",
        "value": "audio-compatible activities while driving",
    }


def test_preference_metadata_ignores_negation_and_impersonal_instructions():
    negated = extract_candidates("I prefer not to use dark mode")
    assert not any("preference_metadata" in item for item in negated)
    assert extract_candidates("I want to fix the auth bug") == []
    assert extract_candidates("Always validate input before storing it") == []
    assert extract_candidates("I prefer history, but Alex prefers comedy") == []


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


# --- Solution/learning patterns ---


def test_extract_trick_was():
    """Detects 'the trick was' pattern."""
    text = "The trick was adding a keepalive ping to the connection pool"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "solution"


def test_extract_had_to():
    """Detects 'had to' pattern for workarounds."""
    text = "Had to pin fastembed to 0.3.1 because 0.4.0 breaks the embedding dimensions"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "solution"


def test_extract_watch_out():
    """Detects gotcha/caveat patterns."""
    text = "Watch out for asyncpg pool going stale after long idle periods"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "solution"


def test_extract_turns_out():
    """Detects 'turns out' discovery patterns."""
    text = "Turns out the Ryuk reaper container needs explicit cleanup in conftest"
    results = extract_candidates(text)
    assert len(results) >= 1
    assert results[0]["type"] == "solution"


def test_extract_learned_from_task():
    """Extracts multiple solution patterns from task-style notes."""
    text = """The fix was using a background keepalive task that pings every 5 minutes
Watch out for testcontainers leaving zombie containers on CI
Had to add retry logic for the initial DB connection on startup"""
    results = extract_candidates(text)
    assert len(results) >= 2
    types = [r["type"] for r in results]
    assert all(t == "solution" for t in types)


# --- Behavior extraction ---


def test_extract_behavior_when_do():
    """Detects 'when X, do Y' pattern."""
    text = "When writing tests, always use pytest fixtures instead of setUp"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert "writing tests" in results[0]["trigger_pattern"].lower()
    assert "pytest" in results[0]["action"].lower()


def test_extract_behavior_if_then():
    """Detects 'if X, then Y' pattern."""
    text = "If the build fails, check the pre-commit hooks first"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert "build fails" in results[0]["trigger_pattern"].lower()
    assert "pre-commit" in results[0]["action"].lower()


def test_extract_behavior_before_after():
    """Detects 'before/after X, Y' pattern."""
    text = "Before committing, always run the full test suite"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert results[0]["confidence"] >= 0.7


def test_extract_behavior_always_when():
    """Detects 'always X when Y' pattern."""
    text = "Always run equivalence tests when touching the primer"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert "primer" in results[0]["action"].lower() or "primer" in results[0]["trigger_pattern"].lower()


def test_extract_behavior_never_without():
    """Detects 'never X without Y' pattern."""
    text = "Never deploy without running the smoke tests"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert results[0]["confidence"] >= 0.7


def test_extract_behavior_make_sure():
    """Detects 'make sure to X before Y' pattern."""
    text = "Make sure to backup the database before running migrations"
    results = extract_behaviors(text)
    assert len(results) == 1


def test_extract_behavior_empty_input():
    """Empty input returns empty list."""
    assert extract_behaviors("") == []
    assert extract_behaviors("   ") == []


def test_extract_behavior_no_match():
    """Non-behavioral text returns empty."""
    text = "The quick brown fox jumps over the lazy dog"
    assert extract_behaviors(text) == []


def test_extract_behavior_deduplicates():
    """Same trigger/action pair only yields one candidate."""
    text = "When deploying, always run smoke tests\nWhen deploying, always run smoke tests"
    results = extract_behaviors(text)
    assert len(results) == 1


def test_extract_behavior_multiple():
    """Extracts multiple behaviors from multi-line text."""
    text = """When writing tests, always use pytest fixtures
If the CI fails, check Docker daemon status first
Never push to main without a PR review"""
    results = extract_behaviors(text)
    assert len(results) >= 2


def test_extract_behavior_short_lines_ignored():
    """Short lines are skipped."""
    text = "if x, do y"
    assert extract_behaviors(text) == []


def test_extract_behavior_has_source_line():
    """Each candidate includes the original source_line."""
    text = "When refactoring, always check for unused imports first"
    results = extract_behaviors(text)
    assert len(results) == 1
    assert "source_line" in results[0]


# --- Regressions for the 2026-05-04 behavior-fragmentation audit ---


def test_no_extract_after_a_fragment():
    """Reproduces weft-715b809b: 'after a test failure. Lesson: ...' must
    not produce trigger='after a' / action='test failure. ...'.

    The non-greedy capture used to grab a single determiner before the
    period; we now require a comma boundary AND reject triggers that
    end in a determiner.
    """
    text = "after a test failure. **Lesson: when two heuristics share semantic ground, share the input list.**"
    results = extract_behaviors(text)
    for r in results:
        assert r["trigger_pattern"].lower() != "after a"
        last = r["trigger_pattern"].split()[-1].lower().strip(".,;:")
        assert last not in {"a", "an", "the", "every"}


def test_no_extract_after_every_fragment():
    """Reproduces weft-c335c36f: 'after every release. To make ...' must
    not produce trigger='after every'."""
    text = "after every release. To make a session-level GUC survive, use setup= callback."
    results = extract_behaviors(text)
    for r in results:
        assert r["trigger_pattern"].lower() != "after every"


def test_required_comma_in_before_after():
    """The before/after pattern requires a comma now. Without one,
    the pattern shouldn't match (was producing fragments before)."""
    text = "after a long deploy run health checks against staging"
    results = extract_behaviors(text)
    # No comma → no match. The regex used to fire with `,?` optional.
    assert all(r["trigger_pattern"].lower() != "after a" for r in results)


# --- validate_memory_content (door-stop for weft_remember) ---


def test_validate_accepts_normal_memory():
    ok, reason = validate_memory_content(
        "We pin asyncpg to 0.30 because the setup= callback is required."
    )
    assert ok is True
    assert reason is None


def test_validate_rejects_too_short():
    ok, reason = validate_memory_content("ok")
    assert ok is False
    assert reason == "content_too_short"


def test_validate_rejects_bare_heading():
    """Reproduces weft-8264fac3 / weft-da90bac1 / weft-f05b4d63 — agents
    chunked docs by markdown heading and stored each title alone."""
    for heading in [
        "### .gitignore pattern for env templates",
        "### PRD_06 §3 Quick Hits — pattern for LLM-narrative scaffolding",
        "## Bootstrap-exception pattern for process docs",
        "# A top-level heading by itself",
    ]:
        ok, reason = validate_memory_content(heading)
        assert ok is False, f"should reject: {heading!r}"
        assert reason == "heading_only"


def test_validate_accepts_heading_with_body():
    """A heading is fine when it has a body — only bare headings get rejected."""
    content = "### Bootstrap-exception pattern\n\nUse this when the install order would otherwise create a chicken-and-egg."
    ok, reason = validate_memory_content(content)
    assert ok is True
    assert reason is None


def test_validate_rejects_trailing_colon():
    """Reproduces weft-ffd96aaa: '**Bool-vs-int validation pattern is now
    canonical** (applied 3x in atomic.py):' — body got truncated upstream."""
    content = "**Bool-vs-int validation pattern is now canonical** (applied 3x in atomic.py):"
    ok, reason = validate_memory_content(content)
    assert ok is False
    assert reason == "trailing_colon"


def test_validate_rejects_empty():
    ok, reason = validate_memory_content("")
    assert ok is False
    ok2, reason2 = validate_memory_content("   \n\n  ")
    assert ok2 is False
