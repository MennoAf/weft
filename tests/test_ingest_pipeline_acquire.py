"""Regression tests for ingest_pipeline.route's acquire() integration.

Before this fix, ``route()`` called ``store_memory(pool, ...)`` with the
raw pool — no ``acquire()`` wrapper, so ``SET LOCAL app.user_id`` never
fired and the migration-34 NOT NULL on ``memories.user_id`` tripped on
every ingest under live traffic.

These tests pin the fix: with the ``current_user_id`` contextvar set
(as the HTTP middleware would), route() persists memories tagged to
the correct user.
"""

from __future__ import annotations

import pytest

from weft.auth import current_user_id
from weft.ingest_pipeline import Intent, route


@pytest.mark.asyncio
async def test_route_persists_memory_with_user_id_stamped(pool):
    """The bug: route() raised NotNullViolation because user_id was NULL.
    The fix: acquire() runs SET LOCAL on the connection, so the GUC the
    INSERT reads from is set to the contextvar's value."""
    intents = [
        Intent(
            type="general_note",
            content="ingest pipeline regression note",
            entities=[],
            confidence=0.8,
        )
    ]

    tok = current_user_id.set("ingest-route-user")
    try:
        result = await route(intents, pool, embedding_provider=None)
    finally:
        current_user_id.reset(tok)

    assert result.errors == []
    assert result.memories_created == 1

    # Confirm the row landed with the right user_id (raw pool query
    # bypasses RLS via testcontainer superuser, which is fine for
    # checking the INSERT's user_id stamping).
    rows = await pool.fetch(
        "SELECT id, user_id, content FROM memories "
        "WHERE content = $1",
        "ingest pipeline regression note",
    )
    assert len(rows) == 1
    assert rows[0]["user_id"] == "ingest-route-user"


@pytest.mark.asyncio
async def test_route_honors_memory_type_hint(pool):
    """The channel-mapping fix: when Intent carries memory_type_hint, it
    overrides _INTENT_MEMORY_TYPE[intent.type]. Without this, the LLM-derived
    intent type would win at write time and the channel mapping would be a
    silent no-op (the substrate-stub gap that put L8 on HOLD)."""
    intents = [
        Intent(
            type="general_note",          # default mapping → fact
            content="brain dump note routed via channel mapping",
            confidence=0.8,
            memory_type_hint="preference",  # hint should win
            extra_topics=["discord", "brain-dump"],
        )
    ]

    tok = current_user_id.set("ingest-hint-user")
    try:
        result = await route(intents, pool, embedding_provider=None)
    finally:
        current_user_id.reset(tok)

    assert result.errors == []
    assert result.memories_created == 1

    rows = await pool.fetch(
        "SELECT type, topic FROM memories WHERE content = $1",
        "brain dump note routed via channel mapping",
    )
    assert len(rows) == 1
    # The hint must beat the default ("fact") from _INTENT_MEMORY_TYPE.
    assert rows[0]["type"] == "preference"
    # Source-supplied topics ride along with the auto-topics.
    assert "discord" in rows[0]["topic"]
    assert "brain-dump" in rows[0]["topic"]
    assert "intent:general_note" in rows[0]["topic"]


@pytest.mark.asyncio
async def test_route_invalid_hint_falls_back_to_default(pool):
    """An invalid memory_type_hint string must not crash ingest. Validation
    fails silently (warning logged) and the LLM default applies — protects
    against a bad config value taking down the whole pipeline."""
    intents = [
        Intent(
            type="general_note",
            content="ingest with bogus hint should still land",
            confidence=0.8,
            memory_type_hint="not_a_real_type",
        )
    ]

    tok = current_user_id.set("ingest-bad-hint-user")
    try:
        result = await route(intents, pool, embedding_provider=None)
    finally:
        current_user_id.reset(tok)

    assert result.errors == []
    assert result.memories_created == 1

    rows = await pool.fetch(
        "SELECT type FROM memories WHERE content = $1",
        "ingest with bogus hint should still land",
    )
    assert len(rows) == 1
    assert rows[0]["type"] == "fact"  # fell back to _INTENT_MEMORY_TYPE['general_note']


@pytest.mark.asyncio
async def test_route_two_intents_each_get_own_acquire_scope(pool):
    """Per-intent acquire() means a failure in one intent doesn't
    poison the others — the existing per-intent try/except still
    works, and each intent's INSERT lands on its own
    transaction-scoped SET LOCAL."""
    intents = [
        Intent(type="general_note", content="route batch first",
               entities=[], confidence=0.8),
        Intent(type="general_note", content="route batch second",
               entities=[], confidence=0.8),
    ]

    tok = current_user_id.set("ingest-batch-user")
    try:
        result = await route(intents, pool, embedding_provider=None)
    finally:
        current_user_id.reset(tok)

    assert result.errors == []
    assert result.memories_created == 2

    rows = await pool.fetch(
        "SELECT user_id FROM memories WHERE content = ANY($1::text[])",
        ["route batch first", "route batch second"],
    )
    assert {r["user_id"] for r in rows} == {"ingest-batch-user"}
