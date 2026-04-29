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
