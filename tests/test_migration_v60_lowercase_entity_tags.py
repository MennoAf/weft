"""Tests for migration 60: lowercase entity:* tags in memories.topic[].

Backfill half of the entity-tag casing fix (weft-6318d198). Verifies the
migration SQL lowercases ONLY ``entity:*`` elements, leaves every other tag
untouched, preserves array order, and is idempotent. Runs against real
Postgres via the testcontainers ``pool`` fixture.
"""

from __future__ import annotations

import uuid

import pytest

from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory


def _uid() -> str:
    return f"test-v60-{uuid.uuid4().hex[:12]}"


def _v60_sql() -> str:
    from weft.db.migrations import MIGRATIONS

    sql = [s for version, _, s in MIGRATIONS if version == 60]
    assert len(sql) == 1, "expected migration 60 in MIGRATIONS list"
    return sql[0]


async def _store(pool, user_id: str, topic: list[str]) -> str:
    """Store a memory with an exact topic list and return its id."""
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            mem = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content="v60 backfill fixture.",
                    topic=topic,
                    source=MemorySource.conversation,
                    confidence=0.8,
                ),
            )
    finally:
        current_user_id.reset(tok)
    return mem.id


async def _topic_of(pool, mem_id: str) -> list[str]:
    return list(
        await pool.fetchval("SELECT topic FROM memories WHERE id = $1", mem_id)
    )


@pytest.mark.asyncio
async def test_v60_lowercases_only_entity_tags_preserving_order(pool):
    """entity:* elements are lowercased; non-entity tags + order are untouched."""
    user_id = _uid()
    mem_id = await _store(
        pool,
        user_id,
        ["intent:foo", "entity:Weft", "custom:Bar", "entity:Windward"],
    )

    # Sanity: store preserved the mixed casing we're backfilling.
    assert await _topic_of(pool, mem_id) == [
        "intent:foo",
        "entity:Weft",
        "custom:Bar",
        "entity:Windward",
    ]

    await pool.execute(_v60_sql())

    assert await _topic_of(pool, mem_id) == [
        "intent:foo",      # non-entity prefix untouched
        "entity:weft",     # lowercased
        "custom:Bar",      # non-entity, mixed-case PRESERVED
        "entity:windward",  # lowercased
    ]


@pytest.mark.asyncio
async def test_v60_leaves_already_lowercase_rows_untouched(pool):
    """A row whose entity tags are already lowercase is not rewritten."""
    user_id = _uid()
    mem_id = await _store(pool, user_id, ["entity:weft", "intent:bar"])

    await pool.execute(_v60_sql())

    assert await _topic_of(pool, mem_id) == ["entity:weft", "intent:bar"]


@pytest.mark.asyncio
async def test_v60_idempotent(pool):
    """Running the migration twice yields the same result (no further change)."""
    user_id = _uid()
    mem_id = await _store(pool, user_id, ["entity:R0.1", "entity:Journey Builder"])

    await pool.execute(_v60_sql())
    after_first = await _topic_of(pool, mem_id)
    assert after_first == ["entity:r0.1", "entity:journey builder"]

    await pool.execute(_v60_sql())
    after_second = await _topic_of(pool, mem_id)
    assert after_second == after_first
