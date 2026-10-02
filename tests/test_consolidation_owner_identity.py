"""Owner-scoped consolidation regression tests.

These tests deliberately use a disposable asyncpg pool without the shared
fixture's session-level app.user_id setup.  Consolidation must establish its
identity scope itself, as the detached prime/background path does in
production.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from weft.auth import current_user_id
from weft.consolidation import (
    ConsolidationConfig,
    consolidate,
    find_contradictions,
)
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.db.connection import _current_conn, _pgvector_codec_init, acquire, get_db
from weft.models import MemoryCreate, MemoryType, RelationType
from weft.store import store_memory


@pytest.fixture
async def raw_identity_pool(pool, pg_dsn) -> AsyncIterator[asyncpg.Pool]:
    """Use a real disposable pool with no identity setup callback."""
    raw = await asyncpg.create_pool(
        pg_dsn,
        min_size=2,
        max_size=4,
        init=_pgvector_codec_init,
    )
    try:
        assert await raw.fetchval("SELECT current_setting('app.user_id', true)") in (None, "")
        yield raw
    finally:
        await raw.close()


@asynccontextmanager
async def _owner(user_id: str, pool: asyncpg.Pool):
    token = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            yield
    finally:
        current_user_id.reset(token)


async def _seed_owner_pair(pool: asyncpg.Pool, user_id: str):
    """Seed one duplicate pair and one contradictory pair for an owner."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")
    duplicate_a = "pgvector uses cosine distance to measure vector similarity"
    duplicate_b = "pgvector uses cosine distance for vector similarity measurement"
    contradiction_a = "pgvector supports HNSW indexing"
    contradiction_b = "pgvector does not support HNSW indexing"

    async with _owner(user_id, pool):
        memories = []
        for content, confidence, topic in (
            (duplicate_a, 0.9, [f"duplicate-a-{user_id}"]),
            (duplicate_b, 0.6, [f"duplicate-b-{user_id}"]),
            (contradiction_a, 0.8, [f"contradiction-a-{user_id}"]),
            (contradiction_b, 0.7, [f"contradiction-b-{user_id}"]),
        ):
            memories.append(
                await store_memory(
                    pool,
                    MemoryCreate(
                        type=MemoryType.fact,
                        content=content,
                        confidence=confidence,
                        topic=topic,
                    ),
                    embedding=await provider.embed(content),
                )
            )
    assert current_user_id.get() is None
    assert _current_conn.get(None) is None
    return memories


@pytest.mark.asyncio
async def test_consolidation_relationships_are_owner_scoped_and_restored(raw_identity_pool):
    """Detached-style consolidation writes both relationship kinds for each owner.

    Each run must read and write only its owner's memories, attribute every
    relationship to that owner, and restore both identity/context state after
    returning to its caller.
    """
    pool = raw_identity_pool
    owner_a = "consolidation-owner-a"
    owner_b = "consolidation-owner-b"
    memories_a = await _seed_owner_pair(pool, owner_a)
    memories_b = await _seed_owner_pair(pool, owner_b)

    config = ConsolidationConfig(
        duplicate_threshold=0.95,
        contradiction_similarity_min=0.7,
        contradiction_similarity_max=0.99,
    )

    for owner, memories in ((owner_a, memories_a), (owner_b, memories_b)):
        duplicate_ids = {memories[0].id, memories[1].id}
        contradiction_ids = {memories[2].id, memories[3].id}
        token = current_user_id.set(owner)
        try:
            # Run the contradiction detector before dedup can consume the
            # highly similar negation pair. This directly exercises the
            # owner-scoped relationship write, while the full consolidate run
            # below exercises the detached-style orchestrator + supersedes write.
            flagged = await find_contradictions(
                pool, sim_min=0.7, sim_max=0.99,
            )
            assert len(flagged) >= 1

            report = await consolidate(pool, config=config)
            assert report.errors == []
            assert len(report.duplicates_merged) >= 1

            async with acquire(pool) as conn:
                rows = await conn.fetch(
                    """
                    SELECT source_id, target_id, relation, user_id
                    FROM memory_relationships
                    WHERE source_id = ANY($1::text[]) OR target_id = ANY($1::text[])
                    """,
                    list(duplicate_ids | contradiction_ids),
                )
                assert rows
                assert {row["user_id"] for row in rows} == {owner}
                assert {
                    row["relation"] for row in rows
                } >= {RelationType.supersedes.value, RelationType.contradicts.value}

                owner_ids = {mem.id for mem in memories}
                other_ids = (
                    {mem.id for mem in memories_b}
                    if owner == owner_a
                    else {mem.id for mem in memories_a}
                )
                reported_ids = {
                    memory_id
                    for pair in report.duplicates_merged
                    for memory_id in pair
                }
                reported_ids.update(
                    memory_id
                    for pair in report.contradictions_flagged
                    for memory_id in pair
                )
                assert reported_ids <= owner_ids
                assert not reported_ids & other_ids
        finally:
            current_user_id.reset(token)

        assert current_user_id.get() is None
        assert _current_conn.get(None) is None
        assert get_db(pool) is pool

    # A raw connection after the scoped run has no transaction-local identity.
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_setting('app.user_id', true)") in (None, "")

    # Both owners received their own duplicate and contradiction relationships;
    # no owner was able to consolidate across the other owner's rows.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT user_id, relation, count(*) AS count FROM memory_relationships "
            "GROUP BY user_id, relation ORDER BY user_id, relation"
        )
    assert {(row["user_id"], row["relation"]) for row in rows} >= {
        (owner_a, RelationType.supersedes.value),
        (owner_a, RelationType.contradicts.value),
        (owner_b, RelationType.supersedes.value),
        (owner_b, RelationType.contradicts.value),
    }


@pytest.mark.asyncio
async def test_failed_subsystem_rolls_back_and_next_consolidation_unlocks(raw_identity_pool):
    """A failed owner pass does not poison later passes or the advisory lock."""
    pool = raw_identity_pool
    owner = "consolidation-failure-owner"
    token = current_user_id.set(owner)
    try:
        async def fail_duplicate_pass(*args, **kwargs):
            async with acquire(pool) as conn:
                await conn.execute("SELECT definitely_missing_consolidation_table")

        with patch("weft.consolidation.find_duplicates", side_effect=fail_duplicate_pass):
            failed = await consolidate(pool, dry_run=True)
        assert any("Duplicate detection failed" in error for error in failed.errors)
        assert any("Contradiction detection failed" not in error for error in failed.errors)

        recovered = await consolidate(pool, dry_run=True)
        assert recovered.skipped is False
        assert not any("Duplicate detection failed" in error for error in recovered.errors)

        async with pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", 839272,
            ) is True
            assert await conn.fetchval(
                "SELECT pg_advisory_unlock($1)", 839272,
            ) is True
    finally:
        current_user_id.reset(token)
    assert _current_conn.get(None) is None
    assert current_user_id.get() is None


@pytest.mark.asyncio
async def test_replay_callback_gets_fresh_connection_for_system_sentinel(raw_identity_pool):
    """Replay cannot inherit the owner connection and sentinel binds its own GUC."""
    pool = raw_identity_pool
    owner = "consolidation-replay-owner"
    observed: dict[str, object] = {}

    async def fake_replay(replay_pool):
        observed["inherited_conn"] = _current_conn.get(None)
        observed["owner_during_callback"] = current_user_id.get(None)
        replay_token = current_user_id.set(SYSTEM_GLOBAL_USER_ID)
        try:
            async with acquire(replay_pool) as conn:
                observed["sentinel_guc"] = await conn.fetchval(
                    "SELECT current_setting('app.user_id', true)"
                )
        finally:
            current_user_id.reset(replay_token)
        return SimpleNamespace(rows_done=0, claims_written=0)

    token = current_user_id.set(owner)
    try:
        with patch(
            "weft.replay_executor.run_replay_executor_batch",
            new=AsyncMock(side_effect=fake_replay),
        ):
            report = await consolidate(pool)
    finally:
        current_user_id.reset(token)

    assert report.errors == []
    assert observed["inherited_conn"] is None
    assert observed["owner_during_callback"] == owner
    assert observed["sentinel_guc"] == SYSTEM_GLOBAL_USER_ID
    assert _current_conn.get(None) is None
    assert current_user_id.get() is None
