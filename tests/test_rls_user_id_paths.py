"""Cross-module user_id propagation integration tests.

Verifies that every INSERT path across all modules correctly propagates
user_id via current_setting('app.user_id', true). Exercises the full flow:
  set app.user_id → acquire() → create memory/behavior/entity/episode →
  revise memory → run consolidation → generate primer →
  verify all rows have correct user_id.

All store operations run inside acquire() which issues SET LOCAL app.user_id,
matching real MCP tool execution paths.

Requires a real PostgreSQL instance (testcontainers via conftest.py).
"""

from __future__ import annotations

import pytest

from weft.auth import current_user_id
from weft.behaviors import store_behavior
from weft.consolidation import consolidate
from weft.db.connection import acquire
from weft.entities import store_entity, link_mention
from weft.episodes import create_episode, add_memory_to_episode
from weft.models import (
    BehaviorCreate,
    EntityCreate,
    EntityType,
    EpisodeCreate,
    MemoryCreate,
    MemoryType,
    RelationType,
)
from weft.primer import build_primer
from weft.revise import revise_memory
from weft.store import store_memory, add_relationship

TEST_USER = "rls-integration-user-1"


@pytest.fixture(autouse=True)
def _reset_contextvar():
    """Ensure contextvar is clean before and after each test."""
    tok = current_user_id.set(None)
    yield
    current_user_id.reset(tok)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _check_user_id(pool, table, id_col, id_val, expected_user):
    """Assert that a row's user_id matches expected_user."""
    row = await pool.fetchrow(
        f"SELECT user_id FROM {table} WHERE {id_col} = $1",  # noqa: S608
        id_val,
    )
    assert row is not None, f"Row {id_val} not found in {table}"
    assert row["user_id"] == expected_user, (
        f"{table}.{id_col}={id_val}: expected user_id={expected_user!r}, got {row['user_id']!r}"
    )


# ---------------------------------------------------------------------------
# Individual module INSERT tests
# ---------------------------------------------------------------------------


class TestMemoryUserIdPropagation:
    async def test_store_memory_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                mem = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="memory user_id test"),
                )
        finally:
            current_user_id.reset(tok)

        await _check_user_id(pool, "memories", "id", mem.id, TEST_USER)

    async def test_store_memory_uses_default_when_no_contextvar(self, pool):
        """Without an explicit contextvar, store_memory writes against the
        session's ``app.user_id`` (set by the test fixture's setup callback).
        Migration 34 killed the legacy NULL = global path; the column DEFAULT
        + NOT NULL constraint guarantee every row carries a real owner."""
        mem = await store_memory(
            pool,
            MemoryCreate(type=MemoryType.fact, content="no user memory"),
        )
        row = await pool.fetchrow("SELECT user_id FROM memories WHERE id = $1", mem.id)
        assert row["user_id"] == "test-user-default"


class TestBehaviorUserIdPropagation:
    async def test_store_behavior_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                beh = await store_behavior(
                    pool,
                    BehaviorCreate(
                        trigger_pattern="when testing",
                        action="use pytest",
                    ),
                )
        finally:
            current_user_id.reset(tok)

        await _check_user_id(pool, "behaviors", "id", beh.id, TEST_USER)


class TestEntityUserIdPropagation:
    async def test_store_entity_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                ent = await store_entity(
                    pool,
                    EntityCreate(name="TestEntity", entity_type=EntityType.concept),
                )
        finally:
            current_user_id.reset(tok)

        await _check_user_id(pool, "entities", "id", ent.id, TEST_USER)

    async def test_entity_mention_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                ent = await store_entity(
                    pool,
                    EntityCreate(name="MentionEntity", entity_type=EntityType.person),
                )
                mem = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="mention test"),
                )
                await link_mention(pool, ent.id, mem.id)
        finally:
            current_user_id.reset(tok)

        row = await pool.fetchrow(
            "SELECT user_id FROM entity_mentions WHERE entity_id = $1 AND memory_id = $2",
            ent.id, mem.id,
        )
        assert row is not None
        assert row["user_id"] == TEST_USER


class TestEpisodeUserIdPropagation:
    async def test_create_episode_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                ep = await create_episode(
                    pool,
                    EpisodeCreate(title="test episode"),
                )
        finally:
            current_user_id.reset(tok)

        await _check_user_id(pool, "episodes", "id", ep.id, TEST_USER)

    async def test_episode_memory_link_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                ep = await create_episode(pool, EpisodeCreate(title="link test"))
                mem = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="episode link test"),
                )
                await add_memory_to_episode(pool, ep.id, mem.id)
        finally:
            current_user_id.reset(tok)

        row = await pool.fetchrow(
            "SELECT user_id FROM episode_memories WHERE episode_id = $1 AND memory_id = $2",
            ep.id, mem.id,
        )
        assert row is not None
        assert row["user_id"] == TEST_USER


class TestRelationshipUserIdPropagation:
    async def test_add_relationship_sets_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                m1 = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="rel source"),
                )
                m2 = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="rel target"),
                )
                await add_relationship(pool, m1.id, m2.id, RelationType.related_to)
        finally:
            current_user_id.reset(tok)

        row = await pool.fetchrow(
            "SELECT user_id FROM memory_relationships WHERE source_id = $1 AND target_id = $2",
            m1.id, m2.id,
        )
        assert row is not None
        assert row["user_id"] == TEST_USER


class TestReviseUserIdPropagation:
    async def test_revise_memory_sets_user_id_on_new_and_relationship(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                orig = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="original content"),
                )
                new_mem, old_mem = await revise_memory(pool, orig.id, "revised content")
        finally:
            current_user_id.reset(tok)

        # New memory has user_id
        await _check_user_id(pool, "memories", "id", new_mem.id, TEST_USER)

        # Supersedes relationship has user_id
        row = await pool.fetchrow(
            "SELECT user_id FROM memory_relationships WHERE source_id = $1 AND target_id = $2",
            new_mem.id, old_mem.id,
        )
        assert row is not None, "supersedes relationship not found"
        assert row["user_id"] == TEST_USER


# ---------------------------------------------------------------------------
# Full end-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEndUserIdFlow:
    """Exercise the complete flow: create across all modules, then verify."""

    async def test_full_flow_all_rows_have_correct_user_id(self, pool):
        tok = current_user_id.set(TEST_USER)
        try:
            # Phase 1: create objects (all inside acquire for RLS context)
            async with acquire(pool):
                # 1. Memory
                mem = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="e2e memory"),
                )

                # 2. Behavior
                beh = await store_behavior(
                    pool,
                    BehaviorCreate(trigger_pattern="e2e trigger", action="e2e action"),
                )

                # 3. Entity + mention link
                ent = await store_entity(
                    pool,
                    EntityCreate(name="E2EEntity", entity_type=EntityType.concept),
                )
                await link_mention(pool, ent.id, mem.id)

                # 4. Episode + memory link
                ep = await create_episode(pool, EpisodeCreate(title="e2e episode"))
                await add_memory_to_episode(pool, ep.id, mem.id)

                # 5. Relationship
                mem2 = await store_memory(
                    pool,
                    MemoryCreate(type=MemoryType.fact, content="e2e memory 2"),
                )
                await add_relationship(pool, mem.id, mem2.id, RelationType.related_to)

            # Phase 2: revise (opens its own acquire internally)
            async with acquire(pool):
                new_mem, _ = await revise_memory(pool, mem2.id, "e2e revised")

            # Phase 3: primer (acquires its own connections from pool)
            primer = await build_primer(pool)
            assert primer is not None

        finally:
            current_user_id.reset(tok)

        # --- Verify all rows ---
        await _check_user_id(pool, "memories", "id", mem.id, TEST_USER)
        await _check_user_id(pool, "memories", "id", mem2.id, TEST_USER)
        await _check_user_id(pool, "memories", "id", new_mem.id, TEST_USER)
        await _check_user_id(pool, "behaviors", "id", beh.id, TEST_USER)
        await _check_user_id(pool, "entities", "id", ent.id, TEST_USER)
        await _check_user_id(pool, "episodes", "id", ep.id, TEST_USER)

        # Junction / relationship tables
        row = await pool.fetchrow(
            "SELECT user_id FROM entity_mentions WHERE entity_id = $1", ent.id,
        )
        assert row["user_id"] == TEST_USER

        row = await pool.fetchrow(
            "SELECT user_id FROM episode_memories WHERE episode_id = $1", ep.id,
        )
        assert row["user_id"] == TEST_USER

        # relates_to relationship
        row = await pool.fetchrow(
            "SELECT user_id FROM memory_relationships WHERE source_id = $1 AND target_id = $2",
            mem.id, mem2.id,
        )
        assert row["user_id"] == TEST_USER

        # supersedes relationship from revise
        row = await pool.fetchrow(
            "SELECT user_id FROM memory_relationships WHERE source_id = $1 AND target_id = $2",
            new_mem.id, mem2.id,
        )
        assert row is not None, "supersedes relationship from revise not found"
        assert row["user_id"] == TEST_USER


class TestConsolidationUserIdPropagation:
    """Consolidation merge creates memory_relationships — verify user_id."""

    async def test_consolidation_merge_relationship_has_user_id(self, pool):
        """Create near-duplicate memories, consolidate, check relationship user_id."""
        tok = current_user_id.set(TEST_USER)
        try:
            async with acquire(pool):
                # Create two identical memories to trigger dedup
                m1 = await store_memory(
                    pool,
                    MemoryCreate(
                        type=MemoryType.fact,
                        content="The project uses PostgreSQL 16 with pgvector extension",
                    ),
                )
                m2 = await store_memory(
                    pool,
                    MemoryCreate(
                        type=MemoryType.fact,
                        content="The project uses PostgreSQL 16 with pgvector extension",
                    ),
                )

            # Consolidation acquires its own connections
            report = await consolidate(pool, dry_run=False)
        finally:
            current_user_id.reset(tok)

        # If duplicates were merged, check the supersedes relationship
        if len(report.duplicates_merged) > 0:
            row = await pool.fetchrow(
                """SELECT user_id FROM memory_relationships
                   WHERE (source_id = $1 AND target_id = $2)
                      OR (source_id = $2 AND target_id = $1)""",
                m1.id, m2.id,
            )
            assert row is not None, "expected supersedes relationship from consolidation"
            assert row["user_id"] == TEST_USER
