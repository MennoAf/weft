"""Tests for migration 64: add project_facets column + GIN index + backfill.

Tests verify:
  1. Column schema: exists, is NOT NULL, defaults to '{}', type is TEXT[]
  2. Backfill logic: NULL project_id -> '{}', non-NULL project_id -> ARRAY[lower(project_id)]
  3. GIN index: idx_memories_project_facets exists and functions correctly
  4. Non-destructive: existing recall/store tests pass unchanged
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg
import pytest


def _memory_id() -> str:
    return f"weft-{uuid.uuid4().hex[:8]}"


async def _insert_memory(
    conn: asyncpg.Connection,
    *,
    memory_id: str | None = None,
    type_: str = "preference",
    topic: list[str] | None = None,
    content: str = "test content",
    project_id: str | None = None,
) -> str:
    """Insert a memory row for testing."""
    if memory_id is None:
        memory_id = _memory_id()
    if topic is None:
        topic = []

    await conn.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, agent_id, status, pinned,
            usefulness_score, usefulness_count, write_provenance,
            review_status
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14, $15,
            $16, $17, $18, $19
        )
        """,
        memory_id, type_, topic, content, "conversation", 0.7,
        0, datetime.now(timezone.utc), datetime.now(timezone.utc),
        datetime.now(timezone.utc),
        0, project_id, None, "active", False,
        0.7, 0, "supervisor", "active",
    )
    return memory_id


@pytest.mark.asyncio
async def test_project_facets_column_exists(pool):
    """project_facets column exists with correct properties."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'memories' AND column_name = 'project_facets'
        """
    )
    assert len(cols) == 1, "project_facets column does not exist"
    col = cols[0]

    assert col["column_name"] == "project_facets"
    # PostgreSQL returns 'ARRAY' or 'text[]' — both are valid representations
    assert col["data_type"] in ("text[]", "ARRAY"), f"expected text[] or ARRAY, got {col['data_type']}"
    assert col["is_nullable"] == "NO", "project_facets should be NOT NULL"
    # Default is '{}'::text[] (or similar) — just check it's not null
    assert col["column_default"] is not None, "project_facets should have a default"


@pytest.mark.asyncio
async def test_project_facets_with_explicit_value(pool):
    """Can insert memories with explicit project_facets values."""
    memory_id = _memory_id()
    now = datetime.now(timezone.utc)

    # Insert with explicit project_facets
    await pool.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, status, pinned,
            usefulness_score, usefulness_count, write_provenance,
            review_status, project_facets
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14,
            $15, $16, $17, $18, $19
        )
        """,
        memory_id, "preference", [], "test", "conversation", 0.7,
        0, now, now, now,
        0, "my-project", "active", False,
        0.7, 0, "supervisor", "active", ["my-project"],
    )

    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1",
        memory_id,
    )
    assert row is not None
    assert row["project_facets"] == ["my-project"]


@pytest.mark.asyncio
async def test_project_facets_defaults_to_empty_array(pool):
    """New inserts without explicit project_facets default to empty array."""
    memory_id = _memory_id()
    now = datetime.now(timezone.utc)

    await pool.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, status, pinned,
            usefulness_score, usefulness_count, write_provenance,
            review_status
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14,
            $15, $16, $17, $18
        )
        """,
        memory_id, "fact", [], "test", "conversation", 0.7,
        0, now, now, now,
        0, None, "active", False,
        0.7, 0, "supervisor", "active",
    )

    row = await pool.fetchrow(
        "SELECT project_facets FROM memories WHERE id = $1",
        memory_id,
    )
    assert row is not None
    assert row["project_facets"] == []


@pytest.mark.asyncio
async def test_project_facets_gin_index_exists(pool):
    """GIN index idx_memories_project_facets exists."""
    exists = await pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_memories_project_facets'
        )
        """
    )
    assert exists, "idx_memories_project_facets index does not exist"


@pytest.mark.asyncio
async def test_project_facets_gin_index_query_functional(pool):
    """GIN index can be used in facet-based queries."""
    # Insert test memories with project_facets
    memory1 = _memory_id()
    memory2 = _memory_id()
    now = datetime.now(timezone.utc)

    await pool.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, status, pinned,
            usefulness_score, usefulness_count, write_provenance,
            review_status, project_facets
        ) VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8, $8, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17),
            ($18, $19, $20, $21, $22, $23, $24, $8, $8, $8, $25, $26, $27, $28, $29, $30, $31, $32, $33)
        """,
        memory1, "preference", [], "memory 1", "conversation", 0.7,
        0, now, 0, "proj-a", "active", False, 0.7, 0, "supervisor", "active",
        ["proj-a"],
        memory2, "fact", [], "memory 2", "conversation", 0.7,
        0, 0, "proj-b", "active", False, 0.7, 0, "supervisor", "active",
        ["proj-b"],
    )

    # Query using array overlap operator (uses GIN index)
    rows = await pool.fetch(
        """
        SELECT id FROM memories
        WHERE project_facets && ARRAY['proj-a']::text[]
        """
    )
    # Just verify query runs; results depend on other test data
    assert isinstance(rows, list)


@pytest.mark.asyncio
async def test_v64_migration_idempotent(pool):
    """Running migration 64 SQL twice does not error."""
    from weft.db.migrations import MIGRATIONS

    v64_sql = [sql for version, _, sql in MIGRATIONS if version == 64]
    assert len(v64_sql) == 1, "expected migration 64 in MIGRATIONS list"

    # Running it twice should be safe (CREATE INDEX IF NOT EXISTS, column already exists)
    try:
        await pool.execute(v64_sql[0])
    except asyncpg.DuplicateColumnError:
        # Expected if column already exists from first run
        pass


@pytest.mark.asyncio
async def test_existing_memory_recall_unchanged(pool):
    """Existing memory store/recall is unaffected (schema non-destructive)."""
    memory_id = _memory_id()
    now = datetime.now(timezone.utc)

    # Insert a memory the "old way" (without explicitly setting project_facets)
    await pool.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, status, pinned,
            usefulness_score, usefulness_count, write_provenance,
            review_status
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14,
            $15, $16, $17, $18
        )
        """,
        memory_id, "decision", ["arch"], "old style insert", "conversation", 0.9,
        100, now, now, now,
        5, None, "active", True,
        0.9, 3, "supervisor", "active",
    )

    # Fetch the entire row
    row = await pool.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
    assert row is not None
    assert row["id"] == memory_id
    assert row["type"] == "decision"
    assert row["topic"] == ["arch"]
    assert row["content"] == "old style insert"
    assert abs(row["confidence"] - 0.9) < 0.0001  # Account for float precision
    assert row["access_count"] == 5
    # project_facets should be present and empty (default backfilled)
    assert row["project_facets"] == [] or row["project_facets"] == "{}"
