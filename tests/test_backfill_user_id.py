"""Tests for weft/db/backfill_user_id.py — TDD, written before implementation."""

from __future__ import annotations

import pytest

from weft.config.user_identity import get_user_id


# audit_backfill_user_id table is created by migration 32 and truncated by the
# pool fixture; no per-test audit setup needed.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert_behavior(pool, *, project_id=None, user_id=None) -> str:
    """Insert a minimal behaviors row; return its id."""
    import uuid
    row_id = uuid.uuid4().hex
    await pool.execute(
        """
        INSERT INTO behaviors (id, trigger_pattern, action, project_id, user_id)
        VALUES ($1, 'test-pattern', 'test-action', $2, $3)
        """,
        row_id, project_id, user_id,
    )
    return row_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_backfill_dual_scoped(pool):
    """Row with project_id set and user_id NULL → assigned config user_id, audit old_scope='dual-scoped'."""
    from weft.db.backfill_user_id import backfill_user_id

    row_id = await _insert_behavior(pool, project_id="proj-x", user_id=None)

    count = await backfill_user_id(pool)
    assert count >= 1

    row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", row_id)
    assert row["user_id"] == get_user_id()

    audit = await pool.fetchrow(
        "SELECT old_scope FROM audit_backfill_user_id WHERE source_table = 'behaviors' AND row_id = $1",
        row_id,
    )
    assert audit is not None
    assert audit["old_scope"] == "dual-scoped"


async def test_backfill_pure_user(pool):
    """Row with project_id NULL and user_id NULL → assigned config user_id, audit old_scope='pure-user'."""
    from weft.db.backfill_user_id import backfill_user_id

    row_id = await _insert_behavior(pool, project_id=None, user_id=None)

    count = await backfill_user_id(pool)
    assert count >= 1

    row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", row_id)
    assert row["user_id"] == get_user_id()

    audit = await pool.fetchrow(
        "SELECT old_scope FROM audit_backfill_user_id WHERE source_table = 'behaviors' AND row_id = $1",
        row_id,
    )
    assert audit is not None
    assert audit["old_scope"] == "pure-user"


async def test_backfill_idempotent(pool):
    """Two rows (one dual, one pure). First run → 2. Second run → 0. Rows unchanged after second call."""
    from weft.db.backfill_user_id import backfill_user_id

    dual_id = await _insert_behavior(pool, project_id="proj-y", user_id=None)
    pure_id = await _insert_behavior(pool, project_id=None, user_id=None)

    first = await backfill_user_id(pool)
    assert first == 2

    second = await backfill_user_id(pool)
    assert second == 0

    # Rows still have the correct user_id after second call
    uid = get_user_id()
    dual_row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", dual_id)
    pure_row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", pure_id)
    assert dual_row["user_id"] == uid
    assert pure_row["user_id"] == uid


async def test_backfill_skips_already_assigned(pool):
    """Row with user_id already set → untouched, no audit row."""
    from weft.db.backfill_user_id import backfill_user_id

    row_id = await _insert_behavior(pool, project_id=None, user_id="some-other-user")

    count = await backfill_user_id(pool)
    assert count == 0

    row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", row_id)
    assert row["user_id"] == "some-other-user"

    audit_count = await pool.fetchval(
        "SELECT COUNT(*) FROM audit_backfill_user_id WHERE row_id = $1", row_id
    )
    assert audit_count == 0
