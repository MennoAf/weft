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


# ---------------------------------------------------------------------------
# Explicit user_id override (admin-safety path)
# ---------------------------------------------------------------------------

async def test_backfill_accepts_explicit_user_id(pool):
    """When user_id is passed explicitly, that value lands in rows + audit —
    NOT whatever get_user_id() would resolve in the current environment."""
    from weft.db.backfill_user_id import backfill_user_id

    row_id = await _insert_behavior(pool, project_id="proj-z", user_id=None)

    count = await backfill_user_id(pool, user_id="explicit-admin-id")
    assert count >= 1

    row = await pool.fetchrow("SELECT user_id FROM behaviors WHERE id = $1", row_id)
    assert row["user_id"] == "explicit-admin-id"

    # Config-resolved get_user_id() must not have been used.
    assert row["user_id"] != get_user_id()


async def test_backfill_explicit_user_id_does_not_change_existing(pool):
    """Override value doesn't touch rows that already have a user_id set."""
    from weft.db.backfill_user_id import backfill_user_id

    protected_id = await _insert_behavior(
        pool, project_id=None, user_id="pre-existing-user"
    )
    null_id = await _insert_behavior(pool, project_id=None, user_id=None)

    await backfill_user_id(pool, user_id="stamp-this")

    protected = await pool.fetchrow(
        "SELECT user_id FROM behaviors WHERE id = $1", protected_id
    )
    stamped = await pool.fetchrow(
        "SELECT user_id FROM behaviors WHERE id = $1", null_id
    )
    assert protected["user_id"] == "pre-existing-user"
    assert stamped["user_id"] == "stamp-this"


# ---------------------------------------------------------------------------
# Dry-run reporting (read-only landscape)
# ---------------------------------------------------------------------------

async def test_dry_run_does_not_mutate(pool):
    """dry_run_backfill_user_id must not touch any table or write audit rows."""
    from weft.db.backfill_user_id import dry_run_backfill_user_id

    null_id = await _insert_behavior(pool, project_id=None, user_id=None)
    seen_id = await _insert_behavior(pool, project_id=None, user_id="existing-user")

    before_null = await pool.fetchval(
        "SELECT user_id FROM behaviors WHERE id = $1", null_id
    )
    before_seen = await pool.fetchval(
        "SELECT user_id FROM behaviors WHERE id = $1", seen_id
    )

    report = await dry_run_backfill_user_id(pool, user_id="dry-run-probe")

    # Nothing mutated.
    after_null = await pool.fetchval(
        "SELECT user_id FROM behaviors WHERE id = $1", null_id
    )
    after_seen = await pool.fetchval(
        "SELECT user_id FROM behaviors WHERE id = $1", seen_id
    )
    assert after_null == before_null  # still NULL
    assert after_seen == before_seen  # still "existing-user"

    # No audit rows written.
    audit_count = await pool.fetchval(
        "SELECT COUNT(*) FROM audit_backfill_user_id"
    )
    assert audit_count == 0

    # Report reflects the landscape.
    assert report.proposed_user_id == "dry-run-probe"
    assert report.total_null_rows >= 1
    beh = report.per_table["behaviors"]
    assert beh["null_count"] >= 1
    assert "existing-user" in beh["distinct_user_ids"]
    assert beh["total_rows"] >= 2


async def test_dry_run_covers_all_scoped_tables(pool):
    """Report includes an entry for every scoped table, even if empty."""
    from weft.db.backfill_user_id import _TABLES, dry_run_backfill_user_id

    report = await dry_run_backfill_user_id(pool)
    expected_tables = {t[0] for t in _TABLES}
    assert set(report.per_table.keys()) == expected_tables
    for name, info in report.per_table.items():
        assert "null_count" in info
        assert "distinct_user_ids" in info
        assert "total_rows" in info


async def test_dry_run_proposed_user_id_defaults_to_get_user_id(pool):
    """When no override is passed, report shows what a real run would use."""
    from weft.db.backfill_user_id import dry_run_backfill_user_id

    report = await dry_run_backfill_user_id(pool)
    assert report.proposed_user_id == get_user_id()
