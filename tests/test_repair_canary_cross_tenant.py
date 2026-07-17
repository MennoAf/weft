"""Regression tests for the journaled cross-tenant canary repair."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts.repair_canary_cross_tenant import (
    RepairMismatch,
    apply_repair,
    inspect_repair,
    resolve_user_id,
    restore_repair,
)


async def _seed_probe_events(
    conn,
    *,
    user_id: str,
    probe_id: str,
    events: list[tuple[datetime, bool]],
) -> None:
    misses = sum(not hit for _, hit in events)
    latest = max((audited_at for audited_at, _ in events), default=None)
    await conn.execute(
        """
        INSERT INTO recall_canary
            (probe_id, memory_id, user_id, probe_text, probe_type,
             audit_count, miss_count, last_audit_at)
        VALUES ($1, $2, $3, 'repair probe', 'active', $4, $5, $6)
        """,
        probe_id,
        f"memory-{probe_id}",
        user_id,
        len(events),
        misses,
        latest,
    )
    for audited_at, hit in events:
        await conn.execute(
            """
            INSERT INTO recall_canary_audit
                (probe_id, user_id, audited_at, hit)
            VALUES ($1, $2, $3, $4)
            """,
            probe_id,
            user_id,
            audited_at,
            hit,
        )
    await conn.execute(
        """
        INSERT INTO weft_counters (name, count, updated_at)
        VALUES ('canary.miss', $1, now())
        ON CONFLICT (name) DO UPDATE
        SET count = weft_counters.count + EXCLUDED.count,
            updated_at = now()
        """,
        misses,
    )


async def test_inspect_is_read_only_and_resolves_token_owner(pool):
    owner = "repair-owner"
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO weft_tokens (token_hash, user_id, caller_mode, label)
            VALUES ('repair-token-hash', $1, 'supervisor', 'codex-desktop')
            """,
            owner,
        )
        await _seed_probe_events(
            conn,
            user_id=owner,
            probe_id="repair-probe-1",
            events=[
                (now - timedelta(hours=3), False),
                (now - timedelta(hours=2), False),
            ],
        )

        resolved = await resolve_user_id(conn, "codex-desktop")
        preview = await inspect_repair(
            conn, user_id=resolved, before=now - timedelta(hours=1)
        )

        assert preview.summary() == {
            "user_id": owner,
            "before": (now - timedelta(hours=1)).isoformat(),
            "events": 2,
            "misses": 2,
            "hits": 0,
            "affected_probes": 1,
            "global_counter_before": 2,
            "global_counter_after": 0,
        }
        assert await conn.fetchval("SELECT count(*) FROM recall_canary_audit") == 2
        assert await conn.fetchval(
            "SELECT count FROM weft_counters WHERE name = 'canary.miss'"
        ) == 2


async def test_apply_repairs_only_pre_cutoff_and_restore_preserves_newer_state(
    pool, tmp_path
):
    owner = "repair-owner"
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    old_miss = now - timedelta(hours=2)
    newer_hit = now
    journal = tmp_path / "canary-repair.json"

    async with pool.acquire() as conn:
        await _seed_probe_events(
            conn,
            user_id=owner,
            probe_id="repair-probe-2",
            events=[(old_miss, False), (newer_hit, True)],
        )

        preview = await apply_repair(
            conn,
            user_id=owner,
            before=cutoff,
            expected_events=1,
            expected_misses=1,
            journal_path=journal,
        )
        assert preview.event_count == 1
        assert journal.exists()
        assert await conn.fetchval("SELECT count(*) FROM recall_canary_audit") == 1
        row = await conn.fetchrow(
            "SELECT audit_count, miss_count, last_audit_at FROM recall_canary "
            "WHERE probe_id = 'repair-probe-2'"
        )
        assert (row["audit_count"], row["miss_count"]) == (1, 0)
        assert row["last_audit_at"] == newer_hit
        assert await conn.fetchval(
            "SELECT count FROM weft_counters WHERE name = 'canary.miss'"
        ) == 0

        restored = await restore_repair(conn, journal)
        assert restored == 1
        assert await conn.fetchval("SELECT count(*) FROM recall_canary_audit") == 2
        row = await conn.fetchrow(
            "SELECT audit_count, miss_count, last_audit_at FROM recall_canary "
            "WHERE probe_id = 'repair-probe-2'"
        )
        assert (row["audit_count"], row["miss_count"]) == (2, 1)
        assert row["last_audit_at"] == newer_hit
        assert await conn.fetchval(
            "SELECT count FROM weft_counters WHERE name = 'canary.miss'"
        ) == 1

        with pytest.raises(RepairMismatch, match="already been restored"):
            await restore_repair(conn, journal)


async def test_apply_count_mismatch_rolls_back_without_journal(pool, tmp_path):
    owner = "repair-owner"
    now = datetime.now(timezone.utc)
    journal = tmp_path / "must-not-exist.json"
    async with pool.acquire() as conn:
        await _seed_probe_events(
            conn,
            user_id=owner,
            probe_id="repair-probe-3",
            events=[(now - timedelta(hours=2), False)],
        )

        with pytest.raises(RepairMismatch, match="event count changed"):
            await apply_repair(
                conn,
                user_id=owner,
                before=now - timedelta(hours=1),
                expected_events=2,
                expected_misses=1,
                journal_path=journal,
            )

        assert not journal.exists()
        assert await conn.fetchval("SELECT count(*) FROM recall_canary_audit") == 1
        row = await conn.fetchrow(
            "SELECT audit_count, miss_count FROM recall_canary "
            "WHERE probe_id = 'repair-probe-3'"
        )
        assert (row["audit_count"], row["miss_count"]) == (1, 1)


async def test_apply_refuses_scope_containing_hits(pool, tmp_path):
    owner = "repair-owner"
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await _seed_probe_events(
            conn,
            user_id=owner,
            probe_id="repair-probe-4",
            events=[(now - timedelta(hours=2), True)],
        )

        with pytest.raises(RepairMismatch, match="hit event"):
            await apply_repair(
                conn,
                user_id=owner,
                before=now - timedelta(hours=1),
                expected_events=1,
                expected_misses=0,
                journal_path=tmp_path / "hits.json",
            )


async def test_restore_recovers_when_applied_marker_was_not_written(pool, tmp_path):
    """Database state, not a post-commit marker, proves that apply succeeded."""
    owner = "repair-owner"
    now = datetime.now(timezone.utc)
    journal = tmp_path / "marker-crash.json"
    async with pool.acquire() as conn:
        await _seed_probe_events(
            conn,
            user_id=owner,
            probe_id="repair-probe-5",
            events=[(now - timedelta(hours=2), False)],
        )
        await apply_repair(
            conn,
            user_id=owner,
            before=now - timedelta(hours=1),
            expected_events=1,
            expected_misses=1,
            journal_path=journal,
        )
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["applied_at"] = None
        journal.write_text(json.dumps(payload), encoding="utf-8")

        assert await restore_repair(conn, journal) == 1
        assert await conn.fetchval("SELECT count(*) FROM recall_canary_audit") == 1
        assert await conn.fetchval(
            "SELECT count FROM weft_counters WHERE name = 'canary.miss'"
        ) == 1
