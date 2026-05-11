"""Tests for migration 48: belief_claims table + prune-guard on episode_turns.

Schema tests mirror the v46 pattern.  Prune-guard tests are the critical
Sieve-audit requirement: turns whose id appears in any active or superseded
belief_claim's evidence_turn_ids must survive all three turn-deletion
functions in weft.episode_turns.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim_id() -> str:
    return f"belief-{uuid.uuid4().hex[:10]}"


async def _insert_claim(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    attribute: str,
    evidence_turn_ids: list[str],
    status: str = "active",
    occurred_at: datetime | None = None,
    scope: str = "global",
) -> str:
    """Low-level INSERT into belief_claims inside the current transaction.

    Caller must have already issued ``SET LOCAL app.user_id = ...`` in the
    enclosing transaction so both the GUC default and the RLS INSERT policy
    are satisfied.
    """
    claim_id = _claim_id()
    if occurred_at is None:
        occurred_at = datetime.now(timezone.utc)
    await conn.execute(
        """
        INSERT INTO belief_claims (
            claim_id, user_id, attribute, value, scope,
            evidence_turn_ids, status,
            occurred_at, source_provenance,
            detector_confidence, detector_version
        ) VALUES (
            $1, $2, $3, $4::jsonb, $5,
            $6, $7,
            $8, 'user_stated',
            1.0, 'v1'
        )
        """,
        claim_id,
        user_id,
        attribute,
        '{"v": 1}',
        scope,
        evidence_turn_ids,
        status,
        occurred_at,
    )
    return claim_id


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_belief_claims_table_exists(pool):
    """belief_claims table is created with the expected columns."""
    cols = await pool.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'belief_claims'
        ORDER BY ordinal_position
        """
    )
    by_name = {r["column_name"]: r for r in cols}
    expected = {
        "claim_id", "user_id", "attribute", "value", "scope",
        "evidence_turn_ids", "superseded_by", "status",
        "created_at", "occurred_at", "source_provenance",
        "detector_confidence", "detector_version",
    }
    assert set(by_name) == expected

    assert by_name["claim_id"]["data_type"] == "text"
    assert by_name["claim_id"]["is_nullable"] == "NO"
    assert by_name["user_id"]["data_type"] == "text"
    assert by_name["user_id"]["is_nullable"] == "NO"
    assert by_name["attribute"]["data_type"] == "text"
    assert by_name["attribute"]["is_nullable"] == "NO"
    assert by_name["value"]["data_type"] == "jsonb"
    assert by_name["value"]["is_nullable"] == "NO"
    assert by_name["scope"]["data_type"] == "text"
    assert by_name["status"]["data_type"] == "text"
    assert by_name["status"]["is_nullable"] == "NO"
    assert by_name["created_at"]["data_type"] == "timestamp with time zone"
    assert by_name["created_at"]["is_nullable"] == "NO"
    assert by_name["occurred_at"]["data_type"] == "timestamp with time zone"
    assert by_name["occurred_at"]["is_nullable"] == "NO"
    assert by_name["source_provenance"]["data_type"] == "text"
    assert by_name["source_provenance"]["is_nullable"] == "NO"
    assert by_name["detector_confidence"]["data_type"] == "real"
    assert by_name["detector_confidence"]["is_nullable"] == "NO"
    assert by_name["detector_version"]["data_type"] == "text"
    assert by_name["detector_version"]["is_nullable"] == "NO"


@pytest.mark.asyncio
async def test_belief_claims_primary_key(pool):
    """PK is claim_id."""
    rows = await pool.fetch(
        """
        SELECT a.attname AS column_name
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = 'belief_claims'::regclass
          AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """
    )
    assert [r["column_name"] for r in rows] == ["claim_id"]


@pytest.mark.asyncio
async def test_belief_claims_indexes_exist(pool):
    """All four named indexes are present."""
    for idx_name in (
        "idx_belief_claims_current",
        "idx_belief_claims_chain",
        "idx_belief_claims_attribute_prefix",
        "idx_belief_claims_evidence_gin",
    ):
        exists = await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
            idx_name,
        )
        assert exists, f"missing index {idx_name}"


@pytest.mark.asyncio
async def test_belief_claims_partial_unique_active(pool):
    """Two 'active' rows for the same (user_id, attribute, scope) raises UniqueViolation.

    Superseding the first (setting status = 'superseded') lets the second
    insert succeed — the partial unique index only covers status = 'active'.
    """
    user_id = "test-user-default"
    now = datetime.now(timezone.utc)

    # Phase 1: commit the first 'active' claim in its own transaction.
    cid1: str = ""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            cid1 = await _insert_claim(
                conn, user_id=user_id,
                attribute="sleep.recent_hours", evidence_turn_ids=["et-aaa"],
                occurred_at=now - timedelta(hours=2),
            )

    # Phase 2: attempt a duplicate 'active' row — must fail.
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            with pytest.raises(asyncpg.UniqueViolationError):
                await _insert_claim(
                    conn, user_id=user_id,
                    attribute="sleep.recent_hours", evidence_turn_ids=["et-bbb"],
                    occurred_at=now,
                )
        # Transaction was aborted — the failed insert is rolled back.

    # Phase 3: supersede the first, then insert the second — must succeed.
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            cid2 = _claim_id()
            await conn.execute(
                "UPDATE belief_claims SET status = 'superseded' WHERE claim_id = $1",
                cid1,
            )
            # Now the second 'active' insert must succeed.
            await conn.execute(
                """
                INSERT INTO belief_claims (
                    claim_id, user_id, attribute, value, scope,
                    evidence_turn_ids, status, occurred_at,
                    source_provenance, detector_confidence, detector_version
                ) VALUES (
                    $1, $2, 'sleep.recent_hours', '{"v":2}'::jsonb, 'global',
                    '{et-bbb}', 'active', $3,
                    'user_stated', 1.0, 'v1'
                )
                """,
                cid2, user_id, now,
            )

    count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE attribute = 'sleep.recent_hours'",
    )
    assert count == 2


@pytest.mark.asyncio
async def test_belief_claims_status_check(pool):
    """Inserting with status = 'bogus' raises a CheckViolationError."""
    user_id = "test-user-default"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO belief_claims (
                        claim_id, user_id, attribute, value, scope,
                        evidence_turn_ids, status, occurred_at,
                        source_provenance, detector_confidence, detector_version
                    ) VALUES (
                        $1, $2, 'attr', '{"v":1}'::jsonb, 'global',
                        '{et-xxx}', 'bogus', $3,
                        'user_stated', 1.0, 'v1'
                    )
                    """,
                    _claim_id(), user_id, now,
                )


@pytest.mark.asyncio
async def test_belief_claims_source_provenance_check(pool):
    """Inserting with an invalid source_provenance raises CheckViolationError."""
    user_id = "test-user-default"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO belief_claims (
                        claim_id, user_id, attribute, value, scope,
                        evidence_turn_ids, status, occurred_at,
                        source_provenance, detector_confidence, detector_version
                    ) VALUES (
                        $1, $2, 'attr', '{"v":1}'::jsonb, 'global',
                        '{et-xxx}', 'active', $3,
                        'robot_decided', 1.0, 'v1'
                    )
                    """,
                    _claim_id(), user_id, now,
                )


@pytest.mark.asyncio
async def test_belief_claims_evidence_non_empty_check(pool):
    """Inserting with evidence_turn_ids = '{}' raises CheckViolationError."""
    user_id = "test-user-default"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO belief_claims (
                        claim_id, user_id, attribute, value, scope,
                        evidence_turn_ids, status, occurred_at,
                        source_provenance, detector_confidence, detector_version
                    ) VALUES (
                        $1, $2, 'attr', '{"v":1}'::jsonb, 'global',
                        '{}', 'active', $3,
                        'user_stated', 1.0, 'v1'
                    )
                    """,
                    _claim_id(), user_id, now,
                )


@pytest.mark.asyncio
async def test_belief_claims_rls_enabled(pool):
    """RLS is enabled on belief_claims."""
    enabled = await pool.fetchval(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'belief_claims'"
    )
    assert enabled is True


@pytest.mark.asyncio
async def test_v48_migration_idempotent(pool):
    """Running migration 48 SQL twice does not error."""
    from weft.db.migrations import MIGRATIONS

    v48_sql = [sql for version, _, sql in MIGRATIONS if version == 48]
    assert len(v48_sql) == 1, "expected migration 48 in MIGRATIONS list"

    # Migration already ran via fixture. Run the raw SQL again — must not raise.
    await pool.execute(v48_sql[0])


# ---------------------------------------------------------------------------
# Prune-guard tests
# ---------------------------------------------------------------------------


async def _setup_graduated_episode(
    pool: asyncpg.Pool,
    *,
    user_id: str = "test-user-default",
    days_ago: int = 60,
) -> tuple:
    """Create a graduated episode backdated by days_ago.

    Returns (episode, turn_a, turn_b) where both turns have
    importance_score=0.1 (below any normal threshold).
    """
    from weft.episode_turns import append_turn
    from weft.episodes import create_episode, graduate_episode
    from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole

    ep = await create_episode(pool, EpisodeCreate(title="prune-guard-test"))
    turn_a = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="turn A"),
    )
    turn_b = await append_turn(
        pool, EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content="turn B"),
    )

    # Graduate the episode so prune functions can touch it.
    ep, _ = await graduate_episode(pool, ep.id)

    # Backdate ended_at so the TTL condition fires.
    ended_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
    await pool.execute(
        "UPDATE episodes SET ended_at = $1 WHERE id = $2",
        ended_at, ep.id,
    )

    # Set low importance scores so both turns qualify for deletion.
    await pool.execute(
        "UPDATE episode_turns SET importance_score = 0.1 WHERE episode_id = $1",
        ep.id,
    )

    return ep, turn_a, turn_b


async def _anchor_turn(
    pool: asyncpg.Pool,
    turn_id: str,
    *,
    user_id: str = "test-user-default",
    status: str = "active",
) -> str:
    """Insert a belief_claim anchoring turn_id and return the claim_id."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            claim_id = await _insert_claim(
                conn,
                user_id=user_id,
                attribute=f"test.attr-{uuid.uuid4().hex[:6]}",
                evidence_turn_ids=[turn_id],
                status=status,
            )
    return claim_id


@pytest.mark.asyncio
async def test_delete_turns_below_importance_skips_claim_anchored(pool):
    """delete_turns_below_importance skips turns anchored by an active claim."""
    from weft.episode_turns import delete_turns_below_importance

    _, turn_a, turn_b = await _setup_graduated_episode(pool)
    await _anchor_turn(pool, turn_a.id, status="active")

    deleted = await delete_turns_below_importance(pool, threshold=0.5, older_than_days=30)

    # turn_b (not anchored) should be deleted; turn_a (anchored) survives.
    assert deleted >= 1
    remaining = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_a.id,
    )
    assert remaining == 1, "anchored turn must survive"
    gone = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_b.id,
    )
    assert gone == 0, "unanchored turn must be pruned"


@pytest.mark.asyncio
async def test_delete_turns_for_graduated_episode_skips_claim_anchored(pool):
    """delete_turns_for_graduated_episode skips turns anchored by an active claim."""
    from weft.episode_turns import delete_turns_for_graduated_episode

    _, turn_a, turn_b = await _setup_graduated_episode(pool)
    await _anchor_turn(pool, turn_a.id, status="active")

    deleted = await delete_turns_for_graduated_episode(pool, older_than_days=30)

    assert deleted >= 1
    remaining = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_a.id,
    )
    assert remaining == 1, "anchored turn must survive"
    gone = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_b.id,
    )
    assert gone == 0, "unanchored turn must be pruned"


@pytest.mark.asyncio
async def test_delete_turns_after_graduation_skips_claim_anchored(pool):
    """delete_turns_after_graduation score-aware path skips anchored turn."""
    from weft.episode_turns import delete_turns_after_graduation

    _, turn_a, turn_b = await _setup_graduated_episode(pool)
    await _anchor_turn(pool, turn_a.id, status="active")

    result = await delete_turns_after_graduation(
        pool,
        high_threshold=0.5,
        ttl_days_scored=30,
        ttl_days_no_score=30,
    )

    assert result["scored_deleted"] >= 1
    remaining = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_a.id,
    )
    assert remaining == 1, "anchored turn must survive score-aware path"
    gone = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_b.id,
    )
    assert gone == 0, "unanchored turn must be pruned by score-aware path"


@pytest.mark.asyncio
async def test_delete_turns_after_graduation_no_score_path_skips_anchored(pool):
    """delete_turns_after_graduation no-score path skips a NULL-score anchored turn."""
    from weft.episode_turns import append_turn, delete_turns_after_graduation
    from weft.episodes import create_episode, graduate_episode
    from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole

    # Create a second episode with NULL-score turns.
    ep2 = await create_episode(pool, EpisodeCreate(title="no-score-prune-guard"))
    turn_null = await append_turn(
        pool,
        EpisodeTurnCreate(episode_id=ep2.id, role=TurnRole.user, content="null-score turn"),
    )
    turn_null_b = await append_turn(
        pool,
        EpisodeTurnCreate(episode_id=ep2.id, role=TurnRole.user, content="null-score turn B"),
    )

    ep2, _ = await graduate_episode(pool, ep2.id)
    ended_at = datetime.now(timezone.utc) - timedelta(days=60)
    await pool.execute(
        "UPDATE episodes SET ended_at = $1 WHERE id = $2",
        ended_at, ep2.id,
    )
    # Explicitly NULL out importance_score so no-score path fires.
    await pool.execute(
        "UPDATE episode_turns SET importance_score = NULL WHERE episode_id = $1",
        ep2.id,
    )

    await _anchor_turn(pool, turn_null.id, status="active")

    result = await delete_turns_after_graduation(
        pool,
        high_threshold=0.5,
        ttl_days_scored=30,
        ttl_days_no_score=30,
    )

    assert result["no_score_deleted"] >= 1
    remaining = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_null.id,
    )
    assert remaining == 1, "anchored NULL-score turn must survive no-score path"
    gone = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_null_b.id,
    )
    assert gone == 0, "unanchored NULL-score turn must be pruned"


@pytest.mark.asyncio
async def test_prune_skips_superseded_claim_anchored(pool):
    """A superseded claim still anchors evidence turns (provenance is immutable)."""
    from weft.episode_turns import delete_turns_below_importance

    _, turn_a, turn_b = await _setup_graduated_episode(pool)
    # Anchor with a *superseded* (not active) claim.
    await _anchor_turn(pool, turn_a.id, status="superseded")

    deleted = await delete_turns_below_importance(pool, threshold=0.5, older_than_days=30)

    assert deleted >= 1
    remaining = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_a.id,
    )
    assert remaining == 1, "superseded-claim-anchored turn must survive"
    gone = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_b.id,
    )
    assert gone == 0, "unanchored turn must be pruned"


@pytest.mark.asyncio
async def test_prune_does_NOT_skip_retracted_claim_anchored(pool):
    """A retracted claim does NOT anchor evidence turns — the turn IS deleted."""
    from weft.episode_turns import delete_turns_below_importance

    _, turn_a, turn_b = await _setup_graduated_episode(pool)
    # Anchor with a *retracted* claim — should NOT protect the turn.
    await _anchor_turn(pool, turn_a.id, status="retracted")

    deleted = await delete_turns_below_importance(pool, threshold=0.5, older_than_days=30)

    # Both turns (turn_a and turn_b) should be deleted since retracted claims
    # do not anchor.
    assert deleted >= 2
    remaining_a = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_a.id,
    )
    remaining_b = await pool.fetchval(
        "SELECT count(*) FROM episode_turns WHERE id = $1", turn_b.id,
    )
    assert remaining_a == 0, "retracted-claim turn must NOT be protected"
    assert remaining_b == 0, "unanchored turn must be pruned"
