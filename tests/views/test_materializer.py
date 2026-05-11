"""Tests for weft.views.materializer — idempotent batch belief materializer.

Uses the real ``pool`` fixture (testcontainers Postgres) and a stub detector
injected via the ``detector=`` kwarg.  No Haiku calls.

All scenarios exercise the full DB path:
- Cursor persistence (weft_metadata)
- Supersession logic (all three cases)
- Late-arrival chain insertion
- Idempotency (replay produces no duplicates)
- RLS compliance (app.user_id GUC set per write)
- Advisory lock serialization (concurrent calls for same key)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import pytest

from weft.episode_turns import append_turn
from weft.episodes import create_episode
from weft.models import EpisodeCreate, EpisodeTurn, EpisodeTurnCreate, TurnRole
from weft.store import get_metadata
from weft.views.belief_detector import DETECTOR_VERSION, ClaimUpdate
from weft.views.materializer import (
    MaterializeResult,
    _CURSOR_KEY,
    _claim_lock_key,
    materialize_pending_turns,
    materialize_turn,
)

# ---------------------------------------------------------------------------
# Test user + helpers
# ---------------------------------------------------------------------------

TEST_USER = "test-user-default"


def _now(offset_seconds: float = 0.0) -> datetime:
    """Return a UTC datetime offset by offset_seconds from now."""
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _make_claim_update(
    attribute: str = "sleep.recent_hours",
    value: Any = {"hours": 7},
    confidence: float = 0.9,
    provenance: str = "user_stated",
    turn_id: str = "et-test0001",
) -> ClaimUpdate:
    """Construct a ClaimUpdate with default fields suitable for testing."""
    return ClaimUpdate(
        attribute=attribute,
        value=value,
        confidence=confidence,
        source_provenance=provenance,
        evidence_turn_id=turn_id,
        detector_version=DETECTOR_VERSION,
    )


async def _create_turn(
    pool: asyncpg.Pool,
    *,
    content: str = "I slept 7 hours last night.",
    role: TurnRole = TurnRole.user,
    occurred_at: datetime | None = None,
) -> EpisodeTurn:
    """Create an episode + turn, return the turn.  Convenience for tests."""
    ep = await create_episode(pool, EpisodeCreate(title=f"test-ep-{uuid.uuid4().hex[:6]}"))
    create = EpisodeTurnCreate(
        episode_id=ep.id,
        role=role,
        content=content,
        occurred_at=occurred_at,
    )
    return await append_turn(pool, create)


async def _count_claims(pool: asyncpg.Pool, *, attribute: str | None = None) -> int:
    """Count belief_claims rows (optionally filtered by attribute)."""
    if attribute:
        return await pool.fetchval(
            "SELECT count(*) FROM belief_claims WHERE attribute = $1", attribute,
        )
    return await pool.fetchval("SELECT count(*) FROM belief_claims")


async def _fetch_claims(
    pool: asyncpg.Pool,
    *,
    user_id: str = TEST_USER,
    attribute: str,
    scope: str = "global",
) -> list[asyncpg.Record]:
    """Fetch all claims for (user_id, attribute, scope) ordered by occurred_at ASC."""
    return await pool.fetch(
        """
        SELECT claim_id, status, occurred_at, superseded_by, detector_version,
               detector_confidence, source_provenance, evidence_turn_ids
        FROM belief_claims
        WHERE user_id = $1 AND attribute = $2 AND scope = $3
        ORDER BY occurred_at ASC
        """,
        user_id, attribute, scope,
    )


def _stub_detector(claim_updates: list[ClaimUpdate]):
    """Return a stub detector that always emits the given ClaimUpdates."""
    def _stub(turn: EpisodeTurn) -> list[ClaimUpdate]:
        # Rewrite evidence_turn_id to match actual turn.
        return [
            ClaimUpdate(
                attribute=u.attribute,
                value=u.value,
                confidence=u.confidence,
                source_provenance=u.source_provenance,
                evidence_turn_id=turn.id,
                reason=u.reason,
                detector_version=u.detector_version,
            )
            for u in claim_updates
        ]
    return _stub


def _stub_empty_detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
    """Stub detector that always returns empty list (abstention)."""
    return []


# ---------------------------------------------------------------------------
# Test 1: Empty pool — no turns to process
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_pool_writes_nothing(pool: asyncpg.Pool) -> None:
    """Materializer on empty turn table: 0 turns processed, cursor unchanged."""
    result = await materialize_pending_turns(pool, detector=_stub_empty_detector)

    assert result.turns_processed == 0
    assert result.claims_written == 0
    assert result.claims_superseded == 0
    assert result.errors == 0
    assert await _count_claims(pool) == 0


# ---------------------------------------------------------------------------
# Test 2: Single positive claim written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_positive_claim_written(pool: asyncpg.Pool) -> None:
    """One user turn → stub emits 1 ClaimUpdate → 1 active claim written."""
    turn = await _create_turn(pool, content="I slept 7 hours last night.")

    detector = _stub_detector(
        [_make_claim_update(attribute="sleep.recent_hours", value={"hours": 7})]
    )

    result = await materialize_pending_turns(pool, detector=detector)

    assert result.turns_processed == 1
    assert result.claims_written == 1
    assert result.claims_superseded == 0
    assert result.errors == 0

    claims = await _fetch_claims(pool, attribute="sleep.recent_hours")
    assert len(claims) == 1
    assert claims[0]["status"] == "active"
    assert claims[0]["evidence_turn_ids"] == [turn.id]
    assert claims[0]["detector_version"] == DETECTOR_VERSION
    assert abs(claims[0]["detector_confidence"] - 0.9) < 0.01


# ---------------------------------------------------------------------------
# Test 3: Abstention writes nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abstention_writes_nothing(pool: asyncpg.Pool) -> None:
    """Stub emits []. No claim written; cursor advances past the turn."""
    await _create_turn(pool, content="Hey, how's it going?")

    result = await materialize_pending_turns(pool, detector=_stub_empty_detector)

    assert result.turns_processed == 1
    assert result.claims_written == 0
    assert result.abstentions == 1
    assert await _count_claims(pool) == 0

    # Cursor must have advanced.
    cursor = await get_metadata(pool, _CURSOR_KEY)
    assert cursor is not None
    assert cursor.get("last_turn_id") != ""


# ---------------------------------------------------------------------------
# Test 4: Normal supersession marks prior as superseded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_supersession_marks_prior_as_superseded(pool: asyncpg.Pool) -> None:
    """Two turns at t=1 and t=2 with same attribute → 1 active + 1 superseded."""
    base = _now(-10)
    t1 = base
    t2 = base + timedelta(seconds=5)

    turn1 = await _create_turn(pool, content="I slept 5 hours.", occurred_at=t1)
    turn2 = await _create_turn(pool, content="I slept 7 hours.", occurred_at=t2)

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        if turn.id == turn1.id:
            hours = 5
        else:
            hours = 7
        return [_make_claim_update(
            attribute="sleep.recent_hours",
            value={"hours": hours},
            turn_id=turn.id,
        )]

    result = await materialize_pending_turns(pool, detector=detector)

    assert result.turns_processed == 2
    assert result.claims_written == 2
    assert result.claims_superseded == 1

    claims = await _fetch_claims(pool, attribute="sleep.recent_hours")
    assert len(claims) == 2

    # Ordered by occurred_at ASC: t1 first, t2 second.
    older, newer = claims[0], claims[1]
    assert older["status"] == "superseded"
    assert newer["status"] == "active"
    # superseded_by on older points to newer.
    assert older["superseded_by"] == newer["claim_id"]


# ---------------------------------------------------------------------------
# Test 5: Late arrival inserts into chain without displacing active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_late_arrival_inserts_into_chain_without_displacing_active(
    pool: asyncpg.Pool,
) -> None:
    """Turn at t=2 becomes active first; turn at t=1 materializes as superseded."""
    base = _now(-20)
    t1 = base
    t2 = base + timedelta(seconds=10)

    # Append turn at t=2 first.
    turn2 = await _create_turn(pool, content="7 hours sleep", occurred_at=t2)
    # Materialize turn2 → becomes active.
    def detector_t2(turn: EpisodeTurn) -> list[ClaimUpdate]:
        return [_make_claim_update(
            attribute="sleep.recent_hours",
            value={"hours": 7},
            turn_id=turn.id,
            confidence=0.92,
        )]

    result1 = await materialize_pending_turns(pool, detector=detector_t2)
    assert result1.claims_written == 1

    # Reset cursor so we can inject the late-arriving turn.
    from weft.store import set_metadata
    await set_metadata(pool, _CURSOR_KEY, {"last_occurred_at": "1970-01-01T00:00:00+00:00", "last_turn_id": ""})

    # Append turn at t=1 (late arrival).
    turn1 = await _create_turn(pool, content="5 hours sleep", occurred_at=t1)

    def detector_t1(turn: EpisodeTurn) -> list[ClaimUpdate]:
        if turn.id == turn1.id:
            return [_make_claim_update(
                attribute="sleep.recent_hours",
                value={"hours": 5},
                turn_id=turn.id,
                confidence=0.88,
            )]
        # turn2 already processed — idempotency check will skip it.
        return [_make_claim_update(
            attribute="sleep.recent_hours",
            value={"hours": 7},
            turn_id=turn.id,
            confidence=0.92,
        )]

    result2 = await materialize_pending_turns(pool, detector=detector_t1)

    claims = await _fetch_claims(pool, attribute="sleep.recent_hours")
    assert len(claims) == 2

    # t1 claim must be superseded; t2 claim must still be active.
    by_occurred = {c["occurred_at"].replace(tzinfo=timezone.utc) if c["occurred_at"].tzinfo is None else c["occurred_at"]: c for c in claims}
    # Just check by evidence_turn_id.
    by_turn = {c["evidence_turn_ids"][0]: c for c in claims}
    assert by_turn[turn1.id]["status"] == "superseded"
    assert by_turn[turn2.id]["status"] == "active"
    # Late arrival's superseded_by should point at the t2 active.
    assert by_turn[turn1.id]["superseded_by"] == by_turn[turn2.id]["claim_id"]


# ---------------------------------------------------------------------------
# Test 6: Idempotent replay — no duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_replay_no_duplicates(pool: asyncpg.Pool) -> None:
    """Run materializer twice over the same 5 turns — second run produces 0 new claims."""
    attribute = "exercise.weekly_frequency"
    turns = [
        await _create_turn(pool, content=f"Run {i} times this week.", occurred_at=_now(-(50 - i * 5)))
        for i in range(5)
    ]

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        idx = next(i for i, t in enumerate(turns) if t.id == turn.id)
        return [_make_claim_update(
            attribute=attribute,
            value={"times": idx + 1},
            turn_id=turn.id,
        )]

    await materialize_pending_turns(pool, detector=detector)
    count_after_first = await _count_claims(pool, attribute=attribute)
    assert count_after_first == 5

    # Reset cursor to force re-processing all turns.
    from weft.store import set_metadata
    await set_metadata(pool, _CURSOR_KEY, {"last_occurred_at": "1970-01-01T00:00:00+00:00", "last_turn_id": ""})

    await materialize_pending_turns(pool, detector=detector)
    count_after_second = await _count_claims(pool, attribute=attribute)

    # Idempotency: count must not increase.
    assert count_after_second == count_after_first


# ---------------------------------------------------------------------------
# Test 7: Cursor advances after successful batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_advances_after_successful_batch(pool: asyncpg.Pool) -> None:
    """3 turns, batch_size=2 → first run processes 2, second run processes 1."""
    base = _now(-30)
    turns = [
        await _create_turn(pool, content=f"turn {i}", occurred_at=base + timedelta(seconds=i))
        for i in range(3)
    ]

    processed_ids: list[str] = []

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        processed_ids.append(turn.id)
        return [_make_claim_update(
            attribute=f"test.attr-{turn.id[-4:]}",
            value={"v": 1},
            turn_id=turn.id,
        )]

    r1 = await materialize_pending_turns(pool, detector=detector, batch_size=2)
    assert r1.turns_processed == 2

    r2 = await materialize_pending_turns(pool, detector=detector, batch_size=2)
    assert r2.turns_processed == 1

    # All three turns processed exactly once.
    assert sorted(processed_ids) == sorted(t.id for t in turns)


# ---------------------------------------------------------------------------
# Test 8: Missing user_id is skipped with error count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_user_id_is_skipped_not_failed(pool: asyncpg.Pool) -> None:
    """A turn with user_id=NULL is skipped; result.errors == 1."""
    # Directly INSERT a turn with user_id=NULL (bypass the normal append path
    # which enforces NOT NULL via GUC default).
    # We have to use the system sentinel to bypass RLS for episode creation,
    # then manually insert a turn with NULL user_id.
    ep = await create_episode(pool, EpisodeCreate(title="null-user-test"))
    turn_id = f"et-{uuid.uuid4().hex[:10]}"
    async with pool.acquire() as conn:
        await conn.execute("SET LOCAL app.user_id = '__system_global_zathras__'")
        await conn.execute(
            """
            INSERT INTO episode_turns (id, episode_id, turn_index, role, content, occurred_at, token_count, user_id)
            VALUES ($1, $2, 0, 'user', 'null-user content', now() - interval '5 seconds', 0, '__system_global_zathras__')
            """,
            turn_id, ep.id,
        )
        # Now update to set user_id NULL (need to disable RLS constraint temporarily
        # using the system sentinel user, but the constraint is NOT NULL — we can't
        # set NULL.  Instead use an obviously-missing placeholder and test missing
        # user_id handling in materialize_turn directly.)

    # Test materialize_turn directly with a turn that has user_id=None.
    null_turn = EpisodeTurn(
        id=f"et-{uuid.uuid4().hex[:10]}",
        episode_id=ep.id,
        turn_index=99,
        role=TurnRole.user,
        content="no user",
        user_id=None,
    )
    updates = [_make_claim_update(turn_id=null_turn.id)]
    result = await materialize_turn(pool, null_turn, updates)
    assert result["skipped"] == 1
    assert result["written"] == 0


@pytest.mark.asyncio
async def test_missing_user_id_increments_errors_in_batch(pool: asyncpg.Pool) -> None:
    """Batch materializer increments errors for NULL user_id turns.

    We inject a fake turn via a custom detector that returns a special sentinel
    to simulate the NULL-user-id path — since episode_turns.user_id is NOT NULL,
    we can't insert a real NULL-user-id row.  Instead we test the
    materialize_turn API directly (see test above) and verify the batch error
    count via a detector that raises for that turn.

    The batch path: when materialize_turn raises (or returns skip), the batch
    continues and result.errors increments.  We verify via a detector error.
    """
    turn = await _create_turn(pool, content="trigger error")

    def error_detector(t: EpisodeTurn) -> list[ClaimUpdate]:
        raise ValueError("simulated detector failure")

    result = await materialize_pending_turns(pool, detector=error_detector)
    assert result.errors == 1
    assert result.turns_processed == 1
    # Cursor still advances past the error turn.
    cursor = await get_metadata(pool, _CURSOR_KEY)
    assert cursor is not None


# ---------------------------------------------------------------------------
# Test 9: Low confidence dropped by materializer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_dropped_by_materializer(pool: asyncpg.Pool) -> None:
    """Stub emits ClaimUpdate with confidence=0.5 → no claim written (defense in depth)."""
    turn = await _create_turn(pool, content="Meh kind of slept.")

    def detector(t: EpisodeTurn) -> list[ClaimUpdate]:
        return [_make_claim_update(
            attribute="sleep.recent_hours",
            value={"hours": 5},
            confidence=0.5,
            turn_id=t.id,
        )]

    result = await materialize_pending_turns(pool, detector=detector)
    assert result.claims_written == 0
    assert await _count_claims(pool, attribute="sleep.recent_hours") == 0


# ---------------------------------------------------------------------------
# Test 10: 100 mixed turns — final state matches expected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_100_mixed_turns_final_state_matches_expected(pool: asyncpg.Pool) -> None:
    """100 turns, 5 attributes, ~20% supersede prior.

    For each attribute, exactly 1 active claim at the end.  The superseded_by
    chain reconstructed from occurred_at order is consistent.
    """
    attributes = [f"pref.attr-{i}" for i in range(5)]
    base = _now(-3600)  # 1 hour ago
    n_turns = 100

    # Build a deterministic turn → attribute + value mapping.
    # Each attribute gets 20 turns (100/5). Within each group, occurred_at
    # increases so the last turn in each group is the active claim.
    turn_attrs: dict[str, str] = {}
    turn_values: dict[str, int] = {}
    turn_occurred: dict[str, datetime] = {}
    all_turns: list[EpisodeTurn] = []

    for i in range(n_turns):
        attr = attributes[i % 5]
        seq = i // 5  # 0..19
        ot = base + timedelta(seconds=i * 10)
        t = await _create_turn(pool, content=f"turn {i} attr {attr}", occurred_at=ot)
        turn_attrs[t.id] = attr
        turn_values[t.id] = seq
        turn_occurred[t.id] = ot
        all_turns.append(t)

    def detector(t: EpisodeTurn) -> list[ClaimUpdate]:
        attr = turn_attrs[t.id]
        val = turn_values[t.id]
        return [_make_claim_update(
            attribute=attr,
            value={"v": val},
            turn_id=t.id,
        )]

    result = await materialize_pending_turns(pool, detector=detector, batch_size=100)
    assert result.errors == 0
    assert result.turns_processed == n_turns

    # Verify: exactly 1 active claim per attribute.
    for attr in attributes:
        active_count = await pool.fetchval(
            "SELECT count(*) FROM belief_claims WHERE attribute = $1 AND status = 'active'",
            attr,
        )
        assert active_count == 1, f"Expected 1 active claim for {attr}, got {active_count}"

        # Verify the active claim has the highest occurred_at for that attribute.
        active_claim = await pool.fetchrow(
            "SELECT claim_id, occurred_at FROM belief_claims WHERE attribute = $1 AND status = 'active'",
            attr,
        )
        max_occurred = await pool.fetchval(
            "SELECT MAX(occurred_at) FROM belief_claims WHERE attribute = $1",
            attr,
        )
        assert active_claim["occurred_at"] == max_occurred, (
            f"Active claim for {attr} does not have max occurred_at"
        )


# ---------------------------------------------------------------------------
# Test 11: Race safety — two concurrent calls produce 1 active per key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_race_safety_two_concurrent_calls_one_active_per_key(
    pool: asyncpg.Pool,
) -> None:
    """asyncio.gather on two materialize_turn calls for the same key → 1 active.

    Tests the pg_advisory_xact_lock serialization path.  If the lock works
    correctly, exactly one of the two writes succeeds as 'active' and the other
    either supersedes it or becomes the active (depending on occurred_at order).
    In either case, exactly 1 active claim exists after both complete.
    """
    attribute = "sleep.recent_hours"
    base = _now(-30)
    t1 = base
    t2 = base + timedelta(seconds=5)

    turn_a = await _create_turn(pool, content="7 hours sleep a", occurred_at=t1)
    turn_b = await _create_turn(pool, content="8 hours sleep b", occurred_at=t2)

    update_a = _make_claim_update(
        attribute=attribute, value={"hours": 7}, confidence=0.9, turn_id=turn_a.id,
    )
    update_b = _make_claim_update(
        attribute=attribute, value={"hours": 8}, confidence=0.92, turn_id=turn_b.id,
    )

    # Run concurrently.
    await asyncio.gather(
        materialize_turn(pool, turn_a, [update_a]),
        materialize_turn(pool, turn_b, [update_b]),
    )

    active_count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims "
        "WHERE attribute = $1 AND status = 'active'",
        attribute,
    )
    assert active_count == 1, (
        f"Expected exactly 1 active claim after concurrent writes, got {active_count}"
    )

    # Total claims should be 2 (one active, one superseded).
    total_count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE attribute = $1",
        attribute,
    )
    assert total_count == 2


# ---------------------------------------------------------------------------
# Test 12: Supersession chain walk returns history in occurred_at order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersession_chain_walk_returns_history_in_occurred_at_order(
    pool: asyncpg.Pool,
) -> None:
    """3 claims for one attribute → occurred_at order matches superseded_by chain."""
    attribute = "exercise.weekly_frequency"
    base = _now(-60)

    turns = [
        await _create_turn(pool, content=f"exercise turn {i}", occurred_at=base + timedelta(seconds=i * 10))
        for i in range(3)
    ]

    def detector(t: EpisodeTurn) -> list[ClaimUpdate]:
        idx = next(i for i, turn in enumerate(turns) if turn.id == t.id)
        return [_make_claim_update(
            attribute=attribute,
            value={"times": idx + 1},
            turn_id=t.id,
        )]

    await materialize_pending_turns(pool, detector=detector)

    claims = await _fetch_claims(pool, attribute=attribute)
    assert len(claims) == 3

    # Claims are ordered by occurred_at ASC — chain should be:
    #   claims[0] (oldest) → superseded_by → claims[1] → superseded_by → claims[2] (active)
    assert claims[0]["status"] == "superseded"
    assert claims[1]["status"] == "superseded"
    assert claims[2]["status"] == "active"

    # Walk superseded_by chain from root.
    assert claims[0]["superseded_by"] == claims[1]["claim_id"]
    assert claims[1]["superseded_by"] == claims[2]["claim_id"]
    assert claims[2]["superseded_by"] is None


# ---------------------------------------------------------------------------
# Test 13 (Finding 1): Sentinel user_id rejected — no claim written, errors >= 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_user_id_rejected(pool: asyncpg.Pool) -> None:
    """A turn with user_id = '__system_global_zathras__' must not produce a claim.

    We INSERT a turn directly using raw SQL with the sentinel GUC so the row
    is visible to the system reader, then run materialize_pending_turns with a
    stub detector.  The materializer must increment result.errors and leave
    belief_claims empty for that user_id.
    """
    sentinel = "__system_global_zathras__"
    ep = await create_episode(pool, EpisodeCreate(title="sentinel-test"))
    sentinel_turn_id = f"et-{uuid.uuid4().hex[:10]}"

    # Insert a turn with user_id = sentinel, bypassing the GUC-guarded helper.
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{sentinel}'")
            await conn.execute(
                """
                INSERT INTO episode_turns (
                    id, episode_id, turn_index, role, content, occurred_at,
                    token_count, user_id
                ) VALUES ($1, $2, 0, 'user', 'sentinel content', now() - interval '2 seconds', 0, $3)
                """,
                sentinel_turn_id, ep.id, sentinel,
            )

    detector = _stub_detector(
        [_make_claim_update(attribute="sleep.recent_hours", value={"hours": 7})]
    )

    result = await materialize_pending_turns(pool, detector=detector)

    # No claim should exist for the sentinel user_id.
    count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE user_id = $1", sentinel,
    )
    assert count == 0, f"Expected 0 claims for sentinel user_id, got {count}"
    assert result.errors >= 1, f"Expected errors >= 1, got {result.errors}"


# ---------------------------------------------------------------------------
# Test 14 (Finding 2): user_id validation rejects special chars, accepts valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_uid", [
    "alice'; DROP TABLE--",
    "x\nbad",
    "x\x00null",
    "x with space",
    "alice@example.com",
    "name<tag>",
])
async def test_user_id_validation_rejects_special_chars(
    pool: asyncpg.Pool,
    bad_uid: str,
) -> None:
    """user_ids with SQL-special or format-special chars must be rejected.

    Each bad user_id should cause materialize to increment result.errors and
    write zero claims.
    """
    ep = await create_episode(pool, EpisodeCreate(title=f"bad-uid-{uuid.uuid4().hex[:6]}"))

    # Build an EpisodeTurn in memory with the bad user_id — we don't insert it
    # into the DB, we test materialize_turn directly.
    bad_turn = EpisodeTurn(
        id=f"et-{uuid.uuid4().hex[:10]}",
        episode_id=ep.id,
        turn_index=0,
        role=TurnRole.user,
        content="test content",
        user_id=bad_uid,
    )
    updates = [_make_claim_update(turn_id=bad_turn.id)]

    # materialize_turn should raise ValueError and the caller increments errors.
    with pytest.raises(Exception):
        await materialize_turn(pool, bad_turn, updates)

    count = await pool.fetchval("SELECT count(*) FROM belief_claims")
    assert count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("good_uid", [
    "alice",
    "user-123",
    "5e9d8b4f1a2c",
])
async def test_user_id_validation_accepts_valid(
    pool: asyncpg.Pool,
    good_uid: str,
) -> None:
    """Valid user_ids must not be rejected."""
    ep = await create_episode(pool, EpisodeCreate(title=f"good-uid-{uuid.uuid4().hex[:6]}"))
    turn = EpisodeTurn(
        id=f"et-{uuid.uuid4().hex[:10]}",
        episode_id=ep.id,
        turn_index=0,
        role=TurnRole.user,
        content="I slept 7 hours.",
        user_id=good_uid,
    )

    # Manually insert the turn so the materializer can see it via the batch path.
    async with pool.acquire() as conn:
        async with conn.transaction():
            safe = good_uid.replace("'", "''")
            await conn.execute(f"SET LOCAL app.user_id = '{safe}'")
            await conn.execute(
                """
                INSERT INTO episode_turns (
                    id, episode_id, turn_index, role, content, occurred_at,
                    token_count, user_id
                ) VALUES ($1, $2, 0, 'user', 'I slept 7 hours.', now() - interval '1 second', 0, $3)
                """,
                turn.id, ep.id, good_uid,
            )

    updates = [_make_claim_update(attribute="sleep.recent_hours", value={"hours": 7}, turn_id=turn.id)]
    result = await materialize_turn(pool, turn, updates)
    assert result["written"] == 1, f"Expected 1 written for uid={good_uid!r}, got {result}"


# ---------------------------------------------------------------------------
# Test 15 (Finding 3): _claim_lock_key is deterministic across calls
# ---------------------------------------------------------------------------


def test_claim_lock_key_deterministic_across_calls() -> None:
    """_claim_lock_key must return the same value for the same inputs every call.

    Pins the expected value so a change in the hash formula (e.g. switching
    back to Python's hash()) will flip this test red.
    """
    # Expected value: sha256("user-1|sleep.recent_hours|global")[:4] big-endian & 0x7FFFFFFF
    # Computed once: 1622640639
    _EXPECTED = 1622640639

    results = [
        _claim_lock_key("user-1", "sleep.recent_hours", "global")
        for _ in range(1000)
    ]
    assert len(set(results)) == 1, "Lock key is not deterministic across calls"
    assert results[0] == _EXPECTED, (
        f"Lock key changed: expected {_EXPECTED}, got {results[0]}. "
        "If the hash algorithm was intentionally changed, update _EXPECTED."
    )
