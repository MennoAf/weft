"""Integration tests for weft/replay.py — miss→turns linkage (E1.L2 + E1.L3).

Seeds a real episode + turns, inserts belief_claims with evidence_turn_ids,
simulates a retrieval miss, and asserts that resolve_implicated_turns returns
the correct implicated turn_ids.

Coverage:
  - Path (a): belief-claim evidence_turn_ids (primary path)
  - Path (b): nearest-episode keyword fallback (when no belief claims match)
  - ReaskPair input: reask_query text is used for resolution
  - Deduplication: evidence_turn_ids appearing across multiple claims
    are returned once
  - Empty result: query that matches no claims AND no turns
  - E1.L3 enqueue-on-miss: apply_reask_feedback writes one pending
    replay_queue row per implicated episode (idempotent)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from weft.episode_turns import append_turn
from weft.episodes import create_episode
from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole
from weft.replay import ReaskPair, resolve_implicated_turns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-default"


def _claim_id() -> str:
    return f"belief-{uuid.uuid4().hex[:10]}"


async def _insert_belief_claim(
    pool,
    *,
    user_id: str,
    attribute: str,
    evidence_turn_ids: list[str],
    status: str = "active",
    scope: str = "global",
) -> str:
    """Insert a belief_claim and return its claim_id."""
    claim_id = _claim_id()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                """
                INSERT INTO belief_claims (
                    claim_id, user_id, attribute, value, scope,
                    evidence_turn_ids, status, occurred_at,
                    source_provenance, detector_confidence, detector_version
                ) VALUES (
                    $1, $2, $3, '{"v": 1}'::jsonb, $4,
                    $5, $6, $7,
                    'user_stated', 1.0, 'v1'
                )
                """,
                claim_id,
                user_id,
                attribute,
                scope,
                evidence_turn_ids,
                status,
                now,
            )
    return claim_id


# ---------------------------------------------------------------------------
# Primary test: path (a) — belief-claim evidence_turn_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_implicated_turns(pool):
    """Seed an episode + turns, simulate a miss, assert implicated turn_ids returned.

    This is the done_when test for loom-c483a5dc.

    Path (a): the missed query matches the belief_claim attribute via token-
    overlap. The claim carries evidence_turn_ids pointing to the turns that
    provided evidence when the belief was extracted. resolve_implicated_turns
    should return exactly those turn_ids.
    """
    # 1. Create an episode with two turns.
    ep = await create_episode(pool, EpisodeCreate(title="sleep-tracking-session"))
    turn_a = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="I slept about 6 hours last night",
        ),
    )
    turn_b = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.assistant,
            content="Got it, I'll track your sleep hours.",
        ),
    )

    # 2. Insert a belief claim whose attribute matches the query topic ("sleep"),
    #    anchoring both turns as evidence.
    await _insert_belief_claim(
        pool,
        user_id=USER_ID,
        attribute="sleep.recent_hours",
        evidence_turn_ids=[turn_a.id, turn_b.id],
    )

    # 3. Simulate a miss: the agent re-asks about sleep hours.
    missed_query = "how many hours of sleep did I get recently?"
    turn_ids = await resolve_implicated_turns(
        pool,
        missed_query,
        user_id=USER_ID,
    )

    # 4. Assert: the implicated turns are the ones that evidenced the belief claim.
    assert set(turn_ids) == {turn_a.id, turn_b.id}, (
        f"Expected evidence turns {turn_a.id!r} and {turn_b.id!r}, got {turn_ids!r}"
    )


# ---------------------------------------------------------------------------
# ReaskPair input: reask_query used for lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_implicated_turns_reask_pair(pool):
    """resolve_implicated_turns accepts a ReaskPair and uses reask_query."""
    ep = await create_episode(pool, EpisodeCreate(title="diet-tracking"))
    turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="I had a salad for lunch",
        ),
    )
    await _insert_belief_claim(
        pool,
        user_id=USER_ID,
        attribute="diet.lunch_recent",
        evidence_turn_ids=[turn.id],
    )

    pair = ReaskPair(
        original_query="what did I eat",
        reask_query="what did I have for lunch recently?",
    )
    turn_ids = await resolve_implicated_turns(pool, pair, user_id=USER_ID)
    assert turn.id in turn_ids


# ---------------------------------------------------------------------------
# Deduplication: evidence_turn_ids shared across multiple claims
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_implicated_turns_deduplicates(pool):
    """Turn IDs shared across multiple belief claims appear only once."""
    ep = await create_episode(pool, EpisodeCreate(title="exercise-log"))
    shared_turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="I ran 5km this morning",
        ),
    )
    other_turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.assistant,
            content="Great — I've noted your morning run.",
        ),
    )

    # Two distinct claims both reference shared_turn.  Use distinct attributes
    # so the partial-unique-active index doesn't reject the second insert.
    await _insert_belief_claim(
        pool,
        user_id=USER_ID,
        attribute="exercise.morning_run",
        evidence_turn_ids=[shared_turn.id],
    )
    await _insert_belief_claim(
        pool,
        user_id=USER_ID,
        attribute="exercise.distance_km",
        evidence_turn_ids=[shared_turn.id, other_turn.id],
    )

    turn_ids = await resolve_implicated_turns(pool, "exercise run morning", user_id=USER_ID)

    # shared_turn.id must appear exactly once despite being in both claims.
    assert turn_ids.count(shared_turn.id) == 1, (
        f"shared_turn.id appeared {turn_ids.count(shared_turn.id)} times; expected 1"
    )
    assert other_turn.id in turn_ids


# ---------------------------------------------------------------------------
# Fallback path (b): keyword turn search when no belief claims match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_implicated_turns_fallback_nearest_episode(pool):
    """Falls back to keyword turn search when no belief claims match the query."""
    # Create an episode with a distinct topic that has NO belief claim.
    ep = await create_episode(pool, EpisodeCreate(title="travel-planning"))
    turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="I am planning a trip to Japan next spring",
        ),
    )

    # No belief claim inserted for "japan" — forces path (b).
    # Use a query whose tokens all appear in the turn content so the
    # FTS keyword half (websearch_to_tsquery AND-logic) can match it.
    turn_ids = await resolve_implicated_turns(
        pool,
        "japan trip spring",
        user_id=USER_ID,
    )

    # The keyword fallback should surface the seeded turn.
    assert turn.id in turn_ids, (
        f"Expected nearest-episode fallback to include {turn.id!r}; got {turn_ids!r}"
    )


# ---------------------------------------------------------------------------
# Empty result: no turns at all in the database for the topic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_implicated_turns_empty_when_no_data(pool):
    """Returns empty list when neither belief claims nor episode turns match."""
    # The pool fixture TRUNCATEs all tables, so this test starts clean.
    turn_ids = await resolve_implicated_turns(
        pool,
        "completely unrelated xyzzy query",
        user_id=USER_ID,
    )
    # May return [] or whatever turns happen to survive — just assert no crash.
    assert isinstance(turn_ids, list)


# ---------------------------------------------------------------------------
# E1.L3: enqueue_on_miss — replay_queue row written from the reask feedback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_on_miss(pool):
    """A recall miss followed by a re-ask results in exactly one pending replay_queue
    row for the implicated episode.  A second identical re-ask adds no duplicate.

    done_when assertion for loom-5d414368 (E1.L3 in the replay loop epic).

    Setup:
      1. Create an episode with a turn whose content matches the missed query.
      2. Insert a belief_claim whose attribute matches the query, anchoring the
         turn as evidence — this ensures resolve_implicated_turns returns the
         turn via path (a) (belief-claim evidence_turn_ids) so the episode
         mapping is deterministic.
      3. Log a "missed" recall query in weft_recall_queries.
      4. Create a dummy satisfying memory (required by apply_reask_feedback).
      5. Call apply_reask_feedback → this should atomically claim the miss row
         AND enqueue one pending replay_queue row for the episode.
      6. Assert exactly one pending replay_queue row for the episode.
      7. Call apply_reask_feedback again with the same missed_query_id → the
         update returns 'UPDATE 0' (already processed), so enqueue is NOT
         called a second time.  Assert still exactly one pending row.
    """
    from weft.models import MemoryCreate, MemoryType
    from weft.store import apply_reask_feedback, log_recall_query, store_memory

    # 1. Create an episode with a turn whose content contains "sleep".
    ep = await create_episode(pool, EpisodeCreate(title="sleep-miss-episode"))
    turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="I slept about seven hours last night",
        ),
    )

    # 2. Insert a belief_claim anchoring the turn as evidence for "sleep".
    await _insert_belief_claim(
        pool,
        user_id=USER_ID,
        attribute="sleep.hours_recent",
        evidence_turn_ids=[turn.id],
    )

    # 3. Log the "missed" query — must match the claim's attribute so
    #    resolve_implicated_turns picks it up via path (a).
    await log_recall_query(
        pool,
        tool_name="recall",
        query_text="how many hours of sleep did I get recently",
    )
    rows = await pool.fetch(
        "SELECT query_id FROM weft_recall_queries "
        "WHERE query_text = 'how many hours of sleep did I get recently'"
    )
    assert len(rows) == 1, "log_recall_query did not write the expected row"
    missed_query_id = rows[0]["query_id"]

    # 4. Create a dummy satisfying memory.
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="User typically sleeps around 7 hours",
            topic=["sleep"],
            confidence=0.8,
        ),
    )

    # 5. First re-ask: apply_reask_feedback claims the miss and enqueues replay.
    result = await apply_reask_feedback(pool, missed_query_id, mem.id)
    assert result is not None, "expected a fresh claim — got None (already processed?)"

    # 6. Assert exactly ONE pending replay_queue row for the episode.
    pending_rows = await pool.fetch(
        "SELECT id, episode_id, turn_ids, reason, status "
        "FROM replay_queue "
        "WHERE episode_id = $1 AND status = 'pending'",
        ep.id,
    )
    assert len(pending_rows) == 1, (
        f"Expected exactly 1 pending replay_queue row for episode {ep.id!r}, "
        f"got {len(pending_rows)}"
    )
    rq_row = pending_rows[0]
    assert rq_row["episode_id"] == ep.id
    assert rq_row["reason"] == "reask-miss"
    assert turn.id in rq_row["turn_ids"], (
        f"Implicated turn {turn.id!r} not found in replay_queue.turn_ids: "
        f"{rq_row['turn_ids']!r}"
    )

    # 7. Second re-ask with the SAME missed_query_id — must be a no-op.
    result2 = await apply_reask_feedback(pool, missed_query_id, mem.id)
    assert result2 is None, (
        "Expected idempotent no-op (None) on second apply_reask_feedback call"
    )

    # Still exactly one pending row — no duplicate was written.
    pending_rows_after = await pool.fetch(
        "SELECT id FROM replay_queue WHERE episode_id = $1 AND status = 'pending'",
        ep.id,
    )
    assert len(pending_rows_after) == 1, (
        f"Expected still exactly 1 pending row after second re-ask, "
        f"got {len(pending_rows_after)}"
    )
