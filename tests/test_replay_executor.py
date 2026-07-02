"""Integration tests for weft.replay_executor — drain → detect → write.

Requires a real PostgreSQL instance (testcontainers via conftest.py). Only the
Haiku client is mocked; everything else (replay_queue lifecycle, RLS context,
the materializer write path, belief_claims) runs against the real DB — this is
the scheduler→detector→materializer→DB integration seam that mocked-layer tests
cannot cover (see anti-pattern weft-e587d0a2).

Spec: Loom task loom-ebef8ec1 (E2.L7).
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.episode_turns import append_turn
from weft.episodes import create_episode
from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole
from weft.views import belief_detector
from weft.counters import COUNTER_REPLAY_STALE_REAPED, get_counter
from weft.replay import REPLAY_QUEUE_STALENESS_DAYS
from weft.replay_executor import (
    REPLAY_AGGREGATE_DETECTOR_VERSION,
    reap_stale_pending_replays,
    run_replay_executor,
)
from tests.conftest import DEFAULT_TEST_USER_ID

pytestmark = pytest.mark.asyncio

_GET_CLIENT = "weft.views.belief_detector._get_client"


def _mock_client(json_data) -> AsyncMock:
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def _model_routed_client(by_model: dict) -> AsyncMock:
    """A client whose messages.create response depends on the ``model`` kwarg.

    Lets escalation tests give Haiku and Sonnet different answers. A model not in
    ``by_model`` raises KeyError if called — surfacing an unexpected escalation.
    """

    def _create(*args, **kwargs):
        payload = by_model[kwargs["model"]]
        response = MagicMock()
        response.content = [MagicMock(text=json.dumps(payload))]
        return response

    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=_create)
    return client


async def _seed_episode_with_turns(pool, contents: list[str]) -> tuple[str, list[str]]:
    """Create an episode + one user turn per content string. Returns (episode_id, turn_ids)."""
    ep = await create_episode(pool, EpisodeCreate(title=f"agg-{uuid.uuid4().hex[:6]}"))
    turn_ids: list[str] = []
    for content in contents:
        turn = await append_turn(
            pool,
            EpisodeTurnCreate(episode_id=ep.id, role=TurnRole.user, content=content),
        )
        turn_ids.append(turn.id)
    return ep.id, turn_ids


async def _enqueue_replay(pool, episode_id: str, turn_ids: list[str]) -> str:
    """Insert a pending replay_queue row (user_id picks up the session GUC default)."""
    rq_id = f"rq-{uuid.uuid4().hex[:10]}"
    await pool.execute(
        """
        INSERT INTO replay_queue (id, episode_id, turn_ids, reason, status)
        VALUES ($1, $2, $3, 'reask-miss', 'pending')
        """,
        rq_id,
        episode_id,
        turn_ids,
    )
    return rq_id


_COUNT_CLAIM_RESPONSE_TEMPLATE = {
    "attribute": "meetings.windward_count",
    "value": {"count": 3, "period": "this week"},
    "confidence": 0.9,
    "source_provenance": "user_stated",
}


async def test_queued_enumeration_yields_replay_claim_and_marks_done(pool):
    """done_when (loom-ebef8ec1): a queued episode with cross-turn enumeration
    yields >=1 aggregate belief_claim (evidence spanning multiple turns), the
    queue row becomes done, and a pre-existing non-flagged claim is untouched."""
    episode_id, turn_ids = await _seed_episode_with_turns(
        pool,
        [
            "Met with the Windward team on Monday.",
            "Had another Windward sync on Wednesday.",
            "Third Windward meeting this week was Friday.",
        ],
    )

    # Pre-existing, non-flagged claim from the single-turn detector path. Must
    # survive untouched (different attribute, ordinary detector_version).
    await pool.execute(
        """
        INSERT INTO belief_claims (
            claim_id, attribute, value, scope, evidence_turn_ids, status,
            occurred_at, source_provenance, detector_confidence, detector_version
        ) VALUES (
            $1, 'sleep.recent_hours', '{"hours": 7}'::jsonb, 'global',
            $2, 'active', now(), 'user_stated', 0.9, 'belief-detector-v1.0'
        )
        """,
        f"belief-{uuid.uuid4().hex[:10]}",
        [turn_ids[0]],
    )

    rq_id = await _enqueue_replay(pool, episode_id, turn_ids)

    llm_response = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    with patch(_GET_CLIENT, return_value=_mock_client(llm_response)):
        result = await run_replay_executor(pool)

    assert result.rows_processed == 1
    assert result.rows_done == 1
    assert result.rows_failed == 0
    assert result.claims_written == 1

    # The aggregate claim landed with the replay- PROOF prefix and full span.
    rows = await pool.fetch(
        "SELECT detector_version, evidence_turn_ids, status, value "
        "FROM belief_claims WHERE attribute = 'meetings.windward_count'"
    )
    assert len(rows) == 1
    claim = rows[0]
    assert claim["detector_version"] == REPLAY_AGGREGATE_DETECTOR_VERSION
    assert claim["detector_version"].startswith("replay-")  # satisfies replay_claims_30d
    assert set(claim["evidence_turn_ids"]) == set(turn_ids)
    assert claim["status"] == "active"

    # Queue row driven to terminal 'done'.
    status = await pool.fetchval("SELECT status FROM replay_queue WHERE id = $1", rq_id)
    assert status == "done"

    # Pre-existing non-flagged claim untouched.
    pre = await pool.fetch(
        "SELECT detector_version, status FROM belief_claims "
        "WHERE attribute = 'sleep.recent_hours'"
    )
    assert len(pre) == 1
    assert pre[0]["detector_version"] == "belief-detector-v1.0"
    assert pre[0]["status"] == "active"


async def test_replay_claim_counts_toward_replay_claims_30d(pool):
    """The written claim matches the replay_claims_30d PROOF query — the whole
    point of the 'replay-' prefix contract."""
    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    await _enqueue_replay(pool, episode_id, turn_ids)
    llm_response = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    with patch(_GET_CLIENT, return_value=_mock_client(llm_response)):
        await run_replay_executor(pool)

    count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE detector_version LIKE 'replay-%'"
    )
    assert count == 1


async def test_no_aggregate_still_marks_row_done(pool):
    """If the detector finds no cross-turn fact (returns []), the row is still
    consumed (terminal 'done') so it cannot pin its turns forever."""
    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["How's it going?", "I slept 7 hours."]
    )
    rq_id = await _enqueue_replay(pool, episode_id, turn_ids)
    with patch(_GET_CLIENT, return_value=_mock_client([])):
        result = await run_replay_executor(pool)

    assert result.rows_done == 1
    assert result.claims_written == 0
    status = await pool.fetchval("SELECT status FROM replay_queue WHERE id = $1", rq_id)
    assert status == "done"


async def test_write_failure_marks_row_failed_and_bumps_counter(pool):
    """An unrecoverable error during the write drives the row to terminal
    'failed' (not left pending) and increments replay.executor.failed."""
    from weft.counters import COUNTER_REPLAY_EXECUTOR_FAILED, get_counter

    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    rq_id = await _enqueue_replay(pool, episode_id, turn_ids)

    before = await get_counter(pool, COUNTER_REPLAY_EXECUTOR_FAILED)

    llm_response = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    with patch(_GET_CLIENT, return_value=_mock_client(llm_response)):
        # Force the materializer write to blow up.
        with patch(
            "weft.replay_executor.materialize_turn",
            side_effect=RuntimeError("simulated write failure"),
        ):
            result = await run_replay_executor(pool)

    assert result.rows_failed == 1
    assert result.rows_done == 0
    status = await pool.fetchval("SELECT status FROM replay_queue WHERE id = $1", rq_id)
    assert status == "failed"
    after = await get_counter(pool, COUNTER_REPLAY_EXECUTOR_FAILED)
    assert after == before + 1


async def test_done_row_not_reprocessed(pool):
    """A second drain does not touch an already-terminal row."""
    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    await _enqueue_replay(pool, episode_id, turn_ids)
    llm_response = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    with patch(_GET_CLIENT, return_value=_mock_client(llm_response)):
        first = await run_replay_executor(pool)
        second = await run_replay_executor(pool)

    assert first.rows_processed == 1
    assert second.rows_processed == 0  # nothing pending the second time
    # Exactly one aggregate claim — no duplicate from a re-drain.
    count = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE attribute = 'meetings.windward_count'"
    )
    assert count == 1


# ---------------------------------------------------------------------------
# E3.L9 — Sonnet 4.6 escalation on Haiku abstention / low confidence (never Opus)
# ---------------------------------------------------------------------------


async def test_haiku_abstention_triggers_one_sonnet_escalation(pool):
    """done_when (loom-fdb7c181): a Haiku abstention escalates the turn-set
    exactly once to Sonnet 4.6; the adopted claim is tagged with the Sonnet tier
    and still carries the replay- PROOF prefix."""
    from weft.replay_executor import (
        REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION,
        SONNET_ESCALATION_MODEL,
    )

    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    await _enqueue_replay(pool, episode_id, turn_ids)

    sonnet_claim = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    client = _model_routed_client(
        {
            belief_detector._MODEL: [],  # Haiku abstains
            SONNET_ESCALATION_MODEL: sonnet_claim,  # Sonnet recovers the aggregate
        }
    )
    with patch(_GET_CLIENT, return_value=client):
        result = await run_replay_executor(pool)

    assert result.rows_done == 1
    assert result.claims_written == 1

    # Exactly one Haiku call then exactly one Sonnet call — a single retry.
    models = [c.kwargs["model"] for c in client.messages.create.call_args_list]
    assert models == [belief_detector._MODEL, SONNET_ESCALATION_MODEL]

    rows = await pool.fetch(
        "SELECT detector_version FROM belief_claims WHERE attribute = 'meetings.windward_count'"
    )
    assert len(rows) == 1
    assert rows[0]["detector_version"] == REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION
    assert rows[0]["detector_version"].startswith("replay-")  # PROOF metric intact


async def test_low_confidence_haiku_triggers_sonnet_escalation(pool):
    """A Haiku claim below the 0.85 review threshold also escalates to Sonnet."""
    from weft.replay_executor import (
        REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION,
        SONNET_ESCALATION_MODEL,
    )

    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    await _enqueue_replay(pool, episode_id, turn_ids)

    low_conf = [
        {**_COUNT_CLAIM_RESPONSE_TEMPLATE, "confidence": 0.7, "evidence_turn_ids": turn_ids}
    ]
    high_conf = [
        {**_COUNT_CLAIM_RESPONSE_TEMPLATE, "confidence": 0.95, "evidence_turn_ids": turn_ids}
    ]
    client = _model_routed_client(
        {belief_detector._MODEL: low_conf, SONNET_ESCALATION_MODEL: high_conf}
    )
    with patch(_GET_CLIENT, return_value=client):
        result = await run_replay_executor(pool)

    assert result.claims_written == 1
    models = [c.kwargs["model"] for c in client.messages.create.call_args_list]
    assert models == [belief_detector._MODEL, SONNET_ESCALATION_MODEL]
    version = await pool.fetchval(
        "SELECT detector_version FROM belief_claims WHERE attribute = 'meetings.windward_count'"
    )
    assert version == REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION


async def test_confident_haiku_skips_escalation(pool):
    """A confident Haiku hit (>= 0.85) triggers zero retries — no Sonnet call."""
    from weft.replay_executor import (
        REPLAY_AGGREGATE_DETECTOR_VERSION,
        SONNET_ESCALATION_MODEL,
    )

    episode_id, turn_ids = await _seed_episode_with_turns(
        pool, ["Met Monday.", "Met Wednesday.", "Met Friday."]
    )
    await _enqueue_replay(pool, episode_id, turn_ids)

    # Only a Haiku route is registered; any Sonnet call would KeyError.
    high_conf = [{**_COUNT_CLAIM_RESPONSE_TEMPLATE, "evidence_turn_ids": turn_ids}]
    client = _model_routed_client({belief_detector._MODEL: high_conf})
    with patch(_GET_CLIENT, return_value=client):
        result = await run_replay_executor(pool)

    assert result.claims_written == 1
    models = [c.kwargs["model"] for c in client.messages.create.call_args_list]
    assert models == [belief_detector._MODEL]  # exactly one call, no escalation
    assert SONNET_ESCALATION_MODEL not in models
    version = await pool.fetchval(
        "SELECT detector_version FROM belief_claims WHERE attribute = 'meetings.windward_count'"
    )
    assert version == REPLAY_AGGREGATE_DETECTOR_VERSION  # Haiku tier


async def test_escalation_never_routes_to_opus():
    """Hard rule (Pinch weft-82b2860a): no Opus model id anywhere in the replay
    or aggregate-detector code paths."""
    import inspect

    from weft.views import aggregate_detector
    import weft.replay_executor as replay_executor

    for module in (replay_executor, aggregate_detector):
        source = inspect.getsource(module)
        assert "claude-opus" not in source, f"opus model id found in {module.__name__}"


# ---------------------------------------------------------------------------
# reap_stale_pending_replays — drain orphaned stale 'pending' rows to 'failed'
# (complements the L4 retention age-bound; loom-d8b41372)
# ---------------------------------------------------------------------------

_STALE_AGE = REPLAY_QUEUE_STALENESS_DAYS + 6  # comfortably past the window


async def _insert_replay_row(
    pool, *, age_days, status="pending", user_id=DEFAULT_TEST_USER_ID
) -> str:
    """Insert one replay_queue row at a controllable age/status/owner."""
    ep = await create_episode(pool, EpisodeCreate(title=f"reap-{uuid.uuid4().hex[:6]}"))
    rq_id = f"rq-{uuid.uuid4().hex[:10]}"
    await pool.execute(
        """
        INSERT INTO replay_queue (id, episode_id, turn_ids, reason, status, user_id, created_at)
        VALUES ($1, $2, $3, 'reap-test', $4, $5, now() - ($6 || ' days')::interval)
        """,
        rq_id, ep.id, ["et-x"], status, user_id, str(age_days),
    )
    return rq_id


async def _status_of(pool, rq_id: str) -> str:
    return await pool.fetchval("SELECT status FROM replay_queue WHERE id = $1", rq_id)


async def test_reap_transitions_stale_pending_to_failed(pool):
    """A pending row past the staleness window is reaped to terminal 'failed'
    and the reaped counter is bumped."""
    rq_id = await _insert_replay_row(pool, age_days=_STALE_AGE)
    before = await get_counter(pool, COUNTER_REPLAY_STALE_REAPED)

    reaped = await reap_stale_pending_replays(pool)

    assert reaped == 1
    assert await _status_of(pool, rq_id) == "failed"
    assert await get_counter(pool, COUNTER_REPLAY_STALE_REAPED) == before + 1


async def test_reap_leaves_fresh_and_terminal_rows_untouched(pool):
    """Fresh pending rows and already-terminal rows are never reaped."""
    fresh = await _insert_replay_row(pool, age_days=1, status="pending")
    done = await _insert_replay_row(pool, age_days=_STALE_AGE, status="done")
    failed = await _insert_replay_row(pool, age_days=_STALE_AGE, status="failed")

    reaped = await reap_stale_pending_replays(pool)

    assert reaped == 0
    assert await _status_of(pool, fresh) == "pending"
    assert await _status_of(pool, done) == "done"
    assert await _status_of(pool, failed) == "failed"


async def test_reap_respects_the_staleness_window_boundary(pool):
    """A row just inside the window survives; aged just outside, it's reaped."""
    rq_id = await _insert_replay_row(
        pool, age_days=REPLAY_QUEUE_STALENESS_DAYS - 2, status="pending"
    )
    assert await reap_stale_pending_replays(pool) == 0
    assert await _status_of(pool, rq_id) == "pending"

    await pool.execute(
        "UPDATE replay_queue SET created_at = now() - ($2 || ' days')::interval WHERE id = $1",
        rq_id, str(REPLAY_QUEUE_STALENESS_DAYS + 2),
    )
    assert await reap_stale_pending_replays(pool) == 1
    assert await _status_of(pool, rq_id) == "failed"


async def test_reap_is_best_effort_per_row(pool):
    """A row whose terminal write throws is logged and skipped — the sweep
    still reaps the healthy rows and never aborts."""
    good = await _insert_replay_row(pool, age_days=_STALE_AGE)
    poison = await _insert_replay_row(pool, age_days=_STALE_AGE)

    from weft import replay_executor as rx

    real_set_status = rx._set_status

    async def _flaky_set_status(pool_, replay_id, user_id, status):
        if replay_id == poison:
            raise RuntimeError("simulated unwritable user_id / poison row")
        return await real_set_status(pool_, replay_id, user_id, status)

    with patch.object(rx, "_set_status", side_effect=_flaky_set_status):
        reaped = await rx.reap_stale_pending_replays(pool)

    assert reaped == 1, "only the healthy row is reaped; the poison row is skipped"
    assert await _status_of(pool, good) == "failed"
    assert await _status_of(pool, poison) == "pending", "poison row survives for a later pass"


async def test_run_replay_executor_reaps_stale_before_draining(pool):
    """The executor pass reaps orphaned stale rows (Phase 0) so they never reach
    the drain — result.rows_reaped reflects it and the row is terminal."""
    rq_id = await _insert_replay_row(pool, age_days=_STALE_AGE)

    # No client needed: the row is reaped before Phase-1 hydration, so the drain
    # finds no pending work and never calls the detector.
    result = await run_replay_executor(pool)

    assert result.rows_reaped == 1
    assert await _status_of(pool, rq_id) == "failed"
