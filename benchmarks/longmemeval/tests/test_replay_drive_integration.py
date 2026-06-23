#!/usr/bin/env python3
"""
test_replay_drive_integration.py — replay-loop harness wiring (recall-gap epic).

The ``--tier replay`` path is the load-bearing fix for the inert-substrate scar:
without an explicit per-question enqueue + drain, the recall-gap replay loop
never runs on the benchmark control flow, so a before/after measures nothing.
These tests prove the loop actually executes end-to-end against a real Postgres
(testcontainers ``pool`` fixture) and that its aggregate claims land on the
recall read path:

  1. drive_replay enqueues + drains, the aggregate detector's multi-turn claim
     is written with the ``replay-`` PROOF prefix, and retrieve(tier='replay')
     reads it back — the chain a naive run skips.
  2. An empty sandbox enqueues nothing and skips the drain (no wasted LLM call).
  3. The executor selector dispatches inline vs batch (both produce the same
     claims; only LLM dispatch differs).
  4. CLI guard rails reject the two misconfigurations that burn Reader spend.

The aggregate detector is faked (no Haiku/Batch calls) — we patch
``replay_executor.detect_aggregate_claims`` at the module boundary, exactly the
seam the inline executor calls. The per-turn materialization detector is faked
too (mirrors test_belief_view_integration). Turns + claims live under the pool
fixture identity ``test-user-default``.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from weft.embeddings import get_provider
from weft.episode_turns import append_turn
from weft.episodes import create_episode
from weft.models import EpisodeCreate, EpisodeTurn, EpisodeTurnCreate, TurnRole
from weft.views.belief_detector import ClaimUpdate

from benchmarks.longmemeval.materialize import materialize_question
from benchmarks.longmemeval.replay_drive import ReplayDriveStats, drive_replay
from benchmarks.longmemeval.router import retrieve

pytestmark = pytest.mark.asyncio

TEST_USER = "test-user-default"  # the pool fixture's app.user_id GUC


# ----------------------------------------------------------------------
# Helpers (mirror test_belief_view_integration)
# ----------------------------------------------------------------------


def _claim(attribute: str, value: object, turn_id: str, confidence: float = 0.9) -> ClaimUpdate:
    return ClaimUpdate(
        attribute=attribute,
        value=value,
        confidence=confidence,
        source_provenance="user_stated",
        evidence_turn_id=turn_id,
    )


async def _episode(pool: asyncpg.Pool, project_id: str) -> str:
    ep = await create_episode(
        pool, EpisodeCreate(title=f"replay-{uuid.uuid4().hex[:6]}", project_id=project_id),
    )
    return ep.id


async def _turn(
    pool: asyncpg.Pool,
    episode_id: str,
    content: str,
    *,
    role: TurnRole = TurnRole.user,
    occurred_at=None,
    embedding: list[float] | None = None,
) -> EpisodeTurn:
    return await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=episode_id, role=role, content=content, occurred_at=occurred_at,
        ),
        embedding=embedding,
    )


async def _claims_by_detector(
    pool: asyncpg.Pool, like: str, user_id: str = TEST_USER
) -> list[asyncpg.Record]:
    return await pool.fetch(
        "SELECT attribute, value, status, detector_version, evidence_turn_ids "
        "FROM belief_claims WHERE user_id = $1 AND detector_version LIKE $2 "
        "AND status = 'active' ORDER BY attribute",
        user_id, like,
    )


# ----------------------------------------------------------------------
# 1. End-to-end: enqueue → drain → aggregate claim written → recall reads it
# ----------------------------------------------------------------------


async def test_drive_replay_writes_aggregate_claims_and_recall_reads_them(
    pool: asyncpg.Pool, monkeypatch,
) -> None:
    """The whole point: the replay loop EXECUTES and its claim hits recall.

    A naive benchmark run never reaches this state — enqueue + drain are the
    steps it skips. We assert (a) a 'replay-' stamped claim exists in
    belief_claims and (b) retrieve(tier='replay') surfaces it to the Reader.
    """
    from benchmarks.longmemeval.ingest import project_id_for

    project_id = project_id_for("replay-e2e")
    ep = await _episode(pool, project_id)
    # Three turns each naming one wedding — the enumeration the single-turn
    # detector can only see piecewise; the aggregate detector spans them.
    t1 = await _turn(pool, ep, "I attended my cousin's wedding in March.")
    t2 = await _turn(pool, ep, "I went to a friend's wedding in June.")
    t3 = await _turn(pool, ep, "I was at a coworker's wedding in September.")
    span = [t1.id, t2.id, t3.id]

    # Per-turn materialization (belief-view baseline) — gives the enqueue
    # belief-claim path real evidence_turn_ids to resolve against the question.
    # DISTINCT attributes per turn (one claim per wedding event): same-attribute
    # claims would supersede each other down to a single active claim, so the
    # enqueue resolution would see only one evidence turn and the aggregate
    # detector could never span the enumeration. Each attribute shares the
    # "weddings" token with the question so all three match resolution.
    def per_turn(turn: EpisodeTurn) -> list[ClaimUpdate]:
        c = turn.content.lower()
        for month in ("march", "june", "september"):
            if month in c:
                return [_claim(f"weddings.attended_{month}", {"detail": turn.content}, turn.id)]
        return []

    await materialize_question(pool, project_id, detector=per_turn)

    # Fake the multi-turn aggregate detector at the executor's import boundary.
    # Confidence ≥ REVIEW_CONFIDENCE (0.85) so the inline path does NOT escalate
    # to Sonnet. Accepts the optional `model` kwarg the escalation path passes.
    async def fake_aggregate(turns, model=None):
        return [
            ClaimUpdate(
                attribute="weddings.attended_count",
                value={"count": len(turns)},
                confidence=0.95,
                source_provenance="user_stated",
                evidence_turn_id=turns[-1].id,
                evidence_turn_ids=[t.id for t in turns],
            )
        ]

    monkeypatch.setattr(
        "weft.replay_executor.detect_aggregate_claims", fake_aggregate,
    )

    stats = await drive_replay(
        pool,
        question="how many weddings have I attended",
        user_id=TEST_USER,
        executor="inline",
    )

    # The loop ran: at least one episode enqueued + drained, one aggregate claim.
    assert isinstance(stats, ReplayDriveStats)
    assert stats.rows_enqueued >= 1
    assert stats.rows_processed >= 1
    assert stats.rows_failed == 0
    assert stats.claims_written >= 1

    # The claim carries the load-bearing 'replay-' PROOF prefix (so the
    # replay_claims_30d health metric counts it) and spans all three turns.
    replay_rows = await _claims_by_detector(pool, "replay-%")
    assert len(replay_rows) == 1
    assert replay_rows[0]["detector_version"] == "replay-aggregate-detector-v1.0"
    assert replay_rows[0]["attribute"] == "weddings.attended_count"
    persisted_span = list(replay_rows[0]["evidence_turn_ids"])
    assert set(persisted_span) == set(span)

    # And it reaches the Reader on the replay tier's recall.
    embedder = get_provider("fastembed", dimensions=768)
    recalls = await retrieve(
        pool, embedder,
        question="how many weddings have I attended",
        question_type="multi-session",
        project_id=project_id,
        tier="replay",
        user_id=TEST_USER,
    )
    assert any("weddings.attended_count" in r.memory.content for r in recalls)
    assert any("count=3" in r.memory.content for r in recalls)


# ----------------------------------------------------------------------
# 2. Empty sandbox — nothing to enqueue, drain is skipped
# ----------------------------------------------------------------------


async def test_drive_replay_empty_sandbox_skips_drain(
    pool: asyncpg.Pool, monkeypatch,
) -> None:
    """No implicated turns → 0 enqueued → executor is never invoked.

    Guards against a wasted Batch-API round-trip on questions the substrate has
    nothing to say about.
    """
    called = {"inline": False, "batch": False}

    async def boom_inline(*a, **k):
        called["inline"] = True
        raise AssertionError("inline executor must not run on an empty enqueue")

    async def boom_batch(*a, **k):
        called["batch"] = True
        raise AssertionError("batch executor must not run on an empty enqueue")

    monkeypatch.setattr(
        "benchmarks.longmemeval.replay_drive.run_replay_executor", boom_inline,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.replay_drive.run_replay_executor_batch", boom_batch,
    )

    stats = await drive_replay(
        pool,
        question="a query about a topic that has no turns at all in this sandbox",
        user_id=TEST_USER,
        executor="inline",
    )

    assert stats.rows_enqueued == 0
    assert stats.rows_processed == 0
    assert stats.claims_written == 0
    assert called == {"inline": False, "batch": False}


# ----------------------------------------------------------------------
# 3. Executor selection — inline vs batch dispatch (no DB needed)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expect_called,expect_other",
    [
        ("inline", "run_replay_executor", "run_replay_executor_batch"),
        ("batch", "run_replay_executor_batch", "run_replay_executor"),
    ],
)
async def test_drive_replay_executor_selection(
    monkeypatch, kind, expect_called, expect_other,
) -> None:
    """``executor=`` routes to the matching entrypoint; the other never runs."""
    from weft.replay_executor import ReplayExecutorResult

    import benchmarks.longmemeval.replay_drive as rd

    calls: list[str] = []

    async def fake_enqueue(pool, question, user_id, *, reason="reask-miss"):
        return 2  # force the drain branch

    async def fake_inline(pool, **k):
        calls.append("run_replay_executor")
        return ReplayExecutorResult(rows_processed=2, rows_done=2, claims_written=1)

    async def fake_batch(pool, **k):
        calls.append("run_replay_executor_batch")
        return ReplayExecutorResult(rows_processed=2, rows_done=2, claims_written=5)

    monkeypatch.setattr(rd, "enqueue_replay_on_miss", fake_enqueue)
    monkeypatch.setattr(rd, "run_replay_executor", fake_inline)
    monkeypatch.setattr(rd, "run_replay_executor_batch", fake_batch)

    stats = await drive_replay(
        object(),  # pool unused — both enqueue + executor are faked
        question="q", user_id=TEST_USER, executor=kind,
    )

    assert calls == [expect_called]
    assert expect_other not in calls
    assert stats.rows_enqueued == 2
    assert stats.rows_processed == 2


# ----------------------------------------------------------------------
# 4. CLI guard rails — replay inherits belief-view's two hard-fails
# ----------------------------------------------------------------------


async def test_cli_rejects_replay_without_turns_mode(tmp_path) -> None:
    from click.testing import CliRunner

    from benchmarks.longmemeval.adapter import cli

    dataset = tmp_path / "ds.json"
    dataset.write_text("[]", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["--dataset", str(dataset), "--tier", "replay", "--mode", "raw"],
    )
    assert result.exit_code != 0
    assert "requires --mode turns" in result.output


async def test_cli_rejects_replay_with_no_cleanup(tmp_path) -> None:
    from click.testing import CliRunner

    from benchmarks.longmemeval.adapter import cli

    dataset = tmp_path / "ds.json"
    dataset.write_text("[]", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "--dataset", str(dataset),
            "--tier", "replay",
            "--mode", "turns",
            "--no-cleanup",
        ],
    )
    assert result.exit_code != 0
    assert "incompatible with --no-cleanup" in result.output
