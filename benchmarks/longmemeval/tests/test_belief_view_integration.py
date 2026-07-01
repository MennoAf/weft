#!/usr/bin/env python3
"""
test_belief_view_integration.py — belief-view harness wiring (loom-1fe75d00).

Exercises the three new pieces that let the LongMemEval harness run the
belief-view end-to-end, against a real Postgres (testcontainers ``pool``
fixture, loaded via the package conftest plugin):

  1. ``materialize.materialize_question`` — sandbox-scoped, cursor-free
     materialization of one question's turns into ``belief_claims``.
  2. ``router.retrieve(tier='belief-view')`` — reads the claim view, falls
     back to turn recall on a miss.
  3. ``ingest.cleanup_haystack`` — deletes a question's claims before the
     episode cascade.

The detector is faked (no Haiku calls). Turns are written under the
``pool`` fixture's GUC identity (``test-user-default``); claims inherit that
identity from the turn rows, so only the read side (``retrieve``) takes an
explicit ``user_id``.
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

from benchmarks.longmemeval.dataset import Instance
from benchmarks.longmemeval.ingest import cleanup_haystack, project_id_for
from benchmarks.longmemeval.materialize import materialize_question
from benchmarks.longmemeval.router import retrieve

pytestmark = pytest.mark.asyncio

# The pool fixture sets app.user_id to this; turns inherit it via the
# episode_turns column GUC default, so claims must be read/written under it.
TEST_USER = "test-user-default"


# ----------------------------------------------------------------------
# Helpers
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
        pool, EpisodeCreate(title=f"bv-{uuid.uuid4().hex[:6]}", project_id=project_id),
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


async def _active_claims(pool: asyncpg.Pool, user_id: str = TEST_USER) -> list[asyncpg.Record]:
    return await pool.fetch(
        "SELECT attribute, value, status, evidence_turn_ids, occurred_at "
        "FROM belief_claims WHERE user_id = $1 AND status = 'active' "
        "ORDER BY attribute",
        user_id,
    )


# ----------------------------------------------------------------------
# 1. materialize_question
# ----------------------------------------------------------------------


async def test_materialize_question_writes_claims_and_skips_no_claim_turns(
    pool: asyncpg.Pool,
) -> None:
    project_id = project_id_for("mat-basic")
    ep = await _episode(pool, project_id)
    sleep_turn = await _turn(pool, ep, "I slept about 7 hours last night.")
    await _turn(pool, ep, "Hey, how's it going today?")  # no-claim

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        if "slept" in turn.content.lower():
            return [_claim("sleep.recent_hours", {"hours": 7}, turn.id)]
        return []  # abstention

    stats = await materialize_question(
        pool, project_id, detector=detector,
    )

    assert stats.claims_written == 1
    assert stats.claims_superseded == 0
    assert stats.abstentions == 1  # the greeting turn

    rows = await _active_claims(pool)
    assert len(rows) == 1
    assert rows[0]["attribute"] == "sleep.recent_hours"
    # Provenance anchor points back at the originating turn.
    assert sleep_turn.id in list(rows[0]["evidence_turn_ids"])


async def test_materialize_question_supersession_collapses_to_latest(
    pool: asyncpg.Pool,
) -> None:
    """Two claims for the same attribute → one active (latest), one superseded.

    This is the knowledge-update mechanism: the Reader sees only the current
    value, never the stale one (the Hawaii-vs-Paris failure mode).
    """
    from datetime import datetime, timezone

    project_id = project_id_for("mat-supersede")
    ep = await _episode(pool, project_id)
    early = await _turn(
        pool, ep, "I've been sleeping 5 hours.",
        occurred_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
    )
    late = await _turn(
        pool, ep, "Now I'm sleeping 7 hours.",
        occurred_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        hours = 5 if "5 hours" in turn.content else 7
        return [_claim("sleep.recent_hours", {"hours": hours}, turn.id)]

    stats = await materialize_question(
        pool, project_id, detector=detector,
    )

    assert stats.claims_written == 2
    assert stats.claims_superseded == 1

    active = await _active_claims(pool)
    assert len(active) == 1
    value = active[0]["value"]
    # asyncpg may return JSONB as str; normalize.
    if isinstance(value, str):
        import json
        value = json.loads(value)
    assert value == {"hours": 7}
    assert late.id in list(active[0]["evidence_turn_ids"])
    assert early.id not in list(active[0]["evidence_turn_ids"])


async def test_materialize_question_empty_sandbox_is_noop(pool: asyncpg.Pool) -> None:
    stats = await materialize_question(
        pool, project_id_for("mat-empty"),
        detector=lambda t: [],
    )
    assert stats.turns_total == 0
    assert stats.turns_processed == 0
    assert stats.claims_written == 0


async def test_materialize_question_counts_fetched_turns(pool: asyncpg.Pool) -> None:
    """turns_total reflects the sandbox fetch, independent of claim outcomes.

    This is the field the adapter's zero-turn RLS/identity warning keys on.
    """
    project_id = project_id_for("mat-total")
    ep = await _episode(pool, project_id)
    await _turn(pool, ep, "I slept 7 hours.")
    await _turn(pool, ep, "Hey there!")

    stats = await materialize_question(pool, project_id, detector=lambda t: [])

    assert stats.turns_total == 2
    assert stats.abstentions == 2


async def test_materialize_question_aborts_on_consecutive_errors(
    pool: asyncpg.Pool,
) -> None:
    """A dead detector (auth/rate-limit burst) must abort, not limp through."""
    from benchmarks.longmemeval.materialize import MaterializationAborted

    project_id = project_id_for("mat-abort")
    ep = await _episode(pool, project_id)
    for i in range(4):
        await _turn(pool, ep, f"I slept {i + 5} hours.")

    def dead_detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        raise ConnectionError("simulated API outage")

    with pytest.raises(MaterializationAborted):
        await materialize_question(
            pool, project_id, detector=dead_detector, max_consecutive_errors=3,
        )


async def test_materialize_question_isolated_errors_do_not_abort(
    pool: asyncpg.Pool,
) -> None:
    """Scattered failures below the consecutive cap are skipped, not fatal —
    successes reset the streak."""
    from datetime import datetime, timezone

    project_id = project_id_for("mat-scattered")
    ep = await _episode(pool, project_id)
    # Explicit occurred_at pins processing order (FAIL, ok, FAIL, ok) so the
    # error streak is provably 1-1, never 2 — ties on occurred_at would break
    # to random short-id order and could flake the cap.
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    contents = ["FAIL one", "I slept 7 hours.", "FAIL two", "I slept 8 hours."]
    for i, content in enumerate(contents):
        await _turn(pool, ep, content, occurred_at=base.replace(hour=i + 1))

    def flaky_detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        if turn.content.startswith("FAIL"):
            raise ConnectionError("transient")
        hours = 7 if "7 hours" in turn.content else 8
        return [_claim("sleep.recent_hours", {"hours": hours}, turn.id)]

    stats = await materialize_question(
        pool, project_id, detector=flaky_detector, max_consecutive_errors=2,
    )

    assert stats.errors == 2  # both failures tolerated, run completed
    assert stats.claims_written == 2
    assert stats.claims_superseded == 1  # 7h → 8h supersession still resolved


# ----------------------------------------------------------------------
# 2. retrieve(tier='belief-view')
# ----------------------------------------------------------------------


async def test_retrieve_belief_view_returns_active_claim(pool: asyncpg.Pool) -> None:
    project_id = project_id_for("ret-claim")
    ep = await _episode(pool, project_id)
    await _turn(pool, ep, "I slept 7 hours last night.")

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        return [_claim("sleep.recent_hours", {"hours": 7}, turn.id)]

    await materialize_question(pool, project_id, detector=detector)

    embedder = get_provider("fastembed", dimensions=768)
    recalls = await retrieve(
        pool, embedder,
        question="how much sleep have I been getting",
        question_type="knowledge-update",
        project_id=project_id,
        tier="belief-view",
        user_id=TEST_USER,
    )

    assert len(recalls) >= 1
    # The claim content carries the attribute + rendered value for the Reader.
    assert any("sleep.recent_hours" in r.memory.content for r in recalls)
    assert any("hours=7" in r.memory.content for r in recalls)


async def test_retrieve_belief_view_falls_back_to_turns_on_miss(
    pool: asyncpg.Pool,
) -> None:
    """No matching claim → augment-not-gate fallback to turn-tier recall."""
    project_id = project_id_for("ret-fallback")
    ep = await _episode(pool, project_id)
    embedder = get_provider("fastembed", dimensions=768)
    content = "My favorite hiking trail is the Skyline Ridge loop."
    vec = await embedder.embed(content)
    turn = await _turn(pool, ep, content, embedding=vec)

    # No claims materialized for this sandbox — search_belief_claims returns []
    # and retrieve must fall through to the turn substrate.
    recalls = await retrieve(
        pool, embedder,
        question="what hiking trail do I like",
        question_type="single-session-user",
        project_id=project_id,
        tier="belief-view",
        user_id=TEST_USER,
    )

    assert len(recalls) >= 1
    # Fallback returns the turn (its id), not a belief claim id.
    assert any(r.memory.id == turn.id for r in recalls)


async def test_retrieve_turns_falls_back_to_belief_on_empty(
    pool: asyncpg.Pool,
) -> None:
    """Empty turns tier → never-miss fallback to belief recall.

    Mirrors the same resilience added to weft_recall: a question routed to the
    turns tier that finds nothing there recovers from the belief substrate
    instead of returning an empty hand. Seeds a belief memory and NO turns, so
    the turns path is empty and only the fallback can surface the answer.
    """
    from weft.models import MemoryCreate, MemorySource, MemoryType
    from weft.store import store_memory

    project_id = project_id_for("ret-turns-to-belief")
    embedder = get_provider("fastembed", dimensions=768)
    content = "My favorite hiking trail is the Skyline Ridge loop."
    vec = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact, content=content, topic=["hiking"],
            source=MemorySource.conversation, confidence=0.9, project_id=project_id,
        ),
        embedding=vec,
    )

    recalls = await retrieve(
        pool, embedder,
        question="what hiking trail do I like",
        question_type="single-session-user",
        project_id=project_id,
        tier="turns",  # no turns exist → fallback must recover the belief memory
        user_id=TEST_USER,
    )

    assert len(recalls) >= 1, "empty turns tier returned nothing — fallback did not fire"
    assert any(r.memory.id == mem.id for r in recalls)


# ----------------------------------------------------------------------
# 3. cleanup deletes belief_claims
# ----------------------------------------------------------------------


async def test_cleanup_haystack_deletes_belief_claims(pool: asyncpg.Pool) -> None:
    qid = "cleanup-claims"
    project_id = project_id_for(qid)
    ep = await _episode(pool, project_id)
    await _turn(pool, ep, "I slept 7 hours.")

    def detector(turn: EpisodeTurn) -> list[ClaimUpdate]:
        return [_claim("sleep.recent_hours", {"hours": 7}, turn.id)]

    await materialize_question(pool, project_id, detector=detector)
    assert len(await _active_claims(pool)) == 1

    instance = Instance(
        question_id=qid,
        question_type="knowledge-update",
        question="how much sleep",
        answer="7 hours",
        question_date="2026/05/26",
        sessions=(),
    )
    await cleanup_haystack(pool, instance)

    remaining = await pool.fetchval(
        "SELECT count(*) FROM belief_claims WHERE user_id = $1", TEST_USER,
    )
    assert remaining == 0


async def test_cleanup_haystack_no_turns_is_noop(pool: asyncpg.Pool) -> None:
    """A raw/extracted-mode question (no episode_turns) cleans up cleanly."""
    instance = Instance(
        question_id="cleanup-noturns",
        question_type="single-session-user",
        question="q",
        answer="a",
        question_date="2026/05/26",
        sessions=(),
    )
    # Should not raise (COALESCE empty-array guard).
    await cleanup_haystack(pool, instance)


# ----------------------------------------------------------------------
# 4. CLI guard rails — misconfigurations that would burn Reader spend
#    producing garbage must fail before any pool/env work. No DB needed.
# ----------------------------------------------------------------------


async def test_cli_rejects_belief_view_without_turns_mode(tmp_path) -> None:
    from click.testing import CliRunner

    from benchmarks.longmemeval.adapter import cli

    dataset = tmp_path / "ds.json"
    dataset.write_text("[]", encoding="utf-8")
    result = CliRunner().invoke(
        cli, ["--dataset", str(dataset), "--tier", "belief-view", "--mode", "raw"],
    )
    assert result.exit_code != 0
    assert "requires --mode turns" in result.output


async def test_cli_rejects_belief_view_with_no_cleanup(tmp_path) -> None:
    from click.testing import CliRunner

    from benchmarks.longmemeval.adapter import cli

    dataset = tmp_path / "ds.json"
    dataset.write_text("[]", encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "--dataset", str(dataset),
            "--tier", "belief-view",
            "--mode", "turns",
            "--no-cleanup",
        ],
    )
    assert result.exit_code != 0
    assert "incompatible with --no-cleanup" in result.output
