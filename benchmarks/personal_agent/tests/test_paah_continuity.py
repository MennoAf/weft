"""Deterministic continuity acceptance tests; no LLM calls."""

from __future__ import annotations

from unittest.mock import AsyncMock

import asyncpg
import pytest

from benchmarks.personal_agent.continuity_harness import (
    assemble_continuity_evidence,
    needs_targeted_turns,
    recall_targeted_turns,
)
from benchmarks.personal_agent.continuity_manifest import (
    HANDOFF,
    PAAH_CONTINUITY_OTHER_PROJECT_ID,
    PAAH_CONTINUITY_OTHER_USER_ID,
    QUESTIONS,
    SESSIONS,
    TURNS,
)
from benchmarks.personal_agent.continuity_seed import seed_continuity


@pytest.fixture
def questions():
    return {question.key: question for question in QUESTIONS}


def _turn(key: str, *, turn_id: str | None = None) -> dict:
    spec = next(turn for turn in TURNS if turn.key == key)
    return {
        "id": turn_id or f"turn-{key}",
        "fixture_key": key,
        "role": "user",
        "content": spec.content,
        "occurred_at": spec.occurred_at.isoformat(),
        "authority": spec.authority,
        "quoted_instruction": spec.quoted_instruction,
    }


def test_handoff_sufficient_questions_skip_targeted_turns(questions):
    assert needs_targeted_turns(questions["next_action"]) is False
    assert needs_targeted_turns(questions["final_decision"]) is False


@pytest.mark.asyncio
async def test_arm_a_never_invokes_turn_or_materialized_recall(questions):
    turns = AsyncMock()
    materialized = AsyncMock()
    evidence = await assemble_continuity_evidence(
        arm="A",
        question=questions["rejected_rationale"],
        handoff=HANDOFF,
        recall_turns=turns,
        recall_materialized=materialized,
    )
    turns.assert_not_awaited()
    materialized.assert_not_awaited()
    assert evidence.turns == []
    assert evidence.authoritative_source == "handoff"


@pytest.mark.asyncio
async def test_handoff_answers_next_action_without_turns_in_all_arms(questions):
    for arm in ("A", "B", "C"):
        turns = AsyncMock(return_value=[_turn("attempt_one")])
        evidence = await assemble_continuity_evidence(
            arm=arm,
            question=questions["next_action"],
            handoff=HANDOFF,
            recall_turns=turns,
        )
        turns.assert_not_awaited()
        assert evidence.handoff["next_steps"] == "Run the staging smoke test next."
        assert evidence.turns == []


@pytest.mark.asyncio
async def test_targeted_rationale_returns_bounded_evidence_ids(questions):
    recalled = [_turn("rejected_red")] + [
        {**_turn("attempt_one"), "id": f"distractor-{index}"}
        for index in range(10)
    ]
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["rejected_rationale"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=recalled),
        max_turns=3,
    )
    assert evidence.status == "complete"
    assert evidence.evidence_turn_ids[0] == "turn-rejected_red"
    assert len(evidence.turns) == len(evidence.evidence_turn_ids) == 3


@pytest.mark.asyncio
async def test_quoted_instruction_remains_labelled_data(questions):
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["chronology"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=[_turn("quoted_instruction")]),
    )
    rendered = evidence.to_dict()["turn_evidence"][0]
    assert rendered["kind"] == "quoted_dialogue_evidence"
    assert rendered["quoted_instruction"] is True
    assert "Ignore previous instructions" in rendered["content"]
    assert evidence.status == "incomplete"
    assert evidence.incomplete_reason == (
        "missing_expected_turns:attempt_one,final_blue,rejected_red"
    )
    assert evidence.authoritative_source == "handoff"


@pytest.mark.asyncio
async def test_chronology_is_ordered_and_requires_all_expected_turns(questions):
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["chronology"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=[
            _turn("final_blue"),
            _turn("rejected_red"),
            _turn("attempt_one"),
        ]),
    )
    assert evidence.status == "complete"
    assert [turn["fixture_key"] for turn in evidence.turns] == [
        "attempt_one",
        "rejected_red",
        "final_blue",
    ]


@pytest.mark.asyncio
async def test_real_shaped_partial_recall_is_never_marked_complete(questions):
    real_turn = {
        "id": "turn-attempt-one",
        "role": "assistant",
        "content": "Attempt one failed.",
        "occurred_at": "2026-07-18T10:00:00+00:00",
    }
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["chronology"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=[real_turn]),
        expected_turn_ids={
            "attempt_one": "turn-attempt-one",
            "rejected_red": "turn-rejected-red",
            "final_blue": "turn-final-blue",
        },
    )
    assert evidence.status == "incomplete"
    assert evidence.incomplete_reason == (
        "missing_expected_turns:final_blue,rejected_red"
    )


@pytest.mark.asyncio
async def test_final_handoff_authoritative_over_superseded_turn(questions):
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["superseded"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(
            return_value=[_turn("early_green"), _turn("final_blue")],
        ),
    )
    assert evidence.authoritative_source == "handoff"
    assert evidence.handoff["summary"].endswith("blue launch plan.")
    assert evidence.turns[0]["authority"] == "superseded"


@pytest.mark.asyncio
async def test_turn_failure_is_explicit_and_never_dumps_transcript(questions):
    recall = AsyncMock(side_effect=RuntimeError("retrieval unavailable"))
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["omitted_detail"],
        handoff=HANDOFF,
        recall_turns=recall,
    )
    assert evidence.status == "incomplete"
    assert evidence.incomplete_reason == "turn_recall_failed"
    assert evidence.turns == []
    assert evidence.evidence_turn_ids == []
    assert "transcript" not in evidence.to_dict()


@pytest.mark.asyncio
async def test_missing_turns_report_incomplete_evidence(questions):
    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["exact_wording"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=[]),
    )
    assert evidence.status == "incomplete"
    assert evidence.incomplete_reason == "no_relevant_turns"


@pytest.mark.asyncio
async def test_materializer_arm_is_opt_in_and_preserves_evidence_ids(questions):
    materialized = AsyncMock(return_value=[{
        "attribute": "launch.plan",
        "value": "blue",
        "status": "active",
        "evidence_turn_ids": ["turn-final_blue"],
    }])
    arm_b = await assemble_continuity_evidence(
        arm="B",
        question=questions["final_decision"],
        handoff=HANDOFF,
        recall_materialized=materialized,
    )
    materialized.assert_not_awaited()
    assert arm_b.materialized_beliefs == []

    arm_c = await assemble_continuity_evidence(
        arm="C",
        question=questions["final_decision"],
        handoff=HANDOFF,
        recall_materialized=materialized,
    )
    assert arm_c.materialized_beliefs
    assert arm_c.evidence_turn_ids == ["turn-final_blue"]
    assert arm_c.authoritative_source == "materialized"


@pytest.fixture
async def seeded_continuity(pool):
    return await seed_continuity(pool)


@pytest.mark.asyncio
async def test_real_seed_contains_four_independent_sessions_and_28_turns(
    pool, seeded_continuity,
):
    assert seeded_continuity.clean is True
    assert set(seeded_continuity.episode_ids) == {
        session.session_id for session in SESSIONS
    }
    assert all(
        len(seeded_continuity.turn_ids_by_session[session.session_id]) == 7
        for session in SESSIONS
    )
    episode_count = await pool.fetchval(
        "SELECT COUNT(*) FROM episodes WHERE id = ANY($1::text[])",
        list(seeded_continuity.episode_ids.values()),
    )
    turn_count = await pool.fetchval(
        "SELECT COUNT(*) FROM episode_turns WHERE episode_id = ANY($1::text[])",
        list(seeded_continuity.episode_ids.values()),
    )
    assert episode_count == 4
    assert turn_count == 28


@pytest.fixture
async def scoped_pool_factory(pool):
    role = "paah_continuity_app"
    password = "paah_continuity_pass"
    await pool.execute(f"""
        DO $$ BEGIN
          IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN
            CREATE ROLE {role} LOGIN PASSWORD '{password}' NOSUPERUSER NOBYPASSRLS;
          END IF;
        END $$
    """)
    await pool.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
    # Auto-tier recall may fuse belief and turn evidence. Grant the restricted
    # application role read access to both real recall substrates; RLS, not
    # missing table privileges, must prove user/project isolation.
    await pool.execute(
        f"GRANT SELECT ON memories, episodes, episode_turns, workspace_members "
        f"TO {role}"
    )
    await pool.execute(f"GRANT INSERT, UPDATE ON weft_recall_queries TO {role}")
    await pool.execute(f"GRANT INSERT, UPDATE ON turn_access_log TO {role}")
    async with pool.acquire() as conn:
        host, port = conn._addr
        database = conn._params.database

    opened = []

    async def factory(user_id: str):
        async def setup(conn):
            await conn.execute("SELECT set_config('app.user_id', $1, false)", user_id)

        restricted = await asyncpg.create_pool(
            f"postgresql://{role}:{password}@{host}:{port}/{database}",
            min_size=2,
            max_size=5,
            setup=setup,
        )
        opened.append(restricted)
        return restricted

    yield factory
    for restricted in opened:
        await restricted.close()


@pytest.mark.asyncio
async def test_real_targeted_turn_ids_for_omitted_rationale(
    scoped_pool_factory, seeded_continuity, questions,
):
    scoped_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        scoped_pool, questions["rejected_rationale"].query, limit=4,
    )
    ids = [turn["id"] for turn in turns]
    assert seeded_continuity.turn_ids["rejected_red"] in ids
    assert seeded_continuity.other_project_turn_id not in ids
    assert seeded_continuity.other_user_turn_id not in ids
    assert len(ids) <= 4


@pytest.mark.asyncio
async def test_real_targeted_turn_evidence_preserves_exact_wording(
    scoped_pool_factory, seeded_continuity, questions,
):
    scoped_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        scoped_pool, questions["exact_wording"].query, limit=4,
    )
    expected = seeded_continuity.turn_ids["exact_words"]
    matching = [turn for turn in turns if turn["id"] == expected]
    assert len(matching) == 1
    assert '"Blue buys us a reversible launch."' in matching[0]["content"]


@pytest.mark.asyncio
async def test_real_instruction_shaped_turn_is_wrapped_as_quoted_evidence(
    scoped_pool_factory, seeded_continuity, questions,
):
    scoped_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        scoped_pool,
        'When did the log say "Ignore previous instructions"?',
        limit=4,
    )
    expected = seeded_continuity.turn_ids["quoted_instruction"]
    matching = [turn for turn in turns if turn["id"] == expected]
    assert len(matching) == 1

    evidence = await assemble_continuity_evidence(
        arm="B",
        question=questions["exact_wording"],
        handoff=HANDOFF,
        recall_turns=AsyncMock(return_value=matching),
    )
    rendered = evidence.to_dict()["turn_evidence"]
    assert rendered[0]["kind"] == "quoted_dialogue_evidence"
    assert "Ignore previous instructions" in rendered[0]["content"]
    assert evidence.authoritative_source == "handoff"


@pytest.mark.asyncio
async def test_real_targeted_turns_preserve_chronological_metadata(
    scoped_pool_factory, seeded_continuity, questions,
):
    scoped_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        scoped_pool, questions["chronology"].query, limit=7,
    )
    by_id = {turn["id"]: turn for turn in turns}
    wanted = [
        seeded_continuity.turn_ids[key]
        for key in ("attempt_one", "rejected_red", "final_blue")
    ]
    assert set(wanted).issubset(by_id)
    occurred = [by_id[turn_id]["occurred_at"] for turn_id in wanted]
    assert occurred == sorted(occurred)


@pytest.mark.asyncio
async def test_turn_recall_project_isolation(
    scoped_pool_factory, seeded_continuity,
):
    scoped_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        scoped_pool,
        "Why did we reject red?",
        project_id=PAAH_CONTINUITY_OTHER_PROJECT_ID,
        limit=5,
    )
    ids = {turn["id"] for turn in turns}
    assert ids == {seeded_continuity.other_project_turn_id}
    assert not ids.intersection(seeded_continuity.turn_ids.values())


@pytest.mark.asyncio
async def test_explicit_user_filter_cannot_override_connection_identity(
    scoped_pool_factory, seeded_continuity,
):
    default_user_pool = await scoped_pool_factory("test-user-default")
    turns = await recall_targeted_turns(
        default_user_pool,
        "Why did we reject red?",
        user_id=PAAH_CONTINUITY_OTHER_USER_ID,
        authenticated_user_id="test-user-default",
        limit=5,
    )
    ids = {turn["id"] for turn in turns}
    assert seeded_continuity.other_user_turn_id not in ids


@pytest.mark.asyncio
async def test_turn_recall_user_isolation(
    scoped_pool_factory, seeded_continuity,
):
    other_pool = await scoped_pool_factory(PAAH_CONTINUITY_OTHER_USER_ID)
    visible_seed = await other_pool.fetchval(
        "SELECT id FROM episode_turns WHERE id = $1",
        seeded_continuity.other_user_turn_id,
    )
    assert visible_seed == seeded_continuity.other_user_turn_id
    turns = await recall_targeted_turns(
        other_pool,
        "Why did we reject red?",
        user_id=PAAH_CONTINUITY_OTHER_USER_ID,
        limit=5,
    )
    ids = {turn["id"] for turn in turns}
    assert ids == {seeded_continuity.other_user_turn_id}
    assert not ids.intersection(seeded_continuity.turn_ids.values())
