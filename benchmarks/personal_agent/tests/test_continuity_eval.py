"""Deterministic A/B/C continuity artifact tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.personal_agent.continuity_eval import (
    _assert_redacted,
    build_fixture_snapshot,
    build_run_artifact,
    evaluate_arm,
    evaluate_sessions,
)
from benchmarks.personal_agent.continuity_manifest import (
    CORE_EPISODIC_SCENARIO_IDS,
    HANDOFF,
    QUESTIONS,
    SESSIONS,
    TURNS,
    validate_manifest,
)


pytestmark = pytest.mark.asyncio


async def test_manifest_has_four_independent_sessions_and_28_stable_scenarios():
    assert validate_manifest() == {
        "sessions": 4,
        "scenarios": 28,
        "core_episodic": 16,
        "handoff_sufficient": 8,
    }
    scenario_ids = [
        session.scenario_id(question)
        for session in SESSIONS
        for question in session.questions
    ]
    assert len(scenario_ids) == len(set(scenario_ids)) == 28
    assert len(CORE_EPISODIC_SCENARIO_IDS) == 16


def _turn(key: str) -> dict:
    spec = next(turn for turn in TURNS if turn.key == key)
    return {
        "id": f"turn-{key}",
        "fixture_key": key,
        "role": "user",
        "content": spec.content,
        "occurred_at": spec.occurred_at.isoformat(),
        "authority": spec.authority,
    }


async def test_evaluator_propagates_ids_for_production_shaped_turns():
    chronology = next(question for question in QUESTIONS if question.key == "chronology")
    expected_ids = {
        key: f"turn-{key}" for key in chronology.expected_turn_keys
    }

    async def recall(_question):
        return [
            {
                "id": expected_ids[key],
                "role": "user",
                "content": next(turn.content for turn in TURNS if turn.key == key),
                "occurred_at": next(
                    turn.occurred_at.isoformat() for turn in TURNS if turn.key == key
                ),
            }
            for key in chronology.expected_turn_keys
        ]

    result = await evaluate_arm(
        arm="B",
        questions=(chronology,),
        handoff=HANDOFF,
        recall_turns=recall,
        expected_turn_ids=expected_ids,
    )
    assert result["episodic_complete_rate"] == 1.0
    assert result["scenarios"][0]["status"] == "complete"


async def test_manifest_rejects_supersession_class_drift():
    first = SESSIONS[0]
    mutated_questions = tuple(
        replace(question, question_class="other_episodic")
        if question.question_class == "superseded" else question
        for question in first.questions
    )
    mutated_sessions = (replace(first, questions=mutated_questions), *SESSIONS[1:])
    with pytest.raises(ValueError, match="exactly one supersession"):
        validate_manifest(mutated_sessions)


async def test_fixed_snapshot_is_synthetic_stable_and_contains_no_db_ids():
    snapshot = build_fixture_snapshot(SESSIONS)
    assert snapshot["synthetic"] is True
    assert snapshot["session_count"] == 4
    assert snapshot["scenario_count"] == 28
    assert all(
        "id" not in turn
        for session in snapshot["sessions"]
        for turn in session["turns"]
    )


@pytest.mark.parametrize("secret", [
    "postgresql://user:pass@host/db",
    "Bearer abc.def.ghi",
    "OPENAI_API_KEY=not-safe",
    "weft-abcdefghijklmnopqrstuvwxyz1234567890",
])
async def test_snapshot_redaction_rejects_secret_shaped_values(secret):
    with pytest.raises(ValueError, match=r"snapshot\.payload"):
        _assert_redacted({"payload": secret})


@pytest.mark.parametrize("key", [
    "password",
    "token",
    "api_key",
    "authorization_header",
    "database_dsn",
])
async def test_snapshot_redaction_rejects_nonempty_secret_bearing_fields(key):
    with pytest.raises(ValueError, match=rf"snapshot\.{key}"):
        _assert_redacted({key: "ordinary-short-value"})


async def test_multi_session_evaluator_preserves_independent_scenario_ids():
    turn_specs = {
        turn.key: turn
        for session in SESSIONS
        for turn in session.turns
    }

    async def recall(_session, question):
        return [
            {
                "id": f"turn-{key}",
                "fixture_key": key,
                "role": "user",
                "content": turn_specs[key].content,
                "occurred_at": turn_specs[key].occurred_at.isoformat(),
                "authority": turn_specs[key].authority,
            }
            for key in question.expected_turn_keys
        ]

    result = await evaluate_sessions(
        arm="B",
        sessions=SESSIONS,
        recall_turns=recall,
    )
    assert result["independent_session_count"] == 4
    assert result["scenario_count"] == 28
    assert result["core_episodic_count"] == 16
    assert result["core_episodic_complete"] == 16
    scenario_ids = [row["scenario_id"] for row in result["scenarios"]]
    assert len(scenario_ids) == len(set(scenario_ids)) == 28
    assert all(session["scenario_count"] == 7 for session in result["sessions"])


async def test_deterministic_artifact_compares_handoff_and_targeted_turn_arms():
    by_question = {
        question.key: [_turn(key) for key in question.expected_turn_keys]
        for question in QUESTIONS
    }

    async def recall(question):
        return by_question[question.key]

    arm_a = await evaluate_arm(
        arm="A",
        questions=QUESTIONS,
        handoff=HANDOFF,
    )
    arm_b = await evaluate_arm(
        arm="B",
        questions=QUESTIONS,
        handoff=HANDOFF,
        recall_turns=recall,
    )
    assert arm_a["activation_precision"] == 1.0
    assert arm_b["activation_precision"] == 1.0
    assert arm_b["episodic_complete_rate"] == 1.0

    artifact = build_run_artifact(
        commit="test-commit",
        fixture_id="continuity-v1",
        arms=[arm_a, arm_b],
    )
    assert artifact["reader_evaluation"]["status"] == "PENDING-PAID-EVALUATION"
    assert artifact["production_wiring_enabled"] is False
    assert artifact["materializer_automatic"] is False
