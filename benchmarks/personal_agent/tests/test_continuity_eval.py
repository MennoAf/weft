"""Deterministic A/B/C continuity artifact tests."""

from __future__ import annotations

import pytest

from benchmarks.personal_agent.continuity_eval import build_run_artifact, evaluate_arm
from benchmarks.personal_agent.continuity_manifest import HANDOFF, QUESTIONS, TURNS


pytestmark = pytest.mark.asyncio


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
