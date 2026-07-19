"""Deterministic continuity run artifacts; no reader-model calls.

This scorer measures retrieval mechanics only. Answer quality remains an
operator-approved repeated paid evaluation and is never inferred here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable

from benchmarks.personal_agent.continuity_harness import (
    Arm,
    assemble_continuity_evidence,
)
from benchmarks.personal_agent.continuity_manifest import ContinuityQuestion

RecallFactory = Callable[[ContinuityQuestion], Awaitable[list[dict]]]


async def evaluate_arm(
    *,
    arm: Arm,
    questions: tuple[ContinuityQuestion, ...],
    handoff: dict,
    recall_turns: RecallFactory | None = None,
    recall_materialized: RecallFactory | None = None,
    expected_turn_ids: dict[str, str] | None = None,
) -> dict:
    """Evaluate deterministic activation/evidence mechanics for one arm."""
    scenarios: list[dict] = []
    for question in questions:
        async def turns(query: str, q: ContinuityQuestion = question) -> list[dict]:
            del query
            return await recall_turns(q) if recall_turns else []

        async def materialized(
            query: str, q: ContinuityQuestion = question,
        ) -> list[dict]:
            del query
            return await recall_materialized(q) if recall_materialized else []

        evidence = await assemble_continuity_evidence(
            arm=arm,
            question=question,
            handoff=handoff,
            recall_turns=turns if recall_turns else None,
            recall_materialized=materialized if recall_materialized else None,
            expected_turn_ids=expected_turn_ids,
        )
        scenarios.append({
            "key": question.key,
            "question_class": question.question_class,
            "handoff_sufficient": question.handoff_sufficient,
            "router_tier": evidence.router_tier,
            "turn_recall_activated": bool(evidence.turns),
            "status": evidence.status,
            "incomplete_reason": evidence.incomplete_reason,
            "evidence_turn_ids": evidence.evidence_turn_ids,
        })

    handoff_cases = [row for row in scenarios if row["handoff_sufficient"]]
    episodic_cases = [row for row in scenarios if not row["handoff_sufficient"]]
    return {
        "arm": arm,
        "scenario_count": len(scenarios),
        "activation_precision": (
            sum(not row["turn_recall_activated"] for row in handoff_cases)
            / len(handoff_cases)
            if handoff_cases else 1.0
        ),
        "episodic_complete_rate": (
            sum(row["status"] == "complete" for row in episodic_cases)
            / len(episodic_cases)
            if episodic_cases else 1.0
        ),
        "scenarios": scenarios,
    }


def build_run_artifact(*, commit: str, fixture_id: str, arms: list[dict]) -> dict:
    """Create a versioned result envelope with the paid gate explicit."""
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "fixture_id": fixture_id,
        "evaluation_kind": "deterministic-retrieval-mechanics",
        "arms": arms,
        "reader_evaluation": {
            "status": "PENDING-PAID-EVALUATION",
            "required_repetitions_per_arm": "3-5",
            "metrics": [
                "answer_correctness",
                "instruction_non_compliance",
                "unsupported_claims",
                "evidence_citation",
                "stale_or_superseded_answers",
                "latency",
                "token_cost",
                "infrastructure_failures",
            ],
        },
        "production_wiring_enabled": False,
        "materializer_automatic": False,
    }
