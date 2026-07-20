"""Deterministic continuity run artifacts; no reader-model calls.

This scorer measures retrieval mechanics only. Answer quality remains an
operator-approved repeated paid evaluation and is never inferred here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Awaitable, Callable

from benchmarks.personal_agent.continuity_harness import (
    Arm,
    assemble_continuity_evidence,
)
from benchmarks.personal_agent.continuity_manifest import (
    CONTINUITY_MANIFEST_ID,
    CORE_EPISODIC_SCENARIO_IDS,
    ContinuityQuestion,
    ContinuitySession,
)

RecallFactory = Callable[[ContinuityQuestion], Awaitable[list[dict]]]

_SECRET_KEY = re.compile(
    r"(?:^|_)(?:authorization|bearer|dsn|api[_-]?key|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:postgres(?:ql)?|redis)://", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~-]+", re.IGNORECASE),
    re.compile(r"\b(?:sk|weft)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*(?:api[_-]?key|password|secret|token)[A-Za-z0-9_]*\s*[:=]", re.IGNORECASE),
)


def _assert_redacted(value: object, *, path: str = "snapshot") -> None:
    """Reject secret-shaped content before a benchmark artifact is written."""
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            raise ValueError(f"secret-shaped value rejected at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY.search(key_text) and item not in (None, "", False, [], {}):
                raise ValueError(f"secret-bearing field rejected at {path}.{key_text}")
            _assert_redacted(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_redacted(item, path=f"{path}[{index}]")


def build_fixture_snapshot(
    sessions: tuple[ContinuitySession, ...],
) -> dict:
    """Serialize fixed synthetic evidence without DB IDs or production content."""
    snapshot = {
        "schema_version": 1,
        "manifest_id": CONTINUITY_MANIFEST_ID,
        "synthetic": True,
        "session_count": len(sessions),
        "scenario_count": sum(len(session.questions) for session in sessions),
        "sessions": [
            {
                "session_id": session.session_id,
                "project_id": session.project_id,
                "handoff": dict(session.handoff),
                "turns": [
                    {
                        "fixture_key": turn.key,
                        "role": "user",
                        "content": turn.content,
                        "occurred_at": turn.occurred_at.isoformat(),
                        "authority": turn.authority,
                        "quoted_instruction": turn.quoted_instruction,
                    }
                    for turn in session.turns
                ],
                "questions": [
                    {
                        "scenario_id": session.scenario_id(question),
                        "key": question.key,
                        "question_class": question.question_class,
                        "query": question.query,
                        "handoff_sufficient": question.handoff_sufficient,
                        "expected_turn_keys": list(question.expected_turn_keys),
                    }
                    for question in session.questions
                ],
            }
            for session in sessions
        ],
    }
    _assert_redacted(snapshot)
    return snapshot


async def evaluate_sessions(
    *,
    arm: Arm,
    sessions: tuple[ContinuitySession, ...],
    recall_turns: Callable[
        [ContinuitySession, ContinuityQuestion], Awaitable[list[dict]]
    ] | None = None,
    recall_materialized: Callable[
        [ContinuitySession, ContinuityQuestion], Awaitable[list[dict]]
    ] | None = None,
    expected_turn_ids: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Evaluate all independent sessions without treating repeats as samples."""
    session_results: list[dict] = []
    flat_scenarios: list[dict] = []
    for session in sessions:
        async def turns(
            question: ContinuityQuestion,
            fixture: ContinuitySession = session,
        ) -> list[dict]:
            return await recall_turns(fixture, question) if recall_turns else []

        async def materialized(
            question: ContinuityQuestion,
            fixture: ContinuitySession = session,
        ) -> list[dict]:
            return (
                await recall_materialized(fixture, question)
                if recall_materialized else []
            )

        result = await evaluate_arm(
            arm=arm,
            questions=session.questions,
            handoff=session.handoff,
            recall_turns=turns if recall_turns else None,
            recall_materialized=materialized if recall_materialized else None,
            expected_turn_ids=(expected_turn_ids or {}).get(session.session_id),
        )
        for scenario in result["scenarios"]:
            scenario["session_id"] = session.session_id
            scenario["scenario_id"] = f"{session.session_id}:{scenario['key']}"
        session_results.append({
            "session_id": session.session_id,
            "activation_precision": result["activation_precision"],
            "episodic_complete_rate": result["episodic_complete_rate"],
            "scenario_count": result["scenario_count"],
        })
        flat_scenarios.extend(result["scenarios"])

    core_ids = set(CORE_EPISODIC_SCENARIO_IDS)
    core = [row for row in flat_scenarios if row["scenario_id"] in core_ids]
    return {
        "arm": arm,
        "manifest_id": CONTINUITY_MANIFEST_ID,
        "independent_session_count": len(sessions),
        "scenario_count": len(flat_scenarios),
        "core_episodic_count": len(core),
        "core_episodic_complete": sum(row["status"] == "complete" for row in core),
        "sessions": session_results,
        "scenarios": flat_scenarios,
    }


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
