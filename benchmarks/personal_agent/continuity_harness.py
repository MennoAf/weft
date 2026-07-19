"""Benchmark-only A/B/C context assembly for session continuity.

Production prime and recall wiring are not changed. This module makes the
handoff-first contract measurable and keeps raw dialogue labelled as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

import asyncpg

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.continuity_manifest import (
    PAAH_CONTINUITY_PROJECT_ID,
    PAAH_CONTINUITY_USER_ID,
    ContinuityQuestion,
)
from weft.turn_recall import route_query_to_tier

Arm = Literal["A", "B", "C"]
TurnRecall = Callable[[str], Awaitable[list[dict]]]
MaterializedRecall = Callable[[str], Awaitable[list[dict]]]


@dataclass(slots=True)
class ContinuityEvidence:
    arm: Arm
    query: str
    router_tier: str
    handoff: dict
    durable_memories: list[dict] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    materialized_beliefs: list[dict] = field(default_factory=list)
    evidence_turn_ids: list[str] = field(default_factory=list)
    status: Literal["complete", "incomplete"] = "complete"
    incomplete_reason: str | None = None
    authoritative_source: Literal["handoff", "durable", "materialized"] = "handoff"

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "query": self.query,
            "router_tier": self.router_tier,
            "handoff": self.handoff,
            "durable_memories": self.durable_memories,
            "turn_evidence": [
                {
                    "kind": "quoted_dialogue_evidence",
                    "id": turn.get("id"),
                    "role": turn.get("role"),
                    "content": turn.get("content"),
                    "occurred_at": turn.get("occurred_at"),
                    "authority": turn.get("authority", "evidence"),
                    "quoted_instruction": bool(turn.get("quoted_instruction", False)),
                }
                for turn in self.turns
            ],
            "materialized_beliefs": self.materialized_beliefs,
            "evidence_turn_ids": self.evidence_turn_ids,
            "status": self.status,
            "incomplete_reason": self.incomplete_reason,
            "authoritative_source": self.authoritative_source,
        }


async def recall_targeted_turns(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str = PAAH_CONTINUITY_PROJECT_ID,
    user_id: str = PAAH_CONTINUITY_USER_ID,
    authenticated_user_id: str | None = None,
    limit: int = 6,
) -> list[dict]:
    """Call real auto-tier recall and normalize only raw turn evidence."""
    from weft.auth import current_user_id
    from weft.mcp.tools import weft_recall

    app = await build_app_context(pool)
    ctx = make_ctx(app)
    token = current_user_id.set(authenticated_user_id or user_id)
    try:
        # Turn recall intentionally fans out across the pool. Do not wrap this
        # in acquire(): sharing one transaction connection across concurrent
        # subqueries produces asyncpg "operation in progress" failures. The
        # pool's setup callback must apply the matching app.user_id GUC.
        response = await weft_recall(
            ctx,
            query=query,
            project_id=project_id,
            user_id=user_id,
            limit=limit,
            tier="auto",
        )
    finally:
        current_user_id.reset(token)

    if response.get("tier_fallback"):
        return []
    if response.get("tier") == "turns":
        return list(response.get("turns", []))
    if response.get("tier") == "both":
        return [
            entry["payload"]
            for entry in response.get("results", [])
            if entry.get("kind") == "turn" and isinstance(entry.get("payload"), dict)
        ]
    return []


def needs_targeted_turns(question: ContinuityQuestion) -> bool:
    """Activate turns only when handoff is insufficient and routing is episodic."""
    if question.handoff_sufficient:
        return False
    return route_query_to_tier(question.query) in {"turns", "both"}


async def assemble_continuity_evidence(
    *,
    arm: Arm,
    question: ContinuityQuestion,
    handoff: dict,
    durable_memories: list[dict] | None = None,
    recall_turns: TurnRecall | None = None,
    recall_materialized: MaterializedRecall | None = None,
    expected_turn_ids: dict[str, str] | None = None,
    max_turns: int = 6,
) -> ContinuityEvidence:
    """Assemble one arm without fabricating evidence on retrieval failure."""
    evidence = ContinuityEvidence(
        arm=arm,
        query=question.query,
        router_tier=route_query_to_tier(question.query),
        handoff=handoff,
        durable_memories=list(durable_memories or []),
    )
    if evidence.durable_memories:
        evidence.authoritative_source = "durable"

    targeted = needs_targeted_turns(question)
    if arm in {"B", "C"} and targeted:
        if recall_turns is None:
            evidence.status = "incomplete"
            evidence.incomplete_reason = "turn_recall_unavailable"
            return evidence
        try:
            evidence.turns = list(await recall_turns(question.query))[:max_turns]
        except Exception:
            evidence.status = "incomplete"
            evidence.incomplete_reason = "turn_recall_failed"
            return evidence
        if not evidence.turns:
            evidence.status = "incomplete"
            evidence.incomplete_reason = "no_relevant_turns"
            return evidence
        if question.question_class == "chronology":
            evidence.turns.sort(key=lambda turn: str(turn.get("occurred_at", "")))
        expected_keys = set(question.expected_turn_keys)
        if expected_keys:
            recalled_keys = {
                turn.get("fixture_key")
                for turn in evidence.turns
                if isinstance(turn.get("fixture_key"), str)
            }
            if expected_turn_ids is not None:
                recalled_ids = {
                    turn["id"]
                    for turn in evidence.turns
                    if isinstance(turn.get("id"), str)
                }
                missing = sorted(
                    key
                    for key in expected_keys
                    if expected_turn_ids.get(key) not in recalled_ids
                )
            elif recalled_keys:
                missing = sorted(expected_keys - recalled_keys)
            else:
                # Production-shaped turns do not carry synthetic fixture keys.
                # Without the seed's stable key→ID map, completeness is unknown
                # and must never be reported as successful.
                missing = sorted(expected_keys)
            if missing:
                evidence.status = "incomplete"
                evidence.incomplete_reason = "missing_expected_turns:" + ",".join(missing)
        evidence.evidence_turn_ids = [
            turn["id"] for turn in evidence.turns if isinstance(turn.get("id"), str)
        ]

    if arm == "C" and recall_materialized is not None:
        try:
            beliefs = list(await recall_materialized(question.query))
        except Exception:
            evidence.status = "incomplete"
            evidence.incomplete_reason = "materialized_recall_failed"
            return evidence
        evidence.materialized_beliefs = beliefs
        claim_ids = {
            turn_id
            for belief in beliefs
            for turn_id in belief.get("evidence_turn_ids", [])
            if isinstance(turn_id, str)
        }
        evidence.evidence_turn_ids = sorted(set(evidence.evidence_turn_ids) | claim_ids)
        if beliefs:
            evidence.authoritative_source = "materialized"

    # Final handoff/durable state outranks superseded dialogue. Raw turns remain
    # evidence, never instructions or an authority override.
    if any(turn.get("authority") == "superseded" for turn in evidence.turns):
        evidence.authoritative_source = (
            "durable" if evidence.durable_memories else "handoff"
        )
    return evidence
