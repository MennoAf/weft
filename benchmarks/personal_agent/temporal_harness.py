#!/usr/bin/env python3
"""
temporal_harness.py — PAAH temporal/dialogue shape (turn-tier, agent-facing).

Drives the real ``weft_recall`` with temporal ("when did X", "how long since X")
and dialogue ("what did I last say about Y") queries and asserts on the response
an agent would actually receive. Two structural signals per probe, across k
phrasings (the spread is the never-miss signal):

  * routed_correct — did route_query_to_tier pick the expected tier
    ('turns'/'both')? A misroute means the turn path never fired.
  * anchor_present — is the ONE seeded turn that answers the probe surfaced (by
    id) in the response? For 'turns' that's response["turns"]; for 'both' it's
    the kind=='turn' payloads. This is the Branch-A "does turn-tier recall
    actually answer" number the roadmap flagged as never measured.

Anchors are matched by turn id (from the seed map), never by content, so a
paraphrase or wrapping can't confuse the match. Retrieval pressure is real:
TEMPORAL_LIMIT (8) < 16 seeded turns, so the anchor must rank into the top-k.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median

import asyncpg

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.seed import SeedTurnsResult
from benchmarks.personal_agent.temporal_manifest import (
    PAAH_TEMPORAL_PROJECT_ID,
    PAAH_USER_ID,
    TEMPORAL_LIMIT,
    TurnProbe,
    get_turn_probes,
)

logger = logging.getLogger(__name__)


def _surfaced_turn_ids(response: dict) -> list[str]:
    """Extract the turn ids an agent would see, whichever tier answered."""
    tier = response.get("tier")
    if tier == "turns":
        return [t.get("id") for t in response.get("turns", []) if t.get("id")]
    if tier == "both":
        return [
            e["payload"].get("id")
            for e in response.get("results", [])
            if e.get("kind") == "turn" and e.get("payload", {}).get("id")
        ]
    # belief legacy (misroute) — no turns surfaced
    return []


@dataclass
class TurnRunRecord:
    query: str
    routed_tier: str | None
    routed_correct: bool
    anchor_present: bool
    surfaced_turns: int


@dataclass
class ProbeStats:
    key: str
    expected_tier: str
    anchor_key: str
    anchor_turn_id: str
    limit: int
    runs: list[TurnRunRecord] = field(default_factory=list)

    @property
    def routed_correct_rate(self) -> float:
        return sum(r.routed_correct for r in self.runs) / len(self.runs)

    @property
    def anchor_present_rate(self) -> float:
        return sum(r.anchor_present for r in self.runs) / len(self.runs)

    @property
    def anchor_present_min(self) -> float:
        # 0/1 per run — min is the never-miss floor (1.0 = never missed).
        return min(float(r.anchor_present) for r in self.runs)

    @property
    def anchor_present_median(self) -> float:
        return median(float(r.anchor_present) for r in self.runs)

    def to_dict(self) -> dict:
        return {
            "probe": self.key,
            "expected_tier": self.expected_tier,
            "anchor_key": self.anchor_key,
            "limit": self.limit,
            "runs": len(self.runs),
            "routed_correct_rate": self.routed_correct_rate,
            "anchor_present": {
                "rate": self.anchor_present_rate,
                "min": self.anchor_present_min,
                "median": self.anchor_present_median,
            },
            "queries": [
                {
                    "query": r.query,
                    "routed_tier": r.routed_tier,
                    "routed_correct": r.routed_correct,
                    "anchor_present": r.anchor_present,
                    "surfaced_turns": r.surfaced_turns,
                }
                for r in self.runs
            ],
        }


async def _run_probe_query(
    ctx, probe: TurnProbe, query: str, anchor_turn_id: str, limit: int
) -> TurnRunRecord:
    from weft.mcp.tools import weft_recall

    response = await weft_recall(
        ctx,
        query=query,
        limit=limit,
        tier="auto",
        project_id=PAAH_TEMPORAL_PROJECT_ID,
    )
    routed_tier = response.get("tier")
    surfaced = _surfaced_turn_ids(response)
    anchor_present = anchor_turn_id in surfaced

    logger.info(
        "paah_temporal: probe=%s query=%r routed=%s (want %s) anchor_present=%s "
        "surfaced=%d",
        probe.key, query, routed_tier, probe.expected_tier, anchor_present,
        len(surfaced),
    )
    return TurnRunRecord(
        query=query,
        routed_tier=routed_tier,
        routed_correct=(routed_tier == probe.expected_tier),
        anchor_present=anchor_present,
        surfaced_turns=len(surfaced),
    )


async def run_temporal_paah(
    pool: asyncpg.Pool,
    seeded: SeedTurnsResult,
    limit: int = TEMPORAL_LIMIT,
) -> list[ProbeStats]:
    """Drive weft_recall over every temporal/dialogue probe; return per-probe stats."""
    app = await build_app_context(pool)
    ctx = make_ctx(app)

    all_stats: list[ProbeStats] = []
    token = current_user_id.set(PAAH_USER_ID)
    try:
        for probe in get_turn_probes():
            anchor_turn_id = seeded.turn_ids[probe.anchor_key]
            stats = ProbeStats(
                key=probe.key,
                expected_tier=probe.expected_tier,
                anchor_key=probe.anchor_key,
                anchor_turn_id=anchor_turn_id,
                limit=limit,
            )
            for phrasing in probe.phrasings:
                stats.runs.append(
                    await _run_probe_query(ctx, probe, phrasing, anchor_turn_id, limit)
                )
            all_stats.append(stats)
            logger.info("paah_temporal stats: %s", stats.to_dict())
    finally:
        current_user_id.reset(token)

    return all_stats
