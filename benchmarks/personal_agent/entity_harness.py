#!/usr/bin/env python3
"""
entity_harness.py — PAAH entity-brief shape (beliefs+graph, agent-facing).

Measures how completely each read path answers "what do I need to know about X":

  * ORACLE   = weft_entity_context(entity_id) — walks the entity→memory edges.
    recall@links should be 1.0 (the complete brief) in one deterministic call.
  * CANDIDATE = weft_recall("brief me on X") — natural-language top-k. Run over k
    phrasings; report min/median/max recall@links. The spread is the never-miss
    signal and the gap vs the oracle is the finding: if NL recall is < 1.0, an
    agent building a brief should walk the entity graph, not trust top-k.

recall@links = |surfaced memory ids ∩ seeded fact ids| / |seeded fact ids|,
matched by id so a distractor entity's facts can't inflate it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median

import asyncpg

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.entity_manifest import (
    ENTITY_LIMIT,
    PAAH_USER_ID,
    get_brief_phrasings,
)
from benchmarks.personal_agent.seed import SeedEntityBriefResult

logger = logging.getLogger(__name__)


@dataclass
class CandidateRun:
    query: str
    routed_tier: str | None    # None = belief legacy path
    fell_back: bool            # True if the turns tier was empty → belief recovery
    recall_at_links: float


@dataclass
class EntityStats:
    name: str
    fact_count: int
    limit: int
    oracle_recall: float = 0.0
    oracle_returned: int = 0
    candidate: list[CandidateRun] = field(default_factory=list)

    @property
    def oracle_complete(self) -> bool:
        return self.oracle_recall == 1.0

    @property
    def _recalls(self) -> list[float]:
        return [c.recall_at_links for c in self.candidate]

    @property
    def candidate_min(self) -> float:
        return min(self._recalls) if self._recalls else 0.0

    @property
    def candidate_median(self) -> float:
        return median(self._recalls) if self._recalls else 0.0

    @property
    def candidate_max(self) -> float:
        return max(self._recalls) if self._recalls else 0.0

    @property
    def fallbacks(self) -> list[CandidateRun]:
        """Runs where the turns tier was empty and belief recovery kicked in."""
        return [c for c in self.candidate if c.fell_back]

    @property
    def never_empty(self) -> bool:
        """Never-miss: every phrasing surfaced the complete brief."""
        return self.candidate_min == 1.0

    def to_dict(self) -> dict:
        return {
            "entity": self.name,
            "fact_count": self.fact_count,
            "limit": self.limit,
            "oracle_entity_context": {
                "returned": self.oracle_returned,
                "recall_at_links": self.oracle_recall,
                "complete": self.oracle_complete,
            },
            "candidate_weft_recall": {
                "runs": len(self.candidate),
                "min_recall_at_links": self.candidate_min,
                "median_recall_at_links": self.candidate_median,
                "max_recall_at_links": self.candidate_max,
                "never_empty": self.never_empty,
                "fallback_recoveries": len(self.fallbacks),
                "per_query": [
                    {
                        "query": c.query,
                        "routed_tier": c.routed_tier or "belief",
                        "fell_back": c.fell_back,
                        "recall_at_links": c.recall_at_links,
                    }
                    for c in self.candidate
                ],
            },
        }


async def run_entity_brief_paah(
    pool: asyncpg.Pool,
    seeded: SeedEntityBriefResult,
    limit: int = ENTITY_LIMIT,
) -> EntityStats:
    """Measure oracle (entity_context) vs candidate (weft_recall) recall@links."""
    from weft.mcp.tools import weft_entity_context, weft_recall

    app = await build_app_context(pool)
    ctx = make_ctx(app)

    brief = seeded.brief
    fact_ids = set(brief.fact_ids)
    stats = EntityStats(
        name=brief.spec.name, fact_count=brief.intended, limit=limit,
    )

    token = current_user_id.set(PAAH_USER_ID)
    try:
        # --- ORACLE: entity_context edge walk ---
        ctx_resp = await weft_entity_context(ctx, entity_id=brief.entity_id)
        oracle_ids = {m.get("id") for m in ctx_resp.get("memories", []) if m.get("id")}
        stats.oracle_returned = len(oracle_ids & fact_ids)
        stats.oracle_recall = (
            len(oracle_ids & fact_ids) / len(fact_ids) if fact_ids else 0.0
        )
        logger.info(
            "paah_entity ORACLE: entity_context returned %d of %d linked facts "
            "(recall@links=%.3f)",
            stats.oracle_returned, len(fact_ids), stats.oracle_recall,
        )

        # --- CANDIDATE: natural-language brief queries ---
        # tier='auto' so the REAL router runs — a brief phrasing that contains a
        # temporal word ("before our meeting") routes to the turns tier and
        # misses the entity's belief-tier facts entirely. Capturing routed_tier
        # turns that into a diagnosable finding instead of a mystery zero.
        for query in get_brief_phrasings():
            resp = await weft_recall(ctx, query=query, limit=limit, tier="auto")
            routed_tier = resp.get("tier")  # None for the belief legacy path
            fell_back = "tier_fallback" in resp  # turns tier was empty → belief
            surfaced = {r.get("id") for r in resp.get("results", []) if r.get("id")}
            recall = len(surfaced & fact_ids) / len(fact_ids) if fact_ids else 0.0
            stats.candidate.append(
                CandidateRun(
                    query=query, routed_tier=routed_tier,
                    fell_back=fell_back, recall_at_links=recall,
                )
            )
            logger.info(
                "paah_entity CANDIDATE: query=%r routed=%s fell_back=%s surfaced "
                "%d of %d (recall@links=%.3f)",
                query, routed_tier or "belief", fell_back,
                len(surfaced & fact_ids), len(fact_ids), recall,
            )
    finally:
        current_user_id.reset(token)

    logger.info("paah_entity stats: %s", stats.to_dict())
    return stats
