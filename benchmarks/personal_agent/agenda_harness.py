#!/usr/bin/env python3
"""
agenda_harness.py — PAAH agenda shape (trackers, agent-facing).

Answers "what's on my plate" two ways and measures the gap between them:

  * ORACLE = weft_tracker_due() — the deterministic due-loop query. It must return
    EXACTLY the seeded open loops (recall@due == 1.0 AND precision == 1.0, i.e. no
    snoozed / future / terminal / no-nudge tracker leaks in), ordered oldest-due
    first so the thing you keep pushing surfaces at the top. Because this is a pure
    SQL predicate — no embeddings — one run is authoritative (the banned
    single-run rule, weft-b015d16a, is about the stochastic recall pipeline, which
    this shape doesn't touch).

  * AGENT-FACING = weft_daily_brief() — the morning digest an agent actually reads
    to answer "what do I have today." The finding this shape looks for: does that
    digest carry the open loops the oracle knows about? weft_tracker_due's own
    docstring says it's "for the daily-brief 'open loops' section," but
    assemble_daily_brief has no such section — a consumption-contract gap directly
    analogous to enumeration (the reconciliation header knew the count; the answer
    the agent read did not). We measure brief coverage of the due set by title so
    the gap is a number, not a hunch.

recall@due / precision are matched by tracker id (from the seed map), so a
distractor tracker's title can't inflate the oracle result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import asyncpg

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.agenda_manifest import PAAH_USER_ID
from benchmarks.personal_agent.seed import SeedAgendaResult

logger = logging.getLogger(__name__)


@dataclass
class AgendaStats:
    due_expected: int = 0
    # ORACLE — weft_tracker_due
    oracle_returned_ids: list[str] = field(default_factory=list)
    oracle_hits: int = 0                 # |returned ∩ expected_due|
    oracle_leaks: list[str] = field(default_factory=list)  # returned but not expected
    keep_pushing_id: str = ""
    keep_pushing_first: bool = False
    # AGENT-FACING — weft_daily_brief
    brief_available: bool = False
    brief_due_covered: int = 0           # due titles that appear in the brief
    brief_sections: list[str] = field(default_factory=list)

    @property
    def oracle_recall(self) -> float:
        return self.oracle_hits / self.due_expected if self.due_expected else 0.0

    @property
    def oracle_complete(self) -> bool:
        return self.oracle_recall == 1.0

    @property
    def oracle_precise(self) -> bool:
        """No non-due tracker leaked into the due queue."""
        return not self.oracle_leaks

    @property
    def brief_coverage(self) -> float:
        return self.brief_due_covered / self.due_expected if self.due_expected else 0.0

    @property
    def brief_surfaces_open_loops(self) -> bool:
        """The agent-facing digest carries every due open loop."""
        return self.brief_available and self.brief_coverage == 1.0

    def to_dict(self) -> dict:
        return {
            "due_expected": self.due_expected,
            "oracle_tracker_due": {
                "returned": len(self.oracle_returned_ids),
                "hits": self.oracle_hits,
                "recall_at_due": self.oracle_recall,
                "complete": self.oracle_complete,
                "precise": self.oracle_precise,
                "leaks": self.oracle_leaks,
                "keep_pushing_first": self.keep_pushing_first,
            },
            "agent_facing_daily_brief": {
                "available": self.brief_available,
                "due_covered": self.brief_due_covered,
                "coverage": self.brief_coverage,
                "surfaces_open_loops": self.brief_surfaces_open_loops,
                "sections": self.brief_sections,
            },
        }


async def run_agenda_paah(
    pool: asyncpg.Pool,
    seeded: SeedAgendaResult,
    limit: int = 100,
) -> AgendaStats:
    """Measure oracle (tracker_due) vs agent-facing (daily_brief) for the plate."""
    from weft.mcp.tools import weft_daily_brief, weft_tracker_due

    app = await build_app_context(pool)
    ctx = make_ctx(app)

    expected_due = seeded.due_ids
    stats = AgendaStats(due_expected=len(expected_due))
    stats.keep_pushing_id = seeded.keep_pushing.tracker_id

    token = current_user_id.set(PAAH_USER_ID)
    try:
        # --- ORACLE: the deterministic due-loop query ---
        due_resp = await weft_tracker_due(ctx, limit=limit)
        returned = [t.get("id") for t in due_resp.get("trackers", []) if t.get("id")]
        returned_set = set(returned)
        stats.oracle_returned_ids = returned
        stats.oracle_hits = len(returned_set & expected_due)
        stats.oracle_leaks = sorted(returned_set - expected_due)
        # Ordering: oldest-due (the thing kept pushing) must come back first.
        stats.keep_pushing_first = bool(returned) and returned[0] == stats.keep_pushing_id
        logger.info(
            "paah_agenda ORACLE tracker_due: returned=%d hits=%d/%d recall=%.3f "
            "leaks=%d keep_pushing_first=%s",
            len(returned), stats.oracle_hits, stats.due_expected,
            stats.oracle_recall, len(stats.oracle_leaks), stats.keep_pushing_first,
        )
        if stats.oracle_leaks:
            leaked_keys = [
                t.spec.key for t in seeded.trackers
                if t.tracker_id in set(stats.oracle_leaks)
            ]
            logger.warning(
                "paah_agenda ORACLE precision miss — non-due trackers leaked: %s",
                leaked_keys,
            )

        # --- AGENT-FACING: does the morning digest carry the open loops? ---
        try:
            brief = await weft_daily_brief(ctx)
            markdown = brief.get("markdown", "") or ""
            stats.brief_available = "error" not in brief
            stats.brief_due_covered = sum(
                1 for t in seeded.due_trackers if t.spec.title in markdown
            )
            # Best-effort section list for the finding (which sections DID render).
            stats.brief_sections = [
                line.lstrip("# ").strip()
                for line in markdown.splitlines()
                if line.startswith("#")
            ]
            logger.info(
                "paah_agenda AGENT-FACING daily_brief: available=%s due_covered=%d/%d "
                "coverage=%.3f",
                stats.brief_available, stats.brief_due_covered, stats.due_expected,
                stats.brief_coverage,
            )
        except Exception:  # noqa: BLE001 — the brief is a downstream consumer; a
            # failure to assemble it must not sink the oracle measurement.
            logger.exception("paah_agenda: daily_brief assembly failed")
            stats.brief_available = False
    finally:
        current_user_id.reset(token)

    logger.info("paah_agenda stats: %s", stats.to_dict())
    return stats
