#!/usr/bin/env python3
"""
harness.py — PAAH enumeration harness (agent-facing, end-to-end).

Unlike enumeration_eval — which measures ``gather_topic_memories`` and
``search_hybrid`` at the FUNCTION level — PAAH drives the real ``weft_recall``
tool with natural-language enumeration queries and asserts on the RESPONSE an
agent would actually receive. That closes the coverage gap the enumeration
router opened: ``resolve_topic`` on the query noun, tier routing (default
``auto``), and the enumeration answer block are all exercised on the hot path.

For each collection and each of k phrasings we record the answers an agent
could read to "how many X do I have?":

  * ``obvious_count``  = response["count"] — the MOST OBVIOUS field. Post-V8 it
                         is corrected to the true membership for a complete
                         enumeration gather (the fix under test).
  * ``enum_count``     = response["enumeration"]["count"] — the explicit,
                         unambiguously-named answer.
  * ``naive_results``  = len(response["results"]) — legacy contrast: the ranked
                         top-k slice, which over/under-counts and should still
                         NOT equal the truth.

Recall@membership is computed over ``enumeration.members`` (the complete list
the agent gets in one field, no assembly), matched by memory id (robust to
face-mode content wrapping). The spread across phrasings is the never-miss
signal; single-run comparisons are banned (pipeline nondeterminism ~37%).

The load-bearing question: is the enumeration answer now something the agent
KNOWS (an explicit correct field) rather than must infer? Post-V8 this harness
is the proof the consumption contract closed — obvious_count and enum_count
should be correct on every run, while naive_results stays wrong.

RENDER ALTITUDE (the loop-closer, issue weft-09ae22a3 item 3). The asserts
above check the tool RESPONSE. But an agent reads the RENDERED context, not the
raw dict — and the response being right ≠ the render showing it. So for each run
we also render the response the way a reading agent consumes it
(``format_recall_context``) at two altitudes, mirroring the agenda shape's
ORACLE + AGENT-FACING split:

  * ``render_legacy_rows``  = rows in the results-only render (use_enumeration
                             =False) — the FINDING: it counts the limit-bounded
                             slice, so it never equals the truth.
  * ``render_enum_rows``    = member rows in the enumeration-aware render — the
                             CLOSE: the complete membership survives rendering.
  * ``render_count_shown``  = the corrected count appears in the rendered header
                             (the agent can READ the number, not just parse it).
  * ``summary_shows_count`` = ``enumeration.summary`` prose carries the count
                             (the PAAH-local prose guard against a regression to
                             len(results) in the summary sentence).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from statistics import median

import asyncpg

from weft.auth import current_user_id

from benchmarks.personal_agent.context import build_app_context, make_ctx
from benchmarks.personal_agent.manifest import PAAH_USER_ID
from benchmarks.personal_agent.seed import SeedResult

logger = logging.getLogger(__name__)


# Enumeration phrasings, keyed off the collection noun (plural). All avoid
# temporal / episodic markers so ``route_query_to_tier`` lands on 'belief'
# (where the enumeration router runs) — and all yield the plural head noun so
# ``resolve_topic`` maps back to the seeded tag. If a phrasing routed away from
# belief or extracted a singular noun, the header would go missing — and the
# harness records exactly that as ``header_present=False``.
_PHRASINGS: tuple[str, ...] = (
    "how many {n} do I have",
    "list all my {n}",
    "enumerate my {n}",
    "what are all the {n} I have",
    "give me all my {n}",
    "show me all my {n}",
)


_ROW_RE = re.compile(r"^\[\d+\]", re.MULTILINE)


def _count_rows(rendered: str) -> int:
    """Count ``[i]`` rows in an agent-facing render (members or results)."""
    return len(_ROW_RE.findall(rendered))


@dataclass
class RunRecord:
    """One weft_recall invocation's agent-facing signals."""

    query: str
    enum_present: bool
    obvious_count: int | None       # response["count"] — the most-obvious field
    enum_count: int | None          # response["enumeration"]["count"]
    naive_results: int              # len(results) — legacy ranked-slice contrast
    recall_at_membership: float     # enumeration.members ∩ manifest / manifest

    # --- render altitude: what the agent actually reads --------------------
    render_legacy_rows: int         # rows in the results-only render (finding)
    render_enum_rows: int           # member rows in the enum-aware render (close)
    render_count_shown: bool        # the corrected count is in the rendered header
    summary_shows_count: bool       # enumeration.summary prose carries the count


@dataclass
class EnumStats:
    """Aggregated enumeration results for one collection across k runs."""

    name: str
    topic_tag: str
    manifest_count: int             # verified stored cardinality (ground truth)
    limit: int
    runs: list[RunRecord] = field(default_factory=list)

    # --- enumeration block presence --------------------------------------
    @property
    def enum_present_rate(self) -> float:
        return sum(r.enum_present for r in self.runs) / len(self.runs)

    # --- the corrected obvious field (response["count"]) -----------------
    @property
    def obvious_count_correct_rate(self) -> float:
        return sum(r.obvious_count == self.manifest_count for r in self.runs) / len(self.runs)

    # --- the explicit enumeration.count ----------------------------------
    @property
    def enum_count_correct_rate(self) -> float:
        return sum(r.enum_count == self.manifest_count for r in self.runs) / len(self.runs)

    # --- legacy contrast: naive len(results) -----------------------------
    @property
    def naive_correct_rate(self) -> float:
        return sum(r.naive_results == self.manifest_count for r in self.runs) / len(self.runs)

    # --- recall@membership over enumeration.members ----------------------
    @property
    def recall_min(self) -> float:
        return min(r.recall_at_membership for r in self.runs)

    @property
    def recall_median(self) -> float:
        return median(r.recall_at_membership for r in self.runs)

    @property
    def recall_max(self) -> float:
        return max(r.recall_at_membership for r in self.runs)

    # --- render altitude: what the agent actually reads -------------------
    @property
    def render_legacy_wrong_rate(self) -> float:
        """FINDING: the results-only render never equals the true count."""
        return sum(
            r.render_legacy_rows != self.manifest_count for r in self.runs
        ) / len(self.runs)

    @property
    def render_close_rate(self) -> float:
        """CLOSE: the enum-aware render surfaces the COMPLETE membership."""
        return sum(
            r.render_enum_rows == self.manifest_count for r in self.runs
        ) / len(self.runs)

    @property
    def render_count_shown_rate(self) -> float:
        """CLOSE: the corrected count is READABLE in the rendered header."""
        return sum(r.render_count_shown for r in self.runs) / len(self.runs)

    @property
    def summary_shows_count_rate(self) -> float:
        """PROSE GUARD: enumeration.summary carries the count as text."""
        return sum(r.summary_shows_count for r in self.runs) / len(self.runs)

    def to_dict(self) -> dict:
        return {
            "collection": self.name,
            "topic_tag": self.topic_tag,
            "manifest_count": self.manifest_count,
            "limit": self.limit,
            "runs": len(self.runs),
            "enumeration_block_present_rate": self.enum_present_rate,
            "correct_count_rate": {
                "obvious_field_response_count": self.obvious_count_correct_rate,
                "explicit_enumeration_count": self.enum_count_correct_rate,
                "naive_len_results": self.naive_correct_rate,
            },
            "recall_at_membership": {
                "min": self.recall_min,
                "median": self.recall_median,
                "max": self.recall_max,
            },
            "render_altitude": {
                "legacy_render_wrong_rate": self.render_legacy_wrong_rate,
                "enum_render_complete_rate": self.render_close_rate,
                "count_readable_in_render_rate": self.render_count_shown_rate,
                "summary_carries_count_rate": self.summary_shows_count_rate,
            },
            "queries": [
                {
                    "query": r.query,
                    "enum_present": r.enum_present,
                    "obvious_count": r.obvious_count,
                    "enum_count": r.enum_count,
                    "naive_results": r.naive_results,
                    "recall_at_membership": r.recall_at_membership,
                    "render_legacy_rows": r.render_legacy_rows,
                    "render_enum_rows": r.render_enum_rows,
                    "render_count_shown": r.render_count_shown,
                    "summary_shows_count": r.summary_shows_count,
                }
                for r in self.runs
            ],
        }


async def _run_one(ctx, query: str, manifest_ids: set[str], limit: int) -> RunRecord:
    """Call the real weft_recall and distill agent-facing signals from it."""
    from weft.mcp.tools import weft_recall

    # The Reader's render is the surface a reading agent actually consumes.
    from benchmarks.longmemeval.reader import format_recall_context

    response = await weft_recall(ctx, query=query, limit=limit, tier="auto")

    results = response.get("results", []) or []
    naive_results = len(results)
    obvious_count = response.get("count")

    enum = response.get("enumeration")
    enum_present = enum is not None
    enum_count = enum["count"] if enum_present else None

    if enum_present:
        # members is the COMPLETE list — no union with results needed.
        member_ids = {m.get("id") for m in enum.get("members", []) if m.get("id")}
    else:
        member_ids = {r.get("id") for r in results if r.get("id")}

    hit = len(member_ids & manifest_ids)
    recall = hit / len(manifest_ids) if manifest_ids else 0.0

    # --- render altitude: render the response the way an agent reads it -----
    # Legacy (results-only) render is the FINDING; enum-aware render is the
    # CLOSE. Both operate on the SAME response, so the delta is purely the
    # enumeration awareness — not a different input.
    legacy_render = format_recall_context(response, use_enumeration=False)
    enum_render = format_recall_context(response, use_enumeration=True)
    render_legacy_rows = _count_rows(legacy_render)
    render_enum_rows = _count_rows(enum_render)
    header = enum_render.splitlines()[0] if enum_render else ""
    render_count_shown = enum_present and str(enum_count) in header
    summary_shows_count = enum_present and str(enum_count) in (enum.get("summary") or "")

    logger.info(
        "paah_enum: query=%r enum=%s obvious_count=%s enum_count=%s "
        "naive_results=%d recall@membership=%.3f render[legacy=%d enum=%d "
        "count_shown=%s summary=%s]",
        query, enum_present, obvious_count, enum_count, naive_results, recall,
        render_legacy_rows, render_enum_rows, render_count_shown,
        summary_shows_count,
    )
    return RunRecord(
        query=query,
        enum_present=enum_present,
        obvious_count=obvious_count,
        enum_count=enum_count,
        naive_results=naive_results,
        recall_at_membership=recall,
        render_legacy_rows=render_legacy_rows,
        render_enum_rows=render_enum_rows,
        render_count_shown=render_count_shown,
        summary_shows_count=summary_shows_count,
    )


async def run_enumeration_paah(
    pool: asyncpg.Pool,
    seed_results: list[SeedResult],
    limit: int = 10,
) -> list[EnumStats]:
    """Drive weft_recall over every seeded collection; return per-collection stats.

    Args:
        pool: testcontainers asyncpg pool (already seeded).
        seed_results: output of ``seed_corpus`` — carries the verified stored
            ids that define ground-truth membership.
        limit: recall limit passed to weft_recall (default 10 = the tool
            default; kept below PLANTS cardinality so the gap is observable).
    """
    app = await build_app_context(pool)
    ctx = make_ctx(app)

    all_stats: list[EnumStats] = []
    token = current_user_id.set(PAAH_USER_ID)
    try:
        for sr in seed_results:
            manifest_ids = set(sr.stored_ids)
            stats = EnumStats(
                name=sr.collection.name,
                topic_tag=sr.collection.topic_tag,
                manifest_count=sr.stored,
                limit=limit,
            )
            for phrasing in _PHRASINGS:
                query = phrasing.format(n=sr.collection.noun)
                stats.runs.append(await _run_one(ctx, query, manifest_ids, limit))
            all_stats.append(stats)
            logger.info("paah_enum stats: %s", stats.to_dict())
    finally:
        current_user_id.reset(token)

    return all_stats
