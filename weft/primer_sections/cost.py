"""Cost section — today's spending posture for the primer.

Tier 2 (deferred under progressive disclosure).  Shows a compact cost
summary so the agent is aware of budget consumption and can self-regulate.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from weft.cost_tracking import get_cost_summary
from weft.primer_sections.context import SECTION_BUDGETS, PrimerContext, SectionResult
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["cost"]
_WINDOW_HOURS = 24  # rolling window for cost summary


async def build_cost_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack a cost posture summary."""
    since = ctx.now - timedelta(hours=_WINDOW_HOURS)
    summary = await get_cost_summary(
        ctx.pool,
        since=since,
        project_id=ctx.project_id,
    )

    if summary.total_entries == 0:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason="No cost entries in window")

    text = (
        f"Cost ({_WINDOW_HOURS}h): ${summary.total_cost_usd:.4f} "
        f"across {summary.total_entries} entries, "
        f"{summary.total_tokens:,} tokens"
    )
    cost = estimate_tokens(text)

    items: list[dict] = []
    section_used = 0

    if ctx.fits_or_guarantee(cost, section_used, _CAP):
        items.append({
            "window_hours": _WINDOW_HOURS,
            "total_entries": summary.total_entries,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": round(summary.total_cost_usd, 4),
        })
        ctx.used_tokens += cost
        section_used += cost

    ctx.section_tokens["cost"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
