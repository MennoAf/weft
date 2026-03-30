"""Calibration section — recent calibration insights for the primer.

Tier 2 (deferred under progressive disclosure).  Shows approval rates
and any action categories that warrant promotion or demotion based on
recent calibration history.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from weft.calibration import evaluate_tier_change, get_calibration_summary
from weft.primer_sections.context import PrimerContext, SectionResult
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = 150   # token budget for calibration section
_EVAL_WINDOW_DAYS = 30
_MIN_RECORDS_TO_SHOW = 1  # Don't show section if fewer records exist


async def build_calibration_section(ctx: PrimerContext) -> SectionResult:
    """Fetch calibration summary and tier change recommendations."""
    since = ctx.now - timedelta(days=_EVAL_WINDOW_DAYS)
    summary = await get_calibration_summary(
        ctx.pool,
        project_id=ctx.project_id,
        since=since,
    )

    total = summary["total"]
    if total < _MIN_RECORDS_TO_SHOW:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason="No calibration records in window")

    items: list[dict] = []
    section_used = 0

    # Overall summary line
    overview = {
        "total": total,
        "approved": summary["approved"],
        "rejected": summary["rejected"],
        "modified": summary["modified"],
        "approval_rate": round(summary["approval_rate"], 2),
        "window_days": _EVAL_WINDOW_DAYS,
    }
    overview_text = (
        f"Calibration: {total} records, "
        f"{summary['approval_rate']:.0%} approval rate (last {_EVAL_WINDOW_DAYS}d)"
    )
    cost = estimate_tokens(overview_text)
    if ctx.fits(cost, section_used, _CAP):
        items.append(overview)
        ctx.used_tokens += cost
        section_used += cost

    # Per-category evaluations (only for categories with records)
    for category, cat_stats in summary.get("by_category", {}).items():
        if cat_stats["total"] < _MIN_RECORDS_TO_SHOW:
            continue
        evaluation = await evaluate_tier_change(
            ctx.pool,
            category,
            project_id=ctx.project_id,
        )
        if evaluation["recommendation"] == "no_change":
            continue

        rec = {
            "action_category": category,
            "recommendation": evaluation["recommendation"],
            "reason": evaluation["reason"],
        }
        rec_text = f"{category}: {evaluation['recommendation']} — {evaluation['reason']}"
        cost = estimate_tokens(rec_text)
        if ctx.fits(cost, section_used, _CAP):
            items.append(rec)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["calibration"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
