"""Calibration section — recent calibration insights for the primer.

Tier 2 (deferred under progressive disclosure).  Shows approval rates
and any action categories that warrant promotion or demotion based on
recent calibration history.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from weft.calibration import evaluate_tier_change, get_calibration_summary
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["calibration"]
_EVAL_WINDOW_DAYS = 30
_MIN_RECORDS_TO_SHOW = 1  # Don't show section if fewer records exist


async def fetch_calibration_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch calibration summary + per-category evaluations (parallel-safe)."""
    since = ctx.now - timedelta(days=_EVAL_WINDOW_DAYS)
    summary = await get_calibration_summary(
        ctx.pool, project_id=ctx.project_id, since=since,
    )

    total = summary["total"]
    if total < _MIN_RECORDS_TO_SHOW:
        return SectionFetch(skipped=True, skip_reason="No calibration records in window")

    # Evaluate each eligible category in parallel.
    categories = [
        c for c, cs in summary.get("by_category", {}).items()
        if cs["total"] >= _MIN_RECORDS_TO_SHOW
    ]
    if categories:
        async with asyncio.TaskGroup() as task_group:
            evaluation_tasks = [
                task_group.create_task(
                    evaluate_tier_change(ctx.pool, c, project_id=ctx.project_id)
                )
                for c in categories
            ]
        evaluations = [task.result() for task in evaluation_tasks]
    else:
        evaluations = []

    return SectionFetch(payload={
        "summary": summary,
        "evaluations": list(zip(categories, evaluations)),
    })


def pack_calibration_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack calibration summary + recommendations against budget."""
    if fetched.skipped:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason=fetched.skip_reason)

    payload = fetched.payload
    summary = payload["summary"]
    total = summary["total"]

    items: list[dict] = []
    section_used = 0

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
    if ctx.fits_or_guarantee(cost, section_used, _CAP):
        items.append(overview)
        ctx.used_tokens += cost
        section_used += cost

    for category, evaluation in payload["evaluations"]:
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


async def build_calibration_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack calibration section (single-shot wrapper)."""
    return pack_calibration_section(ctx, await fetch_calibration_section(ctx))
