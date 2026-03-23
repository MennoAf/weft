"""Wellness snapshot section — check-in pattern summary.

Runs independently of budget packing (not token-budgeted).  Wraps the
check_in_patterns.analyze_all() output into a concise snapshot of trends,
streaks, and rolling averages.  Failure-tolerant: never blocks the primer.

Reference: weft/primer.py lines 701-736.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_wellness_section(ctx: PrimerContext) -> SectionResult:
    """Build a concise wellness snapshot from check-in data."""
    try:
        from weft.check_in_patterns import analyze_all
        from weft.check_ins import list_check_ins

        check_ins = await list_check_ins(ctx.pool, limit=200)
        if not check_ins:
            return SectionResult(items=[], tokens_used=0, skipped=True,
                                 skip_reason="no check-ins")

        report = analyze_all(check_ins)
        snapshot: dict = {}

        if report["trends"]["mood"] or report["trends"]["energy"] or report["trends"]["sleep"]:
            snapshot["trends"] = {
                k: v for k, v in report["trends"].items()
                if k != "period_days" and v is not None
            }

        if report["streaks"]["logging_streak"] > 0:
            snapshot["logging_streak"] = report["streaks"]["logging_streak"]
        if report["streaks"]["good_mood_streaks"]:
            snapshot["good_mood_streak"] = report["streaks"]["good_mood_streaks"][-1]
        if report["streaks"]["low_mood_streaks"]:
            snapshot["low_mood_streak"] = report["streaks"]["low_mood_streaks"][-1]

        series = report["rolling_averages"].get("series", [])
        if series:
            latest = series[-1]
            snapshot["current_averages"] = {
                "mood": latest["avg_mood"],
                "energy": latest["avg_energy"],
                "sleep": latest["avg_sleep"],
                "window_days": report["rolling_averages"]["window_days"],
            }

        if not snapshot:
            return SectionResult(items=[], tokens_used=0, skipped=True,
                                 skip_reason="no meaningful wellness data")

        return SectionResult(items=[snapshot], tokens_used=0, skipped=False)

    except Exception as exc:
        logger.warning("Wellness snapshot failed: %s", exc)
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason=f"error: {exc}")
