"""Health check evaluator aggregator — read-only assessment of all alert subsystems.

Calls the detection logic from each alert subsystem (check-in patterns, Loom
awareness, memory hygiene) without triggering DB writes or alert dispatch.

Each evaluator is wrapped in its own try/except so one failure does not abort
the others. Results are unified into a HealthSummary dataclass.

To add a new evaluator, append an entry to _EVALUATORS and write a corresponding
_evaluate_* async function. No other code changes needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import asyncpg

from weft.loom_query import (
    LoomQueryError,
    get_blocked_pile_ups,
    get_completable_epics,
    get_stale_claimed_tasks,
    loom_tables_exist,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity normalization
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, str] = {
    "info": "info",
    "warning": "warning",
    "critical": "critical",
    # Alternate vocabularies
    "high": "critical",
    "medium": "warning",
    "low": "info",
}


def normalize_severity(raw: str) -> Literal["info", "warning", "critical"]:
    """Map any severity string to the canonical vocabulary."""
    return _SEVERITY_MAP.get(raw.lower(), "info")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthFinding:
    source: str  # e.g. "check_in_alerts", "loom_alerts", "memory_hygiene"
    severity: str  # "info", "warning", "critical"
    message: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HealthSummary:
    findings: list[HealthFinding]
    errors: list[dict]
    evaluated_at: datetime
    total_findings: int = 0

    def __post_init__(self) -> None:
        self.total_findings = len(self.findings)


# ---------------------------------------------------------------------------
# Per-evaluator wrappers (read-only — no create_alert calls)
# ---------------------------------------------------------------------------


async def _evaluate_checkin(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Run check-in pattern detection without creating alerts.

    Calls analyze_all() for pattern detection, then evaluates thresholds
    locally — mirrors evaluate_check_in_alerts logic but skips create_alert.
    """
    from weft.check_in_patterns import (
        _ALERT_LOW_MOOD_STREAK,
        _ALERT_LOW_SLEEP_DAYS,
        _ALERT_LOW_SLEEP_HOURS,
        analyze_all,
    )
    from weft.check_ins import list_check_ins

    check_ins = await list_check_ins(pool, limit=200)
    if not check_ins:
        return []

    report = analyze_all(check_ins, trend_days=90, rolling_days=30)
    findings: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    # 1. Low mood streak
    low_streaks = report["streaks"]["low_mood_streaks"]
    if low_streaks:
        longest = max(low_streaks, key=lambda s: s["length"])
        if longest["length"] >= _ALERT_LOW_MOOD_STREAK:
            findings.append({
                "source": "check_in_alerts",
                "severity": "warning",
                "message": f"Low mood streak: {longest['length']} consecutive days",
                "metadata": {"streak": longest},
            })

    # 2. Low average sleep
    from datetime import timedelta

    sorted_cis = sorted(check_ins, key=lambda ci: ci.logged_at, reverse=True)
    recent_sleep = [
        ci.sleep_hours
        for ci in sorted_cis[:_ALERT_LOW_SLEEP_DAYS * 2]
        if ci.sleep_hours is not None
        and ci.logged_at >= now - timedelta(days=_ALERT_LOW_SLEEP_DAYS + 1)
    ]
    if len(recent_sleep) >= _ALERT_LOW_SLEEP_DAYS:
        avg_sleep = sum(recent_sleep) / len(recent_sleep)
        if avg_sleep < _ALERT_LOW_SLEEP_HOURS:
            findings.append({
                "source": "check_in_alerts",
                "severity": "warning",
                "message": f"Low sleep average: {avg_sleep:.1f}h over last {len(recent_sleep)} entries",
                "metadata": {"avg_sleep": avg_sleep},
            })

    # 3. Declining mood trend
    mood_trend = report["trends"].get("mood")
    if mood_trend and mood_trend["direction"] == "down":
        findings.append({
            "source": "check_in_alerts",
            "severity": "info",
            "message": f"Declining mood trend (slope: {mood_trend['slope']}/day)",
            "metadata": {"trend": mood_trend},
        })

    return findings


async def _evaluate_loom(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Run Loom awareness checks without creating alerts.

    Queries the Loom tasks table for stale claims, epic completion readiness,
    and blocked pile-ups. Read-only — no create_alert calls.
    Delegates all SQL to weft.loom_query.
    """
    try:
        if not await loom_tables_exist(pool):
            return []
    except LoomQueryError:
        return []

    findings: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    # 1. Stale claims
    try:
        rows = await get_stale_claimed_tasks(pool)
        if rows:
            stale_list = []
            for r in rows:
                hours = (now - r["claimed_at"]).total_seconds() / 3600
                stale_list.append(f"{r['title']} ({hours:.0f}h, {r['assignee'] or 'unknown'})")
            findings.append({
                "source": "loom_alerts",
                "severity": "warning",
                "message": f"{len(rows)} stale claimed task(s)",
                "metadata": {"tasks": stale_list},
            })
    except Exception:
        logger.warning("health_check.loom.stale_claims failed", exc_info=True)

    # 2. Epic completion readiness
    try:
        rows = await get_completable_epics(pool)
        if rows:
            findings.append({
                "source": "loom_alerts",
                "severity": "info",
                "message": f"{len(rows)} epic(s) ready to close",
                "metadata": {"epics": [r["title"] for r in rows]},
            })
    except Exception:
        logger.warning("health_check.loom.epic_completion failed", exc_info=True)

    # 3. Blocked pile-ups
    try:
        rows = await get_blocked_pile_ups(pool)
        if rows:
            pile_list = [f"{r['project_name']}: {r['blocked_count']}" for r in rows]
            findings.append({
                "source": "loom_alerts",
                "severity": "warning",
                "message": f"Blocked task pile-up in {len(rows)} project(s)",
                "metadata": {"projects": pile_list},
            })
    except Exception:
        logger.warning("health_check.loom.blocked_pile_up failed", exc_info=True)

    return findings


async def _evaluate_hygiene(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Run memory hygiene checks without creating alerts.

    Checks stale decisions, consolidation overdue, and memory count threshold.
    Read-only — no create_alert calls.
    """
    from weft.db.connection import get_db
    from weft.memory_hygiene_alerts import (
        _CONSOLIDATION_OVERDUE_HOURS,
        _MEMORY_COUNT_THRESHOLD,
        _STALE_DECISION_CONFIDENCE,
        _STALE_DECISION_DAYS,
    )

    findings: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    # 1. Stale decisions
    from datetime import timedelta

    stale_cutoff = now - timedelta(days=_STALE_DECISION_DAYS)
    try:
        overdue_rows = await get_db(pool).fetch(
            """
            SELECT id, content, review_after, confidence
            FROM memories
            WHERE status = 'active'
              AND type = 'decision'
              AND review_after IS NOT NULL
              AND review_after <= $1
            ORDER BY review_after ASC
            LIMIT 20
            """,
            now,
        )
        old_low_rows = await get_db(pool).fetch(
            """
            SELECT id, content, created_at, confidence
            FROM memories
            WHERE status = 'active'
              AND type = 'decision'
              AND review_after IS NULL
              AND created_at < $1
              AND confidence < $2
            ORDER BY confidence ASC
            LIMIT 20
            """,
            stale_cutoff,
            _STALE_DECISION_CONFIDENCE,
        )
        total = len(overdue_rows) + len(old_low_rows)
        if total > 0:
            findings.append({
                "source": "memory_hygiene",
                "severity": "warning",
                "message": f"{total} stale decision(s) need review",
                "metadata": {"overdue": len(overdue_rows), "low_confidence": len(old_low_rows)},
            })
    except Exception:
        logger.warning("health_check.hygiene.stale_decisions failed", exc_info=True)

    # 2. Consolidation overdue
    try:
        from weft.store import get_metadata

        meta = await get_metadata(pool, "last_consolidation_run")
        is_overdue = False
        hours_since = None

        if meta is None:
            count = await get_db(pool).fetchval(
                "SELECT count(*) FROM memories WHERE status = 'active'"
            )
            if count >= 50:
                is_overdue = True
        else:
            ran_at = meta.get("ran_at")
            if ran_at:
                last_run = datetime.fromisoformat(ran_at)
                hours_since = (now - last_run).total_seconds() / 3600
                is_overdue = hours_since >= _CONSOLIDATION_OVERDUE_HOURS

        if is_overdue:
            msg = (
                f"Last consolidation was {hours_since:.0f}h ago"
                if hours_since
                else "Consolidation has never been run"
            )
            findings.append({
                "source": "memory_hygiene",
                "severity": "warning",
                "message": msg,
                "metadata": {"hours_since": hours_since},
            })
    except Exception:
        logger.warning("health_check.hygiene.consolidation failed", exc_info=True)

    # 3. Memory count threshold
    try:
        count = await get_db(pool).fetchval(
            "SELECT count(*) FROM memories WHERE status = 'active'"
        )
        if count >= _MEMORY_COUNT_THRESHOLD:
            findings.append({
                "source": "memory_hygiene",
                "severity": "info",
                "message": f"Active memory count: {count} (threshold: {_MEMORY_COUNT_THRESHOLD})",
                "metadata": {"count": count, "threshold": _MEMORY_COUNT_THRESHOLD},
            })
    except Exception:
        logger.warning("health_check.hygiene.memory_count failed", exc_info=True)

    return findings


# ---------------------------------------------------------------------------
# Evaluator registry — add new evaluators here
# ---------------------------------------------------------------------------

_EVALUATORS: list[tuple[str, str]] = [
    ("check_in_alerts", "_evaluate_checkin"),
    ("loom_alerts", "_evaluate_loom"),
    ("memory_hygiene", "_evaluate_hygiene"),
]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


async def run_all_evaluators(pool: asyncpg.Pool) -> HealthSummary:
    """Run all health evaluators and return a unified summary.

    Each evaluator runs in its own try/except — one failure does not abort
    the others. No alerts are created; this is read-only assessment.
    """
    evaluated_at = datetime.now(timezone.utc)
    logger.info("Running health check")

    all_findings: list[HealthFinding] = []
    errors: list[dict] = []

    import weft.health_check as _mod

    for source, fn_name in _EVALUATORS:
        evaluator_fn = getattr(_mod, fn_name)
        try:
            raw = await evaluator_fn(pool)
            for item in raw or []:
                all_findings.append(HealthFinding(
                    source=item.get("source", source),
                    severity=normalize_severity(item.get("severity", "info")),
                    message=item.get("message", ""),
                    metadata=item.get("metadata", {}),
                ))
        except Exception as exc:
            logger.warning("Evaluator %s failed: %s", source, exc)
            errors.append({
                "source": source,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })

    return HealthSummary(
        findings=all_findings,
        errors=errors,
        evaluated_at=evaluated_at,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def summary_to_dict(summary: HealthSummary) -> dict:
    """Convert HealthSummary to a JSON-serializable dict.

    Includes a human-readable 'summary' string for quick scanning.
    """
    findings_str = f"{summary.total_findings} finding{'s' if summary.total_findings != 1 else ''}"
    errors_str = f"{len(summary.errors)} error{'s' if len(summary.errors) != 1 else ''}"
    evaluator_count = len(_EVALUATORS)

    return {
        "summary": f"{findings_str} across {evaluator_count} evaluators, {errors_str}",
        "total_findings": summary.total_findings,
        "findings": [
            {
                "source": f.source,
                "severity": f.severity,
                "message": f.message,
                "metadata": f.metadata,
            }
            for f in summary.findings
        ],
        "errors": summary.errors,
        "evaluated_at": summary.evaluated_at.isoformat(),
    }
