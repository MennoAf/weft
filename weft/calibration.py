"""Calibration records store — tracks agent action outcomes for tier decisions.

Records whether agent actions were approved, rejected, or modified by users.
Aggregation queries support autonomy tier promotion/demotion decisions.
Includes evaluate_tier_change() for automatic promotion/demotion after each
calibration event.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import asyncpg

from weft.db.connection import get_db
from weft.models import (
    CalibrationCreate,
    CalibrationOutcome,
    CalibrationRecord,
    _weft_id,
)

logger = logging.getLogger(__name__)

# Thresholds for tier promotion/demotion evaluation
_PROMO_MIN_RECORDS = 5          # Minimum calibrations before considering promotion
_PROMO_APPROVAL_RATE = 0.8     # 80% approval rate needed for promotion
_DEMO_MIN_RECORDS = 3           # Minimum calibrations before considering demotion
_DEMO_REJECTION_RATE = 0.5     # 50% rejection rate triggers demotion
_EVAL_WINDOW_DAYS = 30          # Look at records from the last 30 days


# --- Row mapping ---


def _row_to_record(row: asyncpg.Record) -> CalibrationRecord:
    """Convert a database row to a CalibrationRecord model."""
    context = row["context"]
    if isinstance(context, str):
        context = json.loads(context)
    return CalibrationRecord(
        id=row["id"],
        action_category=row["action_category"],
        action_description=row["action_description"],
        outcome=CalibrationOutcome(row["outcome"]),
        agent_id=row["agent_id"],
        project_id=row["project_id"],
        context=context or {},
        user_id=row["user_id"],
        created_at=row["created_at"],
    )


# --- CRUD ---


async def record_calibration(
    pool: asyncpg.Pool,
    create: CalibrationCreate,
) -> CalibrationRecord:
    """Record an agent action calibration event. Returns the created record."""
    record_id = _weft_id()
    context_json = json.dumps(create.context)

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO calibration_records (
            id, action_category, action_description, outcome,
            agent_id, project_id, context, user_id
        )
        VALUES (
            $1, $2, $3, $4,
            $5, $6, $7::jsonb,
            nullif(current_setting('app.user_id', true), '')
        )
        RETURNING *
        """,
        record_id,
        create.action_category,
        create.action_description,
        create.outcome.value,
        create.agent_id,
        create.project_id,
        context_json,
    )
    return _row_to_record(row)


async def get_calibration(
    pool: asyncpg.Pool,
    record_id: str,
) -> CalibrationRecord | None:
    """Fetch a calibration record by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM calibration_records WHERE id = $1",
        record_id,
    )
    return _row_to_record(row) if row else None


async def list_calibrations(
    pool: asyncpg.Pool,
    *,
    action_category: str | None = None,
    outcome: CalibrationOutcome | None = None,
    project_id: str | None = None,
    limit: int = 50,
) -> list[CalibrationRecord]:
    """List calibration records with optional filters."""
    conditions: list[str] = []
    params: list = []
    idx = 1

    if action_category is not None:
        conditions.append(f"action_category = ${idx}")
        params.append(action_category)
        idx += 1

    if outcome is not None:
        conditions.append(f"outcome = ${idx}")
        params.append(outcome.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM calibration_records {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_record(r) for r in rows]


async def get_calibration_summary(
    pool: asyncpg.Pool,
    *,
    action_category: str | None = None,
    project_id: str | None = None,
    since: datetime | None = None,
) -> dict:
    """Return aggregate calibration statistics.

    Returns dict with keys:
        total, approved, rejected, modified, approval_rate, by_category
    """
    conditions: list[str] = []
    params: list = []
    idx = 1

    if action_category is not None:
        conditions.append(f"action_category = ${idx}")
        params.append(action_category)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if since is not None:
        conditions.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    # Overall counts
    row = await get_db(pool).fetchrow(
        f"""
        SELECT
            count(*) AS total,
            count(*) FILTER (WHERE outcome = 'approved') AS approved,
            count(*) FILTER (WHERE outcome = 'rejected') AS rejected,
            count(*) FILTER (WHERE outcome = 'modified') AS modified
        FROM calibration_records {where}
        """,
        *params,
    )

    total = row["total"]
    approved = row["approved"]
    rejected = row["rejected"]
    modified = row["modified"]
    approval_rate = (approved / total) if total > 0 else 0.0

    # Per-category breakdown
    cat_rows = await get_db(pool).fetch(
        f"""
        SELECT
            action_category,
            count(*) AS total,
            count(*) FILTER (WHERE outcome = 'approved') AS approved,
            count(*) FILTER (WHERE outcome = 'rejected') AS rejected,
            count(*) FILTER (WHERE outcome = 'modified') AS modified
        FROM calibration_records {where}
        GROUP BY action_category
        ORDER BY action_category
        """,
        *params,
    )

    by_category = {}
    for cr in cat_rows:
        cat_total = cr["total"]
        by_category[cr["action_category"]] = {
            "total": cat_total,
            "approved": cr["approved"],
            "rejected": cr["rejected"],
            "modified": cr["modified"],
            "approval_rate": (cr["approved"] / cat_total) if cat_total > 0 else 0.0,
        }

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "modified": modified,
        "approval_rate": approval_rate,
        "by_category": by_category,
    }


async def delete_calibration(
    pool: asyncpg.Pool,
    record_id: str,
) -> bool:
    """Delete a calibration record. Returns True if deleted."""
    result = await get_db(pool).execute(
        "DELETE FROM calibration_records WHERE id = $1",
        record_id,
    )
    return result.split()[-1] != "0"


# --- Tier evaluation ---


async def evaluate_tier_change(
    pool: asyncpg.Pool,
    action_category: str,
    *,
    project_id: str | None = None,
) -> dict:
    """Evaluate whether an action category warrants promotion or demotion.

    Looks at calibration records within the evaluation window and compares
    approval/rejection rates against thresholds. Returns a recommendation
    dict with keys: action, recommendation, reason, stats.

    Recommendations: "promote", "demote", or "no_change".
    """
    since = datetime.now(timezone.utc) - timedelta(days=_EVAL_WINDOW_DAYS)
    summary = await get_calibration_summary(
        pool,
        action_category=action_category,
        project_id=project_id,
        since=since,
    )

    total = summary["total"]
    approved = summary["approved"]
    rejected = summary["rejected"]

    result = {
        "action_category": action_category,
        "recommendation": "no_change",
        "reason": "",
        "stats": {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "window_days": _EVAL_WINDOW_DAYS,
        },
    }

    if total == 0:
        result["reason"] = "No calibration records in evaluation window"
        return result

    approval_rate = approved / total
    rejection_rate = rejected / total

    # Check demotion first (safety takes priority)
    if total >= _DEMO_MIN_RECORDS and rejection_rate >= _DEMO_REJECTION_RATE:
        result["recommendation"] = "demote"
        result["reason"] = (
            f"Rejection rate {rejection_rate:.0%} >= {_DEMO_REJECTION_RATE:.0%} "
            f"threshold over {total} records"
        )
        result["stats"]["rejection_rate"] = rejection_rate
        return result

    # Check promotion
    if total >= _PROMO_MIN_RECORDS and approval_rate >= _PROMO_APPROVAL_RATE:
        result["recommendation"] = "promote"
        result["reason"] = (
            f"Approval rate {approval_rate:.0%} >= {_PROMO_APPROVAL_RATE:.0%} "
            f"threshold over {total} records"
        )
        result["stats"]["approval_rate"] = approval_rate
        return result

    result["reason"] = (
        f"Insufficient evidence: {total} records, "
        f"{approval_rate:.0%} approval, {rejection_rate:.0%} rejection"
    )
    result["stats"]["approval_rate"] = approval_rate
    result["stats"]["rejection_rate"] = rejection_rate
    return result
