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

from weft.alerts import create_alert
from weft.autonomy import AutonomyTier, get_policy_by_action, update_policy_tier
from weft.db.connection import get_db
from weft.models import (
    AlertCreate,
    AlertType,
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

# Calibration origins (weft/auth.py caller modes) trusted to drive auto-
# promotion. An approval attested by an untrusted 'agent' caller can never push
# an action's autonomy tier to 'always' — only 'supervisor' (Face / human /
# Orchestrator) approvals count toward promotion. Demotion intentionally counts
# all origins (an untrusted rejection can still raise a human-review alert).
TRUSTED_CALIBRATION_ORIGINS = ("supervisor",)


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
        origin=row["origin"],
        created_at=row["created_at"],
    )


# --- CRUD ---


async def record_calibration(
    pool: asyncpg.Pool,
    create: CalibrationCreate,
) -> CalibrationRecord:
    """Record an agent action calibration event. Returns the created record.

    After recording, evaluates whether the action category warrants an
    autonomy tier promotion. If the evaluate_tier_change recommendation is
    'promote' and a matching policy exists, the tier is auto-applied via
    update_policy_tier (recording a policy_calibration_events row with reason
    prefixed 'auto-calibration'). Demotions are NOT auto-applied here.
    """
    record_id = _weft_id()
    context_json = json.dumps(create.context)

    # Stamp the caller's trust tier (supervisor vs agent) so the promotion path
    # can distinguish attested human approvals from untrusted agent self-reports.
    from weft.auth import get_caller_mode

    origin = get_caller_mode()

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO calibration_records (
            id, action_category, action_description, outcome,
            agent_id, project_id, context, origin, user_id
        )
        VALUES (
            $1, $2, $3, $4,
            $5, $6, $7::jsonb, $8,
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
        origin,
    )
    record = _row_to_record(row)

    # Auto-apply promotions — demotions handled separately via alerts
    await _maybe_auto_promote(
        pool,
        action_category=create.action_category,
        project_id=create.project_id,
        agent_id=create.agent_id,
    )

    # Create a demotion-proposal alert if rejection rate crosses threshold
    await _maybe_alert_demotion(
        pool,
        action_category=create.action_category,
        project_id=create.project_id,
        agent_id=create.agent_id,
    )

    return record


async def _maybe_auto_promote(
    pool: asyncpg.Pool,
    *,
    action_category: str,
    project_id: str | None,
    agent_id: str | None,
) -> None:
    """Evaluate and auto-apply a promotion if thresholds are met.

    Calls evaluate_tier_change; if the recommendation is 'promote', looks up
    the policy for the action category and promotes it via update_policy_tier
    (which records the policy_calibration_events row). Only promotions are
    handled here — demotions must NOT be auto-applied.
    """
    try:
        evaluation = await evaluate_tier_change(
            pool,
            action_category,
            project_id=project_id,
        )
    except Exception:
        logger.exception(
            "evaluate_tier_change failed for action_category=%s; skipping auto-promotion",
            action_category,
        )
        from weft.counters import (
            COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED,
            increment_counter,
        )

        await increment_counter(pool, COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED)
        return

    if evaluation.get("recommendation") != "promote":
        return

    policy = await get_policy_by_action(pool, action_category)
    if policy is None:
        logger.debug(
            "Auto-calibration: no policy found for action_category=%s; skipping",
            action_category,
        )
        return

    if policy.tier == AutonomyTier.always:
        # Already at the highest promotable tier; nothing to do
        return

    reason = f"auto-calibration: {evaluation.get('reason', '')}"
    try:
        await update_policy_tier(
            pool,
            policy.id,
            AutonomyTier.always,
            reason=reason,
            agent_id=agent_id,
            auto_originated=True,
        )
        logger.info(
            "Auto-calibration promoted policy %s (action=%s) to always",
            policy.id,
            action_category,
        )
    except (ValueError, LookupError):
        logger.exception(
            "Auto-calibration: update_policy_tier failed for policy %s; skipping",
            policy.id,
        )
        from weft.counters import (
            COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED,
            increment_counter,
        )

        await increment_counter(pool, COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED)


async def _maybe_alert_demotion(
    pool: asyncpg.Pool,
    *,
    action_category: str,
    project_id: str | None,
    agent_id: str | None,
) -> None:
    """Create a demotion-proposal alert if rejection rate crosses the threshold.

    Calls evaluate_tier_change; if the recommendation is 'demote', inserts an
    alert (via create_alert) proposing the demotion for human review. The alert
    body includes the action category, observed rejection rate, and the
    recommended new tier. The autonomy tier is NEVER changed here — this is the
    human-as-judge half of the loop.

    Errors are swallowed so that calibration recording is never broken by alert
    creation failures.
    """
    try:
        evaluation = await evaluate_tier_change(
            pool,
            action_category,
            project_id=project_id,
        )
    except Exception:
        logger.exception(
            "evaluate_tier_change failed for action_category=%s; skipping demotion alert",
            action_category,
        )
        return

    if evaluation.get("recommendation") != "demote":
        return

    stats = evaluation.get("stats", {})
    rejection_rate = stats.get("rejection_rate", 0.0)
    reason = evaluation.get("reason", "")

    body = (
        f"Action category '{action_category}' has a rejection rate of "
        f"{rejection_rate:.0%} and is recommended for demotion to a lower tier. "
        f"Reason: {reason}. "
        f"Recommended new tier: ask (human review required). "
        f"No tier change has been applied — please review and decide."
    )

    try:
        await create_alert(
            pool,
            AlertCreate(
                alert_type=AlertType.custom,
                title=f"Demotion proposed for action: {action_category}",
                body=body,
                trigger_at=datetime.now(timezone.utc),
                payload={
                    "action_category": action_category,
                    "rejection_rate": rejection_rate,
                    "recommended_tier": "ask",
                    "stats": stats,
                },
                project_id=project_id,
                agent_id=agent_id,
            ),
        )
        logger.info(
            "Demotion alert created for action_category=%s (rejection_rate=%.0f%%)",
            action_category,
            rejection_rate * 100,
        )
    except Exception:
        logger.exception(
            "Failed to create demotion alert for action_category=%s; skipping",
            action_category,
        )


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
    origin_in: tuple[str, ...] | None = None,
) -> dict:
    """Return aggregate calibration statistics.

    ``origin_in`` optionally restricts to records whose ``origin`` is in the
    given trust set (e.g. TRUSTED_CALIBRATION_ORIGINS) — used by the promotion
    path so untrusted approvals don't count toward granting autonomy.

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

    if origin_in is not None:
        conditions.append(f"origin = ANY(${idx})")
        params.append(list(origin_in))
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


async def count_auto_originated_tier_changes(
    pool: asyncpg.Pool,
    *,
    since: datetime | None = None,
    user_id: str | None = None,
    scope_to_user: bool = False,
) -> int:
    """Count auto-originated tier changes from policy_calibration_events.

    Returns the count of rows in policy_calibration_events WHERE
    auto_originated = TRUE within an optional time window. This is the
    calibration loop's aliveness PROOF: >0 per month means the loop fires
    without a human pump; zero while calibration_records grow is the dead tell.

    The filter is the dedicated ``auto_originated`` provenance column (set TRUE
    only by the auto-promotion path), NOT the free-text ``reason`` field — a
    caller cannot spoof the metric by writing a manual event whose reason
    starts with 'auto-calibration'.

    User scoping: with ``scope_to_user`` True, an explicit
    ``user_id IS NOT DISTINCT FROM $user_id`` predicate is added (NULL-safe).
    This is defense-in-depth for system/scheduler callers that may bypass RLS;
    a per-request authenticated caller can leave it False and rely on RLS /
    its app.user_id (the documented trusted-context path — loom-fdd9282a).

    Args:
        pool: Database connection pool.
        since: Optional lower-bound timestamp. If omitted, counts all time.
        user_id: Explicit user scope (used with scope_to_user).
        scope_to_user: When True, restrict the count to ``user_id``.

    Returns:
        Count of auto-originated tier changes.
    """
    conditions: list[str] = []
    params: list = []
    idx = 1

    # Filter on the non-spoofable provenance column
    conditions.append("auto_originated = TRUE")

    # Optional time window
    if since is not None:
        conditions.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1

    # Optional explicit user scope (defense-in-depth beyond RLS)
    if scope_to_user:
        conditions.append(f"user_id IS NOT DISTINCT FROM ${idx}")
        params.append(user_id)
        idx += 1

    where = " AND ".join(conditions)
    row = await get_db(pool).fetchrow(
        f"""
        SELECT count(*) AS total
        FROM policy_calibration_events
        WHERE {where}
        """,
        *params,
    )

    return row["total"] if row else 0


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

    # Check demotion first (safety takes priority). Demotion counts ALL origins:
    # an untrusted agent's rejections should still be able to flag a tier as
    # over-trusted for human review — the fail-safe direction.
    if total >= _DEMO_MIN_RECORDS and rejection_rate >= _DEMO_REJECTION_RATE:
        result["recommendation"] = "demote"
        result["reason"] = (
            f"Rejection rate {rejection_rate:.0%} >= {_DEMO_REJECTION_RATE:.0%} "
            f"threshold over {total} records"
        )
        result["stats"]["rejection_rate"] = rejection_rate
        return result

    # Check promotion using TRUSTED-origin records only. Granting autonomy
    # (tier -> always) is the privilege-escalation direction, so an untrusted
    # 'agent' caller's approvals must never count here — only 'supervisor'
    # (attested human / Face / Orchestrator) approvals do.
    trusted = await get_calibration_summary(
        pool,
        action_category=action_category,
        project_id=project_id,
        since=since,
        origin_in=TRUSTED_CALIBRATION_ORIGINS,
    )
    trusted_total = trusted["total"]
    trusted_approval_rate = trusted["approval_rate"]
    result["stats"]["trusted_total"] = trusted_total
    result["stats"]["trusted_approval_rate"] = trusted_approval_rate

    if (
        trusted_total >= _PROMO_MIN_RECORDS
        and trusted_approval_rate >= _PROMO_APPROVAL_RATE
    ):
        result["recommendation"] = "promote"
        result["reason"] = (
            f"Trusted approval rate {trusted_approval_rate:.0%} >= "
            f"{_PROMO_APPROVAL_RATE:.0%} threshold over {trusted_total} "
            f"trusted-origin records"
        )
        result["stats"]["approval_rate"] = approval_rate
        return result

    result["reason"] = (
        f"Insufficient evidence: {total} records "
        f"({trusted_total} trusted), {approval_rate:.0%} approval "
        f"({trusted_approval_rate:.0%} trusted), {rejection_rate:.0%} rejection"
    )
    result["stats"]["approval_rate"] = approval_rate
    result["stats"]["rejection_rate"] = rejection_rate
    return result
