"""Cost-to-autonomy and cost-to-degradation enforcement loop.

When today's spend crosses a configured threshold band, the loop:

  1. Creates :class:`AutonomyOverride` rows that force the configured
     actions to a stricter tier (typically ``always`` -> ``earned`` at 90%,
     or ``earned`` -> ``never`` at 100%). Overrides expire at the next
     daily reset, so the underlying policies are never demoted — calibration
     remains a separate, evidence-based signal.

  2. Feeds ``cost_pct_used`` into :func:`weft.degradation.update_degradation_state`,
     which fires any degradation policies with ``trigger_type='cost_breach'``.
     Those policies' actions (escalate / pause / restart / restrict) are the
     operator-defined response to a budget breach.

  3. Updates the :class:`CostEnforcementState` row for today with
     ``max_threshold_fired_pct`` and an ``actions_taken`` jsonb log. This
     makes the loop idempotent within a band — a tick that finds the same
     band already active does nothing. A new day = a new state row.

Federation note: every action this loop takes is recorded in two places:
the override row itself (per-action, TTL'd, source-tagged) and the state
row's ``actions_taken`` log (per-day narrative). A foreign agent reading
either table can reconstruct *why* a tier is what it is right now.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import asyncpg

from weft.autonomy import (
    AutonomyOverrideCreate,
    AutonomyTier,
    OverrideSource,
    create_override,
)
from weft.config import CostEnforcementConfig, CostThreshold
from weft.cost_tracking import check_budget
from weft.db.connection import get_db
from weft.degradation import update_degradation_state
from weft.models import _now, _weft_id

logger = logging.getLogger(__name__)


# Config types live in weft/config/__init__.py to avoid a circular
# import — weft.db.connection imports WeftConfig at module load. They're
# re-exported here for callers who think of them as enforcement config.
__all__ = [
    "CostEnforcementConfig",
    "CostThreshold",
    "EnforcementAction",
    "EnforcementReport",
    "cost_enforcement_loop",
    "enforce_cost_thresholds",
]


# ---------------------------------------------------------------------------
# Result + state types
# ---------------------------------------------------------------------------


@dataclass
class EnforcementAction:
    """One action taken by an enforcement tick — for the audit log."""

    action_type: str  # "override_created" | "degradation_fired"
    target: str       # action name or policy_id
    threshold_pct: float
    metadata: dict[str, Any] = field(default_factory=dict)
    fired_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "threshold_pct": self.threshold_pct,
            "metadata": self.metadata,
            "fired_at": self.fired_at.isoformat(),
        }


@dataclass
class EnforcementReport:
    """Result of a single :func:`enforce_cost_thresholds` invocation."""

    pct_used: float = 0.0
    daily_spent_usd: float = 0.0
    daily_limit_usd: float = 0.0
    threshold_band_active: float | None = None  # highest crossed band today
    actions: list[EnforcementAction] = field(default_factory=list)
    state_row_id: str | None = None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pct_used": round(self.pct_used, 2),
            "daily_spent_usd": round(self.daily_spent_usd, 4),
            "daily_limit_usd": self.daily_limit_usd,
            "threshold_band_active": self.threshold_band_active,
            "actions": [a.to_dict() for a in self.actions],
            "state_row_id": self.state_row_id,
            "skipped_reason": self.skipped_reason,
        }


# ---------------------------------------------------------------------------
# Daily reset — overrides expire at the next UTC midnight by default.
# ---------------------------------------------------------------------------


def _next_daily_reset(now: datetime | None = None) -> datetime:
    """Return the next UTC midnight after *now*. Used as override TTL."""
    n = now or datetime.now(timezone.utc)
    next_midnight = datetime.combine(
        n.date() + timedelta(days=1), time.min, tzinfo=timezone.utc
    )
    return next_midnight


# ---------------------------------------------------------------------------
# State row helpers
# ---------------------------------------------------------------------------


async def _get_state(
    pool: asyncpg.Pool, state_date: date,
) -> tuple[str | None, float, dict[str, Any] | None]:
    """Fetch (id, max_threshold_fired_pct, actions_taken) for today's row.

    Returns (None, 0.0, None) if no row exists yet.
    """
    row = await get_db(pool).fetchrow(
        """
        SELECT id, max_threshold_fired_pct, actions_taken
        FROM cost_enforcement_state
        WHERE state_date = $1
          AND user_id = nullif(current_setting('app.user_id', true), '')
        """,
        state_date,
    )
    if row is None:
        return (None, 0.0, None)
    actions = row["actions_taken"]
    if isinstance(actions, str):
        actions = json.loads(actions)
    return (row["id"], float(row["max_threshold_fired_pct"]), actions)


async def _upsert_state(
    pool: asyncpg.Pool,
    *,
    state_date: date,
    daily_limit_usd: float,
    pct_used: float,
    max_threshold_fired_pct: float,
    actions_taken: list[dict[str, Any]],
) -> str:
    """Insert or update today's enforcement state row. Returns the row id."""
    row = await get_db(pool).fetchrow(
        """
        INSERT INTO cost_enforcement_state (
            id, state_date, user_id, max_threshold_fired_pct,
            daily_limit_usd, last_pct_used, last_evaluated_at, actions_taken
        )
        VALUES (
            $1, $2,
            nullif(current_setting('app.user_id', true), ''),
            $3, $4, $5, now(), $6::jsonb
        )
        ON CONFLICT (state_date, user_id) DO UPDATE
            SET max_threshold_fired_pct =
                    GREATEST(cost_enforcement_state.max_threshold_fired_pct,
                             EXCLUDED.max_threshold_fired_pct),
                last_pct_used     = EXCLUDED.last_pct_used,
                last_evaluated_at = now(),
                actions_taken     = EXCLUDED.actions_taken,
                updated_at        = now()
        RETURNING id
        """,
        _weft_id(),
        state_date,
        max_threshold_fired_pct,
        daily_limit_usd,
        pct_used,
        json.dumps(actions_taken),
    )
    return row["id"]


# ---------------------------------------------------------------------------
# Action discovery — when demote_actions=["*"], we need to know which
# actions exist so we can shadow them with overrides.
# ---------------------------------------------------------------------------


async def _resolve_action_targets(
    pool: asyncpg.Pool, demote_actions: list[str],
) -> list[str]:
    """Expand the configured demote_actions list to concrete action names.

    ``["*"]`` -> every distinct action in ``autonomy_policies``. Anything
    else is returned verbatim. Empty list -> empty list.
    """
    if not demote_actions:
        return []
    if demote_actions == ["*"]:
        rows = await get_db(pool).fetch(
            "SELECT DISTINCT action FROM autonomy_policies WHERE enabled = true"
        )
        return [r["action"] for r in rows]
    return list(demote_actions)


# ---------------------------------------------------------------------------
# Core enforcement function
# ---------------------------------------------------------------------------


async def enforce_cost_thresholds(
    pool: asyncpg.Pool,
    *,
    config: CostEnforcementConfig,
    now: datetime | None = None,
) -> EnforcementReport:
    """Single enforcement tick. Idempotent within a daily threshold band.

    Returns an :class:`EnforcementReport` describing what was found and
    what was done. Safe to call repeatedly — only crossings into a new,
    higher band cause new actions to fire.

    Disables cleanly when ``config.enabled`` is False or
    ``config.daily_limit_usd <= 0`` (returns a report with
    ``skipped_reason`` set).
    """
    report = EnforcementReport(daily_limit_usd=config.daily_limit_usd)

    if not config.enabled:
        report.skipped_reason = "disabled"
        return report
    if config.daily_limit_usd <= 0:
        report.skipped_reason = "no_limit"
        return report
    if not config.thresholds:
        report.skipped_reason = "no_thresholds"
        return report

    # Snapshot today's spend.
    status = await check_budget(pool, config.daily_limit_usd)
    report.pct_used = status.pct_used
    report.daily_spent_usd = status.daily_spent_usd

    # Find the highest threshold crossed by current spend.
    sorted_thresholds = sorted(config.thresholds, key=lambda t: t.pct_used)
    crossed: CostThreshold | None = None
    for band in sorted_thresholds:
        if status.pct_used >= band.pct_used:
            crossed = band
    if crossed is None:
        # Below the lowest band — still record state so primer can show
        # current posture. No actions.
        ref_now = now or datetime.now(timezone.utc)
        state_id = await _upsert_state(
            pool,
            state_date=ref_now.date(),
            daily_limit_usd=config.daily_limit_usd,
            pct_used=status.pct_used,
            max_threshold_fired_pct=0.0,
            actions_taken=[],
        )
        report.state_row_id = state_id
        return report

    report.threshold_band_active = crossed.pct_used

    ref_now = now or datetime.now(timezone.utc)
    state_id, prior_max, prior_actions = await _get_state(pool, ref_now.date())

    # Idempotency gate: if today already fired at or above this band, do
    # nothing new. The state row's last_pct_used + last_evaluated_at still
    # update (so the primer reflects the current snapshot).
    if prior_max >= crossed.pct_used:
        new_state_id = await _upsert_state(
            pool,
            state_date=ref_now.date(),
            daily_limit_usd=config.daily_limit_usd,
            pct_used=status.pct_used,
            max_threshold_fired_pct=prior_max,
            actions_taken=prior_actions or [],
        )
        report.state_row_id = new_state_id
        return report

    expires = _next_daily_reset(ref_now)
    new_actions: list[EnforcementAction] = []

    # 1) Autonomy demotions via overrides. We want to fire every band the
    #    spend has crossed since the prior_max — not just the highest —
    #    because each band may have its own demote_to. Sorted ascending,
    #    skip bands at or below prior_max, fire each remaining up to and
    #    including the active band.
    for band in sorted_thresholds:
        if band.pct_used <= prior_max:
            continue
        if band.pct_used > crossed.pct_used:
            break

        # CostThreshold.demote_to is a config-local string enum to keep
        # weft.config free of weft.autonomy imports. Coerce by value.
        demote_tier = AutonomyTier(band.demote_to.value)

        targets = await _resolve_action_targets(pool, band.demote_actions)
        for action_name in targets:
            try:
                ov = await create_override(
                    pool,
                    AutonomyOverrideCreate(
                        action=action_name,
                        effective_tier=demote_tier,
                        source=OverrideSource.cost_enforcement,
                        reason=(
                            f"cost_enforcement: spend at {status.pct_used:.1f}% "
                            f"crossed {band.pct_used:.0f}% band"
                        ),
                        expires_at=expires,
                        metadata={
                            "threshold_pct": band.pct_used,
                            "pct_used_at_fire": status.pct_used,
                            "daily_limit_usd": config.daily_limit_usd,
                        },
                    ),
                )
            except Exception:
                logger.exception(
                    "cost_enforcement.override_failed",
                    extra={"action": action_name, "band_pct": band.pct_used},
                )
                continue

            new_actions.append(EnforcementAction(
                action_type="override_created",
                target=action_name,
                threshold_pct=band.pct_used,
                metadata={
                    "override_id": ov.id,
                    "effective_tier": demote_tier.value,
                    "expires_at": expires.isoformat(),
                },
            ))

        # 2) Feed cost_pct_used into degradation evaluation for this band.
        if band.feed_degradation:
            try:
                triggered = await update_degradation_state(
                    pool,
                    metrics={"cost_pct_used": status.pct_used},
                )
            except Exception:
                logger.exception(
                    "cost_enforcement.degradation_eval_failed",
                    extra={"band_pct": band.pct_used},
                )
                triggered = []

            for fire in triggered:
                new_actions.append(EnforcementAction(
                    action_type="degradation_fired",
                    target=fire["policy_id"],
                    threshold_pct=band.pct_used,
                    metadata={
                        "policy_name": fire.get("name"),
                        "policy_action": fire.get("action"),
                        "reason": fire.get("reason"),
                    },
                ))

    # Persist updated state. Merge new_actions into prior log.
    combined_actions = list(prior_actions or []) + [a.to_dict() for a in new_actions]
    state_id = await _upsert_state(
        pool,
        state_date=ref_now.date(),
        daily_limit_usd=config.daily_limit_usd,
        pct_used=status.pct_used,
        max_threshold_fired_pct=crossed.pct_used,
        actions_taken=combined_actions,
    )
    report.state_row_id = state_id
    report.actions = new_actions

    if new_actions:
        logger.info(
            "cost_enforcement.fired",
            extra={
                "pct_used": round(status.pct_used, 2),
                "band": crossed.pct_used,
                "actions_fired": len(new_actions),
            },
        )
    return report


# ---------------------------------------------------------------------------
# Background loop — wired into MCP lifespan alongside the other 7 loops.
# ---------------------------------------------------------------------------


_MIN_INTERVAL = 30  # seconds; cost spend doesn't change faster than this in practice


async def cost_enforcement_loop(
    pool: asyncpg.Pool,
    *,
    config: CostEnforcementConfig,
) -> None:
    """Background task: re-evaluates cost thresholds on a fixed interval.

    Disables cleanly (returns without raising) when config.enabled is False
    or daily_limit_usd <= 0. Per-cycle errors are logged; the loop continues.
    """
    if not config.enabled:
        logger.info("cost_enforcement.disabled (config.enabled=False)")
        return
    if config.daily_limit_usd <= 0:
        logger.warning(
            "cost_enforcement.no_limit — daily_limit_usd <= 0; loop disabled"
        )
        return

    interval = max(config.interval_seconds, _MIN_INTERVAL)
    logger.info(
        "cost_enforcement.started",
        extra={
            "interval": interval,
            "daily_limit_usd": config.daily_limit_usd,
            "thresholds": [t.pct_used for t in config.thresholds],
        },
    )
    try:
        while True:
            try:
                report = await enforce_cost_thresholds(pool, config=config)
                if report.actions:
                    logger.info(
                        "cost_enforcement.cycle",
                        extra={
                            "pct_used": round(report.pct_used, 2),
                            "band": report.threshold_band_active,
                            "actions": len(report.actions),
                        },
                    )
            except Exception:
                logger.exception("cost_enforcement.cycle_error")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("cost_enforcement.stopped")
        raise
