"""Autonomy policy models for governing agent action permissions.

The three-tier authority model:
- NEVER: Hard stops — actions the agent must never take autonomously.
- EARNED: Actions that start restricted but can be promoted via calibration.
- ALWAYS: Safe-zone actions the agent can always perform without asking.

Philosophy: conservative defaults, iterate from usage. Everything defaults to
the most restricted tier and migrates upward based on real calibration data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

from weft.db.connection import get_db
from weft.models import _now, _weft_id

logger = logging.getLogger(__name__)


class AutonomyTier(str, Enum):
    never = "never"
    earned = "earned"
    always = "always"


# Strictness ordering — when multiple overrides apply to the same action,
# the most restrictive wins. This is the federation-safe default: if cost
# enforcement says "earned" and a degradation policy says "never", the
# agent must ask. The same ordering also picks an override over a more
# permissive baseline policy.
_TIER_STRICTNESS = {
    AutonomyTier.never: 2,
    AutonomyTier.earned: 1,
    AutonomyTier.always: 0,
}


def _stricter(a: AutonomyTier, b: AutonomyTier) -> AutonomyTier:
    """Return the more restrictive of two tiers."""
    return a if _TIER_STRICTNESS[a] >= _TIER_STRICTNESS[b] else b


class OverrideSource(str, Enum):
    """Where the override came from. Used in audit + primer rendering."""

    cost_enforcement = "cost_enforcement"
    degradation_policy = "degradation_policy"
    manual = "manual"


class ActionPolicy(BaseModel):
    """A policy governing whether an agent may perform a specific action."""

    id: str = Field(default_factory=_weft_id)
    action: str  # e.g. "send_slack_message", "create_pr", "deploy"
    tier: AutonomyTier = AutonomyTier.never
    description: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["tier"] = self.tier.value
        return d


class ActionPolicyCreate(BaseModel):
    """Input model for creating an action policy."""

    action: str
    tier: AutonomyTier = AutonomyTier.never
    description: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    enabled: bool = True


class PolicyCalibrationEvent(BaseModel):
    """A record of a tier change or approval/rejection used for calibration."""

    id: str = Field(default_factory=_weft_id)
    policy_id: str
    previous_tier: AutonomyTier
    new_tier: AutonomyTier
    reason: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        d = self.model_dump(mode="json")
        d["previous_tier"] = self.previous_tier.value
        d["new_tier"] = self.new_tier.value
        return d


class PolicyCalibrationEventCreate(BaseModel):
    """Input model for recording a calibration event."""

    policy_id: str
    previous_tier: AutonomyTier
    new_tier: AutonomyTier
    reason: str | None = None
    agent_id: str | None = None
    user_id: str | None = None


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


def _row_to_policy(row: asyncpg.Record) -> ActionPolicy:
    """Convert a database row to an ActionPolicy model."""
    conditions = row["conditions"]
    if isinstance(conditions, str):
        conditions = json.loads(conditions)
    return ActionPolicy(
        id=row["id"],
        action=row["action"],
        tier=row["tier"],
        description=row["description"],
        conditions=conditions or {},
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        enabled=row["enabled"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_calibration_event(row: asyncpg.Record) -> PolicyCalibrationEvent:
    """Convert a database row to a PolicyCalibrationEvent model."""
    return PolicyCalibrationEvent(
        id=row["id"],
        policy_id=row["policy_id"],
        previous_tier=row["previous_tier"],
        new_tier=row["new_tier"],
        reason=row["reason"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
    )


async def create_policy(
    pool: asyncpg.Pool, create: ActionPolicyCreate,
) -> ActionPolicy:
    """Insert a new action policy. Returns the created ActionPolicy."""
    policy_id = _weft_id()
    conditions_json = json.dumps(create.conditions)

    db = get_db(pool)
    row = await db.fetchrow(
        """
        INSERT INTO autonomy_policies (
            id, action, tier, description, conditions,
            project_id, agent_id, user_id, enabled
        )
        VALUES (
            $1, $2, $3, $4, $5::jsonb,
            $6, $7,
            nullif(current_setting('app.user_id', true), ''),
            $8
        )
        RETURNING *
        """,
        policy_id,
        create.action,
        create.tier.value,
        create.description,
        conditions_json,
        create.project_id,
        create.agent_id,
        create.enabled,
    )
    return _row_to_policy(row)


async def get_policy(pool: asyncpg.Pool, policy_id: str) -> ActionPolicy | None:
    """Fetch a policy by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow(
        "SELECT * FROM autonomy_policies WHERE id = $1",
        policy_id,
    )
    return _row_to_policy(row) if row else None


async def get_policy_by_action(
    pool: asyncpg.Pool, action: str,
) -> ActionPolicy | None:
    """Fetch the first enabled policy matching an action name."""
    row = await get_db(pool).fetchrow(
        """
        SELECT * FROM autonomy_policies
        WHERE action = $1 AND enabled = true
        ORDER BY created_at DESC
        LIMIT 1
        """,
        action,
    )
    return _row_to_policy(row) if row else None


async def list_policies(
    pool: asyncpg.Pool,
    *,
    tier: AutonomyTier | None = None,
    enabled_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> list[ActionPolicy]:
    """List policies, optionally filtered by tier."""
    clauses = []
    params: list[Any] = []
    idx = 1

    if tier is not None:
        clauses.append(f"tier = ${idx}")
        params.append(tier.value)
        idx += 1

    if enabled_only:
        clauses.append("enabled = true")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM autonomy_policies
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
        """,
        *params,
        limit,
        offset,
    )
    return [_row_to_policy(r) for r in rows]


async def update_policy_tier(
    pool: asyncpg.Pool,
    policy_id: str,
    new_tier: AutonomyTier,
    *,
    reason: str | None = None,
    agent_id: str | None = None,
) -> ActionPolicy:
    """Change a policy's tier, recording a calibration event.

    Raises ValueError if the policy is a NEVER hard-stop (immutable).
    Raises LookupError if the policy does not exist.
    """
    db = get_db(pool)

    # Fetch current policy
    row = await db.fetchrow(
        "SELECT * FROM autonomy_policies WHERE id = $1",
        policy_id,
    )
    if row is None:
        raise LookupError(f"Policy {policy_id} not found")

    current = _row_to_policy(row)

    if current.tier == AutonomyTier.never:
        raise ValueError(
            f"Policy {policy_id} is a NEVER hard-stop and cannot be changed"
        )

    # Update the tier
    updated_row = await db.fetchrow(
        """
        UPDATE autonomy_policies
        SET tier = $1, updated_at = now()
        WHERE id = $2
        RETURNING *
        """,
        new_tier.value,
        policy_id,
    )

    # Record the calibration event
    event_id = _weft_id()
    await db.execute(
        """
        INSERT INTO policy_calibration_events (
            id, policy_id, previous_tier, new_tier, reason, agent_id, user_id
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            nullif(current_setting('app.user_id', true), '')
        )
        """,
        event_id,
        policy_id,
        current.tier.value,
        new_tier.value,
        reason,
        agent_id,
    )

    return _row_to_policy(updated_row)


async def delete_policy(pool: asyncpg.Pool, policy_id: str) -> bool:
    """Delete a policy. Returns True if deleted."""
    result = await get_db(pool).execute(
        "DELETE FROM autonomy_policies WHERE id = $1",
        policy_id,
    )
    return result.split()[-1] != "0"


async def get_tier_for_action(
    pool: asyncpg.Pool, action: str,
) -> AutonomyTier:
    """Baseline tier for an action — policy table only, no overrides.

    Returns the tier from the most recent enabled policy for the action.
    If no policy exists, defaults to EARNED (conservative but not blocked).

    Use :func:`get_effective_tier` to also consult :class:`AutonomyOverride`
    rows (cost enforcement, degradation locks, manual). For agent runtime
    decisions, always prefer ``get_effective_tier``; this function exists
    for inspecting baseline intent and for tests.
    """
    policy = await get_policy_by_action(pool, action)
    if policy is None:
        return AutonomyTier.earned
    return policy.tier


async def list_calibration_events(
    pool: asyncpg.Pool,
    policy_id: str,
    *,
    limit: int = 50,
) -> list[PolicyCalibrationEvent]:
    """List calibration events for a policy, newest first."""
    rows = await get_db(pool).fetch(
        """
        SELECT * FROM policy_calibration_events
        WHERE policy_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        policy_id,
        limit,
    )
    return [_row_to_calibration_event(r) for r in rows]


# ---------------------------------------------------------------------------
# Autonomy overrides — TTL'd circuit breakers consulted *before* the
# baseline policy table. Federation-safe by construction:
#
#   - Each row is self-describing (source + reason + metadata).
#   - Rows expire on their own — no "remember to undo" code path.
#   - Multiple sources stack; resolution = strictest tier wins.
#   - Underlying policies are never demoted, so calibration-driven
#     promotion/demotion stays a separate signal from cost circuit
#     breakers.
# ---------------------------------------------------------------------------


class AutonomyOverride(BaseModel):
    """A TTL'd override that forces a specific tier for an action."""

    id: str = Field(default_factory=_weft_id)
    action: str
    effective_tier: AutonomyTier
    source: OverrideSource
    reason: str | None = None
    expires_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["effective_tier"] = self.effective_tier.value
        d["source"] = self.source.value
        return d


class AutonomyOverrideCreate(BaseModel):
    """Input model for creating an override."""

    action: str
    effective_tier: AutonomyTier
    source: OverrideSource
    expires_at: datetime
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    agent_id: str | None = None


def _row_to_override(row: asyncpg.Record) -> AutonomyOverride:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return AutonomyOverride(
        id=row["id"],
        action=row["action"],
        effective_tier=AutonomyTier(row["effective_tier"]),
        source=OverrideSource(row["source"]),
        reason=row["reason"],
        expires_at=row["expires_at"],
        metadata=metadata or {},
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
    )


async def create_override(
    pool: asyncpg.Pool, create: AutonomyOverrideCreate,
) -> AutonomyOverride:
    """Insert a new override. ``expires_at`` must be in the future."""
    if create.expires_at <= datetime.now(timezone.utc):
        raise ValueError("expires_at must be in the future")

    override_id = _weft_id()
    metadata_json = json.dumps(create.metadata)

    row = await get_db(pool).fetchrow(
        """
        INSERT INTO autonomy_overrides (
            id, action, effective_tier, source, reason,
            expires_at, metadata, project_id, agent_id, user_id
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6, $7::jsonb, $8, $9,
            nullif(current_setting('app.user_id', true), '')
        )
        RETURNING *
        """,
        override_id,
        create.action,
        create.effective_tier.value,
        create.source.value,
        create.reason,
        create.expires_at,
        metadata_json,
        create.project_id,
        create.agent_id,
    )
    return _row_to_override(row)


async def get_active_overrides(
    pool: asyncpg.Pool,
    *,
    action: str | None = None,
    source: OverrideSource | None = None,
) -> list[AutonomyOverride]:
    """Return overrides whose ``expires_at`` is still in the future.

    Optional filters narrow by action name or source. Newest first.
    """
    clauses = ["expires_at > now()"]
    params: list[Any] = []
    idx = 1

    if action is not None:
        clauses.append(f"action = ${idx}")
        params.append(action)
        idx += 1

    if source is not None:
        clauses.append(f"source = ${idx}")
        params.append(source.value)
        idx += 1

    where = " AND ".join(clauses)
    rows = await get_db(pool).fetch(
        f"""
        SELECT * FROM autonomy_overrides
        WHERE {where}
        ORDER BY created_at DESC
        """,
        *params,
    )
    return [_row_to_override(r) for r in rows]


async def get_effective_tier(
    pool: asyncpg.Pool, action: str,
) -> AutonomyTier:
    """Resolve the *current* tier for an action, taking overrides into account.

    Resolution order:
      1. Live overrides (``expires_at > now()``) — strictest tier wins
         across all matching rows. Strictness: never > earned > always.
      2. Baseline policy from ``autonomy_policies``.
      3. EARNED default (conservative; agent must ask).

    If overrides exist *and* a baseline policy exists, the result is the
    stricter of (strictest override) and (baseline policy). This means an
    override that says ``always`` cannot be used to *escalate* a baseline
    of ``earned`` — overrides are circuit breakers, not promotions.
    """
    overrides = await get_active_overrides(pool, action=action)
    baseline = await get_tier_for_action(pool, action)

    if not overrides:
        return baseline

    strictest_override = overrides[0].effective_tier
    for ov in overrides[1:]:
        strictest_override = _stricter(strictest_override, ov.effective_tier)

    return _stricter(strictest_override, baseline)


async def expire_overrides(
    pool: asyncpg.Pool,
    *,
    source: OverrideSource | None = None,
    older_than_days: int | None = None,
) -> int:
    """Hard-delete overrides whose ``expires_at`` has passed.

    The resolver already filters by ``expires_at > now()`` so cleanup is
    purely a housekeeping operation — the table doesn't grow unbounded
    if this runs occasionally. Returns the count deleted.

    ``source`` narrows to a specific origin (e.g. only sweep
    cost_enforcement leftovers). ``older_than_days`` adds a grace period
    so very recent expirations remain visible in the audit trail for a
    short window.
    """
    clauses = ["expires_at <= now()"]
    params: list[Any] = []
    idx = 1

    if source is not None:
        clauses.append(f"source = ${idx}")
        params.append(source.value)
        idx += 1

    if older_than_days is not None:
        clauses.append(f"expires_at < now() - make_interval(days => ${idx})")
        params.append(older_than_days)
        idx += 1

    where = " AND ".join(clauses)
    result = await get_db(pool).execute(
        f"DELETE FROM autonomy_overrides WHERE {where}",
        *params,
    )
    return int(result.rsplit(" ", 1)[-1]) if result else 0
