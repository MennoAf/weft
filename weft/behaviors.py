"""Behaviors store — CRUD and vector matching for persistent agent rules.

Follows the same patterns as store.py: OR-NULL scoping, soft delete,
vector similarity search, access tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncpg

from weft.db.connection import get_db
from weft.models import Behavior, BehaviorCreate, BehaviorMatch, BehaviorScope, _weft_id
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None


async def store_behavior(
    pool: asyncpg.Pool,
    create: BehaviorCreate,
    embedding: list[float] | None = None,
) -> Behavior:
    """Store a new behavior. Returns the created Behavior."""
    behavior_id = _weft_id()
    now = datetime.now(timezone.utc)
    token_count = estimate_tokens(create.trigger_pattern + " " + create.action)

    await get_db(pool).execute(
        """
        INSERT INTO behaviors (
            id, trigger_pattern, action, confidence, scope,
            project_id, agent_id, user_id, priority, enabled,
            access_count, token_count, created_at, updated_at,
            embedding, status
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, nullif(current_setting('app.user_id', true), ''), $8, $9,
            0, $10, $11, $11,
            $12::vector, 'active'
        )
        """,
        behavior_id,
        create.trigger_pattern,
        create.action,
        create.confidence,
        create.scope.value,
        create.project_id,
        create.agent_id,
        create.priority,
        create.enabled,
        token_count,
        now,
        embedding,
    )

    return Behavior(
        id=behavior_id,
        trigger_pattern=create.trigger_pattern,
        action=create.action,
        confidence=create.confidence,
        scope=create.scope,
        project_id=create.project_id,
        agent_id=create.agent_id,
        user_id=create.user_id,
        priority=create.priority,
        enabled=create.enabled,
        access_count=0,
        token_count=token_count,
        created_at=now,
        updated_at=now,
    )


async def get_behavior(pool: asyncpg.Pool, behavior_id: str) -> Behavior | None:
    """Fetch a single behavior by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow("SELECT * FROM behaviors WHERE id = $1", behavior_id)
    if not row:
        return None
    return _row_to_behavior(row)


async def list_behaviors(
    pool: asyncpg.Pool,
    *,
    scope: BehaviorScope | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    enabled: bool | None = True,
    status: str = "active",
    limit: int = 50,
    offset: int = 0,
) -> list[Behavior]:
    """List behaviors with optional filters.

    Scoping uses OR-NULL logic on project_id/agent_id/user_id (matches value OR global).

    Args:
        pool: Database connection pool.
        scope: Optional scope filter.
        project_id: If provided, filters to behaviors owned by this project OR globally-scoped behaviors (project_id IS NULL). If None, returns all.
        agent_id: If provided, filters to behaviors owned by this agent OR globally-scoped behaviors (agent_id IS NULL). If None, returns all.
        user_id: If provided, filters to behaviors owned by this user OR globally-scoped behaviors (user_id IS NULL). If None, returns all.
        enabled: If True, only enabled behaviors. If False, only disabled. If None, all.
        status: Status filter (default "active").
        limit: Max results (default 50).
        offset: Offset for pagination (default 0).
    """
    conditions = []
    params: list = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if scope is not None:
        conditions.append(f"scope = ${idx}")
        params.append(scope.value)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id = ${idx + 1})")
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

    if enabled is not None:
        conditions.append(f"enabled = ${idx}")
        params.append(enabled)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM behaviors {where}
        ORDER BY priority DESC, confidence DESC, updated_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_behavior(r) for r in rows]


async def match_behaviors(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    limit: int = 10,
    threshold: float = 0.1,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    enabled: bool | None = True,
) -> list[BehaviorMatch]:
    """Match behaviors by vector similarity on trigger_pattern embedding.

    Results ranked by composite score: similarity * confidence * (1 + priority/10).
    Uses OR-NULL scoping on project_id/agent_id/user_id.

    Args:
        pool: Database connection pool.
        embedding: Query embedding vector.
        limit: Max results (default 10).
        threshold: Similarity threshold (default 0.1).
        project_id: If provided, filters to behaviors owned by this project OR globally-scoped behaviors (project_id IS NULL). If None, returns all.
        agent_id: If provided, filters to behaviors owned by this agent OR globally-scoped behaviors (agent_id IS NULL). If None, returns all.
        user_id: If provided, filters to behaviors owned by this user OR globally-scoped behaviors (user_id IS NULL). If None, returns all.
        enabled: If True, only enabled behaviors. If False, only disabled. If None, all.
    """
    conditions = ["embedding IS NOT NULL", "status = 'active'"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    # Similarity threshold
    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        conditions.append(f"(user_id = ${idx} OR user_id = ${idx + 1})")
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

    if enabled is not None:
        conditions.append(f"enabled = ${idx}")
        params.append(enabled)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT *,
               1 - (embedding <=> $1::vector) AS similarity
        FROM behaviors
        {where}
        ORDER BY embedding <=> $1::vector
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(query, *params)

    results = []
    for row in rows:
        behavior = _row_to_behavior(row)
        sim = float(row["similarity"])
        results.append(BehaviorMatch(behavior=behavior, similarity=sim))

    # Re-rank by composite score: similarity * confidence * priority boost
    results.sort(
        key=lambda bm: bm.similarity * bm.behavior.confidence * (1 + bm.behavior.priority / 10),
        reverse=True,
    )

    return results


async def update_behavior(
    pool: asyncpg.Pool,
    behavior_id: str,
    *,
    trigger_pattern: str | None = None,
    action: str | None = None,
    confidence: float | None = None,
    scope: BehaviorScope | None = None,
    project_id: str | None = _UNSET,
    agent_id: str | None = _UNSET,
    priority: int | None = None,
    enabled: bool | None = None,
    status: str | None = None,
    embedding: list[float] | None = None,
) -> Behavior | None:
    """Update mutable fields of a behavior. Returns updated Behavior or None."""
    sets = ["updated_at = now()"]
    params: list = []
    idx = 1

    if trigger_pattern is not None:
        sets.append(f"trigger_pattern = ${idx}")
        params.append(trigger_pattern)
        idx += 1

    if action is not None:
        sets.append(f"action = ${idx}")
        params.append(action)
        idx += 1

    if trigger_pattern is not None or action is not None:
        # Recalculate token count
        # We need current values for the unchanged field
        current = await get_db(pool).fetchrow(
            "SELECT trigger_pattern, action FROM behaviors WHERE id = $1",
            behavior_id,
        )
        if current:
            tp = trigger_pattern or current["trigger_pattern"]
            act = action or current["action"]
            sets.append(f"token_count = ${idx}")
            params.append(estimate_tokens(tp + " " + act))
            idx += 1

    if confidence is not None:
        sets.append(f"confidence = ${idx}")
        params.append(confidence)
        idx += 1

    if scope is not None:
        sets.append(f"scope = ${idx}")
        params.append(scope.value)
        idx += 1

    if project_id is not _UNSET:
        sets.append(f"project_id = ${idx}")
        params.append(project_id)
        idx += 1

    if agent_id is not _UNSET:
        sets.append(f"agent_id = ${idx}")
        params.append(agent_id)
        idx += 1

    if priority is not None:
        sets.append(f"priority = ${idx}")
        params.append(priority)
        idx += 1

    if enabled is not None:
        sets.append(f"enabled = ${idx}")
        params.append(enabled)
        idx += 1

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    if embedding is not None:
        sets.append(f"embedding = ${idx}::vector")
        params.append(embedding)
        idx += 1

    set_clause = ", ".join(sets)
    params.append(behavior_id)

    row = await get_db(pool).fetchrow(
        f"UPDATE behaviors SET {set_clause} WHERE id = ${idx} RETURNING *",
        *params,
    )
    return _row_to_behavior(row) if row else None


async def delete_behavior(
    pool: asyncpg.Pool,
    behavior_id: str,
    *,
    hard: bool = False,
) -> bool:
    """Delete a behavior. Soft-delete (archive) by default."""
    db = get_db(pool)
    if hard:
        result = await db.execute("DELETE FROM behaviors WHERE id = $1", behavior_id)
    else:
        result = await db.execute(
            "UPDATE behaviors SET status = 'archived', updated_at = now() WHERE id = $1",
            behavior_id,
        )
    return result.split()[-1] != "0"


async def touch_behavior(pool: asyncpg.Pool, behavior_id: str) -> None:
    """Increment access_count and update updated_at."""
    await get_db(pool).execute(
        """
        UPDATE behaviors
        SET access_count = access_count + 1,
            updated_at = now()
        WHERE id = $1
        """,
        behavior_id,
    )


# --- Helpers ---


def _row_to_behavior(row: asyncpg.Record) -> Behavior:
    """Convert a database row to a Behavior model."""
    return Behavior(
        id=row["id"],
        trigger_pattern=row["trigger_pattern"],
        action=row["action"],
        confidence=row["confidence"],
        scope=BehaviorScope(row["scope"]),
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        user_id=row["user_id"],
        priority=row["priority"],
        enabled=bool(row["enabled"]),
        access_count=row["access_count"],
        token_count=row["token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=row["status"],
    )
