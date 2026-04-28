"""Tests for behaviors user_id OR-NULL scoping in list and search functions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weft.behaviors import list_behaviors, match_behaviors
from weft.db.connection import get_db
from weft.models import BehaviorScope, _weft_id
from weft.schema import SYSTEM_GLOBAL_USER_ID


def _normalized_embedding(seed: float = 0.1) -> list[float]:
    """Generate a normalized 768-dim embedding (unit vector)."""
    import math
    raw = [math.sin(seed * (i + 1)) for i in range(768)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


async def _insert_behavior(
    pool,
    trigger_pattern: str,
    action: str = "test",
    user_id: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    embedding: list[float] | None = None,
):
    """Helper to insert a behavior with explicit user_id (bypasses current_setting).

    Caller-side ``user_id=None`` means "global row" — stored as the
    SYSTEM_GLOBAL_USER_ID sentinel under the post-mig-36 schema.
    """
    if user_id is None:
        user_id = SYSTEM_GLOBAL_USER_ID
    behavior_id = _weft_id()
    now = datetime.now(timezone.utc)
    await get_db(pool).execute(
        """
        INSERT INTO behaviors (
            id, trigger_pattern, action, confidence, scope,
            project_id, agent_id, user_id, priority, enabled,
            access_count, token_count, created_at, updated_at,
            embedding, status
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            0, $11, $12, $12,
            $13::vector, 'active'
        )
        """,
        behavior_id,
        trigger_pattern,
        action,
        0.7,  # confidence
        BehaviorScope.global_.value,  # scope
        project_id,
        agent_id,
        user_id,
        0,  # priority
        True,  # enabled
        10,  # token_count estimate
        now,
        embedding,
    )
    return behavior_id


# --- list_behaviors with user_id ---


async def test_list_behaviors_user_id_none_returns_all(pool):
    """user_id=None should return all rows (current behavior preserved)."""
    await _insert_behavior(pool, "user-a owned", user_id="user-a")
    await _insert_behavior(pool, "user-b owned", user_id="user-b")
    await _insert_behavior(pool, "global")

    results = await list_behaviors(pool, user_id=None)
    triggers = {b.trigger_pattern for b in results}
    assert len(results) == 3
    assert "user-a owned" in triggers
    assert "user-b owned" in triggers
    assert "global" in triggers


async def test_list_behaviors_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows, excluding user-b."""
    await _insert_behavior(pool, "user-a owned", user_id="user-a")
    await _insert_behavior(pool, "user-b owned", user_id="user-b")
    await _insert_behavior(pool, "global")

    results = await list_behaviors(pool, user_id="user-a")
    triggers = {b.trigger_pattern for b in results}
    assert len(results) == 2
    assert "user-a owned" in triggers
    assert "global" in triggers
    assert "user-b owned" not in triggers


async def test_list_behaviors_user_id_with_only_null(pool):
    """user_id='user-a' with only NULL rows should return those NULL rows."""
    await _insert_behavior(pool, "global 1")
    await _insert_behavior(pool, "global 2")

    results = await list_behaviors(pool, user_id="user-a")
    triggers = {b.trigger_pattern for b in results}
    assert len(results) == 2
    assert "global 1" in triggers
    assert "global 2" in triggers


async def test_list_behaviors_user_id_with_mixed_filters(pool):
    """user_id filter should work alongside other filters (project_id, agent_id, etc)."""
    await _insert_behavior(pool, "user-a proj-1", user_id="user-a", project_id="proj-1")
    await _insert_behavior(pool, "user-a proj-2", user_id="user-a", project_id="proj-2")
    await _insert_behavior(pool, "user-b proj-1", user_id="user-b", project_id="proj-1")
    await _insert_behavior(pool, "global proj-1", project_id="proj-1")

    # user_id="user-a" AND project_id="proj-1"
    results = await list_behaviors(pool, user_id="user-a", project_id="proj-1")
    triggers = {b.trigger_pattern for b in results}
    assert "user-a proj-1" in triggers
    assert "global proj-1" in triggers
    assert "user-a proj-2" not in triggers
    assert "user-b proj-1" not in triggers


# --- match_behaviors with user_id ---


async def test_match_behaviors_user_id_none_returns_all(pool):
    """user_id=None should return all rows from vector search."""
    emb = _normalized_embedding(0.1)

    await _insert_behavior(pool, "user-a rule", user_id="user-a", embedding=emb)
    await _insert_behavior(pool, "user-b rule", user_id="user-b", embedding=emb)
    await _insert_behavior(pool, "global rule", embedding=emb)

    results = await match_behaviors(pool, emb, threshold=0.0, user_id=None)
    triggers = {r.behavior.trigger_pattern for r in results}
    assert len(results) == 3
    assert "user-a rule" in triggers
    assert "user-b rule" in triggers
    assert "global rule" in triggers


async def test_match_behaviors_user_id_filters_or_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows from vector search."""
    emb = _normalized_embedding(0.1)

    await _insert_behavior(pool, "user-a rule", user_id="user-a", embedding=emb)
    await _insert_behavior(pool, "user-b rule", user_id="user-b", embedding=emb)
    await _insert_behavior(pool, "global rule", embedding=emb)

    results = await match_behaviors(pool, emb, threshold=0.0, user_id="user-a")
    triggers = {r.behavior.trigger_pattern for r in results}
    assert len(results) == 2
    assert "user-a rule" in triggers
    assert "global rule" in triggers
    assert "user-b rule" not in triggers


async def test_match_behaviors_user_id_with_mixed_filters(pool):
    """user_id filter should work alongside project_id and agent_id in vector search."""
    emb = _normalized_embedding(0.1)

    await _insert_behavior(
        pool, "user-a proj-1 agent-x", user_id="user-a", project_id="proj-1",
        agent_id="agent-x", embedding=emb,
    )
    await _insert_behavior(
        pool, "user-a proj-2 agent-x", user_id="user-a", project_id="proj-2",
        agent_id="agent-x", embedding=emb,
    )
    await _insert_behavior(
        pool, "user-b proj-1 agent-x", user_id="user-b", project_id="proj-1",
        agent_id="agent-x", embedding=emb,
    )
    await _insert_behavior(
        pool, "global proj-1 agent-x", project_id="proj-1",
        agent_id="agent-x", embedding=emb,
    )

    # user_id="user-a" AND project_id="proj-1" AND agent_id="agent-x"
    results = await match_behaviors(
        pool, emb, threshold=0.0,
        user_id="user-a", project_id="proj-1", agent_id="agent-x",
    )
    triggers = {r.behavior.trigger_pattern for r in results}
    assert "user-a proj-1 agent-x" in triggers
    assert "global proj-1 agent-x" in triggers
    assert "user-a proj-2 agent-x" not in triggers
    assert "user-b proj-1 agent-x" not in triggers
