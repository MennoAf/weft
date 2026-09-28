"""Tests for behaviors store layer (CRUD + vector match)."""

from __future__ import annotations

import pytest

from weft.behaviors import (
    delete_behavior,
    get_behavior,
    list_behaviors,
    match_behaviors,
    store_behavior,
    touch_behavior,
    update_behavior,
)
from weft.models import BehaviorCreate, BehaviorScope
from weft.schema import SYSTEM_GLOBAL_USER_ID


# --- Helpers ---

def _fake_embedding(seed: float = 0.1) -> list[float]:
    """Generate a deterministic 768-dim embedding for testing."""
    import math
    return [math.sin(seed * (i + 1)) for i in range(768)]


def _normalized_embedding(seed: float = 0.1) -> list[float]:
    """Generate a normalized 768-dim embedding (unit vector)."""
    raw = _fake_embedding(seed)
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


# --- store_behavior ---


async def test_store_behavior_minimal(pool):
    bc = BehaviorCreate(
        trigger_pattern="when writing Python",
        action="use type hints",
    )
    b = await store_behavior(pool, bc)
    assert b.id.startswith("weft-")
    assert b.trigger_pattern == "when writing Python"
    assert b.action == "use type hints"
    assert b.confidence == pytest.approx(0.7)
    assert b.scope == BehaviorScope.global_
    assert b.enabled is True
    assert b.access_count == 0
    assert b.token_count > 0


async def test_store_behavior_with_embedding(pool):
    bc = BehaviorCreate(
        trigger_pattern="when deploying",
        action="run tests first",
    )
    emb = _normalized_embedding(0.5)
    b = await store_behavior(pool, bc, embedding=emb)

    row = await pool.fetchrow("SELECT embedding FROM behaviors WHERE id = $1", b.id)
    assert row["embedding"] is not None


async def test_store_behavior_full_fields(pool):
    bc = BehaviorCreate(
        trigger_pattern="when user asks about testing",
        action="recommend pytest",
        confidence=0.9,
        scope=BehaviorScope.project,
        project_id="proj-1",
        agent_id="warp",
        user_id="user-1",
        priority=5,
        enabled=True,
    )
    b = await store_behavior(pool, bc)
    assert b.scope == BehaviorScope.project
    assert b.project_id == "proj-1"
    assert b.agent_id == "warp"
    assert b.user_id == "user-1"
    assert b.priority == 5


# --- get_behavior ---


async def test_get_behavior_found(pool):
    bc = BehaviorCreate(trigger_pattern="t", action="a")
    stored = await store_behavior(pool, bc)

    found = await get_behavior(pool, stored.id)
    assert found is not None
    assert found.id == stored.id
    assert found.trigger_pattern == "t"


async def test_get_behavior_not_found(pool):
    result = await get_behavior(pool, "nonexistent")
    assert result is None


# --- list_behaviors ---


async def test_list_behaviors_empty(pool):
    result = await list_behaviors(pool)
    assert result == []


async def test_list_behaviors_filters_by_enabled(pool):
    await store_behavior(pool, BehaviorCreate(trigger_pattern="t1", action="a1"))
    bc2 = BehaviorCreate(trigger_pattern="t2", action="a2", enabled=False)
    await store_behavior(pool, bc2)

    enabled = await list_behaviors(pool, enabled=True)
    assert len(enabled) == 1
    assert enabled[0].trigger_pattern == "t1"

    disabled = await list_behaviors(pool, enabled=False)
    assert len(disabled) == 1
    assert disabled[0].trigger_pattern == "t2"

    all_behaviors = await list_behaviors(pool, enabled=None)
    assert len(all_behaviors) == 2


async def test_list_behaviors_filters_by_scope(pool):
    await store_behavior(pool, BehaviorCreate(trigger_pattern="g", action="a"))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="p", action="a", scope=BehaviorScope.project, project_id="p1",
    ))

    global_only = await list_behaviors(pool, scope=BehaviorScope.global_)
    assert len(global_only) == 1
    assert global_only[0].trigger_pattern == "g"


async def test_list_behaviors_or_null_scoping(pool):
    """project_id filter uses OR-NULL: matches specific project + global behaviors."""
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="global", action="a",
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="proj-specific", action="a", project_id="proj-1",
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="other-proj", action="a", project_id="proj-2",
    ))

    results = await list_behaviors(pool, project_id="proj-1")
    triggers = {b.trigger_pattern for b in results}
    assert "global" in triggers
    assert "proj-specific" in triggers
    assert "other-proj" not in triggers


async def test_list_behaviors_ordered_by_priority(pool):
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="low", action="a", priority=1,
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="high", action="a", priority=10,
    ))
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="mid", action="a", priority=5,
    ))

    results = await list_behaviors(pool)
    assert results[0].trigger_pattern == "high"
    assert results[1].trigger_pattern == "mid"
    assert results[2].trigger_pattern == "low"


async def test_list_behaviors_respects_limit(pool):
    for i in range(5):
        await store_behavior(pool, BehaviorCreate(
            trigger_pattern=f"t{i}", action="a",
        ))

    results = await list_behaviors(pool, limit=3)
    assert len(results) == 3


async def test_list_behaviors_excludes_archived(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))
    await delete_behavior(pool, b.id)

    results = await list_behaviors(pool)
    assert len(results) == 0


# --- match_behaviors (vector search) ---


async def test_match_behaviors_by_similarity(pool):
    emb1 = _normalized_embedding(0.1)
    emb2 = _normalized_embedding(0.5)

    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when writing tests", action="use pytest",
    ), embedding=emb1)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="when deploying code", action="run CI first",
    ), embedding=emb2)

    # Search with emb close to emb1
    results = await match_behaviors(pool, emb1, limit=10, threshold=0.0)
    assert len(results) >= 1
    assert results[0].behavior.trigger_pattern == "when writing tests"
    assert results[0].similarity > 0.5


async def test_match_behaviors_threshold_filters(pool):
    emb = _normalized_embedding(0.1)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="t", action="a",
    ), embedding=emb)

    # Very high threshold should filter everything when query is different
    different_emb = _normalized_embedding(99.0)
    results = await match_behaviors(pool, different_emb, threshold=0.99)
    # May or may not match depending on how different the embeddings are
    # but the threshold logic is exercised
    for r in results:
        assert r.similarity >= 0.99


async def test_match_behaviors_or_null_scoping(pool):
    emb = _normalized_embedding(0.1)

    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="global rule", action="a",
    ), embedding=emb)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="proj rule", action="a", project_id="proj-1",
    ), embedding=emb)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="other proj", action="a", project_id="proj-2",
    ), embedding=emb)

    results = await match_behaviors(pool, emb, project_id="proj-1", threshold=0.0)
    triggers = {r.behavior.trigger_pattern for r in results}
    assert "global rule" in triggers
    assert "proj rule" in triggers
    assert "other proj" not in triggers


async def test_match_behaviors_excludes_disabled(pool):
    emb = _normalized_embedding(0.1)

    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="enabled", action="a",
    ), embedding=emb)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="disabled", action="a", enabled=False,
    ), embedding=emb)

    results = await match_behaviors(pool, emb, enabled=True, threshold=0.0)
    triggers = {r.behavior.trigger_pattern for r in results}
    assert "enabled" in triggers
    assert "disabled" not in triggers


async def test_match_behaviors_composite_ranking(pool):
    """Higher priority + confidence should rank higher even with similar similarity."""
    emb = _normalized_embedding(0.1)

    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="low priority", action="a", priority=0, confidence=0.5,
    ), embedding=emb)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="high priority", action="a", priority=10, confidence=0.9,
    ), embedding=emb)

    results = await match_behaviors(pool, emb, threshold=0.0)
    assert len(results) == 2
    assert results[0].behavior.trigger_pattern == "high priority"


async def test_match_behaviors_skips_no_embedding(pool):
    """Behaviors without embeddings are excluded from vector search."""
    emb = _normalized_embedding(0.1)

    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="has embedding", action="a",
    ), embedding=emb)
    await store_behavior(pool, BehaviorCreate(
        trigger_pattern="no embedding", action="a",
    ))

    results = await match_behaviors(pool, emb, threshold=0.0)
    triggers = {r.behavior.trigger_pattern for r in results}
    assert "has embedding" in triggers
    assert "no embedding" not in triggers


# --- update_behavior ---


async def test_update_behavior_action(pool):
    b = await store_behavior(pool, BehaviorCreate(
        trigger_pattern="t", action="old action",
    ))

    updated = await update_behavior(pool, b.id, action="new action")
    assert updated is not None
    assert updated.action == "new action"
    assert updated.updated_at > b.updated_at


async def test_update_behavior_confidence(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    updated = await update_behavior(pool, b.id, confidence=0.95)
    assert updated.confidence == pytest.approx(0.95, abs=0.01)


async def test_update_behavior_priority(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    updated = await update_behavior(pool, b.id, priority=10)
    assert updated.priority == 10


async def test_update_behavior_enabled(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    updated = await update_behavior(pool, b.id, enabled=False)
    assert updated.enabled is False


async def test_update_behavior_scope(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    updated = await update_behavior(pool, b.id, scope=BehaviorScope.project)
    assert updated.scope == BehaviorScope.project


async def test_update_behavior_not_found(pool):
    result = await update_behavior(pool, "nonexistent", action="a")
    assert result is None


async def test_update_behavior_recalculates_token_count(pool):
    b = await store_behavior(pool, BehaviorCreate(
        trigger_pattern="short", action="a",
    ))
    original_tokens = b.token_count

    updated = await update_behavior(
        pool, b.id, trigger_pattern="a much longer trigger pattern description",
    )
    assert updated.token_count > original_tokens


# --- delete_behavior ---


async def test_delete_behavior_soft(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    deleted = await delete_behavior(pool, b.id)
    assert deleted is True

    found = await get_behavior(pool, b.id)
    assert found is not None
    assert found.status == "archived"


async def test_delete_behavior_hard(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))

    deleted = await delete_behavior(pool, b.id, hard=True)
    assert deleted is True

    found = await get_behavior(pool, b.id)
    assert found is None


async def test_delete_behavior_not_found(pool):
    deleted = await delete_behavior(pool, "nonexistent")
    assert deleted is False


# --- touch_behavior ---


async def test_touch_behavior_increments_access_count(pool):
    b = await store_behavior(pool, BehaviorCreate(trigger_pattern="t", action="a"))
    assert b.access_count == 0

    await touch_behavior(pool, b.id)
    await touch_behavior(pool, b.id)

    found = await get_behavior(pool, b.id)
    assert found.access_count == 2
    assert found.updated_at > b.updated_at


# --- store_behavior user_id resolution ---


async def test_store_behavior_falls_back_to_system_global_when_guc_empty(pg_container):
    """When app.user_id GUC is empty AND no explicit user_id is supplied,
    store_behavior must default to the SYSTEM_GLOBAL sentinel — not NULL.

    Without this fallback a supervisor caller with no GUC would write a
    row that's invisible to its own subsequent match/list queries
    (the user_id filter excludes "neither mine nor sentinel").

    Builds a dedicated pool whose setup callback does NOT pre-populate
    the GUC, so we can exercise the empty-GUC code path that the default
    test pool's setup hides.
    """
    import asyncpg

    from weft.db.connection import _pgvector_codec_init, register_pgvector_codec
    from weft.db.migrations import run_migrations

    dsn = pg_container.get_connection_url().replace("+psycopg2", "")

    async def _no_op_setup(conn):
        # Intentionally do NOT set app.user_id — leaves it at its
        # default (empty), which is what we want to test.
        pass

    p = await asyncpg.create_pool(
        dsn, min_size=1, max_size=2,
        init=_pgvector_codec_init, setup=_no_op_setup,
    )
    try:
        await run_migrations(p)
        await register_pgvector_codec(p)
        await p.execute("TRUNCATE behaviors CASCADE")

        bc = BehaviorCreate(
            trigger_pattern="empty-guc fallback",
            action="land on sentinel",
        )
        b = await store_behavior(p, bc)
        assert b.user_id == SYSTEM_GLOBAL_USER_ID

        # Confirm at the DB row level too.
        row = await p.fetchrow(
            "SELECT user_id FROM behaviors WHERE id = $1", b.id,
        )
        assert row["user_id"] == SYSTEM_GLOBAL_USER_ID
    finally:
        await p.close()


async def test_store_behavior_explicit_user_id_takes_precedence(pool):
    """When BehaviorCreate.user_id is set, that value wins over the GUC."""
    bc = BehaviorCreate(
        trigger_pattern="explicit-user trigger",
        action="a",
        user_id="explicit-user-id",
    )
    b = await store_behavior(pool, bc)
    assert b.user_id == "explicit-user-id"

    # Verify at the DB level too, not just the returned object.
    row = await pool.fetchrow(
        "SELECT user_id FROM behaviors WHERE id = $1", b.id,
    )
    assert row["user_id"] == "explicit-user-id"


async def test_store_behavior_uses_guc_when_no_explicit_user_id(pool):
    """When BehaviorCreate.user_id is None, store falls through to the GUC.

    The pool fixture's setup callback sets app.user_id='test-user-default',
    so a behavior stored here should land under that user.
    """
    bc = BehaviorCreate(trigger_pattern="guc-fallback", action="a")
    b = await store_behavior(pool, bc)
    assert b.user_id == "test-user-default"


# --- truly-global behavior fires across project_ids ---


async def test_match_behaviors_truly_global_fires_cross_project(pool):
    """A behavior with scope='global', project_id=NULL, user_id=SYSTEM_GLOBAL
    must surface for match_behaviors callers in any project_id.

    Regression: scope='global' rules silently bound to their birth-project
    when ``weft_behavior_add`` auto-resolved project_id from CWD. The
    fix sets project_id=NULL for scope='global' so cross-project firing
    works as the docstring promises.
    """
    emb = _normalized_embedding(0.1)

    bc = BehaviorCreate(
        trigger_pattern="user says save this finding",
        action="weft_remember",
        scope=BehaviorScope.global_,
        user_id=SYSTEM_GLOBAL_USER_ID,
        # project_id intentionally None — truly global
    )
    stored = await store_behavior(pool, bc, embedding=emb)
    assert stored.project_id is None
    assert stored.user_id == SYSTEM_GLOBAL_USER_ID

    # Querying from project_id="warp" with a different user_id should
    # still surface the global rule.
    results = await match_behaviors(
        pool, emb,
        project_id="warp",
        user_id="warp-user-id",
        threshold=0.0,
    )
    triggers = {r.behavior.trigger_pattern for r in results}
    assert "user says save this finding" in triggers

    # Querying from a third project_id should also surface it.
    results_muttr = await match_behaviors(
        pool, emb,
        project_id="muttr",
        user_id="muttr-user-id",
        threshold=0.0,
    )
    triggers_muttr = {r.behavior.trigger_pattern for r in results_muttr}
    assert "user says save this finding" in triggers_muttr
