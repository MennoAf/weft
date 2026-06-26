"""Tests for gather_topic_memories (Tier-1 complete topic gather, V1).

done_when assertions:
  (1) Seed 60 active memories under one tag; gather returns all 60 (no limit=10/50 cap).
  (2) Excludes memories with status != 'active'.
  (3) Results ordered by created_at ascending.
  (4) RLS isolation — another user's tagged memory is NOT returned.
  (5) Entity-graph secondary merge sets truncated=True when entity-linked set
      would hit the LIMIT 100 cap in get_entity_memories.

HARD CONSTRAINT: does NOT modify conftest.py.
Each test uses a UNIQUE per-test user_id (uuid-derived) to be self-isolating.
RLS isolation test uses two distinct uuid-derived user_ids.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from weft.auth import current_user_id
from weft.db.connection import acquire, get_db
from weft.entities import link_mention, store_entity
from weft.models import EntityCreate, EntityType, MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory
from weft.topic_gather import gather_topic_memories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_user_id() -> str:
    """Generate a unique per-test user_id to ensure test isolation."""
    return f"tg-test-{uuid.uuid4().hex[:12]}"


async def _seed_memory(
    pool,
    user_id: str,
    tag: str,
    content: str,
    created_at: datetime | None = None,
    status: str = "active",
) -> str:
    """Seed a memory for user_id with the given tag. Returns memory id.

    Uses store_memory (real write path) so the row is schema-valid.
    For archived rows we call store_memory then patch status directly,
    since MemoryCreate always writes 'active'.
    """
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            mem = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content=content,
                    topic=[tag],
                    source=MemorySource.conversation,
                    confidence=0.7,
                ),
            )
    finally:
        current_user_id.reset(tok)

    # Patch created_at and/or status if needed (superuser pool bypasses RLS)
    if created_at is not None or status != "active":
        updates = []
        params = []
        idx = 1
        if created_at is not None:
            updates.append(f"created_at = ${idx}")
            params.append(created_at)
            idx += 1
        if status != "active":
            updates.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        params.append(mem.id)
        await pool.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ${idx}",
            *params,
        )

    return mem.id


# ---------------------------------------------------------------------------
# (1) 60-memory completeness — no limit cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_returns_all_60_no_limit_cap(pool):
    """Seed 60 active memories under one tag; gather must return all 60.

    Proves Tier-1 does NOT route through the limit=10/50-capped search
    functions in store.py. V1 completeness assertion.
    """
    user_id = _unique_user_id()
    tag = f"tg-tag-{uuid.uuid4().hex[:8]}"

    # Seed 60 memories
    seeded_ids: set[str] = set()
    for i in range(60):
        mid = await _seed_memory(pool, user_id, tag, f"memory content {i}")
        seeded_ids.add(mid)

    result = await gather_topic_memories(pool, [tag], user_id)

    returned_ids = {m.id for m in result["memories"]}
    assert len(result["memories"]) == 60, (
        f"Expected 60 memories, got {len(result['memories'])}. "
        "Likely hit a limit=10/50 cap from store.py search functions."
    )
    assert returned_ids == seeded_ids, "Returned IDs do not match seeded IDs"
    assert result["complete"] is True


# ---------------------------------------------------------------------------
# (2) Excludes status != 'active'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_excludes_archived_memories(pool):
    """Active memories are returned; archived memories are excluded."""
    user_id = _unique_user_id()
    tag = f"tg-tag-{uuid.uuid4().hex[:8]}"

    active_id = await _seed_memory(pool, user_id, tag, "active memory")
    await _seed_memory(pool, user_id, tag, "archived memory", status="archived")

    result = await gather_topic_memories(pool, [tag], user_id)

    returned_ids = {m.id for m in result["memories"]}
    assert active_id in returned_ids, "Active memory should be in results"
    assert len(result["memories"]) == 1, (
        f"Expected 1 active memory, got {len(result['memories'])}. "
        "Archived memory should be excluded."
    )
    # All returned memories must have status active
    for mem in result["memories"]:
        assert mem.status.value == "active", (
            f"Non-active memory {mem.id} with status {mem.status} was returned"
        )


# ---------------------------------------------------------------------------
# (3) Ordered by created_at ascending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_ordered_by_created_at_asc(pool):
    """Memories are returned ordered by created_at ascending."""
    user_id = _unique_user_id()
    tag = f"tg-tag-{uuid.uuid4().hex[:8]}"

    base_time = datetime.now(timezone.utc) - timedelta(hours=5)

    # Seed in reverse order (newest first) to confirm ordering is not by insertion
    ordered_ids = []
    for i in range(5):
        # Seed with explicit created_at going forward in time
        mid = await _seed_memory(
            pool,
            user_id,
            tag,
            f"ordered memory {i}",
            created_at=base_time + timedelta(hours=i),
        )
        ordered_ids.append(mid)

    result = await gather_topic_memories(pool, [tag], user_id)

    assert len(result["memories"]) == 5, f"Expected 5 memories, got {len(result['memories'])}"

    returned_ids = [m.id for m in result["memories"]]
    assert returned_ids == ordered_ids, (
        f"Expected created_at ASC order {ordered_ids}, got {returned_ids}"
    )

    # Also verify timestamps are monotonically non-decreasing
    timestamps = [m.created_at for m in result["memories"]]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1], (
            f"created_at not ascending at position {i}: {timestamps[i]} > {timestamps[i+1]}"
        )


# ---------------------------------------------------------------------------
# (4) RLS isolation — another user's memory is not returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_rls_isolation_other_user_excluded(pool):
    """Another user's tagged memory must NOT appear in gather results.

    Uses two distinct uuid-derived user_ids to guarantee no cross-contamination
    from other tests.
    """
    user_a = _unique_user_id()
    user_b = _unique_user_id()
    shared_tag = f"tg-shared-{uuid.uuid4().hex[:8]}"

    # Both users have memories under the same tag
    a_id = await _seed_memory(pool, user_a, shared_tag, "user A memory")
    b_id = await _seed_memory(pool, user_b, shared_tag, "user B memory")

    # Gather as user A
    result_a = await gather_topic_memories(pool, [shared_tag], user_a)
    returned_ids_a = {m.id for m in result_a["memories"]}

    assert a_id in returned_ids_a, "User A's own memory should be returned"
    assert b_id not in returned_ids_a, (
        f"User B's memory {b_id} must NOT appear in User A's gather (RLS isolation failure)"
    )

    # Gather as user B
    result_b = await gather_topic_memories(pool, [shared_tag], user_b)
    returned_ids_b = {m.id for m in result_b["memories"]}

    assert b_id in returned_ids_b, "User B's own memory should be returned"
    assert a_id not in returned_ids_b, (
        f"User A's memory {a_id} must NOT appear in User B's gather (RLS isolation failure)"
    )


# ---------------------------------------------------------------------------
# (5) Entity secondary merge sets truncated=True at 100-cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_entity_secondary_truncated_when_at_limit(pool):
    """Entity-graph secondary merge sets truncated=True when entity memories >= 100.

    Strategy:
    - Create one primary memory with tag X for user U (so the gather has 1 hit)
    - Create an entity
    - Link 100 distinct memories (without tag X) to the entity via entity_mentions
    - Also link the primary memory to the entity
    - gather_topic_memories(tags=[X], user_id=U) should:
        a) find the primary memory via tag query
        b) find the entity via entity_mentions on the primary memory
        c) call get_entity_memories(entity_id) → returns 100 rows (the cap)
        d) set truncated=True because len(entity_mems) >= 100
    """
    user_id = _unique_user_id()
    tag = f"tg-tag-{uuid.uuid4().hex[:8]}"

    # Seed the primary memory with the target tag (user context set in _seed_memory)
    primary_id = await _seed_memory(pool, user_id, tag, "primary tagged memory")

    # Create an entity. Pass user_id explicitly so EntityCreate.user_id wins
    # over the session contextvar (store_entity uses COALESCE($8, current_setting)).
    entity = await store_entity(
        pool,
        EntityCreate(
            name=f"test-entity-{uuid.uuid4().hex[:8]}",
            entity_type=EntityType.concept,
            user_id=user_id,
        ),
    )

    # Link primary memory to entity (superuser pool, no RLS restriction)
    await link_mention(pool, entity.id, primary_id)

    # Seed 100 more memories WITHOUT the target tag, link them to the entity.
    # These are entity-linked but not tag-matched. This brings entity total to 101
    # (primary + 100 others), ensuring get_entity_memories hits the 100 cap.
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            for i in range(100):
                mem = await store_memory(
                    pool,
                    MemoryCreate(
                        type=MemoryType.fact,
                        content=f"entity-linked memory {i} (no tag)",
                        topic=["unrelated-topic"],
                        source=MemorySource.conversation,
                        confidence=0.7,
                    ),
                )
                await link_mention(pool, entity.id, mem.id)
    finally:
        current_user_id.reset(tok)

    # Now gather — entity secondary should hit the 100 cap
    result = await gather_topic_memories(pool, [tag], user_id)

    assert result["truncated"] is True, (
        "truncated must be True when entity-linked set hits the LIMIT 100 cap. "
        f"Got truncated={result['truncated']}, memories count={len(result['memories'])}"
    )
    # Primary memory must still be present (it was tag-matched)
    returned_ids = {m.id for m in result["memories"]}
    assert primary_id in returned_ids, "Primary tag-matched memory must be in results"


# ---------------------------------------------------------------------------
# (6) Entity-secondary user isolation — user B's memory not returned via entity path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_secondary_does_not_leak_other_user_memory(pool):
    """Entity-secondary merge must NOT return another user's memory.

    Strategy:
    - User A: seed a memory with tag T (enters primary set), create an entity,
      link user A's memory to that entity.
    - User B: seed a memory WITHOUT tag T (so it cannot enter via the primary
      path), link it to the SAME entity.
    - gather_topic_memories(tags=[T], user_id=user_A) must:
        a) return user A's tagged memory (primary path)
        b) find the entity via entity_mentions on user A's memory (secondary path)
        c) NOT include user B's entity-linked memory (user-visibility filter)

    This test catches the superuser-pool blind spot: testcontainers bypasses
    RLS entirely, so without the application-level visibility predicate on the
    secondary query, user B's memory would be returned.
    """
    user_a = _unique_user_id()
    user_b = _unique_user_id()
    tag = f"tg-tag-{uuid.uuid4().hex[:8]}"

    # User A: seed a primary memory with tag T
    a_primary_id = await _seed_memory(pool, user_a, tag, "user A primary tagged memory")

    # Create an entity owned by user A
    tok_a = current_user_id.set(user_a)
    try:
        async with acquire(pool):
            entity = await store_entity(
                pool,
                EntityCreate(
                    name=f"shared-entity-{uuid.uuid4().hex[:8]}",
                    entity_type=EntityType.concept,
                    user_id=user_a,
                ),
            )
            # Link user A's memory to the entity
            await link_mention(pool, entity.id, a_primary_id)
    finally:
        current_user_id.reset(tok_a)

    # User B: seed a memory WITHOUT tag T (so it ONLY enters via entity-secondary)
    b_id = await _seed_memory(pool, user_b, "unrelated-tag", "user B entity-linked memory")

    # Link user B's memory to the SAME entity (as user B)
    tok_b = current_user_id.set(user_b)
    try:
        async with acquire(pool):
            await link_mention(pool, entity.id, b_id)
    finally:
        current_user_id.reset(tok_b)

    # Gather as user A — entity secondary should find the entity but must NOT
    # include user B's memory due to the user-visibility predicate.
    result = await gather_topic_memories(pool, [tag], user_a)
    returned_ids = {m.id for m in result["memories"]}

    assert a_primary_id in returned_ids, "User A's own tagged memory must be returned"
    assert b_id not in returned_ids, (
        f"User B's memory {b_id} must NOT appear in user A's gather via the "
        "entity-secondary path (entity-secondary user-visibility filter failure)"
    )
