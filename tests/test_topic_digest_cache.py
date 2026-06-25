"""Tests for weft/topic_digest_cache.py — done_when assertions:

  (1) write_digest then read_digest returns the fresh row.
  (2) A digest with stale=true is NOT served as fresh (read returns None).
  (3) Writing an active memory tagged with topic T flips T's digest stale=true
      within the same transaction boundary (V3 write-invalidation hook).
  (4) An unrelated topic's digest is untouched by that write.

Each test uses a unique per-test user_id for self-isolation.
Pool fixture comes from tests/conftest.py (real Postgres testcontainer).
"""

from __future__ import annotations

import uuid

import pytest

from weft.auth import current_user_id
from weft.db.connection import acquire
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory
from weft.topic_digest_cache import mark_stale_for_tags, read_digest, write_digest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    """Unique per-test user_id prefix to guarantee self-isolation."""
    return f"tdc-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# (1) write_digest then read_digest returns the fresh row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_then_read_returns_fresh_row(pool):
    """write_digest followed by read_digest returns the written row (stale=False)."""
    user_id = _uid()
    topic = f"topic-{uuid.uuid4().hex[:8]}"

    # Set user context so RLS INSERT policy passes
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            digest_id = await write_digest(
                pool,
                user_id=user_id,
                topic=topic,
                content="This is a synthesized digest about the topic.",
                detector_version="v1-test",
                scope="global",
                provenance={"mem-abc": ["span1"]},
            )
    finally:
        current_user_id.reset(tok)

    assert digest_id.startswith("td-"), f"digest_id should be prefixed td-, got: {digest_id}"

    # read_digest — user context needed for RLS SELECT
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            row = await read_digest(pool, user_id=user_id, topic=topic, scope="global")
    finally:
        current_user_id.reset(tok)

    assert row is not None, "read_digest should return a row after write_digest"
    assert row["digest_id"] == digest_id
    assert row["topic"] == topic
    assert row["content"] == "This is a synthesized digest about the topic."
    assert row["stale"] is False
    assert row["detector_version"] == "v1-test"
    assert row["provenance"] is not None


# ---------------------------------------------------------------------------
# (2) A digest with stale=true is NOT served as fresh (read returns None)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_digest_returns_none(pool):
    """read_digest returns None when the digest row has stale=True."""
    user_id = _uid()
    topic = f"topic-{uuid.uuid4().hex[:8]}"

    # Write a fresh digest first
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            digest_id = await write_digest(
                pool,
                user_id=user_id,
                topic=topic,
                content="Fresh digest content.",
                detector_version="v1-test",
            )
    finally:
        current_user_id.reset(tok)

    # Manually flip stale=True on the row (simulating an invalidation)
    # Use the superuser pool (bypasses RLS) to patch the row directly
    await pool.execute(
        "UPDATE topic_digests SET stale = true WHERE digest_id = $1",
        digest_id,
    )

    # read_digest must now return None (stale row is not fresh)
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            row = await read_digest(pool, user_id=user_id, topic=topic, scope="global")
    finally:
        current_user_id.reset(tok)

    assert row is None, (
        f"read_digest must return None for a stale digest, got: {row}"
    )


# ---------------------------------------------------------------------------
# (3) Writing a memory tagged with topic T flips T's digest to stale=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_memory_invalidates_matching_digest(pool):
    """store_memory with topic T flips the digest for T to stale=True (V3 hook)."""
    user_id = _uid()
    topic = f"topic-{uuid.uuid4().hex[:8]}"

    # 1. Write a fresh digest for the topic
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            digest_id = await write_digest(
                pool,
                user_id=user_id,
                topic=topic,
                content="Digest before memory write.",
                detector_version="v1-test",
            )
    finally:
        current_user_id.reset(tok)

    # Confirm digest is fresh
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            row_before = await read_digest(pool, user_id=user_id, topic=topic)
    finally:
        current_user_id.reset(tok)
    assert row_before is not None, "Digest should be fresh before memory write"

    # 2. Write a memory tagged with the same topic — V3 hook must fire
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content="New memory that invalidates the digest.",
                    topic=[topic],
                    source=MemorySource.conversation,
                    confidence=0.8,
                ),
            )
    finally:
        current_user_id.reset(tok)

    # 3. read_digest must now return None (stale after write-invalidation)
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            row_after = await read_digest(pool, user_id=user_id, topic=topic)
    finally:
        current_user_id.reset(tok)

    assert row_after is None, (
        "read_digest must return None after store_memory tagged with the same topic "
        f"(V3 write-invalidation hook did not flip stale=True). Got: {row_after}"
    )

    # Confirm stale=True in the DB directly (superuser pool bypasses RLS)
    stale_val = await pool.fetchval(
        "SELECT stale FROM topic_digests WHERE digest_id = $1",
        digest_id,
    )
    assert stale_val is True, (
        f"topic_digests row for {digest_id} must have stale=True after memory write, "
        f"got stale={stale_val}"
    )


# ---------------------------------------------------------------------------
# (4) An unrelated topic's digest is untouched by the write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unrelated_topic_digest_untouched(pool):
    """Writing a memory with topic A does NOT flip stale on digest for topic B."""
    user_id = _uid()
    topic_a = f"topic-a-{uuid.uuid4().hex[:8]}"
    topic_b = f"topic-b-{uuid.uuid4().hex[:8]}"

    # Write digests for both topics
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            digest_id_a = await write_digest(
                pool,
                user_id=user_id,
                topic=topic_a,
                content="Digest for topic A.",
                detector_version="v1-test",
            )
            digest_id_b = await write_digest(
                pool,
                user_id=user_id,
                topic=topic_b,
                content="Digest for topic B.",
                detector_version="v1-test",
            )
    finally:
        current_user_id.reset(tok)

    # Write a memory tagged ONLY with topic A
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content="Memory tagged only with topic A.",
                    topic=[topic_a],
                    source=MemorySource.conversation,
                    confidence=0.8,
                ),
            )
    finally:
        current_user_id.reset(tok)

    # Topic A's digest should be stale
    stale_a = await pool.fetchval(
        "SELECT stale FROM topic_digests WHERE digest_id = $1",
        digest_id_a,
    )
    assert stale_a is True, (
        f"Digest for topic A ({digest_id_a}) must be stale=True after memory write, "
        f"got stale={stale_a}"
    )

    # Topic B's digest must be UNTOUCHED (stale=False still)
    stale_b = await pool.fetchval(
        "SELECT stale FROM topic_digests WHERE digest_id = $1",
        digest_id_b,
    )
    assert stale_b is False, (
        f"Digest for topic B ({digest_id_b}) must remain stale=False (unrelated topic), "
        f"got stale={stale_b}"
    )

    # read_digest for B must still return the fresh row
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            row_b = await read_digest(pool, user_id=user_id, topic=topic_b)
    finally:
        current_user_id.reset(tok)

    assert row_b is not None, (
        "read_digest for topic B must still return a fresh row "
        "(unrelated topic must not be affected by topic A's memory write)"
    )
    assert row_b["digest_id"] == digest_id_b
