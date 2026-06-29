"""Tests for weft_fsck — orphan memory detection (provenance-first spec).

Tests the full lifecycle: seed orphan (no tags/entities/episodes) →
verify orphan detection → add tag edge → verify orphan is removed.

Corrected orphan definition (weft-ede3f70a / weft-ab1e37c0 / weft-f0768396):
  (a) Merge candidate (review_status='pending_review') — NOT a defect.
  (b) Dream-link (memory_relationships row) — NOT a defect.
  Genuine similarity-only (no structural edge, no merge, no link) — IS a defect.

Synthetic persona: "Jim Boblaw" (test data only — never a real user).
"""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.entities import link_mention, store_entity
from weft.episodes import add_memory_to_episode, create_episode
from weft.fsck import list_orphan_memories
from weft.models import (
    EntityCreate,
    EntityType,
    EpisodeCreate,
    MemoryCreate,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from weft.store import add_relationship, store_memory, update_memory


@pytest.fixture
def provider():
    """Get a test embedding provider (fastembed)."""
    return get_provider("fastembed")


async def test_orphan_detection_seed_to_add_tag(pool, provider):
    """Test the full lifecycle: seed orphan → detect → add tag → no longer orphan."""
    # Step 1: Create a memory with no topic tags and get an embedding
    content = "This is a test memory about distributed systems"
    embedding = await provider.embed(content)

    memory_create = MemoryCreate(
        type=MemoryType.fact,
        content=content,
        topic=[],  # Empty topic array — orphan candidate
        confidence=0.9,
    )
    memory = await store_memory(pool, memory_create, embedding=embedding)
    memory_id = memory.id

    # Step 2: Verify the memory is detected as orphan
    orphans = await list_orphan_memories(pool)
    orphan_ids = [o["memory_id"] for o in orphans]
    assert memory_id in orphan_ids, f"Memory {memory_id} should be detected as orphan"
    assert len([o for o in orphans if o["memory_id"] == memory_id]) == 1

    # Verify the reason is correct
    orphan_entry = [o for o in orphans if o["memory_id"] == memory_id][0]
    assert orphan_entry["reason"] == "vector-only reachable"

    # Step 3: Add a topic tag via update
    updated = await update_memory(
        pool,
        memory_id,
        topic=["distributed-systems"],
    )
    assert updated.topic == ["distributed-systems"]

    # Step 4: Verify the memory is NO LONGER detected as orphan
    orphans_after = await list_orphan_memories(pool)
    orphan_ids_after = [o["memory_id"] for o in orphans_after]
    assert memory_id not in orphan_ids_after, (
        f"Memory {memory_id} should NOT be orphan after adding topic tag"
    )


async def test_orphan_not_detected_with_entity_mention(pool, provider):
    """Test that a memory with entity_mentions is not detected as orphan."""
    content = "Memory about Alice who works on ML systems"
    embedding = await provider.embed(content)

    memory_create = MemoryCreate(
        type=MemoryType.fact,
        content=content,
        topic=[],  # Empty topic array
        confidence=0.9,
    )
    memory = await store_memory(pool, memory_create, embedding=embedding)
    memory_id = memory.id

    # Add entity mention
    entity_create = EntityCreate(
        name="Alice",
        entity_type=EntityType.person,
    )
    entity = await store_entity(pool, entity_create)

    # Link the memory to the entity
    await link_mention(pool, entity.id, memory_id)

    # Verify the memory is NOT detected as orphan
    orphans = await list_orphan_memories(pool)
    orphan_ids = [o["memory_id"] for o in orphans]
    assert memory_id not in orphan_ids, (
        f"Memory {memory_id} with entity_mentions should NOT be orphan"
    )


async def test_orphan_not_detected_with_episode_membership(pool, provider):
    """Test that a memory added to an episode is not detected as orphan."""
    content = "Memory about a design decision"
    embedding = await provider.embed(content)

    memory_create = MemoryCreate(
        type=MemoryType.decision,
        content=content,
        topic=[],  # Empty topic array
        confidence=0.85,
    )
    memory = await store_memory(pool, memory_create, embedding=embedding)
    memory_id = memory.id

    # Create an episode and add the memory to it
    episode_create = EpisodeCreate(
        title="Design Review Session",
        status="active",
    )
    episode = await create_episode(pool, episode_create)

    # Add memory to episode
    await add_memory_to_episode(pool, episode.id, memory_id)

    # Verify the memory is NOT detected as orphan
    orphans = await list_orphan_memories(pool)
    orphan_ids = [o["memory_id"] for o in orphans]
    assert memory_id not in orphan_ids, (
        f"Memory {memory_id} in episode should NOT be orphan"
    )


async def test_orphan_empty_topic_array(pool, provider):
    """Test that a memory with empty topic array [] is detected as orphan."""
    content = "Another test memory"
    embedding = await provider.embed(content)

    memory_create = MemoryCreate(
        type=MemoryType.pattern,
        content=content,
        topic=[],  # Empty array — orphan candidate
        confidence=0.75,
    )
    memory = await store_memory(pool, memory_create, embedding=embedding)
    memory_id = memory.id

    # Verify the memory is detected as orphan
    orphans = await list_orphan_memories(pool)
    orphan_ids = [o["memory_id"] for o in orphans]
    assert memory_id in orphan_ids, (
        f"Memory {memory_id} with empty topic array should be detected as orphan"
    )


async def test_inactive_memory_not_orphan(pool, provider):
    """Test that inactive memories are not detected as orphans."""
    content = "Archived memory content"
    embedding = await provider.embed(content)

    memory_create = MemoryCreate(
        type=MemoryType.fact,
        content=content,
        topic=[],
        confidence=0.8,
    )
    memory = await store_memory(pool, memory_create, embedding=embedding)
    memory_id = memory.id

    # Archive the memory (change status from active to archived)
    await update_memory(pool, memory_id, status=MemoryStatus.archived)

    # Verify the memory is NOT detected as orphan (inactive memories excluded)
    orphans = await list_orphan_memories(pool)
    orphan_ids = [o["memory_id"] for o in orphans]
    assert memory_id not in orphan_ids, (
        f"Archived memory {memory_id} should NOT be detected as orphan"
    )


async def test_multiple_orphans_detected(pool, provider):
    """Test that multiple orphans are all detected correctly."""
    orphan_ids = []

    # Create three orphan memories
    for i in range(3):
        content = f"Orphan memory {i}"
        embedding = await provider.embed(content)
        memory_create = MemoryCreate(
            type=MemoryType.fact,
            content=content,
            topic=[],
            confidence=0.8,
        )
        memory = await store_memory(pool, memory_create, embedding=embedding)
        orphan_ids.append(memory.id)

    # Create one memory with topic (not orphan)
    tagged_content = "Memory with topic"
    tagged_embedding = await provider.embed(tagged_content)
    tagged_memory_create = MemoryCreate(
        type=MemoryType.fact,
        content=tagged_content,
        topic=["test-topic"],
        confidence=0.8,
    )
    tagged_memory = await store_memory(pool, tagged_memory_create, embedding=tagged_embedding)
    tagged_memory_id = tagged_memory.id

    # Verify all orphans are detected and tagged memory is not
    orphans = await list_orphan_memories(pool)
    detected_orphan_ids = {o["memory_id"] for o in orphans}

    for orphan_id in orphan_ids:
        assert orphan_id in detected_orphan_ids, (
            f"Orphan memory {orphan_id} should be detected"
        )

    assert tagged_memory_id not in detected_orphan_ids, (
        f"Tagged memory {tagged_memory_id} should NOT be detected as orphan"
    )

    # Verify count matches
    assert len([o for o in orphans if o["memory_id"] in orphan_ids]) == 3, (
        "Should detect exactly 3 orphans"
    )


# ---------------------------------------------------------------------------
# Provenance-first exclusions (corrected spec)
# ---------------------------------------------------------------------------


async def test_merge_candidate_not_orphan(pool, provider):
    """Exclusion (a): a memory pending a duplicate-belief merge is NOT an orphan.

    The L2 dedup path marks a new memory as review_status='pending_review' when
    it is a cross-project merge candidate (0.6 ≤ sim < 0.85).  Fsck must read
    review_status FIRST and skip such memories — they are a planned merge step,
    not a graph defect.

    Synthetic persona: Jim Boblaw.
    """
    content = "Jim Boblaw prefers dark mode in all editors and IDEs"
    embedding = await provider.embed(content)

    memory = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.preference,
            content=content,
            topic=[],  # No structural edges — would be orphan if not pending_review
            confidence=0.7,
        ),
        embedding=embedding,
    )
    memory_id = memory.id

    # Simulate L2 marking this memory as a cross-project merge candidate.
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1",
        memory_id,
    )

    # Provenance-first exclusion (a): review_status='pending_review' → NOT orphan.
    orphans = await list_orphan_memories(pool)
    orphan_ids = {o["memory_id"] for o in orphans}
    assert memory_id not in orphan_ids, (
        f"Merge candidate {memory_id} (pending_review) must NOT be flagged as orphan"
    )


async def test_dream_link_not_orphan(pool, provider):
    """Exclusion (b): a memory with a dream-link (memory_relationships row) is NOT an orphan.

    A 'dream-link' is a blessed similarity-inferred relationship between two
    distinct memories stored in memory_relationships.  Any memory with such a
    link is reachable via the structural graph, not purely by vector similarity.
    Fsck must recognise it and exclude the memory from the orphan list.

    Asymmetric scoping: the memory_relationships check carries no project_id
    filter so cross-project dream-links are correctly excluded for the
    associative half (beliefs).  This test exercises that path with two
    memories that would otherwise appear as genuine orphans.

    Synthetic persona: Jim Boblaw.
    """
    # Memory A: has topic tags — not at risk of being orphan; anchors the link.
    content_a = "Jim Boblaw works on distributed systems reliability at scale"
    embedding_a = await provider.embed(content_a)
    memory_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["distributed-systems", "reliability"],
            confidence=0.9,
        ),
        embedding=embedding_a,
    )

    # Memory B: no structural edges — orphan candidate until dream-link is added.
    content_b = "Jim Boblaw's systems work prioritises fault tolerance over throughput"
    embedding_b = await provider.embed(content_b)
    memory_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            topic=[],  # Empty — orphan without the dream-link
            confidence=0.8,
        ),
        embedding=embedding_b,
    )

    # Confirm B is initially orphan (no dream-link yet — genuine similarity-only).
    orphans_before = await list_orphan_memories(pool)
    orphan_ids_before = {o["memory_id"] for o in orphans_before}
    assert memory_b.id in orphan_ids_before, (
        f"Memory B {memory_b.id} should be orphan before dream-link is added"
    )

    # Add the dream-link: a memory_relationships row from A → B.
    await add_relationship(pool, memory_a.id, memory_b.id, RelationType.related_to)

    # Provenance-first exclusion (b): dream-link present → B is NOT orphan.
    orphans_after = await list_orphan_memories(pool)
    orphan_ids_after = {o["memory_id"] for o in orphans_after}
    assert memory_b.id not in orphan_ids_after, (
        f"Memory B {memory_b.id} with dream-link must NOT be flagged as orphan"
    )
    assert memory_a.id not in orphan_ids_after, (
        f"Memory A {memory_a.id} with topic tags must NOT be flagged as orphan"
    )


async def test_genuine_similarity_only_is_orphan(pool, provider):
    """A memory with no structural edges AND no merge/link explanation IS an orphan.

    This is the positive case: similarity-only reachable memories with no
    pending_review status and no memory_relationships row are genuine defects.

    Synthetic persona: Jim Boblaw.
    """
    content = "Jim Boblaw's general preference for minimalist tooling"
    embedding = await provider.embed(content)

    memory = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.preference,
            content=content,
            topic=[],          # No topic
            confidence=0.75,
        ),
        embedding=embedding,
    )
    memory_id = memory.id

    # No pending_review, no memory_relationships → genuine orphan.
    orphans = await list_orphan_memories(pool)
    orphan_ids = {o["memory_id"] for o in orphans}
    assert memory_id in orphan_ids, (
        f"Genuine similarity-only memory {memory_id} must be flagged as orphan"
    )
