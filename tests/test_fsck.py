"""Tests for weft_fsck — orphan memory detection in vector index only.

Tests the full lifecycle: seed orphan (no tags/entities/episodes) →
verify orphan detection → add tag edge → verify orphan is removed.
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
)
from weft.store import store_memory, update_memory


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
