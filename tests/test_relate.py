"""Tests for relationship CRUD through the store layer."""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemoryType, RelationType
from weft.store import (
    add_relationship,
    get_relationships,
    remove_relationship,
    store_memory,
)


async def _make_memory(pool, content: str):
    """Helper: create a minimal test memory."""
    return await store_memory(pool, MemoryCreate(type=MemoryType.fact, content=content))


async def test_add_and_get_related_to(pool):
    """Create two memories, add a related_to relationship, verify via get_relationships."""
    m1 = await _make_memory(pool, "memory one")
    m2 = await _make_memory(pool, "memory two")

    rel = await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.related_to)
    assert rel.source_id == m1.id
    assert rel.target_id == m2.id
    assert rel.relation == RelationType.related_to

    rels = await get_relationships(pool, m1.id)
    assert len(rels) == 1
    assert rels[0].relation == RelationType.related_to


async def test_multiple_relationships(pool):
    """Add related_to and contradicts, verify both are returned."""
    m1 = await _make_memory(pool, "claim A")
    m2 = await _make_memory(pool, "claim B")

    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.related_to)
    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.contradicts)

    rels = await get_relationships(pool, m1.id)
    assert len(rels) == 2
    relation_types = {r.relation for r in rels}
    assert relation_types == {RelationType.related_to, RelationType.contradicts}


async def test_remove_relationship(pool):
    """Remove a relationship and verify it's gone."""
    m1 = await _make_memory(pool, "old version")
    m2 = await _make_memory(pool, "new version")

    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.supersedes)
    assert len(await get_relationships(pool, m1.id)) == 1

    removed = await remove_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.supersedes)
    assert removed is True

    rels = await get_relationships(pool, m1.id)
    assert len(rels) == 0


async def test_remove_nonexistent_relationship(pool):
    """Removing a relationship that doesn't exist returns False."""
    m1 = await _make_memory(pool, "lonely memory")
    removed = await remove_relationship(pool, source_id=m1.id, target_id="weft-nonexist", relation=RelationType.related_to)
    assert removed is False


async def test_get_relationships_with_filter(pool):
    """Filter get_relationships by relation type."""
    m1 = await _make_memory(pool, "base memory")
    m2 = await _make_memory(pool, "related memory")
    m3 = await _make_memory(pool, "contradicting memory")

    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.related_to)
    await add_relationship(pool, source_id=m1.id, target_id=m3.id, relation=RelationType.contradicts)

    # Without filter: both relationships
    all_rels = await get_relationships(pool, m1.id)
    assert len(all_rels) == 2

    # Filter to related_to only
    related = await get_relationships(pool, m1.id, relation=RelationType.related_to)
    assert len(related) == 1
    assert related[0].target_id == m2.id

    # Filter to contradicts only
    contradicts = await get_relationships(pool, m1.id, relation=RelationType.contradicts)
    assert len(contradicts) == 1
    assert contradicts[0].target_id == m3.id


async def test_all_relation_types(pool):
    """Verify every relation type works: supersedes, related_to, contradicts,
    derived_from, merge_candidate."""
    m1 = await _make_memory(pool, "source memory")
    targets = {}
    for rtype in RelationType:
        t = await _make_memory(pool, f"target for {rtype.value}")
        targets[rtype] = t
        await add_relationship(pool, source_id=m1.id, target_id=t.id, relation=rtype)

    rels = await get_relationships(pool, m1.id)
    assert len(rels) == len(RelationType)

    found_types = {r.relation for r in rels}
    assert found_types == set(RelationType)


async def test_get_relationships_as_target(pool):
    """Relationships are returned whether the memory is source or target."""
    m1 = await _make_memory(pool, "source")
    m2 = await _make_memory(pool, "target")

    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.derived_from)

    # Query from the target side
    rels = await get_relationships(pool, m2.id)
    assert len(rels) == 1
    assert rels[0].source_id == m1.id
    assert rels[0].target_id == m2.id


async def test_duplicate_relationship_idempotent(pool):
    """Adding the same relationship twice is idempotent (ON CONFLICT DO NOTHING)."""
    m1 = await _make_memory(pool, "alpha")
    m2 = await _make_memory(pool, "beta")

    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.related_to)
    await add_relationship(pool, source_id=m1.id, target_id=m2.id, relation=RelationType.related_to)

    rels = await get_relationships(pool, m1.id)
    assert len(rels) == 1
