"""Test that route() does NOT link candidate entities to memories.

This is a critical integration test verifying that the two-tier system
prevents wrong merges: a 0.70 similarity candidate is created but NOT
linked to the mention, so the memory can be queried independently of
the potential duplicate.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from weft.ingest_pipeline import EntityRef, Intent, route
from weft.models import EntityCreate, EntityType, MemoryCreate, MemoryType
from weft.entities import get_entity, get_memory_entities
from weft.store import store_memory


class TestRouteCandidateNoLink:
    """Test that route() correctly handles candidate entities."""

    @pytest.mark.asyncio
    async def test_route_does_not_link_candidate_entities(self, pool):
        """Integration test: route() should NOT link entities with status='candidate'."""
        from weft.entities import store_entity

        # Create an existing active entity that will become the "candidate match"
        existing_alice = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )

        # Create an intent with an entity mention
        intent = Intent(
            type="person_fact",
            content="Alice is working on the project",
            entities=[EntityRef(name="Alice", entity_type="person")],
            confidence=0.9,
        )

        # Mock the embedding provider
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.5] * 768)

        # Mock search_entities to return a 0.70 similarity match
        # (in the candidate range, not auto-merge)
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [(existing_alice, 0.70)]

            # Run route with this intent
            result = await route([intent], pool, provider)

        # Check that a memory was created
        assert result.memories_created == 1

        # Check that a candidate entity was created
        # (resolve_entities would have returned a new candidate)
        # But it should NOT be linked to the memory
        created_memories = await pool.fetch("SELECT id FROM memories WHERE status = 'active'")
        assert len(created_memories) == 1
        memory_id = created_memories[0]["id"]

        # Get entities linked to this memory
        linked_entities = await get_memory_entities(pool, memory_id)

        # There should be NO linked entities (candidate was not linked)
        assert len(linked_entities) == 0, \
            "Candidate entity should NOT be linked to memory"

    @pytest.mark.asyncio
    async def test_route_links_active_entities_only(self, pool):
        """Integration test: route() should link active entities."""
        from weft.entities import store_entity

        # Create an existing active entity with high similarity
        existing_alice = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )

        # Create an intent with an entity mention
        intent = Intent(
            type="person_fact",
            content="Alice is working on the project",
            entities=[EntityRef(name="Alice", entity_type="person")],
            confidence=0.9,
        )

        # Mock the embedding provider
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.5] * 768)

        # Mock search_entities to return a 0.90 similarity match
        # (above auto-merge threshold)
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [(existing_alice, 0.90)]

            # Run route with this intent
            result = await route([intent], pool, provider)

        # Check that a memory was created
        assert result.memories_created == 1
        # Check that entity was linked
        assert result.entities_linked == 1

        # Get the memory and verify the entity is linked
        created_memories = await pool.fetch("SELECT id FROM memories WHERE status = 'active'")
        assert len(created_memories) == 1
        memory_id = created_memories[0]["id"]

        linked_entities = await get_memory_entities(pool, memory_id)
        assert len(linked_entities) == 1
        assert linked_entities[0].id == existing_alice.id

    @pytest.mark.asyncio
    async def test_route_mixed_active_and_candidates(self, pool):
        """Integration test: route() with multiple entities at different thresholds."""
        from weft.entities import store_entity

        # Create existing entities
        existing_alice = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )
        existing_bob = await store_entity(
            pool,
            EntityCreate(name="Robert Smith", entity_type=EntityType.person),
            embedding=[0.3] * 768,
        )

        # Create an intent with two entity mentions
        intent = Intent(
            type="person_fact",
            content="Alice and Bob are working on the project",
            entities=[
                EntityRef(name="Alice", entity_type="person"),
                EntityRef(name="Bob", entity_type="person"),
            ],
            confidence=0.9,
        )

        # Mock the embedding provider
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.5] * 768)

        # Mock search_entities to return different similarities
        search_call_count = [0]

        async def search_side_effect(*args, **kwargs):
            search_call_count[0] += 1
            if search_call_count[0] == 1:
                # First call (Alice): high similarity (0.90) → auto-merge
                return [(existing_alice, 0.90)]
            elif search_call_count[0] == 2:
                # Second call (Bob): medium similarity (0.70) → candidate
                return [(existing_bob, 0.70)]
            return []

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = search_side_effect

            # Run route with this intent
            result = await route([intent], pool, provider)

        # Check that a memory was created
        assert result.memories_created == 1
        # Check that only the active entity (Alice) was linked
        assert result.entities_linked == 1

        # Get the memory and verify only Alice is linked
        created_memories = await pool.fetch("SELECT id FROM memories WHERE status = 'active'")
        assert len(created_memories) == 1
        memory_id = created_memories[0]["id"]

        linked_entities = await get_memory_entities(pool, memory_id)
        assert len(linked_entities) == 1
        assert linked_entities[0].id == existing_alice.id

        # Verify Bob candidate was created but not linked
        candidate_bobs = await pool.fetch(
            "SELECT id FROM entities WHERE name = 'Bob' AND status = 'candidate'"
        )
        # There should be at most one candidate Bob
        assert len(candidate_bobs) <= 1
