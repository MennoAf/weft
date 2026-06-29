"""Tests for two-tier resolve_entities (auto-merge / candidate / new).

Tests verify that resolve_entities implements the correct similarity thresholds:
- cosine ≥0.85: auto-merge to existing entity
- 0.6 ≤ cosine <0.85: create candidate entity (NOT auto-linked)
- cosine <0.6: create new active entity
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from weft.ingest_pipeline import EntityRef, resolve_entities
from weft.models import EntityCreate, EntityType, _weft_id


def _mock_entity(entity_id: str, name: str, entity_type: str = "concept", status: str = "active"):
    """Create a mock Entity object."""
    from weft.models import Entity
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return Entity(
        id=entity_id,
        name=name,
        entity_type=EntityType(entity_type),
        aliases=[],
        description=None,
        project_id=None,
        agent_id=None,
        status=status,
        mention_count=0,
        created_at=now,
        updated_at=now,
    )


def _mock_embedding_provider():
    """Create a mock embedding provider."""
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[0.5] * 768)
    return provider


class TestResolveEntitiesTwoTier:
    """Test the two-tier entity resolution system."""

    @pytest.mark.asyncio
    async def test_high_similarity_auto_merges(self):
        """Cosine ≥0.85 should auto-merge to existing entity."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Return a match with 0.90 similarity
            mock_search.return_value = [(existing, 0.90)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should return the existing entity ID (auto-merged)
        assert result == {"Alice": "weft-existing"}
        # Should NOT create a new entity
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_medium_similarity_creates_candidate(self):
        """Cosine 0.6-0.85 should create a candidate entity (NOT auto-linked).

        A 0.70 similarity match is a near-duplicate that needs human review.
        We create a candidate entity but don't link it to the memory.
        """
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Return a match with 0.70 similarity (in candidate range)
            mock_search.return_value = [(existing, 0.70)]

            # Mock the candidate creation
            candidate = _mock_entity("weft-candidate", "Alice", status="candidate")
            mock_store.return_value = candidate

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should return the candidate entity ID
        assert result == {"Alice": "weft-candidate"}
        # Should have called store_entity to create candidate
        mock_store.assert_called_once()
        # Check that store_entity was called with status='candidate'
        call_args = mock_store.call_args
        assert call_args.kwargs["status"] == "candidate"

    @pytest.mark.asyncio
    async def test_low_similarity_creates_new_entity(self):
        """Cosine <0.6 should create a new active entity.

        A 0.55 similarity match is too weak to be a candidate.
        We create a new entity and it will be auto-linked to the memory.
        """
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Return a match with 0.55 similarity (below candidate threshold)
            mock_search.return_value = [(existing, 0.55)]

            # Mock the new entity creation
            new_entity = _mock_entity("weft-new", "Alice", status="active")
            mock_store.return_value = new_entity

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should return the new entity ID
        assert result == {"Alice": "weft-new"}
        # Should have called store_entity to create new entity
        mock_store.assert_called_once()
        # Check that store_entity was called with status='active'
        call_args = mock_store.call_args
        assert call_args.kwargs["status"] == "active"

    @pytest.mark.asyncio
    async def test_no_match_creates_new_entity(self):
        """No match should create a new active entity."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Return no matches
            mock_search.return_value = []

            # Mock the new entity creation
            new_entity = _mock_entity("weft-new", "Charlie", status="active")
            mock_store.return_value = new_entity

            result = await resolve_entities(
                [EntityRef(name="Charlie", entity_type="person")],
                pool,
                provider,
            )

        # Should return the new entity ID
        assert result == {"Charlie": "weft-new"}
        # Should have called store_entity once
        mock_store.assert_called_once()
        # Check that store_entity was called with status='active'
        call_args = mock_store.call_args
        assert call_args.kwargs["status"] == "active"

    @pytest.mark.asyncio
    async def test_boundary_0_85_auto_merges(self):
        """Boundary at 0.85 should auto-merge."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Exactly 0.85
            mock_search.return_value = [(existing, 0.85)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        assert result == {"Alice": "weft-existing"}
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_boundary_0_6_creates_candidate(self):
        """Boundary at 0.6 should create a candidate (not auto-merge)."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Exactly 0.6
            mock_search.return_value = [(existing, 0.6)]

            candidate = _mock_entity("weft-candidate", "Alice", status="candidate")
            mock_store.return_value = candidate

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        assert result == {"Alice": "weft-candidate"}
        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args.kwargs["status"] == "candidate"

    @pytest.mark.asyncio
    async def test_boundary_just_below_0_6(self):
        """Just below 0.6 should create a new active entity."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()
        existing = _mock_entity("weft-existing", "Alice", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Slightly below 0.6
            mock_search.return_value = [(existing, 0.59)]

            new_entity = _mock_entity("weft-new", "Alice", status="active")
            mock_store.return_value = new_entity

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        assert result == {"Alice": "weft-new"}
        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args.kwargs["status"] == "active"

    @pytest.mark.asyncio
    async def test_multiple_entities_mixed_thresholds(self):
        """Test multiple entities with different similarity scores."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()

        existing_alice = _mock_entity("weft-alice-existing", "Alice", status="active")
        existing_bob = _mock_entity("weft-bob-existing", "Bob", status="active")
        existing_charlie = _mock_entity("weft-charlie-existing", "Charlie", status="active")

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock) as mock_store:
            # Set up return values for each search
            def search_side_effect(*args, **kwargs):
                query_embedding = args[1]
                # Simple mock: return different entities based on call order
                if mock_search.call_count == 1:  # Alice - high similarity (0.90)
                    return [(existing_alice, 0.90)]
                elif mock_search.call_count == 2:  # Bob - medium similarity (0.70)
                    return [(existing_bob, 0.70)]
                elif mock_search.call_count == 3:  # Charlie - low similarity (0.55)
                    return [(existing_charlie, 0.55)]
                return []

            mock_search.side_effect = search_side_effect

            # Set up store_entity side effects
            candidate_bob = _mock_entity("weft-bob-candidate", "Bob", status="candidate")
            new_charlie = _mock_entity("weft-charlie-new", "Charlie", status="active")

            def store_side_effect(*args, **kwargs):
                if mock_store.call_count == 1:  # Bob candidate
                    return candidate_bob
                elif mock_store.call_count == 2:  # Charlie new
                    return new_charlie
                return None

            mock_store.side_effect = store_side_effect

            result = await resolve_entities(
                [
                    EntityRef(name="Alice", entity_type="person"),
                    EntityRef(name="Bob", entity_type="person"),
                    EntityRef(name="Charlie", entity_type="person"),
                ],
                pool,
                provider,
            )

        # Alice: high similarity (0.90) → auto-merge to existing
        assert result["Alice"] == "weft-alice-existing"
        # Bob: medium similarity (0.70) → candidate entity
        assert result["Bob"] == "weft-bob-candidate"
        # Charlie: low similarity (0.55) → new active entity
        assert result["Charlie"] == "weft-charlie-new"

    @pytest.mark.asyncio
    async def test_search_threshold_is_0_4(self):
        """Verify that search_entities is called with threshold=0.4 to catch candidates."""
        pool = AsyncMock()
        provider = _mock_embedding_provider()

        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search, \
             patch("weft.entities.store_entity", new_callable=AsyncMock):
            mock_search.return_value = []

            await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Verify search_entities was called with threshold=0.4
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["threshold"] == 0.4


class TestResolveEntitiesIntegration:
    """Integration tests with real DB (using testcontainers)."""

    @pytest.mark.asyncio
    async def test_high_similarity_match_uses_existing(self, pool):
        """Integration: high similarity should use existing entity (0.90 ≥ 0.85)."""
        from weft.entities import store_entity

        # Create an existing active entity
        existing = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )

        provider = _mock_embedding_provider()
        # Mock to return high similarity match
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [(existing, 0.90)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should use existing entity
        assert result["Alice"] == existing.id

    @pytest.mark.asyncio
    async def test_candidate_range_creates_candidate_entity(self, pool):
        """Integration: 0.70 similarity creates candidate entity (0.6 ≤ 0.70 < 0.85)."""
        from weft.entities import get_entity, store_entity

        # Create an existing active entity
        existing = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )

        provider = _mock_embedding_provider()
        # Mock to return candidate-range similarity
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [(existing, 0.70)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should have created a candidate entity
        assert "Alice" in result
        candidate_id = result["Alice"]
        assert candidate_id != existing.id  # Should be a different entity

        # Verify the created entity has status='candidate'
        candidate = await get_entity(pool, candidate_id)
        assert candidate is not None
        assert candidate.status == "candidate"

    @pytest.mark.asyncio
    async def test_low_similarity_creates_active_entity(self, pool):
        """Integration: 0.55 similarity creates new active entity (<0.6)."""
        from weft.entities import get_entity, store_entity

        # Create an existing active entity
        existing = await store_entity(
            pool,
            EntityCreate(name="Alice Johnson", entity_type=EntityType.person),
            embedding=[0.5] * 768,
        )

        provider = _mock_embedding_provider()
        # Mock to return low similarity
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [(existing, 0.55)]

            result = await resolve_entities(
                [EntityRef(name="Alice", entity_type="person")],
                pool,
                provider,
            )

        # Should have created a new active entity
        assert "Alice" in result
        new_id = result["Alice"]
        assert new_id != existing.id

        # Verify the created entity has status='active'
        new_entity = await get_entity(pool, new_id)
        assert new_entity is not None
        assert new_entity.status == "active"

    @pytest.mark.asyncio
    async def test_no_match_creates_active_entity(self, pool):
        """Integration: no match creates new active entity."""
        from weft.entities import get_entity

        provider = _mock_embedding_provider()
        # Mock to return no matches
        with patch("weft.entities.search_entities", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []

            result = await resolve_entities(
                [EntityRef(name="Brand New Entity", entity_type="concept")],
                pool,
                provider,
            )

        # Should have created a new entity
        assert "Brand New Entity" in result
        new_id = result["Brand New Entity"]

        # Verify it's active
        entity = await get_entity(pool, new_id)
        assert entity is not None
        assert entity.status == "active"
