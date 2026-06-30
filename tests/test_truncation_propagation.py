"""Tests for truncated flag propagation through MCP responses.

done_when assertions:
  (1) weft_status response includes truncated field from gather_topic_memories
  (2) weft_entity_context response includes truncated field when entity memory set hits 100-cap
  (3) weft_entity_context response includes truncated=false when under limit
  (4) Truncated flag accurately reflects database-level caps, not token budget truncation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.db.connection import acquire
from weft.entities import link_mention, store_entity
from weft.mcp.server import AppContext
from weft.models import EntityCreate, EntityType, MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING = [0.1] * 768  # matches OpenAI text-embedding-3-small @ 768 dims


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests."""

    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return list(_FAKE_EMBEDDING)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(_FAKE_EMBEDDING) for _ in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    """Build a mock FastMCP Context that carries our AppContext."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


def _unique_user_id() -> str:
    """Generate a unique per-test user_id to ensure test isolation."""
    return f"trunc-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def app(pool):
    """AppContext wired to real DB pool + fake embedding + NullCache."""
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# ---------------------------------------------------------------------------
# Test 1: weft_status surfaces truncated from gather_topic_memories
# ---------------------------------------------------------------------------


class TestWeftStatusTruncation:
    async def test_status_surfaces_truncated_when_entity_secondary_hits_100_cap(self, ctx, pool):
        """weft_status includes truncated=true when entity-secondary hits 100-cap.

        Verifies the truncated flag computed by gather_topic_memories reaches
        the agent through the MCP response.
        """
        from unittest.mock import patch
        from weft.mcp.tools import weft_status

        user_id = _unique_user_id()
        tag = f"status-trunc-{uuid.uuid4().hex[:8]}"

        # Use the same user_id consistently for all operations
        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                # Store a primary memory under the known topic
                primary = await store_memory(
                    pool,
                    MemoryCreate(
                        type=MemoryType.fact,
                        content="Primary status memory for truncation test",
                        topic=[tag],
                        source=MemorySource.conversation,
                        confidence=0.7,
                    ),
                )
                primary_id = primary.id

                # Create an entity and link primary memory to it
                entity = await store_entity(
                    pool,
                    EntityCreate(
                        name=f"trunc-entity-{uuid.uuid4().hex[:8]}",
                        entity_type=EntityType.concept,
                        user_id=user_id,
                    ),
                )
                await link_mention(pool, entity.id, primary_id)

                # Seed 100 memories without the topic tag and link them to the entity
                for i in range(100):
                    mem = await store_memory(
                        pool,
                        MemoryCreate(
                            type=MemoryType.fact,
                            content=f"entity-linked truncation test memory {i} (no tag)",
                            topic=["unrelated-topic"],
                            source=MemorySource.conversation,
                            confidence=0.7,
                        ),
                    )
                    await link_mention(pool, entity.id, mem.id)
        finally:
            current_user_id.reset(tok)

        # Call weft_status and verify truncated surfaces
        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        assert "truncated" in result, "MCP response must include truncated field"
        assert result["truncated"] is True, (
            f"truncated must be True when entity-secondary hit 100-cap. "
            f"Got truncated={result['truncated']}"
        )
        assert "memories" in result
        assert result["memories"], "Should have at least the primary memory"

    async def test_status_returns_truncated_false_when_under_limit(self, ctx):
        """weft_status includes truncated=false when entity set is under 100."""
        from unittest.mock import patch
        from weft.mcp.tools import weft_remember, weft_status

        user_id = _unique_user_id()
        tag = f"status-no-trunc-{uuid.uuid4().hex[:8]}"

        # Store just one memory
        result = await weft_remember(
            ctx,
            content="Single memory for status no-truncation test",
            type="fact",
            topic=[tag],
            source="conversation",
        )

        # Call weft_status and verify truncated=false
        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        assert "truncated" in result, "MCP response must include truncated field"
        assert result["truncated"] is False, (
            "truncated must be False when entity-secondary does not hit cap"
        )


# ---------------------------------------------------------------------------
# Test 2: weft_entity_context surfaces truncated flag
# ---------------------------------------------------------------------------


class TestWeftEntityContextTruncation:
    async def test_entity_context_surfaces_truncated_when_at_100_cap(self, ctx, pool):
        """weft_entity_context includes truncated=true when hitting 100-memory cap.

        Verifies the truncated flag computed by get_entity_memories reaches
        the agent through the MCP response.
        """
        user_id = _unique_user_id()
        from weft.mcp.tools import weft_entity_context

        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                # Create entity
                entity = await store_entity(
                    pool,
                    EntityCreate(
                        name=f"entity-context-trunc-{uuid.uuid4().hex[:8]}",
                        entity_type=EntityType.concept,
                        user_id=user_id,
                    ),
                )

                # Link exactly 100 memories to the entity
                for i in range(100):
                    mem = await store_memory(
                        pool,
                        MemoryCreate(
                            type=MemoryType.fact,
                            content=f"entity context truncation test memory {i}",
                            topic=["test-topic"],
                            source=MemorySource.conversation,
                            confidence=0.7,
                        ),
                    )
                    await link_mention(pool, entity.id, mem.id)

                # Call entity_context and verify truncated surfaces
                result = await weft_entity_context(ctx, entity.id)

                assert "truncated" in result, (
                    "MCP response from weft_entity_context must include truncated field"
                )
                assert result["truncated"] is True, (
                    f"truncated must be True when entity memories >= 100. "
                    f"Got truncated={result['truncated']}"
                )
                assert "memories" in result
                assert result["memory_count"] > 0
        finally:
            current_user_id.reset(tok)

    async def test_entity_context_surfaces_truncated_false_under_limit(self, ctx, pool):
        """weft_entity_context includes truncated=false when under 100 limit."""
        user_id = _unique_user_id()
        from weft.mcp.tools import weft_entity_context

        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                # Create entity
                entity = await store_entity(
                    pool,
                    EntityCreate(
                        name=f"entity-context-no-trunc-{uuid.uuid4().hex[:8]}",
                        entity_type=EntityType.concept,
                        user_id=user_id,
                    ),
                )

                # Link only 5 memories to the entity (well under 100)
                for i in range(5):
                    mem = await store_memory(
                        pool,
                        MemoryCreate(
                            type=MemoryType.fact,
                            content=f"entity context no truncation test memory {i}",
                            topic=["test-topic"],
                            source=MemorySource.conversation,
                            confidence=0.7,
                        ),
                    )
                    await link_mention(pool, entity.id, mem.id)

                # Call entity_context and verify truncated=false
                result = await weft_entity_context(ctx, entity.id)

                assert "truncated" in result, (
                    "MCP response from weft_entity_context must include truncated field"
                )
                assert result["truncated"] is False, (
                    "truncated must be False when entity memories < 100"
                )
                assert result["memory_count"] == 5
        finally:
            current_user_id.reset(tok)

    async def test_entity_context_truncated_vs_token_truncated_are_distinct(self, ctx, pool):
        """Verify truncated (database cap) and memories_truncated (token budget) are distinct.

        memories_truncated = memories dropped due to token budget
        truncated = memories dropped due to database-level 100-row cap
        Both can be True simultaneously.
        """
        user_id = _unique_user_id()
        from weft.mcp.tools import weft_entity_context

        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                # Create entity
                entity = await store_entity(
                    pool,
                    EntityCreate(
                        name=f"entity-context-both-{uuid.uuid4().hex[:8]}",
                        entity_type=EntityType.concept,
                        user_id=user_id,
                    ),
                )

                # Link 100 memories with large content (will hit both caps)
                for i in range(100):
                    mem = await store_memory(
                        pool,
                        MemoryCreate(
                            type=MemoryType.fact,
                            content=f"Large memory content for test {i}: " + "x" * 500,
                            topic=["test-topic"],
                            source=MemorySource.conversation,
                            confidence=0.7,
                        ),
                    )
                    await link_mention(pool, entity.id, mem.id)

                # Call entity_context with tight token budget
                result = await weft_entity_context(ctx, entity.id, budget_tokens=1000)

                # Both flags should be present
                assert "truncated" in result
                assert "memories_truncated" in result

                # truncated (db cap) should be True
                assert result["truncated"] is True, (
                    "truncated should be True (100-row database cap)"
                )

                # memories_truncated (token budget) should be > 0
                assert result["memories_truncated"] > 0, (
                    "memories_truncated should be > 0 (token budget filtering)"
                )

                # But memory_count should be < memory count from db (due to token budget)
                assert result["memory_count"] < 100, (
                    "memory_count should be < 100 (filtered by token budget)"
                )
        finally:
            current_user_id.reset(tok)
