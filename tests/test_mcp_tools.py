"""Tests for MCP tool invocation — happy path, input errors, fallback paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


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
    # list_roots returns empty so _detect_project_id yields None
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


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
# weft_remember
# ---------------------------------------------------------------------------


class TestWeftRemember:
    async def test_happy_path(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx,
            content="Postgres uses MVCC for concurrency control",
            type="fact",
            topic=["postgres"],
            confidence=0.8,
        )

        assert "id" in result
        assert result["content"] == "Postgres uses MVCC for concurrency control"
        assert result["type"] == "fact"
        assert result["confidence"] == 0.8

    async def test_invalid_type(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx, content="A long enough memory body", type="not_a_real_type",
        )
        assert "error" in result
        assert result["error"] == "Invalid input"

    async def test_invalid_source(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx, content="A long enough memory body", source="not_a_source",
        )
        assert "error" in result
        assert result["error"] == "Invalid input"

    async def test_coerce_topic_json_string(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx,
            content="A coercion test memory body for topic JSON",
            topic='["postgres", "testing"]',
        )
        assert "id" in result
        assert result["topic"] == ["postgres", "testing"]

    async def test_check_contradictions(self, ctx):
        from weft.mcp.tools import weft_remember

        # Store two memories with opposing content
        r1 = await weft_remember(ctx, content="We use PostgreSQL 15", topic=["database"])
        assert "id" in r1

        r2 = await weft_remember(
            ctx,
            content="We do not use PostgreSQL 15",
            topic=["database"],
            check_contradictions=True,
        )
        # Should succeed regardless (contradictions are warnings, not errors)
        assert "id" in r2

    async def test_pinned_memory(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx, content="Always check CI before merging", pinned=True,
        )
        assert result.get("pinned") is True

    async def test_review_after_relative(self, ctx):
        from weft.mcp.tools import weft_remember

        result = await weft_remember(
            ctx, content="Revisit caching strategy", review_after="30d",
        )
        assert "id" in result
        assert result.get("review_after") is not None


# ---------------------------------------------------------------------------
# weft_recall
# ---------------------------------------------------------------------------


class TestWeftRecall:
    async def test_happy_path(self, ctx):
        from weft.mcp.tools import weft_recall, weft_remember

        await weft_remember(ctx, content="Redis uses single-threaded event loop")

        result = await weft_recall(ctx, query="Redis architecture")
        assert "results" in result
        assert result["count"] >= 0  # may or may not match depending on cosine sim

    async def test_with_filters(self, ctx):
        from weft.mcp.tools import weft_recall, weft_remember

        await weft_remember(ctx, content="Test pattern memory body", type="pattern", topic=["testing"])

        result = await weft_recall(ctx, query="test", type="pattern")
        assert "results" in result

    async def test_invalid_type_filter(self, ctx):
        from weft.mcp.tools import weft_recall

        result = await weft_recall(ctx, query="test", type="bogus_type")
        assert "error" in result
        assert result["error"] == "Invalid input"

    async def test_empty_results(self, ctx):
        from weft.mcp.tools import weft_recall

        result = await weft_recall(ctx, query="something nobody remembers")
        assert result["count"] == 0
        assert result["results"] == []

    async def test_total_matches_when_more_exist(self, ctx):
        """When more matches exist than limit, total_matches and showing are returned."""
        from weft.mcp.tools import weft_recall, weft_remember

        # Store several memories on the same topic
        for i in range(5):
            await weft_remember(ctx, content=f"Database optimization technique {i}")

        result = await weft_recall(ctx, query="database optimization", limit=2, threshold=0.0)
        assert result["count"] <= 2
        if result["count"] > 0:
            # total_matches should be present when more results exist beyond limit
            assert "total_matches" in result
            assert result["total_matches"] >= result["count"]
            assert "showing" in result

    async def test_no_total_matches_when_all_fit(self, ctx):
        """When all matches fit within limit, total_matches is not returned."""
        from weft.mcp.tools import weft_recall, weft_remember

        await weft_remember(ctx, content="Unique memory about platypus biology")

        result = await weft_recall(ctx, query="platypus", limit=10, threshold=0.0)
        # When count == total, no need for total_matches
        if result["count"] > 0 and "total_matches" not in result:
            pass  # correct — all results fit
        elif "total_matches" in result:
            assert result["total_matches"] > result["count"]

    async def test_retrieval_telemetry_bumped_for_returned_results(self, ctx, app):
        """v49 compounding-loop Step 1: returned memories get retrieval_count++ and last_retrieved_at stamped.

        Uses mode='keyword' so the bump assertion doesn't depend on
        FakeEmbeddingProvider similarity scores — BM25 over the literal
        content gives deterministic matches.

        ``user_id`` is passed explicitly because weft_remember (in this test
        harness) writes under the pool's app.user_id GUC ("test-user-default")
        while weft_recall's default resolves get_user_id() to the real
        installation UUID. The two must match for the row to be visible.
        """
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.mcp.tools import weft_recall, weft_remember

        store_result = await weft_remember(
            ctx, content="Bumper memory about narwhals and arctic ecosystems"
        )
        memory_id = store_result["id"]

        before = await app.pool.fetchrow(
            "SELECT last_retrieved_at, retrieval_count FROM memories WHERE id = $1",
            memory_id,
        )
        assert before["last_retrieved_at"] is None
        assert before["retrieval_count"] == 0

        result = await weft_recall(
            ctx,
            query="narwhals arctic",
            limit=10,
            mode="keyword",
            tier="belief",
            user_id=DEFAULT_TEST_USER_ID,
        )
        assert result["count"] >= 1
        assert any(r["id"] == memory_id for r in result["results"])

        after = await app.pool.fetchrow(
            "SELECT last_retrieved_at, retrieval_count FROM memories WHERE id = $1",
            memory_id,
        )
        assert after["last_retrieved_at"] is not None
        assert after["retrieval_count"] == 1

    async def test_retrieval_telemetry_not_bumped_when_empty(self, ctx, app):
        """Empty result sets don't issue spurious bumps — guards against accidental wildcard updates."""
        from tests.conftest import DEFAULT_TEST_USER_ID
        from weft.mcp.tools import weft_recall, weft_remember

        store_result = await weft_remember(
            ctx, content="Tangential memory about volcanic geology"
        )
        memory_id = store_result["id"]

        result = await weft_recall(
            ctx,
            query="entirely unrelated quantum chromodynamics topic",
            mode="keyword",
            tier="belief",
            user_id=DEFAULT_TEST_USER_ID,
        )
        assert result["count"] == 0

        row = await app.pool.fetchrow(
            "SELECT last_retrieved_at, retrieval_count FROM memories WHERE id = $1",
            memory_id,
        )
        assert row["last_retrieved_at"] is None
        assert row["retrieval_count"] == 0


# ---------------------------------------------------------------------------
# weft_status
# ---------------------------------------------------------------------------


class TestWeftStatus:
    async def test_recent_writes_in_status(self, app):
        """weft_status includes recent_writes with provenance."""
        from weft.mcp.tools import weft_remember, weft_status

        ctx = _make_ctx(app)
        result = await weft_remember(ctx, content="Status test memory", source="conversation")
        assert "id" in result, f"weft_remember failed: {result}"

        # Verify memory actually exists in DB
        count = await app.pool.fetchval("SELECT COUNT(*) FROM memories")
        assert count >= 1

        result = await weft_status(ctx)
        assert "recent_writes" in result
        assert len(result["recent_writes"]) >= 1

        write = result["recent_writes"][0]
        assert "id" in write
        assert "type" in write
        assert "source" in write
        assert "created_at" in write
        assert "content" in write


# ---------------------------------------------------------------------------
# weft_prime
# ---------------------------------------------------------------------------


class TestWeftPrime:
    async def test_happy_path(self, ctx):
        from weft.mcp.tools import weft_prime

        result = await weft_prime(ctx)
        # Prime returns structured sections
        assert "rules" in result
        assert "recent_work" in result
        assert "total_tokens" in result
        assert "budget_tokens" in result

    async def test_with_project_id(self, ctx):
        from weft.mcp.tools import weft_prime, weft_remember

        await weft_remember(ctx, content="Project-scoped memory", project_id="testproj")

        result = await weft_prime(ctx, project_id="testproj")
        assert "rules" in result

    async def test_with_query_bias(self, ctx):
        from weft.mcp.tools import weft_prime

        result = await weft_prime(ctx, query="database optimization")
        assert "rules" in result

    async def test_custom_budget(self, ctx):
        from weft.mcp.tools import weft_prime

        result = await weft_prime(ctx, budget_tokens=500)
        assert result["budget_tokens"] == 500


# ---------------------------------------------------------------------------
# weft_forget
# ---------------------------------------------------------------------------


class TestWeftForget:
    async def test_soft_delete(self, ctx):
        from weft.mcp.tools import weft_forget, weft_remember

        mem = await weft_remember(ctx, content="Temporary note for soft-delete test")
        result = await weft_forget(ctx, memory_id=mem["id"])
        assert result["deleted"] is True
        assert result["hard"] is False

    async def test_hard_delete(self, ctx):
        from weft.mcp.tools import weft_forget, weft_remember

        mem = await weft_remember(ctx, content="Delete me permanently")
        result = await weft_forget(ctx, memory_id=mem["id"], hard=True)
        assert result["deleted"] is True
        assert result["hard"] is True


# ---------------------------------------------------------------------------
# _coerce_list
# ---------------------------------------------------------------------------


class TestCoerceList:
    def test_none(self):
        from weft.mcp.tools import _coerce_list

        assert _coerce_list(None) is None

    def test_list_passthrough(self):
        from weft.mcp.tools import _coerce_list

        assert _coerce_list(["a", "b"]) == ["a", "b"]

    def test_json_string(self):
        from weft.mcp.tools import _coerce_list

        assert _coerce_list('["x","y"]') == ["x", "y"]

    def test_invalid_json(self):
        from weft.mcp.tools import _coerce_list

        assert _coerce_list("not json") == "not json"

    def test_json_non_list(self):
        from weft.mcp.tools import _coerce_list

        assert _coerce_list('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------------------------------
# DB unavailability fallback
# ---------------------------------------------------------------------------


class TestDbFallback:
    async def test_recall_degrades_gracefully(self, app):
        """When the pool is closed, weft_recall returns degraded fallback."""
        from weft.mcp.tools import weft_recall

        # Close the pool to simulate DB unavailability
        await app.pool.close()
        ctx = _make_ctx(app)

        result = await weft_recall(ctx, query="anything")
        assert result.get("degraded") is True

    async def test_prime_degrades_gracefully(self, app):
        """When the pool is closed, weft_prime returns degraded fallback."""
        from weft.mcp.tools import weft_prime

        await app.pool.close()
        ctx = _make_ctx(app)

        result = await weft_prime(ctx)
        assert result.get("degraded") is True

    async def test_prime_fallback_truncates_oversized_export(self, app, tmp_path):
        """Fallback path must truncate huge exported files to stay within budget."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_prime

        # Create a massive fallback file (~200K chars)
        huge_content = "This is a very long memory. " * 10_000
        fallback_file = tmp_path / "weft_export.md"
        fallback_file.write_text(huge_content)

        await app.pool.close()
        ctx = _make_ctx(app)

        with patch("weft.fallback.read_fallback", return_value=huge_content):
            result = await weft_prime(ctx, budget_tokens=2400)

        assert result.get("degraded") is True
        # Handoff should exist but be truncated, not the full 200K+ chars
        if result["handoff"]:
            handoff_content = result["handoff"][0]["content"]
            assert len(handoff_content) < 20_000  # well under 200K+


# ---------------------------------------------------------------------------
# user_id filter tests — Phase 1 multi-user scoping
# Orthogonality contract: retrieval_mode (face/code/all), scope (user/project/agent),
# and user_id are THREE INDEPENDENT KNOBS. None implies the other.
# ---------------------------------------------------------------------------


class TestUserIdFiltering:
    """
    Tests that user_id is plumbed through to the underlying store/domain functions.

    Orthogonality contract (load-bearing): retrieval_mode (face/code/all),
    scope (user/project/agent), and user_id are THREE INDEPENDENT knobs —
    no two imply the other; all three compose freely.
    """

    # ------------------------------------------------------------------
    # weft_recall
    # ------------------------------------------------------------------

    async def test_recall_without_user_id_defaults_to_get_user_id(self, app, monkeypatch):
        """Calling weft_recall without user_id uses get_user_id() automatically."""
        from unittest.mock import AsyncMock, patch

        from weft.mcp.tools import weft_recall

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "patched-uid")

        captured = {}

        original_search_hybrid = __import__("weft.store", fromlist=["search_hybrid"]).search_hybrid

        async def spy_search_hybrid(pool, query, embedding, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original_search_hybrid(pool, query, embedding, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.search_hybrid", side_effect=spy_search_hybrid):
            await weft_recall(ctx, query="anything")

        assert captured.get("user_id") == "patched-uid"

    async def test_recall_explicit_none_defaults_to_get_user_id(self, app, monkeypatch):
        """Passing user_id=None explicitly also falls through to get_user_id()."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_recall

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "patched-uid-2")

        captured = {}

        original = __import__("weft.store", fromlist=["search_hybrid"]).search_hybrid

        async def spy(pool, query, embedding, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original(pool, query, embedding, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.search_hybrid", side_effect=spy):
            await weft_recall(ctx, query="anything", user_id=None)

        assert captured.get("user_id") == "patched-uid-2"

    async def test_recall_byte_identical_default_vs_explicit(self, app, monkeypatch):
        """Default user_id path produces same results as explicit user_id path."""
        from weft.mcp.tools import weft_recall, weft_remember

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "user-abc")

        ctx = _make_ctx(app)
        await weft_remember(ctx, content="Test memory for recall identity check", topic=["identity"])

        result_default = await weft_recall(ctx, query="identity check", threshold=0.0)
        result_explicit = await weft_recall(ctx, query="identity check", threshold=0.0, user_id="user-abc")

        assert result_default["count"] == result_explicit["count"]

    async def test_recall_retrieval_mode_and_user_id_compose(self, app, monkeypatch):
        """retrieval_mode and user_id are orthogonal — both are applied independently."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_recall

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "compose-uid")

        captured = {}

        original = __import__("weft.store", fromlist=["search_hybrid"]).search_hybrid

        async def spy(pool, query, embedding, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            captured["sources"] = kwargs.get("sources")
            return await original(pool, query, embedding, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.search_hybrid", side_effect=spy):
            await weft_recall(ctx, query="compose test", retrieval_mode="code", user_id="explicit-uid")

        # Both filters applied: user_id is set AND sources is set (not None, for 'code' mode)
        assert captured.get("user_id") == "explicit-uid"
        assert captured.get("sources") is not None

    # ------------------------------------------------------------------
    # weft_behavior_list
    # ------------------------------------------------------------------

    async def test_behavior_list_without_user_id_defaults_to_get_user_id(self, app, monkeypatch):
        """weft_behavior_list without user_id calls list_behaviors with get_user_id()."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_behavior_list

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "behaviors-uid")

        captured = {}

        original = __import__("weft.behaviors", fromlist=["list_behaviors"]).list_behaviors

        async def spy(pool, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original(pool, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.list_behaviors_store", side_effect=spy):
            await weft_behavior_list(ctx)

        assert captured.get("user_id") == "behaviors-uid"

    async def test_behavior_list_explicit_none_defaults_to_get_user_id(self, app, monkeypatch):
        """Passing user_id=None to weft_behavior_list also falls through to get_user_id()."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_behavior_list

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "behaviors-uid-2")

        captured = {}

        original = __import__("weft.behaviors", fromlist=["list_behaviors"]).list_behaviors

        async def spy(pool, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original(pool, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.list_behaviors_store", side_effect=spy):
            await weft_behavior_list(ctx, user_id=None)

        assert captured.get("user_id") == "behaviors-uid-2"

    # ------------------------------------------------------------------
    # weft_entity_search
    # ------------------------------------------------------------------

    async def test_entity_search_without_user_id_defaults_to_get_user_id(self, app, monkeypatch):
        """weft_entity_search without user_id calls search_entities with get_user_id()."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_entity_search

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "entities-uid")

        captured = {}

        original = __import__("weft.entities", fromlist=["search_entities"]).search_entities

        async def spy(pool, embedding, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original(pool, embedding, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.search_entities", side_effect=spy):
            await weft_entity_search(ctx, query="test entity")

        assert captured.get("user_id") == "entities-uid"

    async def test_entity_search_explicit_none_defaults_to_get_user_id(self, app, monkeypatch):
        """Passing user_id=None to weft_entity_search also falls through to get_user_id()."""
        from unittest.mock import patch

        from weft.mcp.tools import weft_entity_search

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "entities-uid-2")

        captured = {}

        original = __import__("weft.entities", fromlist=["search_entities"]).search_entities

        async def spy(pool, embedding, **kwargs):
            captured["user_id"] = kwargs.get("user_id")
            return await original(pool, embedding, **kwargs)

        ctx = _make_ctx(app)
        with patch("weft.mcp.tools.search_entities", side_effect=spy):
            await weft_entity_search(ctx, query="test entity", user_id=None)

        assert captured.get("user_id") == "entities-uid-2"

    async def test_entity_search_byte_identical_default_vs_explicit(self, app, monkeypatch):
        """Default user_id path produces same results as explicit user_id path for entities."""
        from weft.mcp.tools import weft_entity_search

        monkeypatch.setattr("weft.mcp.tools.get_user_id", lambda: "entity-id-match")

        ctx = _make_ctx(app)
        result_default = await weft_entity_search(ctx, query="some concept", threshold=0.0)
        result_explicit = await weft_entity_search(ctx, query="some concept", threshold=0.0, user_id="entity-id-match")

        assert result_default["count"] == result_explicit["count"]
