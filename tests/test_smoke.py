"""Smoke tests — verify core infrastructure works."""

from __future__ import annotations

import pytest


async def test_pool_connects(pool):
    """Verify asyncpg pool connects and migrations ran."""
    result = await pool.fetchval("SELECT COUNT(*) FROM memories")
    assert result == 0


async def test_tables_exist(pool):
    """Verify all expected tables were created by migrations."""
    tables = await pool.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    names = {t["tablename"] for t in tables}
    assert "memories" in names
    assert "memory_relationships" in names
    assert "schema_migrations" in names


async def test_pgvector_enabled(pool):
    """Verify pgvector extension is loaded."""
    version = await pool.fetchval(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    assert version is not None


async def test_hnsw_index_exists(pool):
    """Verify HNSW index was created on embedding column by migration 7."""
    index = await pool.fetchrow(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE tablename = 'memories' AND indexname = 'idx_memories_embedding_hnsw'"
    )
    assert index is not None
    assert "hnsw" in index["indexdef"].lower()
    assert "vector_cosine_ops" in index["indexdef"]


async def test_embedding_column_has_dimensions(pool):
    """Verify embedding column was typed to vector(384) by migration 7."""
    udt = await pool.fetchval(
        "SELECT format_type(atttypid, atttypmod) "
        "FROM pg_attribute "
        "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'"
    )
    assert udt == "vector(384)"


async def test_redis_connects(redis_conn):
    """Verify Redis connection works."""
    await redis_conn.set("test_key", "test_value")
    val = await redis_conn.get("test_key")
    assert val == "test_value"


async def test_embedding_provider():
    """Verify FastEmbed provider generates correct dimensions."""
    from weft.embeddings import get_provider

    provider = get_provider("fastembed")
    vec = await provider.embed("test embedding")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


async def test_memory_model():
    """Verify Pydantic model creation and serialization."""
    from weft.models import Memory, MemoryType

    m = Memory(type=MemoryType.fact, content="test memory", topic=["test"])
    assert m.id.startswith("weft-")
    assert len(m.id) == 13  # "weft-" + 8 hex chars
    d = m.to_dict()
    assert d["type"] == "fact"
    assert d["status"] == "active"


async def test_invalid_enum_raises_valueerror():
    """Invalid enum values should raise ValueError, not pass silently."""
    from weft.models import MemoryType, MemorySource, MemoryStatus

    with pytest.raises(ValueError):
        MemoryType("not_a_type")
    with pytest.raises(ValueError):
        MemorySource("observation")  # the original bug report example
    with pytest.raises(ValueError):
        MemoryStatus("bogus")


async def test_input_error_response_format():
    """_input_error_response returns structured error with detail."""
    from weft.mcp.tools import _input_error_response

    result = _input_error_response("weft_remember", ValueError("'observation' is not a valid MemorySource"))
    assert result["error"] == "Invalid input"
    assert "observation" in result["detail"]
    assert result["tool"] == "weft_remember"
    assert "degraded" not in result  # input errors are not degraded mode


class TestDetectProjectId:
    """Auto-detect project_id from MCP client roots."""

    def test_extracts_directory_name(self):
        from weft.mcp.tools import _detect_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        root = MagicMock()
        root.uri = "file:///Users/jason/Projects/Weft"
        ctx.list_roots = AsyncMock(return_value=[root])

        result = asyncio.get_event_loop().run_until_complete(_detect_project_id(ctx))
        assert result == "weft"

    def test_lowercases_name(self):
        from weft.mcp.tools import _detect_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        root = MagicMock()
        root.uri = "file:///Users/jason/Projects/MyProject"
        ctx.list_roots = AsyncMock(return_value=[root])

        result = asyncio.get_event_loop().run_until_complete(_detect_project_id(ctx))
        assert result == "myproject"

    def test_empty_roots_returns_none(self):
        from weft.mcp.tools import _detect_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        ctx.list_roots = AsyncMock(return_value=[])

        result = asyncio.get_event_loop().run_until_complete(_detect_project_id(ctx))
        assert result is None

    def test_exception_returns_none(self):
        from weft.mcp.tools import _detect_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        ctx.list_roots = AsyncMock(side_effect=Exception("not supported"))

        result = asyncio.get_event_loop().run_until_complete(_detect_project_id(ctx))
        assert result is None


class TestResolveProjectId:
    """Explicit project_id takes precedence over auto-detect."""

    def test_explicit_wins(self):
        from weft.mcp.tools import _resolve_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        ctx.list_roots = AsyncMock(return_value=[])

        result = asyncio.get_event_loop().run_until_complete(
            _resolve_project_id(ctx, "my-project")
        )
        assert result == "my-project"

    def test_falls_back_to_detect(self):
        from weft.mcp.tools import _resolve_project_id
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        ctx = MagicMock()
        root = MagicMock()
        root.uri = "file:///Users/jason/Projects/Weft"
        ctx.list_roots = AsyncMock(return_value=[root])

        result = asyncio.get_event_loop().run_until_complete(
            _resolve_project_id(ctx, None)
        )
        assert result == "weft"


class TestCoerceList:
    """MCP transport sometimes serializes list params as JSON strings."""

    def test_none_passthrough(self):
        from weft.mcp.tools import _coerce_list
        assert _coerce_list(None) is None

    def test_list_passthrough(self):
        from weft.mcp.tools import _coerce_list
        assert _coerce_list(["a", "b"]) == ["a", "b"]

    def test_json_string_to_list(self):
        from weft.mcp.tools import _coerce_list
        assert _coerce_list('["weft", "bug"]') == ["weft", "bug"]

    def test_empty_json_array(self):
        from weft.mcp.tools import _coerce_list
        assert _coerce_list("[]") == []

    def test_non_json_string_passthrough(self):
        from weft.mcp.tools import _coerce_list
        # Non-JSON strings pass through so Pydantic can raise properly
        result = _coerce_list("not json")
        assert result == "not json"

    def test_json_string_not_array_passthrough(self):
        from weft.mcp.tools import _coerce_list
        # A JSON string that parses to non-list passes through
        result = _coerce_list('"just a string"')
        assert result == '"just a string"'


class TestParseReviewAfter:
    """Parse review_after: ISO timestamps and relative durations."""

    def test_none_returns_none(self):
        from weft.mcp.tools import _parse_review_after
        assert _parse_review_after(None) is None

    def test_relative_days(self):
        from weft.mcp.tools import _parse_review_after
        from datetime import datetime, timezone
        result = _parse_review_after("30d")
        assert result is not None
        # Should be ~30 days from now
        delta = (result - datetime.now(timezone.utc)).total_seconds()
        assert 29 * 86400 < delta < 31 * 86400

    def test_relative_weeks(self):
        from weft.mcp.tools import _parse_review_after
        from datetime import datetime, timezone
        result = _parse_review_after("2w")
        assert result is not None
        delta = (result - datetime.now(timezone.utc)).total_seconds()
        assert 13 * 86400 < delta < 15 * 86400

    def test_relative_months(self):
        from weft.mcp.tools import _parse_review_after
        from datetime import datetime, timezone
        result = _parse_review_after("3m")
        assert result is not None
        delta = (result - datetime.now(timezone.utc)).total_seconds()
        assert 89 * 86400 < delta < 91 * 86400

    def test_relative_with_words(self):
        from weft.mcp.tools import _parse_review_after
        result = _parse_review_after("7 days")
        assert result is not None

    def test_iso_timestamp(self):
        from weft.mcp.tools import _parse_review_after
        result = _parse_review_after("2026-06-01T00:00:00+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
