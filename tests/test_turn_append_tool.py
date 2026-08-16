"""Tests for the weft_turn_append MCP tool (loom-bc8d9801).

Covers the four contract points called out in the Loom task:
  - happy path: turn written, embedding non-null, response shape correct
  - invalid role rejected with input-error response
  - bad timestamp rejected with input-error response
  - missing episode rejected with input-error response

Plus a fallback test: when the embedding provider raises, the turn is
still written and the response carries a ``warning`` field (matches the
weft_remember contract for embedding failures).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_turn_append
from weft.models import EpisodeCreate


_FAKE_EMBEDDING = [0.1] * 768


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return list(_FAKE_EMBEDDING)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(_FAKE_EMBEDDING) for _ in texts]


class _BrokenEmbedding:
    """Embedding provider that always raises — for the failure-fallback test."""

    provider_name = "broken"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding service unreachable")


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
async def app(pool):
    # Re-init pgvector codec on every pooled connection. The conftest's
    # ``init=`` callback fired BEFORE migrations created the vector type,
    # so connections opened during pool warm-up cached the failure. The
    # post-migration ``register_pgvector_codec`` call only touches one
    # conn, leaving the others codec-less. Without this re-init, the first
    # weft_turn_append call against a no-codec conn rejects the list-shaped
    # embedding with `expected str, got list`.
    from weft.db.connection import _pgvector_codec_init
    conns = [await pool.acquire() for _ in range(pool.get_size())]
    try:
        for c in conns:
            await _pgvector_codec_init(c)
    finally:
        for c in conns:
            await pool.release(c)
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbedding(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


@pytest.fixture
async def episode(pool):
    return await create_episode(pool, EpisodeCreate(title="turn-append-test"))


# --- happy path ---


async def test_happy_path_writes_turn_with_embedding(ctx, app, episode):
    result = await weft_turn_append(
        ctx,
        episode_id=episode.id,
        role="user",
        content="hello there",
    )

    assert "error" not in result, result
    assert result["episode_id"] == episode.id
    assert result["turn_index"] == 0
    assert result["turn_id"].startswith("et-")
    assert result["occurred_at"]  # ISO string

    # Verify the row landed and the embedding is non-null (vector column).
    # Read via the same pool the AppContext used so codec state matches.
    row = await app.pool.fetchrow(
        "SELECT episode_id, role, content, turn_index, embedding IS NOT NULL "
        "AS has_embedding FROM episode_turns WHERE id = $1",
        result["turn_id"],
    )
    assert row is not None
    assert row["episode_id"] == episode.id
    assert row["role"] == "user"
    assert row["content"] == "hello there"
    assert row["turn_index"] == 0
    assert row["has_embedding"] is True


async def test_sequential_appends_increment_turn_index(ctx, episode):
    r0 = await weft_turn_append(ctx, episode_id=episode.id, role="user", content="a")
    r1 = await weft_turn_append(ctx, episode_id=episode.id, role="assistant", content="b")
    r2 = await weft_turn_append(ctx, episode_id=episode.id, role="tool", content="c")
    assert (r0["turn_index"], r1["turn_index"], r2["turn_index"]) == (0, 1, 2)


async def test_explicit_occurred_at_is_persisted(ctx, pool, episode):
    result = await weft_turn_append(
        ctx,
        episode_id=episode.id,
        role="user",
        content="historical turn",
        occurred_at="2024-06-15T12:00:00+00:00",
    )
    assert "error" not in result
    row = await pool.fetchrow(
        "SELECT occurred_at FROM episode_turns WHERE id = $1", result["turn_id"],
    )
    assert row["occurred_at"].isoformat() == "2024-06-15T12:00:00+00:00"


async def test_source_session_id_is_persisted(ctx, pool, episode):
    result = await weft_turn_append(
        ctx,
        episode_id=episode.id,
        role="user",
        content="source session turn",
        source_session_id="conversation-42",
    )
    assert "error" not in result
    row = await pool.fetchrow(
        "SELECT source_session_id FROM episode_turns WHERE id = $1",
        result["turn_id"],
    )
    assert row["source_session_id"] == "conversation-42"


async def test_naive_iso_timestamp_is_treated_as_utc(ctx, pool, episode):
    """Wick may send timestamps without an explicit zone; assume UTC."""
    result = await weft_turn_append(
        ctx,
        episode_id=episode.id,
        role="user",
        content="naive ts",
        occurred_at="2024-06-15T12:00:00",
    )
    assert "error" not in result
    row = await pool.fetchrow(
        "SELECT occurred_at FROM episode_turns WHERE id = $1", result["turn_id"],
    )
    assert row["occurred_at"].tzinfo is not None
    assert row["occurred_at"].isoformat() == "2024-06-15T12:00:00+00:00"


# --- input errors ---


async def test_invalid_role_returns_input_error(ctx, episode):
    result = await weft_turn_append(
        ctx, episode_id=episode.id, role="banana", content="x",
    )
    assert result["error"] == "Invalid input"
    assert "role must be one of" in result["detail"]
    assert result["tool"] == "weft_turn_append"


async def test_bad_timestamp_returns_input_error(ctx, episode):
    result = await weft_turn_append(
        ctx,
        episode_id=episode.id,
        role="user",
        content="x",
        occurred_at="yesterday at noon",
    )
    assert result["error"] == "Invalid input"
    assert "ISO-8601" in result["detail"]
    assert result["tool"] == "weft_turn_append"


async def test_missing_episode_returns_input_error(ctx):
    result = await weft_turn_append(
        ctx, episode_id="ep-does-not-exist", role="user", content="x",
    )
    assert result["error"] == "Invalid input"
    assert "episode not found" in result["detail"]


# --- embedding failure fallback ---


async def test_embedding_failure_still_writes_turn_with_warning(pool, episode):
    """Match the weft_remember contract: embedding failure does not block
    the write; the response carries a 'warning' field instead."""
    app = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_BrokenEmbedding(),
        config=WeftConfig(),
    )
    ctx = _make_ctx(app)

    result = await weft_turn_append(
        ctx, episode_id=episode.id, role="user", content="no embedding for me",
    )
    assert "error" not in result
    assert result["turn_id"].startswith("et-")
    assert "warning" in result
    assert "embedding failed" in result["warning"]

    row = await pool.fetchrow(
        "SELECT embedding IS NULL AS no_embedding FROM episode_turns "
        "WHERE id = $1",
        result["turn_id"],
    )
    assert row["no_embedding"] is True
