"""Tests for migration v47: episodes.embedding column + HNSW index + backfill.

Schema-side checks confirm the column and HNSW index exist after migrations
run. Backfill-side checks exercise ``reembed_table("episodes", ...)``
with a fake provider so we don't need a real embedder API key. The
embedder lookup helper (``resolve_episode_embedder``) is unit-tested
directly with env-var monkeypatching.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from weft.config import WeftConfig
from weft.db.reembed import (
    TABLE_TEXT_COLUMNS,
    TABLE_TEXT_EXPRESSIONS,
    reembed_table,
    resolve_episode_embedder,
)


class FakeProvider:
    """Deterministic embedding provider for tests; mirrors test_reembed.FakeProvider."""

    def __init__(self, dimensions: int = 768):
        self._dimensions = dimensions
        self.call_count = 0
        self.last_texts: list[str] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "fake"

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return [0.1] * self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.last_texts = list(texts)
        return [[0.1] * self._dimensions for _ in texts]


# ---------------------------------------------------------------------------
# Schema (column + index)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_episodes_has_embedding_column(pool):
    """Migration adds embedding vector(768) NULL column to episodes."""
    row = await pool.fetchrow(
        """
        SELECT a.atttypmod AS dim, a.attnotnull AS notnull
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_type t ON a.atttypid = t.oid
        WHERE c.relname = 'episodes'
          AND a.attname = 'embedding'
          AND t.typname = 'vector'
        """
    )
    assert row is not None, "episodes.embedding column missing after v47"
    assert row["dim"] == 768, f"expected vector(768), got vector({row['dim']})"
    assert row["notnull"] is False, "embedding column must be NULL-able"


@pytest.mark.asyncio
async def test_episodes_has_hnsw_index(pool):
    """Migration creates idx_episodes_embedding_hnsw on the embedding column."""
    row = await pool.fetchrow(
        """
        SELECT indexdef
        FROM pg_indexes
        WHERE tablename = 'episodes'
          AND indexname = 'idx_episodes_embedding_hnsw'
        """
    )
    assert row is not None, "HNSW index idx_episodes_embedding_hnsw missing"
    indexdef = row["indexdef"].lower()
    assert "hnsw" in indexdef
    assert "vector_cosine_ops" in indexdef


@pytest.mark.asyncio
async def test_v47_is_idempotent(pool):
    """Re-running the v47 SQL is a no-op (IF NOT EXISTS clauses guard both ops)."""
    from weft.db.migrations.v47_episode_embedding import SQL

    # Should not raise even though the column + index already exist.
    await pool.execute(SQL)


# ---------------------------------------------------------------------------
# Backfill via reembed_table("episodes", ...)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_writes_embeddings_to_null_rows(pool):
    """Episodes with NULL embeddings get vectors written; existing ones untouched."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO episodes (id, title, summary) VALUES ($1, $2, $3)",
            "ep-1",
            "Refactor token bucket",
            "Pulled the rate limiter into its own module.",
        )
        await conn.execute(
            "INSERT INTO episodes (id, title) VALUES ($1, $2)",
            "ep-2",
            "Title-only episode",  # summary is NULL
        )
        await conn.execute(
            "INSERT INTO episodes (id, title, embedding) VALUES ($1, $2, $3::vector)",
            "ep-3",
            "Already embedded",
            [0.5] * 768,
        )

    provider = FakeProvider()
    count = await reembed_table(pool, "episodes", provider)
    assert count == 2  # ep-1 and ep-2; ep-3 skipped

    # Confirm the actual embeddings were written.
    rows = await pool.fetch(
        "SELECT id, embedding FROM episodes ORDER BY id"
    )
    by_id = {r["id"]: r["embedding"] for r in rows}
    assert by_id["ep-1"] is not None
    assert len(by_id["ep-1"]) == 768
    assert by_id["ep-2"] is not None
    # ep-3 should still have its original [0.5]*768 vector, not the FakeProvider's [0.1]*768.
    assert by_id["ep-3"][0] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_backfill_text_includes_title_and_summary(pool):
    """The text fed to the embedder is title || ' ' || COALESCE(summary, '')."""
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO episodes (id, title, summary) VALUES ($1, $2, $3)",
            "ep-text",
            "TITLE",
            "BODY",
        )
        await conn.execute(
            "INSERT INTO episodes (id, title) VALUES ($1, $2)",
            "ep-no-summary",
            "JUSTITLE",
        )

    provider = FakeProvider()
    await reembed_table(pool, "episodes", provider)

    # FakeProvider records the texts of its last batch.
    assert "TITLE BODY" in provider.last_texts
    assert "JUSTITLE " in provider.last_texts  # trailing space from COALESCE('')


@pytest.mark.asyncio
async def test_backfill_is_idempotent(pool):
    """Running backfill twice does not re-embed already-embedded rows."""
    async with pool.acquire() as conn:
        for i in range(3):
            await conn.execute(
                "INSERT INTO episodes (id, title) VALUES ($1, $2)",
                f"idem-{i}",
                f"Episode {i}",
            )

    provider = FakeProvider()
    first = await reembed_table(pool, "episodes", provider)
    second = await reembed_table(pool, "episodes", provider)
    assert first == 3
    assert second == 0


@pytest.mark.asyncio
async def test_backfill_respects_batch_size(pool):
    """Large episode tables process in batches, not a single embed_batch call."""
    async with pool.acquire() as conn:
        for i in range(7):
            await conn.execute(
                "INSERT INTO episodes (id, title) VALUES ($1, $2)",
                f"batch-{i}",
                f"Episode {i}",
            )

    provider = FakeProvider()
    count = await reembed_table(pool, "episodes", provider, batch_size=3)
    assert count == 7
    # 7 rows / batch=3 -> 3 batches (3, 3, 1)
    assert provider.call_count == 3


# ---------------------------------------------------------------------------
# Embedder lookup helper
# ---------------------------------------------------------------------------


def test_resolve_episode_embedder_falls_back_to_memory_provider(monkeypatch):
    """Without WEFT_EPISODE_EMBEDDER, helper uses config.embedding.provider."""
    monkeypatch.delenv("WEFT_EPISODE_EMBEDDER", raising=False)

    config = WeftConfig()
    config.embedding.provider = "fastembed"

    captured: dict = {}

    def fake_get_provider(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    with patch("weft.embeddings.get_provider", fake_get_provider):
        resolve_episode_embedder(config)

    assert captured["name"] == "fastembed"
    assert captured["kwargs"]["dimensions"] == config.embedding.dimensions


def test_resolve_episode_embedder_honors_env_override(monkeypatch):
    """Setting WEFT_EPISODE_EMBEDDER picks the named provider, not the memory one."""
    monkeypatch.setenv("WEFT_EPISODE_EMBEDDER", "google")

    config = WeftConfig()
    config.embedding.provider = "openai"  # memory tier

    captured: dict = {}

    def fake_get_provider(name, **kwargs):
        captured["name"] = name
        return object()

    with patch("weft.embeddings.get_provider", fake_get_provider):
        resolve_episode_embedder(config)

    assert captured["name"] == "google"


# ---------------------------------------------------------------------------
# Reembed registry sanity
# ---------------------------------------------------------------------------


def test_episodes_registered_in_reembed_maps():
    """episodes shows up in both TABLE_TEXT_COLUMNS and TABLE_TEXT_EXPRESSIONS."""
    assert "episodes" in TABLE_TEXT_COLUMNS
    assert TABLE_TEXT_COLUMNS["episodes"] == "title"
    assert "episodes" in TABLE_TEXT_EXPRESSIONS
    assert "title" in TABLE_TEXT_EXPRESSIONS["episodes"]
    assert "summary" in TABLE_TEXT_EXPRESSIONS["episodes"]
    assert "COALESCE" in TABLE_TEXT_EXPRESSIONS["episodes"].upper() or \
           "coalesce" in TABLE_TEXT_EXPRESSIONS["episodes"]
