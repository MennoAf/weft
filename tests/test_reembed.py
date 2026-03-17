"""Tests for auto re-embed after dimension migration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from weft.db.reembed import (
    TABLE_TEXT_COLUMNS,
    auto_reembed,
    reembed_table,
)


class FakeProvider:
    """Test embedding provider that returns deterministic vectors."""

    def __init__(self, dimensions: int = 768):
        self._dimensions = dimensions
        self.call_count = 0

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
        return [[0.1] * self._dimensions for _ in texts]


class FailingProvider(FakeProvider):
    """Provider that fails on the Nth call to embed_batch."""

    def __init__(self, fail_on_call: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._fail_on = fail_on_call

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self.call_count == self._fail_on:
            raise RuntimeError("Simulated embedding failure")
        return [[0.1] * self._dimensions for _ in texts]


# ---------------------------------------------------------------------------
# reembed_table
# ---------------------------------------------------------------------------


class TestReembedTable:
    @pytest.mark.asyncio
    async def test_reembed_null_embeddings(self, pool):
        """Rows with NULL embeddings get re-embedded."""
        # Insert rows without embeddings
        async with pool.acquire() as conn:
            for i in range(5):
                await conn.execute(
                    "INSERT INTO memories (id, type, content) VALUES ($1, 'fact', $2)",
                    f"test-{i}",
                    f"Test content {i}",
                )

        provider = FakeProvider()
        count = await reembed_table(pool, "memories", provider)
        assert count == 5

        # Verify embeddings are set
        row = await pool.fetchrow("SELECT embedding FROM memories WHERE id = 'test-0'")
        assert row["embedding"] is not None
        assert len(row["embedding"]) == 768

    @pytest.mark.asyncio
    async def test_skip_already_embedded(self, pool):
        """Rows with existing embeddings are skipped (force=False)."""
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memories (id, type, content, embedding) "
                "VALUES ('has-emb', 'fact', 'content', $1::vector)",
                [0.2] * 768,
            )
            await conn.execute(
                "INSERT INTO memories (id, type, content) VALUES ('no-emb', 'fact', 'content')",
            )

        provider = FakeProvider()
        count = await reembed_table(pool, "memories", provider)
        assert count == 1  # Only the NULL one

    @pytest.mark.asyncio
    async def test_force_reembeds_all(self, pool):
        """With force=True, re-embeds even rows that already have embeddings."""
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memories (id, type, content, embedding) "
                "VALUES ('has-emb', 'fact', 'content', $1::vector)",
                [0.2] * 768,
            )

        provider = FakeProvider()
        count = await reembed_table(pool, "memories", provider, force=True)
        assert count == 1

    @pytest.mark.asyncio
    async def test_empty_table(self, pool):
        """Empty table returns 0 without calling provider."""
        provider = FakeProvider()
        count = await reembed_table(pool, "memories", provider)
        assert count == 0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_batch_boundary(self, pool):
        """Processes rows in batches correctly."""
        async with pool.acquire() as conn:
            for i in range(5):
                await conn.execute(
                    "INSERT INTO memories (id, type, content) VALUES ($1, 'fact', $2)",
                    f"batch-{i}",
                    f"Content {i}",
                )

        provider = FakeProvider()
        count = await reembed_table(pool, "memories", provider, batch_size=3)
        assert count == 5
        assert provider.call_count == 2  # 3 + 2

    @pytest.mark.asyncio
    async def test_batch_failure_continues(self, pool):
        """Failed batch is skipped, other batches still process."""
        async with pool.acquire() as conn:
            for i in range(6):
                await conn.execute(
                    "INSERT INTO memories (id, type, content) VALUES ($1, 'fact', $2)",
                    f"fail-{i}",
                    f"Content {i}",
                )

        # Fail on second batch (rows 3-5)
        provider = FailingProvider(fail_on_call=2)
        count = await reembed_table(pool, "memories", provider, batch_size=3)
        assert count == 3  # First batch succeeded, second failed

    @pytest.mark.asyncio
    async def test_null_text_column_skipped(self, pool):
        """Rows with NULL text column are excluded."""
        async with pool.acquire() as conn:
            # Entity with NULL description
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type) VALUES ('e1', 'Test', 'person')",
            )
            # Entity with description
            await conn.execute(
                "INSERT INTO entities (id, name, entity_type, description) "
                "VALUES ('e2', 'Test2', 'person', 'Has a description')",
            )

        provider = FakeProvider()
        count = await reembed_table(pool, "entities", provider)
        assert count == 1  # Only e2

    @pytest.mark.asyncio
    async def test_unknown_table_raises(self, pool):
        with pytest.raises(ValueError, match="Unknown table"):
            await reembed_table(pool, "not_a_table", FakeProvider())

    @pytest.mark.asyncio
    async def test_behaviors_table(self, pool):
        """Behaviors table uses 'action' column for text."""
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO behaviors (id, trigger_pattern, action) "
                "VALUES ('b1', 'when testing', 'use pytest')",
            )

        provider = FakeProvider()
        count = await reembed_table(pool, "behaviors", provider)
        assert count == 1


# ---------------------------------------------------------------------------
# auto_reembed
# ---------------------------------------------------------------------------


class TestAutoReembed:
    @pytest.mark.asyncio
    async def test_reembeds_multiple_tables(self, pool):
        """Processes all specified tables."""
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memories (id, type, content) VALUES ('m1', 'fact', 'memory')",
            )
            await conn.execute(
                "INSERT INTO behaviors (id, trigger_pattern, action) "
                "VALUES ('b1', 'trigger', 'action')",
            )

        provider = FakeProvider()
        results = await auto_reembed(pool, provider, ["memories", "behaviors"])
        assert results["memories"] == 1
        assert results["behaviors"] == 1

    @pytest.mark.asyncio
    async def test_defaults_to_all_tables(self, pool):
        """Without explicit tables, processes all known tables."""
        provider = FakeProvider()
        results = await auto_reembed(pool, provider)
        assert set(results.keys()) == set(TABLE_TEXT_COLUMNS.keys())

    @pytest.mark.asyncio
    async def test_empty_tables_return_zero(self, pool):
        """Empty tables return 0 in results."""
        provider = FakeProvider()
        results = await auto_reembed(pool, provider)
        assert all(v == 0 for v in results.values())
