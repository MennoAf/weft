"""Tests for the weft_status MCP tool — topic-digest recall (Tier-1 + Tier-2).

done_when assertions (synthesis mocked, no live API):

  (1) synthesize=False returns {topic, resolved_tags, memories (complete, ordered by
      created_at), complete, truncated} and calls synthesize_digest ZERO times (V2).

  (2) synthesize=True with a synthesized result returns a digest object
      (content+provenance) AND records ONE cost_entries row with
      entry_type='topic_synthesis' and estimated_cost_usd==result.cost_usd.

  (3) On a fresh cached digest, synthesize=True issues ZERO model calls AND records
      ZERO cost_entries rows (cache hit, V6).

  (4) Responses are RLS-scoped to the caller (V7) — another user's memories are NOT
      returned.

  (5) When synthesize_digest returns status='abstained', weft_status returns a
      graceful non-synthesized response (no crash, no cache write) AND records ONE
      cost_entries row with estimated_cost_usd==0.0 and metadata.abstained==True
      carrying projected + memory_count.

Each test uses a unique per-test user_id derived from a uuid prefix so tests are
self-isolating without touching conftest.py.  Pool fixture comes from tests/conftest.py
(real Postgres testcontainer).  Synthesis is mocked via unittest.mock.patch — no live
Anthropic calls.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.cost_tracking import CostEntryType, list_cost_entries
from weft.db.connection import acquire
from weft.mcp.server import AppContext
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory
from weft.topic_digest_cache import write_digest
from weft.views.topic_synthesis import SynthesisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    """Unique per-test user_id prefix for self-isolation."""
    return f"ws-test-{uuid.uuid4().hex[:12]}"


class FakeEmbeddingProvider:
    """Deterministic stub — no network calls."""

    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    """Build a mock FastMCP Context carrying our AppContext."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
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


async def _seed_memory(pool, user_id: str, tag: str, content: str = "Test content") -> str:
    """Seed a memory for user_id with the given tag. Returns memory id."""
    tok = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            mem = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content=content,
                    topic=[tag],
                    source=MemorySource.conversation,
                    confidence=0.8,
                ),
            )
    finally:
        current_user_id.reset(tok)
    return mem.id


async def _count_cost_entries(pool, entry_type: str, reference_id: str) -> list:
    """Count cost_entries rows for a given entry_type and reference_id."""
    rows = await pool.fetch(
        """
        SELECT * FROM cost_entries
        WHERE entry_type = $1
          AND reference_id = $2
        ORDER BY created_at DESC
        """,
        entry_type,
        reference_id,
    )
    return list(rows)


async def _recall_query_rows(pool, tool_name: str, query_text: str) -> list:
    """Fetch weft_recall_queries rows for a tool_name + query_text.

    The testcontainer role is a superuser, so this bypasses RLS — we read the
    rows regardless of which user_id they were attributed to.
    """
    rows = await pool.fetch(
        """
        SELECT * FROM weft_recall_queries
        WHERE tool_name = $1
          AND query_text = $2
        ORDER BY created_at DESC
        """,
        tool_name,
        query_text,
    )
    return list(rows)


# ---------------------------------------------------------------------------
# (1) synthesize=False returns Tier-1 shape, ZERO synthesize_digest calls (V2)
# ---------------------------------------------------------------------------


class TestTier1NoSynthesis:
    async def test_returns_expected_shape(self, ctx, pool):
        """synthesize=False returns {topic, resolved_tags, memories, complete, truncated}."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Memory content alpha")
        await _seed_memory(pool, user_id, tag, "Memory content beta")

        # V2: patch synthesize_digest to verify it is never called on synthesize=False
        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock
        ) as mock_synth:
            result = await weft_status(ctx, topic=tag, synthesize=False)

        # Shape assertions
        assert "topic" in result
        assert result["topic"] == tag
        assert "resolved_tags" in result
        assert isinstance(result["resolved_tags"], list)
        assert len(result["resolved_tags"]) > 0
        assert "memories" in result
        assert "complete" in result
        assert "truncated" in result

        # V2: synthesize_digest must NOT have been called
        mock_synth.assert_not_called()

    async def test_memories_ordered_by_created_at(self, ctx, pool):
        """Memories are ordered by created_at ascending."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        mem_ids = []
        for i in range(3):
            mid = await _seed_memory(pool, user_id, tag, f"Memory content {i}")
            mem_ids.append(mid)

        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        memories = result["memories"]
        assert len(memories) == 3
        # Verify created_at ordering (ascending)
        dates = [m["created_at"] for m in memories]
        assert dates == sorted(dates), "memories must be ordered by created_at ASC"

    async def test_memories_have_required_fields(self, ctx, pool):
        """Each memory dict includes id, type, content, topic, created_at."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Content check memory")

        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        assert len(result["memories"]) >= 1
        mem = result["memories"][0]
        assert "id" in mem
        assert "type" in mem
        assert "content" in mem
        assert "topic" in mem
        assert "created_at" in mem

    async def test_synthesize_false_zero_synth_calls(self, ctx, pool):
        """synthesize=False: synthesize_digest called zero times regardless of memory count."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag)

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest"
        ) as mock_synth:
            await weft_status(ctx, topic=tag, synthesize=False)

        mock_synth.assert_not_called()


# ---------------------------------------------------------------------------
# (2) synthesize=True with synthesized result → digest + ONE cost_entries row
# ---------------------------------------------------------------------------


class TestTier2Synthesized:
    async def test_synthesized_returns_digest_and_records_cost(self, ctx, pool):
        """synthesize=True with a synthesized result: digest dict + ONE cost_entries row."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        mem_id = await _seed_memory(pool, user_id, tag, "Relevant memory about the topic")

        synth_result = SynthesisResult(
            status="synthesized",
            memory_count=1,
            projected_cost_usd=0.001,
            content="This topic covers relevant material.",
            provenance={mem_id: ["relevant material"]},
            cost_usd=0.0025,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=synth_result
        ):
            result = await weft_status(ctx, topic=tag, synthesize=True)

        # Digest must be populated
        assert result["digest"] is not None, "digest must be non-null on synthesized result"
        digest = result["digest"]
        assert digest["content"] == "This topic covers relevant material."
        assert isinstance(digest["provenance"], dict)
        assert mem_id in digest["provenance"]

        # ONE cost_entries row with entry_type=topic_synthesis and correct cost
        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 1, f"Expected 1 cost_entries row, got {len(rows)}"
        row = rows[0]
        assert float(row["estimated_cost_usd"]) == pytest.approx(0.0025)

    async def test_synthesized_cost_entry_has_correct_entry_type(self, ctx, pool):
        """cost_entries row has entry_type='topic_synthesis'."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        mem_id = await _seed_memory(pool, user_id, tag)

        synth_result = SynthesisResult(
            status="synthesized",
            memory_count=1,
            projected_cost_usd=0.001,
            content="Summary narrative.",
            provenance={mem_id: ["narrative"]},
            cost_usd=0.003,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=synth_result
        ):
            await weft_status(ctx, topic=tag, synthesize=True)

        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 1
        assert rows[0]["entry_type"] == "topic_synthesis"

    async def test_synthesized_cost_entry_carries_token_counts(self, ctx, pool):
        """loom-97e0e019: the synthesized cost row carries the result's real
        input/output token counts (not 0/0), and total = input + output."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        mem_id = await _seed_memory(pool, user_id, tag, "Token-threading memory")

        synth_result = SynthesisResult(
            status="synthesized",
            memory_count=1,
            projected_cost_usd=0.001,
            content="Narrative with real usage.",
            provenance={mem_id: ["usage"]},
            cost_usd=0.0031,
            input_tokens=2487,
            output_tokens=613,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=synth_result
        ):
            await weft_status(ctx, topic=tag, synthesize=True)

        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 1
        row = rows[0]
        assert row["input_tokens"] == 2487
        assert row["output_tokens"] == 613
        assert row["total_tokens"] == 2487 + 613
        # weft-49bd0550: the cost row is attributed to the caller, not NULL.
        assert row["user_id"] == user_id


# ---------------------------------------------------------------------------
# (3) Cache hit: synthesize=True issues ZERO model calls + ZERO cost_entries rows (V6)
# ---------------------------------------------------------------------------


class TestTier2CacheHit:
    async def test_cache_hit_zero_model_calls_zero_cost_rows(self, ctx, pool):
        """Fresh cached digest: synthesize=True issues NO model calls and NO cost rows."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Cached topic memory")

        # Write a fresh (non-stale) digest into the cache
        mem_id = f"mem-{uuid.uuid4().hex[:8]}"
        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                await write_digest(
                    pool,
                    user_id=user_id,
                    topic=tag,
                    content="Cached narrative for this topic.",
                    detector_version="topic-synthesis-v1.0",
                    scope="global",
                    provenance={mem_id: ["cached span"]},
                )
        finally:
            current_user_id.reset(tok)

        # synthesize=True — should hit cache and skip the synthesizer
        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock
        ) as mock_synth:
            result = await weft_status(ctx, topic=tag, synthesize=True)

        # Zero model calls (V6)
        mock_synth.assert_not_called()

        # Digest served from cache
        assert result["digest"] is not None
        assert result["digest"]["content"] == "Cached narrative for this topic."

        # Zero cost_entries rows (V6 — cache hit records nothing)
        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 0, (
            f"Cache hit must record ZERO cost_entries rows, got {len(rows)}"
        )


# ---------------------------------------------------------------------------
# (4) RLS isolation: caller only sees their own memories (V7)
# ---------------------------------------------------------------------------


class TestRLSIsolation:
    async def test_other_users_memories_not_returned(self, ctx, pool):
        """V7: another user's memories tagged with the same topic are NOT returned."""
        from weft.mcp.tools import weft_status

        user_a = _uid()
        user_b = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"

        # User A writes 2 memories
        await _seed_memory(pool, user_a, tag, "User A memory one")
        await _seed_memory(pool, user_a, tag, "User A memory two")

        # User B writes 1 memory with the SAME tag
        await _seed_memory(pool, user_b, tag, "User B secret memory")

        # Query as user A
        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_a):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        # Must return exactly user_a's memories, not user_b's
        contents = [m["content"] for m in result["memories"]]
        assert "User B secret memory" not in contents, (
            "V7 violated: another user's memory was returned in weft_status"
        )
        assert "User A memory one" in contents
        assert "User A memory two" in contents
        assert len(result["memories"]) == 2


# ---------------------------------------------------------------------------
# (5) synthesize_digest returns abstained → graceful response + abstain cost row
# ---------------------------------------------------------------------------


class TestAbstention:
    async def test_abstained_response_is_graceful(self, ctx, pool):
        """When synthesize_digest returns abstained: no crash, no cache write, graceful response."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Large memory content for abstained path")

        abstain_result = SynthesisResult(
            status="abstained",
            memory_count=500,
            projected_cost_usd=0.15,  # exceeds cap
            content=None,
            provenance=None,
            cost_usd=None,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=abstain_result
        ):
            result = await weft_status(ctx, topic=tag, synthesize=True)

        # Must not crash and digest must be null
        assert "error" not in result or result.get("degraded") is None, (
            "abstained path must not return an error response"
        )
        assert result.get("digest") is None, (
            "abstained path must return digest=null (no synthesis)"
        )
        assert result.get("synthesis_status") == "abstained"

        # Tier-1 memories must still be present
        assert "memories" in result
        assert len(result["memories"]) >= 1

    async def test_abstained_records_cost_entry_with_zero_cost(self, ctx, pool):
        """Abstained path records ONE cost_entries row with estimated_cost_usd==0.0."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Memory that triggers abstention")

        projected_cost = 0.12
        memory_count = 750

        abstain_result = SynthesisResult(
            status="abstained",
            memory_count=memory_count,
            projected_cost_usd=projected_cost,
            content=None,
            provenance=None,
            cost_usd=None,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=abstain_result
        ):
            await weft_status(ctx, topic=tag, synthesize=True)

        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 1, f"Expected 1 cost_entries row for abstain, got {len(rows)}"

        row = rows[0]
        assert float(row["estimated_cost_usd"]) == pytest.approx(0.0), (
            f"Abstained cost must be 0.0, got {row['estimated_cost_usd']}"
        )
        assert row["entry_type"] == "topic_synthesis"
        # weft-49bd0550: the abstain row is attributed to the caller, not NULL.
        assert row["user_id"] == user_id

    async def test_abstained_cost_entry_has_correct_metadata(self, ctx, pool):
        """Abstained cost_entries row carries metadata.abstained=True + projected + memory_count."""
        from weft.mcp.tools import weft_status
        import json

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "Memory for metadata test")

        projected = 0.18
        mem_count = 900

        abstain_result = SynthesisResult(
            status="abstained",
            memory_count=mem_count,
            projected_cost_usd=projected,
            content=None,
            provenance=None,
            cost_usd=None,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=abstain_result
        ):
            await weft_status(ctx, topic=tag, synthesize=True)

        rows = await _count_cost_entries(pool, "topic_synthesis", tag)
        assert len(rows) == 1

        raw_meta = rows[0]["metadata"]
        if isinstance(raw_meta, str):
            meta = json.loads(raw_meta)
        else:
            meta = dict(raw_meta)

        assert meta.get("abstained") is True, (
            f"metadata.abstained must be True, got: {meta}"
        )
        assert "projected" in meta, f"metadata must carry 'projected', got: {meta}"
        assert meta["projected"] == pytest.approx(projected)
        assert "memory_count" in meta, f"metadata must carry 'memory_count', got: {meta}"
        assert meta["memory_count"] == mem_count

    async def test_abstained_no_cache_write(self, ctx, pool):
        """Abstained path does NOT write a digest to the cache."""
        from weft.mcp.tools import weft_status
        from weft.topic_digest_cache import read_digest

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag)

        abstain_result = SynthesisResult(
            status="abstained",
            memory_count=1000,
            projected_cost_usd=0.20,
            content=None,
            provenance=None,
            cost_usd=None,
        )

        with patch(
            "weft.mcp.tools.resolve_caller_user_id", return_value=user_id
        ), patch(
            "weft.views.topic_synthesis.synthesize_digest", new_callable=AsyncMock, return_value=abstain_result
        ):
            await weft_status(ctx, topic=tag, synthesize=True)

        # Confirm no digest was written
        tok = current_user_id.set(user_id)
        try:
            async with acquire(pool):
                cached = await read_digest(pool, user_id=user_id, topic=tag, scope="global")
        finally:
            current_user_id.reset(tok)

        assert cached is None, (
            "Abstained path must NOT write a digest to the cache"
        )


# ---------------------------------------------------------------------------
# (6) weft_status logs a recall_query row carrying was_empty (loom-8bfddc55)
# ---------------------------------------------------------------------------


class TestRecallQueryLog:
    async def test_empty_topic_logs_one_row_was_empty(self, ctx, pool):
        """A weft_status ask on a topic with NO memories writes exactly ONE
        recall_query row with result_count == 0 (the was_empty signal)."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        # Unique topic that resolves to nothing for this user.
        tag = f"empty-{uuid.uuid4().hex[:8]}"

        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            result = await weft_status(ctx, topic=tag, synthesize=False)

        assert result["memories"] == []

        rows = await _recall_query_rows(pool, "status", tag)
        assert len(rows) == 1, f"expected exactly 1 recall_query row, got {len(rows)}"
        row = rows[0]
        assert row["tool_name"] == "status"
        assert row["result_count"] == 0, "was_empty signal: result_count must be 0"
        # Attributed to the caller, not NULL (RLS user context was applied).
        assert row["user_id"] == user_id

    async def test_nonempty_topic_logs_row_with_count(self, ctx, pool):
        """A weft_status ask on a topic WITH memories writes a recall_query row
        whose result_count reflects the gathered count (was_empty == False)."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "one")
        await _seed_memory(pool, user_id, tag, "two")

        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            await weft_status(ctx, topic=tag, synthesize=False)

        rows = await _recall_query_rows(pool, "status", tag)
        assert len(rows) == 1
        assert rows[0]["result_count"] == 2
        assert rows[0]["result_count"] != 0  # was_empty == False

    async def test_each_call_logs_exactly_one_row(self, ctx, pool):
        """Two weft_status asks on the same topic produce two rows (one per call)."""
        from weft.mcp.tools import weft_status

        user_id = _uid()
        tag = f"tag-{uuid.uuid4().hex[:8]}"
        await _seed_memory(pool, user_id, tag, "content")

        with patch("weft.mcp.tools.resolve_caller_user_id", return_value=user_id):
            await weft_status(ctx, topic=tag, synthesize=False)
            await weft_status(ctx, topic=tag, synthesize=False)

        rows = await _recall_query_rows(pool, "status", tag)
        assert len(rows) == 2
