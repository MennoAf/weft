"""Binding acceptance/dogfood gate for Topic-Digest Recall (loom-54d9a1e5).

Proves weft_status answers "what's the status of <topic>?" over real memory:

  1. Tier-1 completeness (no LLM): weft_status(topic, synthesize=False) returns
     EVERY active memory carrying a resolved topic tag — count == a baseline
     SELECT over the same predicate, and > 0 (non-degenerate). PRD §V1.
  2. Tier-2 synthesis (LIVE Haiku): weft_status(topic, synthesize=True) returns
     non-empty narrative content, provenance cites >=1 real gathered memory id,
     and the recorded cost is > 0 and <= MAX_SYNTH_COST_PER_CALL_USD. PRD §V5.
  3. L1 Resolution Ratchet (no LLM): tokens whose naive normalization misses
     come back was_empty; after record_alias they resolve via the alias path,
     the topic_resolution.alias_hits counter advances once per alias hit, and a
     control token still resolves via naive normalization. PRD §Compounding.

Test substrate is a real Postgres (testcontainers `pool` fixture) seeded with a
synthetic 'weft' corpus — the repo's "live DB" pattern. The Tier-2 case makes a
REAL Haiku call and is gated behind WEFT_RUN_LLM_EVAL=1 + ANTHROPIC_API_KEY,
mirroring weft.views.belief_detector's gated integration test.

Synthetic persona: Jim Boblaw (never real names from conversation).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.cost_tracking import CostEntryType
from weft.counters import get_counter
from weft.db.connection import acquire
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_status
from weft.models import MemoryCreate, MemoryType
from weft.store import store_memory
from weft.topic_resolution import (
    COUNTER_TOPIC_RESOLUTION_ALIAS_HITS,
    _normalize_token,
    record_alias,
)
from weft.views.topic_synthesis import MAX_SYNTH_COST_PER_CALL_USD

_TEST_USER = "acceptance-topic-digest-user"

_LIVE = os.getenv("WEFT_RUN_LLM_EVAL") == "1" and bool(os.getenv("ANTHROPIC_API_KEY"))
_LIVE_REASON = "live Haiku gate off (set WEFT_RUN_LLM_EVAL=1 + ANTHROPIC_API_KEY)"


# ---------------------------------------------------------------------------
# Fixtures — real DB pool + fake embedding + NullCache, caller == _TEST_USER.
# ---------------------------------------------------------------------------


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


@pytest.fixture
def ctx(pool, monkeypatch):
    # weft_status resolves the caller via get_user_id() → WEFT_USER_ID env.
    monkeypatch.setenv("WEFT_USER_ID", _TEST_USER)
    app = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbedding(),
        config=WeftConfig(),
    )
    c = MagicMock()
    c.request_context.lifespan_context = app
    c.list_roots = AsyncMock(return_value=[])
    return c


async def _seed(pool, content: str, topics: list[str]) -> str:
    """Store an active memory under _TEST_USER with the given topic tags."""
    tok = current_user_id.set(_TEST_USER)
    try:
        async with acquire(pool):
            mem = await store_memory(
                pool,
                MemoryCreate(type=MemoryType.fact, content=content, topic=topics),
            )
    finally:
        current_user_id.reset(tok)
    return mem.id


# ---------------------------------------------------------------------------
# (1) Tier-1 completeness — no LLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier1_complete_gather_matches_baseline(ctx, pool):
    seeded = [
        await _seed(pool, "Jim Boblaw shipped the weft facet-recall epic", ["weft"]),
        await _seed(pool, "Jim Boblaw deployed weft to fly", ["weft", "deploy"]),
        await _seed(pool, "Weft uses pgvector for embeddings", ["weft"]),
        await _seed(pool, "The weft entity was mentioned in review", ["entity:weft"]),
    ]
    # A memory under an unrelated topic must NOT be gathered.
    await _seed(pool, "Jim Boblaw likes loom orchestration", ["loom"])

    baseline = await pool.fetchval(
        """
        SELECT count(*) FROM memories
        WHERE status = 'active'
          AND ('weft' = ANY(topic) OR 'entity:weft' = ANY(topic))
          AND user_id = $1
        """,
        _TEST_USER,
    )

    resp = await weft_status(ctx, topic="weft", synthesize=False)

    assert baseline == len(seeded) > 0
    assert len(resp["memories"]) == baseline, (
        f"Tier-1 must be complete: tool returned {len(resp['memories'])}, "
        f"baseline SELECT counted {baseline}"
    )
    assert resp["complete"] is True
    assert resp["truncated"] is False
    assert resp["digest"] is None  # synthesize=False
    returned_ids = {m["id"] for m in resp["memories"]}
    assert returned_ids == set(seeded)


# ---------------------------------------------------------------------------
# (3) L1 Resolution Ratchet — no LLM.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l1_resolution_ratchet(ctx, pool):
    # A real tag with content that the alias will point at.
    await _seed(pool, "Jim Boblaw's canonical weft notes", ["weft"])

    # 5 tokens whose naive normalization (lower / entity:lower) hits nothing.
    miss_tokens = ["WeftDB", "WeftMCP", "WeftHTTP", "WeftRLS", "WeftGIN"]

    # Before aliasing: each resolves to nothing → was_empty (no memories).
    for tok in miss_tokens:
        resp = await weft_status(ctx, topic=tok, synthesize=False)
        assert len(resp["memories"]) == 0, f"{tok} should miss before aliasing"

    # Record an alias mapping each token to the real 'weft' tag.
    for tok in miss_tokens:
        await record_alias(_normalize_token(tok), ["weft"], "manual", _TEST_USER, pool)

    before = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)

    # After aliasing: each resolves via the alias path → non-empty.
    for tok in miss_tokens:
        resp = await weft_status(ctx, topic=tok, synthesize=False)
        assert len(resp["memories"]) > 0, f"{tok} should resolve after aliasing"

    after = await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS)
    assert after - before == len(miss_tokens), (
        f"alias_hits must advance once per alias hit: {after - before} != {len(miss_tokens)}"
    )

    # Control: a 6th token resolves via NAIVE normalization (no alias row),
    # and does not touch the alias counter.
    control = await weft_status(ctx, topic="weft", synthesize=False)
    assert len(control["memories"]) > 0
    assert await get_counter(pool, COUNTER_TOPIC_RESOLUTION_ALIAS_HITS) == after


# ---------------------------------------------------------------------------
# (2) Tier-2 synthesis — LIVE Haiku (gated).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _LIVE, reason=_LIVE_REASON)
async def test_tier2_live_haiku_synthesis(ctx, pool):
    seeded = {
        await _seed(pool, "Jim Boblaw shipped facet-based recall to production", ["weft"]),
        await _seed(pool, "Weft runs as an MCP server on fly.io", ["weft"]),
        await _seed(pool, "Weft's memory substrate is Postgres + pgvector", ["weft"]),
    }

    resp = await weft_status(ctx, topic="weft", synthesize=True)

    digest = resp["digest"]
    assert digest is not None, f"expected a synthesized digest, got {resp.get('synthesis_status')}"
    assert isinstance(digest["content"], str) and digest["content"].strip()

    # Provenance is {memory_id: [spans]} keyed ONLY on real gathered ids.
    cited = set(digest["provenance"].keys())
    assert cited & seeded, f"provenance must cite >=1 gathered memory id; cited={cited}"

    # Recorded cost: 0 < cost <= cap.
    cost = await pool.fetchval(
        """
        SELECT estimated_cost_usd FROM cost_entries
        WHERE entry_type = $1 AND reference_id = 'weft'
        ORDER BY created_at DESC LIMIT 1
        """,
        CostEntryType.topic_synthesis.value,
    )
    assert cost is not None, "a topic_synthesis cost row must be recorded"
    assert 0 < cost <= MAX_SYNTH_COST_PER_CALL_USD, (
        f"recorded cost {cost} must be in (0, {MAX_SYNTH_COST_PER_CALL_USD}]"
    )
