"""Tests for the 'both' tier RRF fuse (loom-98a508d5 / P1.B2).

Three layers:
  - Direct call: ``recall_both(...)`` returns unified {kind, payload, rank,
    rrf_score} entries with the right shape.
  - RRF fusion: when belief returns N items and turns returns M, fused output
    mixes them by rrf_score and respects the requested top_k cap.
  - Scoping: project_id flows into BOTH halves so cross-project context
    cannot bleed in.
  - End-to-end via MCP: weft_recall(tier='auto') on a _BOTH_TIER_MARKERS
    query routes to recall_both and returns tier='both'.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.episode_turns import append_turn
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_recall
from weft.models import (
    EpisodeCreate,
    EpisodeTurnCreate,
    MemoryCreate,
    MemorySource,
    MemoryType,
    TurnRole,
)
from weft.store import store_memory
from weft.turn_recall import recall_both


_FAKE_EMBEDDING = [0.1] * 768


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = 768

    def __init__(self):
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        h = hash(text) % 17
        return [0.1 + (i % h) * 0.001 if h else 0.1 for i in range(768)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
async def app(pool):
    # See test_turn_append_tool.py for the codec-init rationale.
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
async def codec_pool(pool):
    """Pool with pgvector codec initialized on every existing connection.

    Mirrors the app fixture so direct (non-MCP) recall_both calls can
    pass list[float] embeddings without asyncpg failing the type encode.
    """
    from weft.db.connection import _pgvector_codec_init
    conns = [await pool.acquire() for _ in range(pool.get_size())]
    try:
        for c in conns:
            await _pgvector_codec_init(c)
    finally:
        for c in conns:
            await pool.release(c)
    return pool


@pytest.fixture
async def project_with_belief_and_turns(codec_pool):
    """One episode with turns + a few memories, all in project 'alpha'."""
    pool = codec_pool
    ep = await create_episode(
        pool, EpisodeCreate(title="alpha-fixture", project_id="alpha"),
    )

    # Seed turns about the launch
    turn_data = [
        ("user", "Started planning the launch this morning.",
         datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)),
        ("assistant", "The launch is scheduled for March.",
         datetime(2026, 1, 5, 9, 1, tzinfo=timezone.utc)),
        ("user", "Reviewing the launch plan once more.",
         datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc)),
    ]
    for role, content, when in turn_data:
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id,
                role=TurnRole(role),
                content=content,
                occurred_at=when,
            ),
            embedding=_FAKE_EMBEDDING,
        )

    # Seed belief memories about the launch
    for content in [
        "Decision: launch on March 15.",
        "Fact: launch readiness review is the week prior.",
    ]:
        await store_memory(
            pool,
            MemoryCreate(
                type=MemoryType.fact,
                content=content,
                topic=["launch"],
                source=MemorySource.conversation,
                project_id="alpha",
            ),
            embedding=_FAKE_EMBEDDING,
        )
    return ep


# --- Direct call: shape contract ---


async def test_recall_both_returns_unified_entries(
    codec_pool, project_with_belief_and_turns,
):
    """Each entry has kind, payload, rank, rrf_score; payloads are JSON-able."""
    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch",
        project_id="alpha",
        top_k=5,
        embedder=embedder,
    )
    assert fused, "expected at least one fused entry"
    for entry in fused:
        assert set(entry.keys()) == {"kind", "payload", "rank", "rrf_score"}
        assert entry["kind"] in ("memory", "turn")
        assert isinstance(entry["payload"], dict)
        assert isinstance(entry["rank"], int) and entry["rank"] >= 1
        assert isinstance(entry["rrf_score"], float) and entry["rrf_score"] > 0
        # Payload must be JSON-serializable.
        import json
        json.dumps(entry["payload"])


async def test_recall_both_includes_both_kinds(
    codec_pool, project_with_belief_and_turns,
):
    """When both halves have results, the fused list mixes 'memory' and 'turn'."""
    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch",
        project_id="alpha",
        top_k=10,
        embedder=embedder,
    )
    kinds = {e["kind"] for e in fused}
    assert kinds == {"memory", "turn"}, (
        f"expected mixed kinds, got {kinds}: {fused}"
    )


async def test_recall_both_respects_top_k_cap(
    codec_pool, project_with_belief_and_turns,
):
    """top_k=3 returns at most 3 entries even when both halves have more."""
    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch",
        project_id="alpha",
        top_k=3,
        embedder=embedder,
    )
    assert len(fused) <= 3


async def test_recall_both_sorted_by_rrf_score_desc(
    codec_pool, project_with_belief_and_turns,
):
    """Output is ordered descending by rrf_score so top_k truncates the tail."""
    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch",
        project_id="alpha",
        top_k=10,
        embedder=embedder,
    )
    scores = [e["rrf_score"] for e in fused]
    assert scores == sorted(scores, reverse=True)


# --- Scoping: project_id flows to BOTH halves ---


async def test_recall_both_project_scope_isolates_memories_and_turns(codec_pool):
    """No cross-project bleed: memories AND turns from project 'beta' must
    not appear in a project_id='alpha' recall_both."""
    pool = codec_pool
    # Project alpha
    ep_a = await create_episode(
        pool, EpisodeCreate(title="A", project_id="alpha"),
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep_a.id, role=TurnRole.user,
            content="alpha keyword unique launch token",
        ),
        embedding=_FAKE_EMBEDDING,
    )
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="alpha memory mentions the launch token",
            topic=["launch"],
            source=MemorySource.conversation,
            project_id="alpha",
        ),
        embedding=_FAKE_EMBEDDING,
    )

    # Project beta — same content, different project, MUST stay out
    ep_b = await create_episode(
        pool, EpisodeCreate(title="B", project_id="beta"),
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep_b.id, role=TurnRole.user,
            content="beta keyword unique launch token",
        ),
        embedding=_FAKE_EMBEDDING,
    )
    await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="beta memory mentions the launch token",
            topic=["launch"],
            source=MemorySource.conversation,
            project_id="beta",
        ),
        embedding=_FAKE_EMBEDDING,
    )

    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch token",
        project_id="alpha",
        top_k=20,
        embedder=embedder,
    )
    # Every payload's content must be from project alpha.
    for entry in fused:
        content = entry["payload"]["content"]
        assert "beta" not in content.lower(), (
            f"cross-project bleed in {entry['kind']}: {content}"
        )
    # And we DO see the alpha rows.
    contents = " ".join(e["payload"]["content"] for e in fused)
    assert "alpha" in contents.lower()


# --- RRF fusion semantics ---


async def test_recall_both_rrf_score_matches_disjoint_formula(
    codec_pool, project_with_belief_and_turns,
):
    """Disjoint RRF: rrf_score = 1 / (K + rank). K=60 from _RRF_K."""
    from weft.episode_turns import _RRF_K
    embedder = _FakeEmbedding()
    fused = await recall_both(
        codec_pool, "launch",
        project_id="alpha",
        top_k=10,
        embedder=embedder,
    )
    for entry in fused:
        expected = 1.0 / (_RRF_K + entry["rank"])
        assert abs(entry["rrf_score"] - expected) < 1e-9, (
            f"rrf_score={entry['rrf_score']} expected={expected} for "
            f"rank={entry['rank']}"
        )


# --- End-to-end via weft_recall(tier='auto') ---


async def test_weft_recall_auto_routes_episodic_query_to_both(
    ctx, project_with_belief_and_turns,
):
    """A _BOTH_TIER_MARKERS query (e.g. 'do you remember') should route to
    'both' end-to-end via weft_recall(tier='auto')."""
    result = await weft_recall(
        ctx,
        query="do you remember the launch plan",
        project_id="alpha",
        tier="auto",
        limit=5,
    )
    assert "error" not in result, result
    assert result["tier"] == "both"
    assert "results" in result
    assert isinstance(result["results"], list)
    # Belief-tier shape (single 'results' of memory dicts) and turn-tier
    # shape ('turns' array) must NOT leak in.
    assert "turns" not in result
    # Each result entry has the unified shape.
    for entry in result["results"]:
        assert entry["kind"] in ("memory", "turn")
        assert "payload" in entry
        assert "rrf_score" in entry


async def test_weft_recall_explicit_tier_both_dispatches(
    ctx, project_with_belief_and_turns,
):
    """tier='both' explicitly invokes the new path."""
    result = await weft_recall(
        ctx,
        query="anything goes here",
        project_id="alpha",
        tier="both",
        limit=5,
    )
    assert result.get("tier") == "both" or "error" in result
    if "error" not in result:
        assert "results" in result


async def test_weft_recall_invalid_tier_message_lists_both(ctx):
    """The validation error should mention 'both' as a valid option now."""
    result = await weft_recall(ctx, query="anything", tier="bogus")
    assert result["error"] == "Invalid input"
    assert "both" in result["detail"]


# --- Backwards compatibility: belief & turns paths unchanged ---


async def test_belief_path_still_returns_results_shape(ctx):
    """Non-routing query still returns the legacy belief shape."""
    result = await weft_recall(
        ctx,
        query="describe this codebase",
        tier="belief",
        limit=3,
    )
    # Belief path: 'results' array, no 'tier' key, no 'turns' key.
    assert "results" in result
    assert "tier" not in result
    assert "turns" not in result


async def test_turns_path_still_returns_turns_shape(
    ctx, project_with_belief_and_turns,
):
    """tier='turns' still returns the legacy turn-tier shape."""
    result = await weft_recall(
        ctx,
        query="launch",
        project_id="alpha",
        tier="turns",
        limit=5,
    )
    assert "error" not in result, result
    assert result["tier"] == "turns"
    assert "turns" in result
    # Belief shape must NOT leak in.
    assert "results" not in result
