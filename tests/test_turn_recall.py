"""Tests for turn-tier recall (loom-d9ac7e18).

Three layers:
  - store layer: ``recall_turns`` hybrid scoring + filters + project scope
  - planner: ``route_query_to_tier`` + ``extract_anchors`` (pure regex,
    no DB, no fixtures needed)
  - MCP tool: ``weft_recall(tier='turns')`` happy path + auto-routing
    on representative LongMemEval failure-pattern queries
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.episode_turns import (
    append_turn,
    list_recent_turns,
    recall_turns,
)
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_recall
from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole
from weft.turn_recall import (
    extract_anchors,
    route_query_to_tier,
    temporal_anchor,
)


# --- Test fixtures ---


_FAKE_EMBEDDING = [0.1] * 768


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = 768

    def __init__(self):
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Deterministic per-text variation so different anchors don't
        # collapse to identical vectors. Hash → sign-folded to stay in
        # the unit cube.
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
async def episode_with_turns(pool):
    """Episode with five turns covering three topical anchors so recall
    can distinguish them."""
    ep = await create_episode(pool, EpisodeCreate(title="recall-fixture"))

    turns_data = [
        ("user", "Started planning the Q2 product launch this morning.",
         datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)),
        ("assistant", "Noted. The launch is scheduled for March.",
         datetime(2026, 1, 5, 9, 1, tzinfo=timezone.utc)),
        ("user", "Just finished the customer demo with Acme Corp.",
         datetime(2026, 1, 12, 14, 0, tzinfo=timezone.utc)),
        ("assistant", "Demo went well. Followups due next week.",
         datetime(2026, 1, 12, 14, 5, tzinfo=timezone.utc)),
        ("user", "Internal retrospective on the launch was today, learned a lot.",
         datetime(2026, 4, 2, 16, 0, tzinfo=timezone.utc)),
    ]
    embedding = [0.1] * 768
    for role, content, when in turns_data:
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep.id,
                role=TurnRole(role),
                content=content,
                occurred_at=when,
            ),
            embedding=embedding,
        )
    return ep


# --- Planner: pure regex, no DB ---


class TestRouteQueryToTier:
    def test_temporal_markers_route_to_turns(self):
        for q in [
            "how many days between A and B",
            "how long since I last shipped",
            "when did the launch happen",
            "what happened before the demo",
            "what date was the retro",
        ]:
            assert route_query_to_tier(q) == "turns", f"failed for: {q}"

    def test_semantic_queries_route_to_belief(self):
        for q in [
            "describe the project architecture",
            "what does SSL mean",
            "tell me about Bob",
            "summarize the recent changes",
        ]:
            assert route_query_to_tier(q) == "belief", f"failed for: {q}"

    def test_case_insensitive(self):
        assert route_query_to_tier("HOW MANY DAYS BETWEEN X AND Y") == "turns"
        assert route_query_to_tier("Describe Architecture") == "belief"

    def test_episodic_recall_routes_to_both(self):
        # Explicit episodic asks fuse the canonical fact (belief) with
        # the dialogue evidence (turns) via RRF — return 'both'.
        for q in [
            "do you remember the demo",
            "did we discuss the rollout plan",
            "have we discussed the new API",
            "have I mentioned the migration",
            "what did I say about the retro",
            "what did I last say about the launch",
            "what did I decide about scope",
            "my decision on the new architecture",
            "our position on the deprecation",
            "my opinion about the design",
            "remind me about the customer call",
            "remind me when the demo is scheduled",
        ]:
            assert route_query_to_tier(q) == "both", f"failed for: {q}"

    def test_both_takes_priority_over_turns(self):
        # Ordering invariant: queries that hit both _BOTH_TIER_MARKERS
        # and _TURN_TIER_MARKERS resolve to 'both', not 'turns'.
        # "do you recall when did" matches both `do you ... recall`
        # (BOTH) and `when did` (TURN); the BOTH list wins.
        assert (
            route_query_to_tier("do you recall when did the launch ship")
            == "both"
        )
        assert (
            route_query_to_tier("have we discussed before the demo")
            == "both"
        )

    def test_both_markers_case_insensitive(self):
        assert route_query_to_tier("Do You Remember The Demo") == "both"
        assert (
            route_query_to_tier("WHAT DID I LAST SAY ABOUT THE LAUNCH")
            == "both"
        )


class TestExtractAnchors:
    def test_between_x_and_y(self):
        anchors = extract_anchors("how many days between the launch and the demo?")
        assert anchors == ["the launch", "the demo"]

    def test_from_x_to_y(self):
        anchors = extract_anchors("from January to March, what shipped?")
        assert anchors == ["January", "March"]

    def test_after_x_but_before_y(self):
        anchors = extract_anchors(
            "what happened after the kickoff but before the demo?"
        )
        assert anchors == ["the kickoff", "the demo"]

    def test_single_anchor_returns_empty(self):
        # Single-temporal-marker queries don't trigger multi-anchor split.
        assert extract_anchors("when did I last commit") == []
        assert extract_anchors("how many days since launch") == []

    def test_no_temporal_pattern_returns_empty(self):
        assert extract_anchors("describe the architecture") == []


# --- Store: hybrid recall_turns ---


async def test_recall_turns_keyword_hits(pool, episode_with_turns):
    """BM25 half should surface turns whose content matches the query
    even when no embedding is supplied."""
    results = await recall_turns(
        pool, "demo Acme",
        project_id=episode_with_turns.project_id,
        top_k=5,
    )
    assert results, "expected at least one keyword hit"
    # The Acme demo turn should rank first.
    assert "Acme Corp" in results[0].content


async def test_recall_turns_filters_by_time_range(pool, episode_with_turns):
    """``since`` / ``until`` are SQL filters applied before scoring."""
    results = await recall_turns(
        pool, "launch",
        project_id=episode_with_turns.project_id,
        since=datetime(2026, 4, 1, tzinfo=timezone.utc),
        until=datetime(2026, 4, 30, tzinfo=timezone.utc),
        top_k=10,
    )
    # Only the April 2 retrospective turn matches both the keyword and
    # the time window.
    assert len(results) == 1
    assert "retrospective" in results[0].content


async def test_recall_turns_project_scope_isolates(pool):
    """A turn in a different project must not leak into a project-scoped
    recall."""
    ep_a = await create_episode(pool, EpisodeCreate(title="A", project_id="proj-a"))
    ep_b = await create_episode(pool, EpisodeCreate(title="B", project_id="proj-b"))
    embedding = [0.1] * 768
    await append_turn(
        pool,
        EpisodeTurnCreate(episode_id=ep_a.id, role=TurnRole.user,
                          content="alpha keyword unique"),
        embedding=embedding,
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(episode_id=ep_b.id, role=TurnRole.user,
                          content="alpha keyword unique"),
        embedding=embedding,
    )

    a_only = await recall_turns(pool, "alpha", project_id="proj-a", top_k=5)
    assert len(a_only) == 1
    assert a_only[0].episode_id == ep_a.id


async def test_recall_turns_no_embedding_falls_back_to_keyword(pool, episode_with_turns):
    """Skipping the embedding arg should still return BM25 results."""
    results = await recall_turns(
        pool, "demo",
        project_id=episode_with_turns.project_id,
        top_k=5,
        embedding=None,
    )
    assert any("demo" in r.content.lower() for r in results)


async def test_list_recent_turns_orders_descending(pool, episode_with_turns):
    """The fallback recent-turns helper orders by occurred_at DESC."""
    results = await list_recent_turns(
        pool, project_id=episode_with_turns.project_id, limit=10,
    )
    times = [t.occurred_at for t in results]
    assert times == sorted(times, reverse=True)


# --- temporal_anchor ---


async def test_temporal_anchor_splits_multi_anchor_query(pool, episode_with_turns):
    embedder = _FakeEmbedding()
    result = await temporal_anchor(
        pool,
        "how many days between the launch and the demo?",
        project_id=episode_with_turns.project_id,
        top_k_per_anchor=3,
        embedder=embedder,
    )
    assert set(result.keys()) == {"the launch", "the demo"}
    # Each anchor should pull at least one matching turn.
    assert any("launch" in t.content.lower() for t in result["the launch"])
    assert any("demo" in t.content.lower() for t in result["the demo"])


async def test_temporal_anchor_no_split_keys_under_query(pool, episode_with_turns):
    """When no multi-anchor pattern fires, the dict has the original
    query as its sole key — caller never has to special-case empty."""
    embedder = _FakeEmbedding()
    result = await temporal_anchor(
        pool,
        "when did I last commit",
        project_id=episode_with_turns.project_id,
        embedder=embedder,
    )
    assert list(result.keys()) == ["when did I last commit"]


# --- MCP tool: weft_recall(tier=...) ---


async def test_weft_recall_tier_turns_returns_turns_array(ctx, episode_with_turns):
    result = await weft_recall(
        ctx,
        query="Acme demo",
        project_id=episode_with_turns.project_id,
        tier="turns",
        limit=5,
    )
    assert "error" not in result, result
    assert result["tier"] == "turns"
    assert "turns" in result
    assert "results" not in result  # belief-tier shape must NOT leak in
    assert any("Acme" in t["content"] for t in result["turns"])


async def test_weft_recall_tier_auto_routes_temporal_to_turns(
    ctx, episode_with_turns,
):
    result = await weft_recall(
        ctx,
        query="how many days between the launch and the demo?",
        project_id=episode_with_turns.project_id,
        tier="auto",
        limit=10,
    )
    assert result["tier"] == "turns"
    # Multi-anchor queries surface the per-anchor mapping.
    assert "anchors" in result
    assert set(result["anchors"].keys()) == {"the launch", "the demo"}


async def test_weft_recall_tier_auto_routes_semantic_to_belief(ctx):
    """A non-temporal query must NOT short-circuit to turns even with
    tier='auto' — it should fall through to the belief-tier flow."""
    result = await weft_recall(
        ctx,
        query="describe the project architecture",
        tier="auto",
        limit=5,
    )
    # Belief-tier response has 'results' (memories), no 'tier' field, no
    # 'turns' field.
    assert "tier" not in result
    assert "turns" not in result
    assert "results" in result


async def test_weft_recall_tier_belief_explicit_skips_router(
    ctx, episode_with_turns,
):
    """tier='belief' must use the belief path even when the query has
    temporal markers — the user is explicitly overriding."""
    result = await weft_recall(
        ctx,
        query="how many days between launch and demo",
        tier="belief",
        limit=5,
    )
    assert "results" in result
    assert "turns" not in result


async def test_weft_recall_invalid_tier_returns_input_error(ctx):
    result = await weft_recall(ctx, query="anything", tier="bogus")
    assert result["error"] == "Invalid input"
    assert "tier must be" in result["detail"]


async def test_weft_recall_tier_default_auto_is_backwards_compatible(ctx):
    """Existing callers that don't pass tier= must keep getting belief-
    tier behavior on non-temporal queries (default tier='auto')."""
    result = await weft_recall(ctx, query="describe this codebase", limit=3)
    assert "results" in result
    assert "turns" not in result
