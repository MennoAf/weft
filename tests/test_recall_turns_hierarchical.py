"""Tests for hierarchical retrieval descent (loom-0a4e6852, P2.4).

Two layers:

* Direct: ``recall_turns_hierarchical(...)`` ranks episodes first, then
  fans out to ``recall_turns`` scoped to those episode_ids. The
  ``episode_ids`` filter on ``recall_turns`` itself is exercised here.
* Dispatch: ``WEFT_HIERARCHICAL=1`` swaps the flat ``recall_turns`` for
  the hierarchical descent in ``_weft_recall_turns`` (the MCP path) and
  in ``recall_both`` (the both-tier RRF fuse).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.episode_turns import (
    append_turn,
    recall_turns,
    recall_turns_hierarchical,
)
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_recall
from weft.models import EpisodeCreate, EpisodeTurnCreate, TurnRole


# --- Fixtures ---

# Vector dimensions match the v47 episode embedding column.
_DIM = 768


def _vec(seed: float) -> list[float]:
    """Deterministic vector with strong directional signal at index 0.

    Matches the helper in ``test_recall_episodes.py`` so cosine ordering
    is consistent across the two test modules — a vector close to
    ``_vec(0.99)`` ranks high against ``_vec(0.99)`` and low against
    ``_vec(-0.99)``.
    """
    base = [0.0] * _DIM
    base[0] = seed
    base[1] = 1.0 - abs(seed)
    return base


_QUERY_LAUNCH = _vec(0.99)
_QUERY_BUDGET = _vec(-0.99)


class _FakeEmbedding:
    provider_name = "fake"
    dimensions = _DIM

    def __init__(self, vec: list[float] | None = None):
        self.calls: list[str] = []
        # Default: produces the launch-aligned vector. Tests that need
        # something else can pass a different vector.
        self._vec = vec if vec is not None else _vec(0.99)

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vec)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
async def codec_pool(pool):
    """Pool with pgvector codec on every existing connection.

    Direct (non-MCP) calls into recall_turns_hierarchical pass list[float]
    embeddings; without the codec init, asyncpg fails the type encode.
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
async def app(codec_pool):
    return AppContext(
        pool=codec_pool,
        cache=NullCache(),
        embedding=_FakeEmbedding(_vec(0.99)),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


@pytest.fixture
async def two_topical_episodes(codec_pool):
    """Two episodes — one about the launch, one about the budget — each
    with two turns. Embeddings put them on opposite axes in cosine space
    so a query close to one vector cleanly favors that episode.

    Returns ``(launch_ep, budget_ep)``.
    """
    pool = codec_pool

    launch_ep = await create_episode(
        pool,
        EpisodeCreate(title="launch planning", summary="launch logistics"),
        embedding=_QUERY_LAUNCH,
    )
    budget_ep = await create_episode(
        pool,
        EpisodeCreate(title="quarterly budget", summary="budget review"),
        embedding=_QUERY_BUDGET,
    )

    # Launch turns
    for content in [
        "Started planning the launch this morning.",
        "The launch is scheduled for March 15.",
    ]:
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=launch_ep.id, role=TurnRole.user, content=content,
            ),
            embedding=_QUERY_LAUNCH,
        )

    # Budget turns — also contain the word 'launch' in passing so the
    # flat keyword half would surface them, but the hierarchical path
    # should NOT pull them in for a launch-aligned query because the
    # episode-tier vector ranking filters out the budget episode first.
    for content in [
        "Reviewing the launch costs against the budget.",
        "The budget for the launch line item is finalized.",
    ]:
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=budget_ep.id, role=TurnRole.user, content=content,
            ),
            embedding=_QUERY_BUDGET,
        )
    return launch_ep, budget_ep


# --- episode_ids filter on recall_turns directly ---


async def test_recall_turns_episode_ids_filter_scopes_results(
    codec_pool, two_topical_episodes,
):
    """A direct ``recall_turns(episode_ids=[...])`` call returns only
    turns whose episode_id is in the list."""
    launch_ep, _budget_ep = two_topical_episodes
    results = await recall_turns(
        codec_pool, "launch",
        episode_ids=[launch_ep.id],
        top_k=10,
    )
    assert results, "expected at least one turn for the launch episode"
    assert all(t.episode_id == launch_ep.id for t in results), (
        f"unexpected episode_id leak: {[t.episode_id for t in results]}"
    )


async def test_recall_turns_episode_ids_filter_empty_list_short_circuits(
    codec_pool, two_topical_episodes,
):
    """``episode_ids=[]`` returns ``[]`` without hitting the DB."""
    results = await recall_turns(
        codec_pool, "launch",
        episode_ids=[],
        top_k=10,
    )
    assert results == []


async def test_recall_turns_episode_ids_filter_none_preserves_legacy(
    codec_pool, two_topical_episodes,
):
    """``episode_ids=None`` is the default and matches pre-filter
    behavior — turns from BOTH episodes can surface."""
    results = await recall_turns(
        codec_pool, "launch",
        top_k=10,
    )
    episode_ids_seen = {t.episode_id for t in results}
    assert len(episode_ids_seen) >= 1
    # Both episodes have 'launch' in their turn content, so unfiltered
    # we expect to see both.
    launch_ep, budget_ep = two_topical_episodes
    assert launch_ep.id in episode_ids_seen
    assert budget_ep.id in episode_ids_seen


async def test_recall_turns_episode_ids_filter_with_vector_half(
    codec_pool, two_topical_episodes,
):
    """``episode_ids`` filter applies to BOTH halves of recall_turns —
    a vector-driven recall scoped to a single episode returns turns from
    only that episode."""
    launch_ep, _budget_ep = two_topical_episodes
    results = await recall_turns(
        codec_pool, "launch",
        episode_ids=[launch_ep.id],
        embedding=_QUERY_LAUNCH,
        top_k=10,
    )
    assert results
    assert all(t.episode_id == launch_ep.id for t in results)


# --- recall_turns_hierarchical happy path ---


async def test_recall_turns_hierarchical_descends_to_relevant_episode(
    codec_pool, two_topical_episodes,
):
    """A query whose embedding is close to one episode's vector should
    pull turns from that episode and exclude turns from off-topic
    episodes — even if those off-topic episodes contain the query
    keyword."""
    launch_ep, budget_ep = two_topical_episodes
    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    results = await recall_turns_hierarchical(
        codec_pool, "launch",
        embedder=embedder,
        top_k_episodes=1,  # Force descent into the single best episode.
        top_k_turns=10,
    )
    assert results, "expected hierarchical descent to surface launch turns"
    episode_ids_seen = {t.episode_id for t in results}
    assert launch_ep.id in episode_ids_seen
    assert budget_ep.id not in episode_ids_seen, (
        "budget episode leaked into hierarchical descent: "
        f"{episode_ids_seen}"
    )


async def test_recall_turns_hierarchical_multi_episode_fanout(
    codec_pool, two_topical_episodes,
):
    """When ``top_k_episodes`` is broad enough to admit both episodes,
    the descent fans out to turns across both."""
    launch_ep, budget_ep = two_topical_episodes
    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    results = await recall_turns_hierarchical(
        codec_pool, "launch",
        embedder=embedder,
        top_k_episodes=5,  # Both episodes are eligible.
        top_k_turns=20,
    )
    episode_ids_seen = {t.episode_id for t in results}
    assert launch_ep.id in episode_ids_seen
    assert budget_ep.id in episode_ids_seen


async def test_recall_turns_hierarchical_empty_episodes_falls_through(
    codec_pool,
):
    """When ``recall_episodes`` returns nothing, fall through to a flat
    ``recall_turns`` so the caller still gets something. We force the
    empty-episodes path by giving the episode no embedding and using a
    query that doesn't match its title — keyword half misses, vector
    half is skipped (no embedding), so recall_episodes returns []."""
    pool = codec_pool

    # Episode whose title/summary lexically misses the query, no embedding.
    ep = await create_episode(
        pool,
        EpisodeCreate(title="zzz", summary="zzz"),
        embedding=None,
    )
    # Turn whose content contains the query keyword so the flat
    # recall_turns fallback still surfaces it.
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="rare-fallback-token-9001 evidence",
        ),
        embedding=None,
    )

    # No embedding, no embedder → recall_episodes runs keyword-only,
    # which doesn't match 'zzz' against the rare keyword → empty.
    results = await recall_turns_hierarchical(
        codec_pool, "rare-fallback-token-9001",
        top_k_episodes=5,
        top_k_turns=10,
    )
    assert results, (
        "fallback: empty episode set should fall through to flat recall_turns"
    )
    assert any("rare-fallback-token-9001" in t.content for t in results)


async def test_recall_turns_hierarchical_embedder_unavailable_keyword_only(
    codec_pool, two_topical_episodes,
):
    """No ``embedding`` and no ``embedder`` — both halves degrade to
    keyword-only and the descent still returns turns."""
    launch_ep, _budget_ep = two_topical_episodes
    results = await recall_turns_hierarchical(
        codec_pool, "launch",
        top_k_episodes=5,
        top_k_turns=10,
    )
    # Keyword-only: both episodes contain 'launch' in title/summary or
    # turn content, so we expect at least the launch episode to surface.
    assert results
    episode_ids_seen = {t.episode_id for t in results}
    assert launch_ep.id in episode_ids_seen


# --- Scoping passthrough ---


async def test_recall_turns_hierarchical_project_scope_propagates(codec_pool):
    """``project_id`` flows into BOTH halves: the episode rank stays
    inside the project AND the turn fan-out stays inside the project.
    A turn in another project must not leak in even if its episode
    embedding ranks high."""
    pool = codec_pool

    in_proj_ep = await create_episode(
        pool,
        EpisodeCreate(title="alpha launch", project_id="proj-a"),
        embedding=_QUERY_LAUNCH,
    )
    other_proj_ep = await create_episode(
        pool,
        EpisodeCreate(title="alpha launch", project_id="proj-b"),
        embedding=_QUERY_LAUNCH,
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=in_proj_ep.id, role=TurnRole.user,
            content="proj-a launch content",
        ),
        embedding=_QUERY_LAUNCH,
    )
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=other_proj_ep.id, role=TurnRole.user,
            content="proj-b launch content",
        ),
        embedding=_QUERY_LAUNCH,
    )

    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    results = await recall_turns_hierarchical(
        codec_pool, "launch",
        embedder=embedder,
        project_id="proj-a",
        top_k_episodes=10,
        top_k_turns=10,
    )
    episode_ids_seen = {t.episode_id for t in results}
    assert in_proj_ep.id in episode_ids_seen
    assert other_proj_ep.id not in episode_ids_seen, (
        f"cross-project leak: {episode_ids_seen}"
    )


async def test_recall_turns_hierarchical_since_until_propagates(codec_pool):
    """``since`` / ``until`` flows into BOTH halves so a turn whose
    occurred_at is outside the window is dropped, AND an episode whose
    started_at is outside the window doesn't seed the descent."""
    pool = codec_pool

    # Episode in 2020 — outside the since window.
    old_ep = await create_episode(
        pool,
        EpisodeCreate(title="ancient launch"),
        embedding=_QUERY_LAUNCH,
    )
    await pool.execute(
        "UPDATE episodes SET started_at = $1 WHERE id = $2",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        old_ep.id,
    )

    # Episode in 2026 — inside the window.
    new_ep = await create_episode(
        pool,
        EpisodeCreate(title="recent launch"),
        embedding=_QUERY_LAUNCH,
    )
    # New episode's turns occurred recently.
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=new_ep.id, role=TurnRole.user,
            content="recent launch content",
            occurred_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        ),
        embedding=_QUERY_LAUNCH,
    )
    # Old episode's turns occurred long ago.
    await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=old_ep.id, role=TurnRole.user,
            content="ancient launch content",
            occurred_at=datetime(2020, 1, 5, tzinfo=timezone.utc),
        ),
        embedding=_QUERY_LAUNCH,
    )

    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    results = await recall_turns_hierarchical(
        codec_pool, "launch",
        embedder=embedder,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        top_k_episodes=10,
        top_k_turns=10,
    )
    contents = {t.content for t in results}
    assert "recent launch content" in contents
    assert "ancient launch content" not in contents


# --- Flag dispatch in the MCP path ---


async def test_flag_off_uses_flat_recall_turns(
    monkeypatch, ctx, codec_pool, two_topical_episodes,
):
    """Default (flag unset) — ``_weft_recall_turns`` flows through
    ``temporal_anchor`` (which calls flat ``recall_turns``). Both
    episodes' turns are eligible to surface for a query that hits both
    via keyword."""
    monkeypatch.delenv("WEFT_HIERARCHICAL", raising=False)

    result = await weft_recall(
        ctx,
        query="launch",
        tier="turns",
        limit=20,
    )
    assert "error" not in result, result
    assert result["tier"] == "turns"
    episode_ids_seen = {t["episode_id"] for t in result["turns"]}
    launch_ep, budget_ep = two_topical_episodes
    # Flat path: both episodes' turns are eligible (keyword match in
    # both episodes' turn content).
    assert launch_ep.id in episode_ids_seen
    assert budget_ep.id in episode_ids_seen


async def test_flag_on_routes_through_hierarchical(
    monkeypatch, ctx, codec_pool, two_topical_episodes,
):
    """``WEFT_HIERARCHICAL=1`` routes through hierarchical descent.

    With ``top_k_episodes`` defaulting to 10 the descent admits both
    episodes here, so we can't assert exclusion easily — instead we
    spy on ``recall_turns_hierarchical`` to confirm it was called.
    """
    monkeypatch.setenv("WEFT_HIERARCHICAL", "1")

    called: dict[str, int] = {"n": 0}

    from weft.episode_turns import (
        recall_turns_hierarchical as real_hier,
    )

    async def spy(*args, **kwargs):
        called["n"] += 1
        return await real_hier(*args, **kwargs)

    monkeypatch.setattr(
        "weft.mcp.tools.recall_turns_hierarchical",
        spy,
        raising=False,
    )
    # Also patch the import-bound reference inside the function via
    # the lazy ``from weft.episode_turns import recall_turns_hierarchical``
    # — that line resolves the symbol at call time from the
    # ``weft.episode_turns`` module, so patch there.
    monkeypatch.setattr(
        "weft.episode_turns.recall_turns_hierarchical",
        spy,
        raising=False,
    )

    result = await weft_recall(
        ctx,
        query="launch",
        tier="turns",
        limit=20,
    )
    assert "error" not in result, result
    assert result["tier"] == "turns"
    assert called["n"] == 1, (
        "expected hierarchical descent under WEFT_HIERARCHICAL=1, "
        f"got {called['n']} calls"
    )


async def test_flag_value_other_than_one_uses_flat(
    monkeypatch, ctx, codec_pool, two_topical_episodes,
):
    """Only the literal ASCII '1' enables hierarchical. 'true', 'yes',
    '0', '' — all keep the flat path."""
    for sentinel in ("0", "true", "yes", "", "TRUE"):
        monkeypatch.setenv("WEFT_HIERARCHICAL", sentinel)

        called: dict[str, int] = {"n": 0}
        from weft.episode_turns import (
            recall_turns_hierarchical as real_hier,
        )

        async def spy(*args, **kwargs):
            called["n"] += 1
            return await real_hier(*args, **kwargs)

        monkeypatch.setattr(
            "weft.episode_turns.recall_turns_hierarchical",
            spy,
            raising=False,
        )

        result = await weft_recall(
            ctx,
            query="launch",
            tier="turns",
            limit=20,
        )
        assert "error" not in result, result
        assert called["n"] == 0, (
            f"flag={sentinel!r} should NOT enable hierarchical, "
            f"got {called['n']} calls"
        )


# --- Flag dispatch in recall_both (turn half) ---


async def test_recall_both_turn_half_uses_hierarchical_when_flag_on(
    monkeypatch, codec_pool, two_topical_episodes,
):
    """When ``WEFT_HIERARCHICAL=1`` is set, the turn half of
    ``recall_both`` descends hierarchically instead of running the flat
    ``recall_turns``."""
    monkeypatch.setenv("WEFT_HIERARCHICAL", "1")

    called: dict[str, int] = {"n": 0}
    from weft.episode_turns import (
        recall_turns_hierarchical as real_hier,
    )

    async def spy(*args, **kwargs):
        called["n"] += 1
        return await real_hier(*args, **kwargs)

    monkeypatch.setattr(
        "weft.episode_turns.recall_turns_hierarchical",
        spy,
        raising=False,
    )

    from weft.turn_recall import recall_both
    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    fused = await recall_both(
        codec_pool, "launch",
        top_k=10,
        embedder=embedder,
    )
    assert called["n"] == 1, (
        f"expected hierarchical descent in recall_both turn half; "
        f"got {called['n']} calls"
    )


async def test_recall_both_turn_half_flat_when_flag_off(
    monkeypatch, codec_pool, two_topical_episodes,
):
    """Default — ``recall_both`` runs flat ``recall_turns`` for the
    turn half."""
    monkeypatch.delenv("WEFT_HIERARCHICAL", raising=False)

    called: dict[str, int] = {"n": 0}
    from weft.episode_turns import (
        recall_turns_hierarchical as real_hier,
    )

    async def spy(*args, **kwargs):
        called["n"] += 1
        return await real_hier(*args, **kwargs)

    monkeypatch.setattr(
        "weft.episode_turns.recall_turns_hierarchical",
        spy,
        raising=False,
    )

    from weft.turn_recall import recall_both
    embedder = _FakeEmbedding(_QUERY_LAUNCH)
    await recall_both(
        codec_pool, "launch",
        top_k=10,
        embedder=embedder,
    )
    assert called["n"] == 0, (
        f"expected NO hierarchical descent when flag is unset; "
        f"got {called['n']} calls"
    )
