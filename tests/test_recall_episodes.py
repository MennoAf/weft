"""Tests for ``weft.episodes.recall_episodes`` (loom-e78c948d, P2.3).

Episode-tier hybrid recall: cosine + ts_rank, RRF-fused, recency-weighted.
Mirrors the test layout used in tests/test_turn_recall.py for the turn tier
sibling — same fake-embedder pattern, same fixture shape.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from weft.episodes import create_episode, recall_episodes
from weft.models import EpisodeCreate, EpisodeStatus


# --- Test fixtures ---

# Two embeddings that are nearly orthogonal in cosine space — used to
# simulate a "semantic" hit. The vector that aligns with the query
# embedding ranks first via cosine distance; the off-axis one ranks
# behind it. 768 dimensions matches the migration-47 column type.
_DIM = 768


def _vec(seed: float) -> list[float]:
    """Deterministic vector with strong directional signal at index 0."""
    base = [0.0] * _DIM
    base[0] = seed
    base[1] = 1.0 - abs(seed)
    return base


_QUERY_EMBED_LAUNCH = _vec(0.99)   # close to "launch" episode's vector
_QUERY_EMBED_BUDGET = _vec(-0.99)  # close to "budget" episode's vector


# --- Smoke ---


async def test_recall_episodes_empty_pool_returns_empty(pool):
    out = await recall_episodes(pool, "anything")
    assert out == []


async def test_recall_episodes_zero_top_k_returns_empty(pool):
    await create_episode(pool, EpisodeCreate(title="x", summary="y"))
    out = await recall_episodes(pool, "x", top_k_episodes=0)
    assert out == []


# --- Vector half ---


async def test_recall_episodes_vector_half_ranks_semantic_match_first(pool):
    """Episode whose embedding is closest in cosine space ranks first
    even when the keyword half misses (no overlapping tokens)."""
    on_topic = await create_episode(
        pool,
        EpisodeCreate(title="alpha", summary="alpha details"),
        embedding=_vec(0.99),
    )
    off_topic = await create_episode(
        pool,
        EpisodeCreate(title="beta", summary="beta details"),
        embedding=_vec(-0.99),
    )

    # Use a query that is NOT lexically present in either episode so
    # the keyword half misses — vector half is the only signal.
    results = await recall_episodes(
        pool,
        "zzzzz_no_keyword_match",
        embedding=_vec(0.99),
        top_k_episodes=5,
    )
    ids = [e.id for e in results]
    assert on_topic.id in ids
    assert ids.index(on_topic.id) < ids.index(off_topic.id)


# --- Keyword half ---


async def test_recall_episodes_keyword_half_ranks_title_match_first(pool):
    """Keyword half (ts_rank over title || summary) surfaces the
    episode whose title contains the query token over one that doesn't."""
    target = await create_episode(
        pool, EpisodeCreate(title="quarterly budget review", summary="finance"),
    )
    distractor = await create_episode(
        pool, EpisodeCreate(title="customer demo", summary="sales call"),
    )
    results = await recall_episodes(pool, "budget", top_k_episodes=5)
    ids = [e.id for e in results]
    assert target.id in ids
    # The distractor either ranks below or doesn't appear at all.
    if distractor.id in ids:
        assert ids.index(target.id) < ids.index(distractor.id)


# --- RRF fusion ---


async def test_recall_episodes_rrf_fuses_both_halves(pool):
    """An episode that hits BOTH halves outranks one that hits only one."""
    # Hits both halves: matching keyword AND matching embedding direction.
    both = await create_episode(
        pool,
        EpisodeCreate(title="launch", summary="launch logistics"),
        embedding=_QUERY_EMBED_LAUNCH,
    )
    # Hits only the vector half: same embedding, different keyword.
    vector_only = await create_episode(
        pool,
        EpisodeCreate(title="zzz", summary="zzz unrelated content"),
        embedding=_QUERY_EMBED_LAUNCH,
    )
    # Hits only the keyword half: matching keyword, off-axis embedding.
    keyword_only = await create_episode(
        pool,
        EpisodeCreate(title="launch", summary="launch logistics"),
        embedding=_QUERY_EMBED_BUDGET,
    )

    results = await recall_episodes(
        pool,
        "launch",
        embedding=_QUERY_EMBED_LAUNCH,
        top_k_episodes=5,
    )
    ids = [e.id for e in results]
    assert both.id in ids
    # Both-halves episode beats either single-half episode.
    if vector_only.id in ids:
        assert ids.index(both.id) < ids.index(vector_only.id)
    if keyword_only.id in ids:
        assert ids.index(both.id) < ids.index(keyword_only.id)


# --- Recency ---


async def test_recall_episodes_recency_breaks_ties(pool):
    """Two episodes with identical content — the more recent one wins."""
    older = await create_episode(
        pool,
        EpisodeCreate(title="release notes", summary="release details"),
        embedding=_vec(0.5),
    )
    newer = await create_episode(
        pool,
        EpisodeCreate(title="release notes", summary="release details"),
        embedding=_vec(0.5),
    )
    # Backdate the older episode so the recency factor differs even though
    # both halves rank them identically. We can't pass started_at through
    # EpisodeCreate, so reach into the table directly post-insert.
    far_past = datetime.now(timezone.utc) - timedelta(days=400)
    await pool.execute(
        "UPDATE episodes SET started_at = $1 WHERE id = $2",
        far_past,
        older.id,
    )

    results = await recall_episodes(
        pool, "release", embedding=_vec(0.5), top_k_episodes=5,
    )
    ids = [e.id for e in results]
    assert newer.id in ids and older.id in ids
    assert ids.index(newer.id) < ids.index(older.id)


# --- Scoping ---


async def test_recall_episodes_project_scope_isolates(pool):
    """Project filter must not leak episodes from other projects."""
    in_proj = await create_episode(
        pool,
        EpisodeCreate(title="alpha keyword", project_id="proj-a"),
        embedding=_vec(0.5),
    )
    other_proj = await create_episode(
        pool,
        EpisodeCreate(title="alpha keyword", project_id="proj-b"),
        embedding=_vec(0.5),
    )
    global_ep = await create_episode(
        pool, EpisodeCreate(title="alpha keyword"),
        embedding=_vec(0.5),
    )

    results = await recall_episodes(
        pool, "alpha", embedding=_vec(0.5), project_id="proj-a", top_k_episodes=10,
    )
    ids = {e.id for e in results}
    assert in_proj.id in ids
    assert other_proj.id not in ids
    # Globals (project_id IS NULL) still pass through (OR-NULL semantic
    # mirrors list_episodes).
    assert global_ep.id in ids


async def test_recall_episodes_status_filter_excludes_others(pool):
    """``status='closed'`` must exclude open / graduated / expired episodes."""
    open_ep = await create_episode(
        pool, EpisodeCreate(title="alpha"), embedding=_vec(0.5),
    )
    closed_ep = await create_episode(
        pool, EpisodeCreate(title="alpha"), embedding=_vec(0.5),
    )
    # Manually transition one to closed.
    await pool.execute(
        "UPDATE episodes SET status = 'closed' WHERE id = $1",
        closed_ep.id,
    )

    results = await recall_episodes(
        pool, "alpha", embedding=_vec(0.5),
        status=EpisodeStatus.closed, top_k_episodes=10,
    )
    ids = {e.id for e in results}
    assert closed_ep.id in ids
    assert open_ep.id not in ids


async def test_recall_episodes_since_until_filter(pool):
    """``since`` / ``until`` filter on started_at."""
    old = await create_episode(
        pool, EpisodeCreate(title="alpha"), embedding=_vec(0.5),
    )
    new = await create_episode(
        pool, EpisodeCreate(title="alpha"), embedding=_vec(0.5),
    )
    await pool.execute(
        "UPDATE episodes SET started_at = $1 WHERE id = $2",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        old.id,
    )
    await pool.execute(
        "UPDATE episodes SET started_at = $1 WHERE id = $2",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        new.id,
    )

    results = await recall_episodes(
        pool, "alpha", embedding=_vec(0.5),
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        top_k_episodes=10,
    )
    ids = {e.id for e in results}
    assert new.id in ids
    assert old.id not in ids


# --- Embedder auto-resolution ---


class _FakeEmbedder:
    provider_name = "fake"
    dimensions = _DIM

    def __init__(self):
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return _vec(0.5)

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


async def test_recall_episodes_resolves_embedder_when_no_embedding(pool):
    """If ``embedding`` is None and an ``embedder`` is supplied, the
    embedder is invoked once on the query string."""
    await create_episode(
        pool, EpisodeCreate(title="alpha"), embedding=_vec(0.5),
    )
    embedder = _FakeEmbedder()
    results = await recall_episodes(
        pool, "alpha", embedder=embedder, top_k_episodes=5,
    )
    assert results, "expected vector half to surface the alpha episode"
    assert embedder.calls == ["alpha"]


async def test_recall_episodes_no_embedding_no_embedder_keyword_only(pool):
    """No ``embedding`` and no ``embedder`` — vector half skipped, BM25
    half still runs (mirrors recall_turns)."""
    target = await create_episode(
        pool, EpisodeCreate(title="alpha keyword"),
        embedding=None,
    )
    results = await recall_episodes(pool, "alpha", top_k_episodes=5)
    ids = [e.id for e in results]
    assert target.id in ids


# --- Performance ---


async def test_recall_episodes_under_100ms_with_1000_rows(pool):
    """Rough latency floor: ≥1000 episodes, single-statement recall in
    under 100ms. HNSW + GIN-less ts_rank against a small corpus is well
    inside this on testcontainer Postgres."""
    # Bulk-insert 1000 episodes via executemany to keep fixture cost
    # bounded. We bypass create_episode (which does a row-at-a-time INSERT)
    # and rely on table defaults. The pgvector codec is registered on the
    # pool fixture so passing a Python list[float] lets asyncpg encode it.
    embedding = _vec(0.5)
    rows = [
        (
            f"weft-perf-{i:04d}",
            f"perf episode {i}",
            f"summary for episode {i} containing keyword filler",
            datetime.now(timezone.utc),
            embedding,
        )
        for i in range(1000)
    ]
    await pool.executemany(
        """
        INSERT INTO episodes (
            id, title, summary, started_at, embedding,
            status, token_count, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5::vector,
                'open', 0, $4, $4)
        """,
        rows,
    )

    # Warm-up call (HNSW index build can lazily materialize on first scan).
    await recall_episodes(pool, "keyword", embedding=embedding, top_k_episodes=10)

    start = time.perf_counter()
    results = await recall_episodes(
        pool, "keyword", embedding=embedding, top_k_episodes=10,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 10
    print(f"\n[perf] recall_episodes over 1000 rows: {elapsed_ms:.1f}ms")
    assert elapsed_ms < 100, f"recall_episodes took {elapsed_ms:.1f}ms (>100)"
