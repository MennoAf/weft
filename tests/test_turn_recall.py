"""Tests for turn-tier recall (loom-d9ac7e18).

Three layers:
  - store layer: ``recall_turns`` hybrid scoring + filters + project scope
  - planner: ``route_query_to_tier`` + ``extract_anchors`` (pure regex,
    no DB, no fixtures needed)
  - MCP tool: ``weft_recall(tier='turns')`` happy path + auto-routing
    on representative LongMemEval failure-pattern queries
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.episode_turns import (
    TurnRetrievalError,
    append_turn,
    list_recent_turns,
    recall_turns,
    summarize_occurrences,
)
from weft.episodes import create_episode
from weft.mcp.server import AppContext
from weft.mcp.tools import weft_count_occurrences, weft_recall
from weft.models import EpisodeCreate, EpisodeTurn, EpisodeTurnCreate, TurnRole
from weft.turn_recall import (
    extract_anchors,
    route_query_to_tier,
    temporal_anchor,
    temporal_query_variants,
)


# --- Test fixtures ---


@pytest.mark.parametrize(
    ("embedding", "phase"),
    [([0.1] * 768, "vector"), (None, "keyword")],
)
async def test_recall_turns_raises_structured_sql_failure(embedding, phase):
    cause = RuntimeError("database query failed")

    class FailingExecutor:
        async def fetch(self, *args):
            raise cause

    with pytest.raises(TurnRetrievalError) as raised:
        await recall_turns(
            None, "query", embedding=embedding, executor=FailingExecutor(),
        )

    error = raised.value
    assert error.phase == phase
    assert error.code == f"{phase}_search_failed"
    assert error.cause_type == "RuntimeError"
    assert str(error) == f"{phase}_search_failed: RuntimeError"
    assert error.__cause__ is cause


_FAKE_EMBEDDING = [0.1] * 768


def test_summarize_occurrences_deduplicates_turns_by_episode():
    turns = [
        EpisodeTurnCreate(
            episode_id="episode-a", role=TurnRole.user, content="first mention",
            occurred_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        EpisodeTurnCreate(
            episode_id="episode-a", role=TurnRole.assistant, content="follow-up mention",
            occurred_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        ),
        EpisodeTurnCreate(
            episode_id="episode-b", role=TurnRole.user, content="separate occurrence",
            occurred_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
        ),
    ]
    # The pure helper consumes EpisodeTurn records; construct lightweight
    # validated records from the same fixture inputs.
    records = [
        EpisodeTurn(
            id=f"turn-{i}", episode_id=t.episode_id, turn_index=0,
            role=t.role, content=t.content, occurred_at=t.occurred_at,
        )
        for i, t in enumerate(turns)
    ]
    summary = summarize_occurrences(records)
    assert summary.count == 2
    assert [o.occurrence_id for o in summary.occurrences] == ["episode-a", "episode-b"]
    assert summary.occurrences[0].turn_ids == ("turn-0", "turn-1")
    assert "first mention" in summary.occurrences[0].evidence


def test_summarize_occurrences_distinct_days_uses_utc_calendar_date():
    records = [
        EpisodeTurn(
            id="turn-day-a-1", episode_id="episode-a", turn_index=0,
            role=TurnRole.user, content="morning mention",
            occurred_at=datetime(2026, 1, 2, 23, 30, tzinfo=timezone.utc),
        ),
        EpisodeTurn(
            id="turn-day-a-2", episode_id="episode-b", turn_index=0,
            role=TurnRole.user, content="same UTC day",
            occurred_at=datetime(2026, 1, 2, 23, 45, tzinfo=timezone.utc),
        ),
        EpisodeTurn(
            id="turn-day-b", episode_id="episode-c", turn_index=0,
            role=TurnRole.user, content="next day",
            occurred_at=datetime(2026, 1, 3, 0, 5, tzinfo=timezone.utc),
        ),
    ]
    summary = summarize_occurrences(records, basis="distinct_days")
    assert summary.count == 2
    assert [o.occurrence_id for o in summary.occurrences] == [
        "2026-01-02", "2026-01-03",
    ]
    assert set(summary.occurrences[0].turn_ids) == {
        "turn-day-a-1", "turn-day-a-2",
    }


def test_summarize_occurrences_rejects_unknown_basis():
    with pytest.raises(ValueError, match="basis must be"):
        summarize_occurrences([], basis="incidents")


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
            "how many times have we hit this issue",
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

    def test_three_stage_chronology_preserves_middle_anchor(self):
        anchors = extract_anchors(
            "What was the chronology from the first attempt through "
            "rejecting red to the final blue decision?"
        )
        assert anchors == [
            "the first attempt",
            "rejecting red",
            "the final blue decision",
        ]

    def test_after_x_but_before_y(self):
        anchors = extract_anchors(
            "what happened after the kickoff but before the demo?"
        )
        assert anchors == ["the kickoff", "the demo"]

    def test_single_anchor_returns_empty(self):
        # Single-temporal-marker queries don't trigger multi-anchor split.
        assert extract_anchors("when did I last commit") == []
        assert extract_anchors("how many days since launch") == []

    def test_temporal_query_variants_preserve_event_anchor(self):
        variants = temporal_query_variants(
            "How many weeks ago did I attend the friends and family sale at Nordstrom?"
        )
        assert variants == [
            "How many weeks ago did I attend the friends and family sale at Nordstrom?",
            "attend the friends and family sale at Nordstrom",
        ]
        assert temporal_query_variants("How many times have we hit this issue?") == [
            "How many times have we hit this issue?",
            "hit this issue",
        ]
        assert temporal_query_variants(
            "What is the order of the three trips I took in the past three months, from earliest to latest?"
        )[-1] == "trips I took in the past three months"

    def test_no_temporal_pattern_returns_empty(self):
        assert extract_anchors("describe the architecture") == []


# --- Store: hybrid recall_turns ---


async def test_turn_keyword_query_uses_or_joined_content_lexemes():
    from weft.store import build_or_tsquery

    assert build_or_tsquery("autographed baseballs collection first three months how many added") == (
        "autographed | baseballs | collection | first | three | months | how | many | added"
    )
    # PostgreSQL's english configuration stems these forms during to_tsquery;
    # the query builder itself preserves safe content lexemes without ANDing.
    assert build_or_tsquery("added many") == "added | many"


async def test_recall_turns_keyword_or_matches_separate_terms(pool):
    episode = await create_episode(
        pool, EpisodeCreate(title="OR keyword fixture", project_id="proj-or-keyword")
    )
    first = await append_turn(
        pool, EpisodeTurnCreate(
            episode_id=episode.id, role=TurnRole.user,
            content="added to the baseball collection",
        ),
    )
    second = await append_turn(
        pool, EpisodeTurnCreate(
            episode_id=episode.id, role=TurnRole.user,
            content="many months passed before collecting",
        ),
    )

    results = await recall_turns(
        pool, "added many", project_id="proj-or-keyword", top_k=5, embedding=None,
    )
    assert {turn.id for turn in results} == {first.id, second.id}


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


async def test_recall_turns_reranks_by_usefulness(pool):
    """Two turns with similar BM25/cosine scores but diverging
    ``usefulness_score`` should rank high-useful first after the P1.A3
    rerank step."""
    ep = await create_episode(pool, EpisodeCreate(title="rerank", project_id="proj-rerank"))
    embedding = [0.1] * 768

    # Two turns w/ identical text → identical RRF rank-space contribution.
    # Same occurred_at → identical recency. The only diverging factor is
    # usefulness_score (mutated post-insert to bypass model defaults).
    same_when = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    low_turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="rerankprobe sentinel content",
            occurred_at=same_when,
        ),
        embedding=embedding,
    )
    high_turn = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="rerankprobe sentinel content",
            occurred_at=same_when,
        ),
        embedding=embedding,
    )

    # Diverge usefulness_score post-insert.
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE episode_turns SET usefulness_score = $1 WHERE id = $2",
            0.2, low_turn.id,
        )
        await conn.execute(
            "UPDATE episode_turns SET usefulness_score = $1, last_boosted_at = now() "
            "WHERE id = $2",
            1.0, high_turn.id,
        )

    results = await recall_turns(
        pool,
        "rerankprobe",
        project_id="proj-rerank",
        top_k=5,
    )
    assert len(results) == 2
    # High-useful should outrank low-useful.
    assert results[0].id == high_turn.id
    assert results[1].id == low_turn.id


async def test_recall_turns_disable_rerank_flag(pool, monkeypatch):
    """``WEFT_TURN_RERANK_DISABLE=1`` short-circuits the P1.A3 rerank.

    Setup gives turn A higher BM25 (denser keyword match) but lower
    usefulness, and turn B lower BM25 but higher usefulness. Under the
    default rerank, usefulness flips the order and B wins. With the flag
    on, the RRF order is preserved and A wins. Used by the P1.A5
    warm-boost harness to A/B without a code revert.
    """
    ep = await create_episode(
        pool, EpisodeCreate(title="rerank-flag", project_id="proj-rerank-flag"),
    )
    same_when = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    # turn_a: dense keyword content (higher ts_rank), low usefulness.
    turn_a = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="flagprobe flagprobe flagprobe sentinel",
            occurred_at=same_when,
        ),
    )
    # turn_b: sparser match (lower ts_rank), high usefulness.
    turn_b = await append_turn(
        pool,
        EpisodeTurnCreate(
            episode_id=ep.id,
            role=TurnRole.user,
            content="flagprobe sentinel content",
            occurred_at=same_when,
        ),
    )

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE episode_turns SET usefulness_score = $1 WHERE id = $2",
            0.2, turn_a.id,
        )
        await conn.execute(
            "UPDATE episode_turns SET usefulness_score = $1, last_boosted_at = now() "
            "WHERE id = $2",
            1.0, turn_b.id,
        )

    # Default (rerank ON): usefulness wins, turn_b first.
    monkeypatch.delenv("WEFT_TURN_RERANK_DISABLE", raising=False)
    results_on = await recall_turns(
        pool, "flagprobe",
        project_id="proj-rerank-flag",
        top_k=5,
    )
    assert len(results_on) == 2
    assert results_on[0].id == turn_b.id, (
        "with rerank ON, high-useful turn should win"
    )

    # Flag ON: rerank skipped, RRF order preserved, turn_a first.
    monkeypatch.setenv("WEFT_TURN_RERANK_DISABLE", "1")
    results_off = await recall_turns(
        pool, "flagprobe",
        project_id="proj-rerank-flag",
        top_k=5,
    )
    assert len(results_off) == 2
    assert results_off[0].id == turn_a.id, (
        "with rerank DISABLED, RRF/BM25 order should win"
    )


async def test_recall_turns_populates_usefulness_columns(pool, episode_with_turns):
    """``_row_to_turn`` should hydrate the v46 boost-loop columns onto the
    EpisodeTurn model so callers can introspect them (e.g., for
    diagnostics)."""
    results = await recall_turns(
        pool, "launch",
        project_id=episode_with_turns.project_id,
        top_k=5,
    )
    assert results
    # Defaults from v46: usefulness_score=0.7, usefulness_count=0.
    for t in results:
        assert hasattr(t, "usefulness_score")
        assert 0.0 <= t.usefulness_score <= 1.0
        assert hasattr(t, "usefulness_count")
        assert hasattr(t, "last_boosted_at")


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


async def test_weft_count_occurrences_validates_input(ctx):
    empty = await weft_count_occurrences(ctx, query="   ")
    assert empty == {"error": "query must not be empty"}
    bad_limit = await weft_count_occurrences(ctx, query="issue", limit=0)
    assert bad_limit == {"error": "limit must be between 1 and 500"}
    bad_basis = await weft_count_occurrences(ctx, query="issue", basis="bad")
    assert bad_basis == {"error": "basis must be distinct_conversations or distinct_days"}
    bad_range = await weft_count_occurrences(
        ctx,
        query="issue",
        since=datetime(2026, 2, 1, tzinfo=timezone.utc),
        until=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert bad_range == {"error": "since must be before or equal to until"}


async def test_weft_count_occurrences_returns_distinct_conversations(
    ctx, episode_with_turns,
):
    result = await weft_count_occurrences(
        ctx,
        query="Acme demo",
        project_id=episode_with_turns.project_id,
        limit=10,
    )
    assert result["count_basis"] == "distinct_conversations"
    assert result["count"] == 1
    assert result["candidate_turn_count"] >= 1
    assert result["occurrences"][0]["occurrence_id"] == episode_with_turns.id
    assert result["occurrences"][0]["evidence"]
    day_result = await weft_count_occurrences(
        ctx,
        query="Acme demo",
        project_id=episode_with_turns.project_id,
        basis="distinct_days",
        limit=10,
    )
    assert day_result["count_basis"] == "distinct_days"
    assert day_result["count"] >= 1
    assert all(
        len(occurrence["occurrence_id"]) == 10
        for occurrence in day_result["occurrences"]
    )


async def test_weft_recall_uses_wider_bounded_turn_window(ctx, monkeypatch):
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_temporal_anchor(*args, **kwargs):
        captured.update(kwargs)
        return {args[1]: []}

    monkeypatch.setattr(turn_recall_module, "temporal_anchor", fake_temporal_anchor)
    result = await weft_recall(
        ctx, query="saved recall query", project_id="proj-window",
        tier="turns", limit=10,
    )
    assert result["count"] == 0
    assert captured["top_k_per_anchor"] == 10
    assert captured["candidate_sql_limit"] == 50
    assert captured["anchor_result_limit"] == 10


def _fusion_row(turn_id: str) -> dict:
    """Minimal episode_turns row accepted by _rrf_fuse_turn_rows."""
    return {
        "id": turn_id,
        "episode_id": "ep-fusion",
        "turn_index": 0,
        "role": "user",
        "content": f"fusion fixture {turn_id}",
        "occurred_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
        "trace_id": None,
        "importance_score": None,
        "token_count": 4,
        "user_id": "test-user-default",
        "created_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
    }


def test_rrf_fuse_vector_dominant_weights_keep_vector_only_gold():
    """A vector-only gold turn survives a populous keyword half only when
    keyword RRF contributions are weighted below vector contributions.

    49 double-listed noise turns outrank the vector-rank-1 gold when the
    halves are equally weighted (the cycle-2 dilution pathology); with
    vector 1.0 / keyword 0.3 the gold turn re-enters the top-10 window.
    """
    from weft.episode_turns import _rrf_fuse_turn_rows

    gold = _fusion_row("et-gold-vector-only")
    doubles = [_fusion_row(f"et-double-{i:02d}") for i in range(49)]
    keyword_only = [_fusion_row(f"et-kw-{i:02d}") for i in range(5)]
    vector_rows = [gold, *doubles]
    keyword_rows = [*doubles, *keyword_only]

    def fused_ids(vector_weight: float, keyword_weight: float) -> list[str]:
        pairs = _rrf_fuse_turn_rows(
            vector_rows,
            keyword_rows,
            candidate_limit=50,
            top_k=10,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )
        return [turn.id for turn, _ in pairs]

    equal_weight = fused_ids(0.5, 0.5)
    weighted = fused_ids(1.0, 0.3)

    assert "et-gold-vector-only" not in equal_weight
    assert "et-gold-vector-only" in weighted
    assert weighted.index("et-gold-vector-only") < 10


async def test_temporal_anchor_forwards_vector_dominant_fusion_weights(monkeypatch):
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_recall_turns(pool, query, **kwargs):
        captured.update(kwargs)
        return [
            EpisodeTurn(
                id="et-anchor-hit", episode_id="ep", turn_index=0,
                role=TurnRole.user, content="anchor hit",
                occurred_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            )
        ]

    monkeypatch.setattr(turn_recall_module, "recall_turns", fake_recall_turns)
    result = await turn_recall_module.temporal_anchor(
        None,
        "quarterly planning notes without anchors",
        top_k_per_anchor=5,
    )

    assert captured["vector_weight"] == 1.0
    assert captured["keyword_weight"] == 0.3
    assert [t.id for t in result["quarterly planning notes without anchors"]] == [
        "et-anchor-hit"
    ]


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


# --- Round-7: additive session-expansion slots ---


def test_select_session_siblings_orders_by_priority_then_proximity():
    from weft.episode_turns import _select_session_siblings

    window = ["w1", "w2"]
    session_of = {"w1": "sess-a", "w2": "sess-b"}
    index_of = {"w1": 2, "w2": 5}
    session_turns = {
        "sess-a": [("a1", 0), ("w1", 2), ("a3", 3)],
        "sess-b": [("w2", 5), ("b1", 6)],
    }

    siblings = _select_session_siblings(window, 5, session_of, index_of, session_turns)

    # sess-a outranks sess-b (best window rank 0 beats 1); within a session
    # siblings order by |turn_index - hit| then turn_index; window members
    # are never returned.
    assert siblings == ["a3", "a1", "b1"]


def test_select_session_siblings_skips_null_sessions_and_caps():
    from weft.episode_turns import _select_session_siblings

    assert _select_session_siblings(["w1"], 3, {"w1": None}, {"w1": 0}, {}) == []


async def test_recall_turns_session_expansion_preserves_prefix(pool):
    """Mandatory prefix-preservation: with expansion_slots > 0 the first
    `limit` returns are bit-identical to expansion_slots=0, and only
    same-session sibling turns are appended after the protected prefix."""
    ep_a = await create_episode(pool, EpisodeCreate(title="expansion-a"))
    ep_b = await create_episode(pool, EpisodeCreate(title="expansion-b"))
    rows = [
        ("sess-alpha", "zeta expansion anchor one"),
        ("sess-alpha", "filler alpha neighbor two"),
        ("sess-alpha", "filler alpha neighbor three"),
        ("sess-alpha", "filler alpha neighbor four"),
        ("sess-beta", "unrelated beta content five"),
    ]
    for sess, text in rows:
        await append_turn(
            pool,
            EpisodeTurnCreate(
                episode_id=ep_a.id if sess == "sess-alpha" else ep_b.id,
                role=TurnRole.user,
                content=text,
                source_session_id=sess,
            ),
            embedding=[0.1] * 768,
        )

    embedding = [0.1] * 768
    base = await recall_turns(
        pool, "zeta expansion anchor", top_k=2, embedding=embedding,
        expansion_slots=0,
    )
    expanded = await recall_turns(
        pool, "zeta expansion anchor", top_k=2, embedding=embedding,
        expansion_slots=5,
    )

    base_ids = [t.id for t in base]
    expanded_ids = [t.id for t in expanded]
    assert base_ids, "expected a non-empty baseline window"
    assert expanded_ids[: len(base_ids)] == base_ids

    # Additive: the expanded window is a superset that reaches limit + 5
    # slots (here the project has exactly 5 turns, so all of them surface).
    assert len(expanded_ids) == 5

    # Siblings are session-scoped: every appended ep_a turn is a filler from
    # the anchor's own session, in turn_index proximity order to the hit
    # (the hit is the rank-1 "zeta expansion anchor" turn at index 0).
    appended = expanded_ids[len(base_ids):]
    assert appended, "expected session siblings to be appended"
    from weft.episode_turns import list_turns

    all_turns = await list_turns(pool, ep_a.id)
    index_by_id = {t.id: t.turn_index for t in all_turns}
    ep_a_siblings = [tid for tid in appended if tid in index_by_id]
    assert ep_a_siblings, "expected at least one same-session sibling appended"
    assert ep_a_siblings == sorted(ep_a_siblings, key=lambda tid: index_by_id[tid])


async def test_temporal_anchor_forwards_expansion_slots(monkeypatch):
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_recall_turns(pool, query, **kwargs):
        captured.update(kwargs)
        return [
            EpisodeTurn(
                id="et-x", episode_id="ep", turn_index=0,
                role=TurnRole.user, content="hit",
                occurred_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            )
        ]

    monkeypatch.setattr(turn_recall_module, "recall_turns", fake_recall_turns)
    await turn_recall_module.temporal_anchor(
        None, "plain query without anchors", top_k_per_anchor=5,
        expansion_slots=4,
    )
    assert captured["expansion_slots"] == 4


async def test_weft_recall_turn_tier_expansion_slots_default_on(ctx, monkeypatch):
    """The MCP turn-tier tool defaults expansion on (5 slots) and honors the
    returned-length contract: limit + expansion_slots."""
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_temporal_anchor(*args, **kwargs):
        captured.update(kwargs)
        anchor_limit = kwargs["anchor_result_limit"]
        turns = [
            EpisodeTurn(
                id=f"et-exp-{i}", episode_id="ep", turn_index=i,
                role=TurnRole.user,
                content=f"saved recall query expansion turn {i}",
                occurred_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            )
            for i in range(anchor_limit + 5)
        ]
        return {args[1]: turns}

    monkeypatch.setattr(turn_recall_module, "temporal_anchor", fake_temporal_anchor)
    result = await weft_recall(
        ctx, query="saved recall query", project_id="proj-expansion",
        tier="turns", limit=3,
    )
    assert captured["expansion_slots"] == 5
    assert result["count"] == 8
    assert len(result["turns"]) == 8


async def test_weft_recall_expansion_slots_omitted_uses_constant(ctx, monkeypatch):
    """Omitting expansion_slots preserves the round-7 constant behavior (5)."""
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_temporal_anchor(*args, **kwargs):
        captured["expansion_slots"] = kwargs.get("expansion_slots")
        return {args[1]: []}

    monkeypatch.setattr(turn_recall_module, "temporal_anchor", fake_temporal_anchor)
    result = await weft_recall(
        ctx, query="saved recall query", project_id="proj-omit",
        tier="turns", limit=3,
    )
    assert captured["expansion_slots"] == 5
    assert result["count"] == 0


async def test_weft_recall_expansion_slots_explicit_overrides(ctx, monkeypatch):
    """Explicit expansion_slots (8, or 0 = off) reaches recall_turns as-is."""
    from weft import turn_recall as turn_recall_module

    captured = {}

    async def fake_temporal_anchor(*args, **kwargs):
        captured["expansion_slots"] = kwargs.get("expansion_slots")
        return {args[1]: []}

    monkeypatch.setattr(turn_recall_module, "temporal_anchor", fake_temporal_anchor)
    await weft_recall(
        ctx, query="saved recall query", project_id="proj-pinned",
        tier="turns", limit=3, expansion_slots=8,
    )
    assert captured["expansion_slots"] == 8
    await weft_recall(
        ctx, query="saved recall query", project_id="proj-pinned",
        tier="turns", limit=3, expansion_slots=0,
    )
    assert captured["expansion_slots"] == 0


async def test_weft_recall_expansion_slots_negative_is_input_error(ctx):
    result = await weft_recall(
        ctx, query="saved recall query", project_id="proj-neg",
        tier="turns", limit=3, expansion_slots=-2,
    )
    assert result["error"] == "Invalid input"
    assert "expansion_slots" in result["detail"]
