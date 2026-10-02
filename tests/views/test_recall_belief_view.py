"""Tests for the belief-view query helper and its integration with weft_recall.

All tests use the real ``pool`` fixture (testcontainers Postgres).  No Haiku
calls.  The materializer is driven by a stub detector injected via the
``detector=`` kwarg.

Scenarios covered:
1. Hawaii→Paris supersession: most-recent query returns Paris with provenance.
2. Fallback when no belief_claims match: legacy memories path is used.
3. Error in search_belief_claims: falls back gracefully, no exception to caller.
4. Empty / stop-word-only query: returns [] from belief-view, falls to legacy.
5. user_id scope isolation: two users' claims are kept separate.
6. Token-overlap ranking: higher-overlap result comes first.

Reader-only fail #3 (qid 9ea5eabc):
    The LongMemEval failure bucket for this qid is ``knowledge-update`` /
    ``HIT_FULL_FAIL``: recall retrieved both the Hawaii and Paris sessions but
    the Reader answered "Hawaii" rather than "Paris" because it couldn't
    distinguish which trip was more recent.  The Hawaii→Paris supersession
    test below directly targets this failure shape — after materialisation the
    belief-view returns only the active (Paris) claim, so the Reader is
    presented with unambiguous evidence.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
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
from weft.views.belief_detector import DETECTOR_VERSION, ClaimUpdate
from weft.views.belief_query import BeliefClaimResult, search_belief_claims
from weft.views.materializer import materialize_pending_turns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_USER = "test-user-default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now(offset_seconds: float = 0.0) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _dt(iso: str) -> datetime:
    """Parse an ISO datetime string to a timezone-aware datetime."""
    return datetime.fromisoformat(iso)


async def _create_turn(
    pool: asyncpg.Pool,
    *,
    content: str,
    role: TurnRole = TurnRole.user,
    occurred_at: datetime | None = None,
    user_id: str = TEST_USER,
) -> "weft.models.EpisodeTurn":  # type: ignore[name-defined]
    """Create an episode + turn, return the turn."""
    ep = await create_episode(pool, EpisodeCreate(title=f"test-ep-{uuid.uuid4().hex[:6]}"))
    create = EpisodeTurnCreate(
        episode_id=ep.id,
        role=role,
        content=content,
        occurred_at=occurred_at,
    )
    return await append_turn(pool, create)


def _stub_detector_from_fixture(
    fixture_entries: list[dict],
    turn_map: dict[int, "weft.models.EpisodeTurn"],  # type: ignore[name-defined]
) -> Any:
    """Return a stub detector that emits the claim_update from a fixture entry.

    ``turn_map`` maps fixture turn_index → real EpisodeTurn, so the stub can
    rewrite ``evidence_turn_id`` to the real turn's id.
    """
    # Build a reverse map: turn.id → fixture claim_update
    turn_id_to_claim: dict[str, dict] = {}
    for entry in fixture_entries:
        t = turn_map.get(entry["turn_index"])
        if t is not None:
            turn_id_to_claim[t.id] = entry["claim_update"]

    def _stub(turn: "weft.models.EpisodeTurn") -> list[ClaimUpdate]:  # type: ignore[name-defined]
        claim_def = turn_id_to_claim.get(turn.id)
        if claim_def is None:
            return []
        return [
            ClaimUpdate(
                attribute=claim_def["attribute"],
                value=claim_def["value"],
                confidence=claim_def["confidence"],
                source_provenance=claim_def["source_provenance"],
                evidence_turn_id=turn.id,
                detector_version=DETECTOR_VERSION,
            )
        ]

    return _stub


async def _fetch_active_claim(
    pool: asyncpg.Pool,
    *,
    attribute: str,
    user_id: str = TEST_USER,
    scope: str = "global",
) -> asyncpg.Record | None:
    return await pool.fetchrow(
        """
        SELECT claim_id, attribute, value, occurred_at, evidence_turn_ids,
               source_provenance, detector_confidence, status
        FROM belief_claims
        WHERE user_id = $1 AND attribute = $2 AND scope = $3
          AND status = 'active'
        """,
        user_id, attribute, scope,
    )


async def _count_active_claims(
    pool: asyncpg.Pool,
    *,
    attribute: str,
    user_id: str = TEST_USER,
) -> int:
    return await pool.fetchval(
        "SELECT count(*) FROM belief_claims "
        "WHERE attribute = $1 AND user_id = $2 AND status = 'active'",
        attribute, user_id,
    )


# ---------------------------------------------------------------------------
# MCP context helper
# ---------------------------------------------------------------------------


class _FakeEmbedding:
    """Minimal embedding stub: returns a deterministic vector for any text."""

    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        h = hash(text) % 17 or 1
        return [0.1 + (i % h) * 0.001 for i in range(768)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _make_app(pool: asyncpg.Pool) -> AppContext:
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbedding(),
        config=WeftConfig(),
    )


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


# ---------------------------------------------------------------------------
# Test 1: Hawaii→Paris supersession — most recent query returns Paris
#
# Shape: knowledge-update (qid 9ea5eabc): recall retrieved both sessions but
# Reader answered "Hawaii" (the earlier trip) instead of "Paris" (the later).
# After belief-view materialisation, weft_recall(tier='belief') returns only
# the active Paris claim so the Reader receives unambiguous evidence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hawaii_paris_supersession_returns_paris_with_provenance(
    pool: asyncpg.Pool,
) -> None:
    """search_belief_claims returns Paris (active) not Hawaii (superseded).

    Steps:
    1. Load hawaii_paris_supersession.json fixture.
    2. Append both turns with their fixture occurred_at timestamps.
    3. Materialise via stub detector that emits the fixture claim_updates.
    4. Assert: 1 active claim, attribute=trip.recent-family, destination=Paris.
    5. Assert: Hawaii claim is superseded, NOT returned by search_belief_claims.
    6. Assert: evidence_turn_ids contains the Paris turn's id.
    """
    # Load fixture
    fixture_path = (
        __file__.replace(
            "test_recall_belief_view.py", "fixtures/hawaii_paris_supersession.json"
        )
    )
    with open(fixture_path) as f:
        fixture = json.load(f)

    # Append turns in fixture order
    hawaii_entry = fixture[0]  # turn_index=0, older
    paris_entry = fixture[1]   # turn_index=1, newer

    hawaii_turn = await _create_turn(
        pool,
        content=hawaii_entry["content"],
        occurred_at=_dt(hawaii_entry["occurred_at"]),
    )
    paris_turn = await _create_turn(
        pool,
        content=paris_entry["content"],
        occurred_at=_dt(paris_entry["occurred_at"]),
    )

    turn_map = {0: hawaii_turn, 1: paris_turn}
    detector = _stub_detector_from_fixture(fixture, turn_map)

    # Materialise — stub emits Hawaii claim then Paris claim; Paris supersedes.
    result = await materialize_pending_turns(pool, detector=detector)
    assert result.errors == 0
    assert result.claims_written == 2
    assert result.claims_superseded == 1

    # --- Direct search_belief_claims assertion ---
    results = await search_belief_claims(
        pool,
        query="most recent family trip",
        user_id=TEST_USER,
        scope="global",
    )

    assert len(results) == 1, f"Expected 1 active claim, got {len(results)}"
    claim = results[0]
    assert claim.attribute == "trip.recent-family"
    value = claim.value if isinstance(claim.value, dict) else {}
    assert value.get("destination") == "Paris", (
        f"Expected Paris as destination, got {value.get('destination')!r}"
    )
    # Provenance points to the Paris turn
    assert paris_turn.id in claim.evidence_turn_ids, (
        f"Paris turn id {paris_turn.id!r} not in evidence_turn_ids {claim.evidence_turn_ids}"
    )
    assert claim.source_provenance == "user_stated"

    # Hawaii claim must be superseded (NOT in active results)
    hawaii_in_results = any(
        r.attribute == "trip.recent-family"
        and isinstance(r.value, dict)
        and r.value.get("destination") == "Hawaii"
        for r in results
    )
    assert not hawaii_in_results, "Hawaii (superseded) should not appear in results"

    # --- to_recall_dict shape validation ---
    d = claim.to_recall_dict()
    assert d["tier"] == "belief-view"
    assert d["kind"] == "belief_claim"
    assert d["attribute"] == "trip.recent-family"
    assert d["source_provenance"] == "user_stated"
    assert paris_turn.id in d["evidence_turn_ids"]


# ---------------------------------------------------------------------------
# Test 2: Fallback when no claims match — legacy memories path is used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_when_no_claims_match(pool: asyncpg.Pool) -> None:
    """When belief_claims has no match for the query, legacy memories are returned.

    Steps:
    1. Materialise a claim for a different attribute (no match for "family trip").
    2. Insert a memory directly into memories about "family trip".
    3. Call weft_recall(tier='belief').
    4. Assert: response uses legacy memories path (no 'tier' key, has 'results').
    """
    # Append and materialise a turn for a NON-matching attribute
    unrelated_turn = await _create_turn(pool, content="I slept 7 hours last night.")

    def _unrelated_detector(turn):  # type: ignore[no-untyped-def]
        return [
            ClaimUpdate(
                attribute="sleep.recent-hours",
                value={"hours": 7},
                confidence=0.9,
                source_provenance="user_stated",
                evidence_turn_id=turn.id,
                detector_version=DETECTOR_VERSION,
            )
        ]

    await materialize_pending_turns(pool, detector=_unrelated_detector)

    # Insert a legacy memory about "family trip"
    await store_memory(
        pool,
        MemoryCreate(
            content="The family trip to Hawaii was wonderful.",
            type=MemoryType.fact,
            source=MemorySource.conversation,
        ),
        embedding=[0.1] * 768,
    )

    app = _make_app(pool)
    ctx = _make_ctx(app)

    response = await weft_recall(
        ctx,
        query="family trip",
        tier="belief",
        limit=5,
    )

    # Must not error
    assert "error" not in response, f"Unexpected error: {response}"
    # Legacy path: 'results' array present, no 'tier' key
    assert "results" in response
    assert response.get("tier") != "belief-view", (
        "belief-view should NOT activate when no claims match 'family trip'"
    )


# ---------------------------------------------------------------------------
# Test 3: Error in search_belief_claims falls back gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_belief_view_error_falls_back_gracefully(pool: asyncpg.Pool) -> None:
    """If search_belief_claims raises, the legacy memories path still runs.

    Monkeypatches search_belief_claims to raise an exception.  The caller
    (weft_recall) must catch this and produce a valid response from the
    legacy path — no exception should propagate.
    """
    # Seed a memory so the legacy path has something to return
    await store_memory(
        pool,
        MemoryCreate(
            content="Casey Example enjoys running three times a week.",
            type=MemoryType.fact,
            source=MemorySource.conversation,
        ),
        embedding=[0.1] * 768,
    )

    app = _make_app(pool)
    ctx = _make_ctx(app)

    with patch(
        "weft.views.belief_query.search_belief_claims",
        side_effect=asyncio.TimeoutError("injected transient timeout for test"),
    ):
        response = await weft_recall(
            ctx,
            query="running exercise",
            tier="belief",
            limit=5,
        )

    # Must not bubble the injected exception
    assert "error" not in response, f"Exception propagated: {response}"
    # Legacy path must have run
    assert "results" in response
    assert response.get("tier") != "belief-view"


# ---------------------------------------------------------------------------
# Test 4: Empty / stop-word-only query returns [] from belief-view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_query_returns_empty_belief_results(
    pool: asyncpg.Pool,
) -> None:
    """Queries with no meaningful tokens after stop-word removal return [].

    The empty-result path causes the caller to fall through to the legacy
    memories search — we verify search_belief_claims itself returns [].
    """
    # Seed a claim so the table isn't empty
    turn = await _create_turn(pool, content="I got 7 hours of sleep.")

    def _det(t):  # type: ignore[no-untyped-def]
        return [
            ClaimUpdate(
                attribute="sleep.recent-hours",
                value={"hours": 7},
                confidence=0.9,
                source_provenance="user_stated",
                evidence_turn_id=t.id,
                detector_version=DETECTOR_VERSION,
            )
        ]

    await materialize_pending_turns(pool, detector=_det)

    # Empty query
    r1 = await search_belief_claims(
        pool, query="", user_id=TEST_USER,
    )
    assert r1 == [], f"Empty query should return [], got {r1}"

    # Stop-word-only query: "the a an" — all stripped, no tokens
    r2 = await search_belief_claims(
        pool, query="the a an", user_id=TEST_USER,
    )
    assert r2 == [], f"Stop-word-only query should return [], got {r2}"

    # Short single-char token: "a i" — dropped by min-length guard
    r3 = await search_belief_claims(
        pool, query="a i", user_id=TEST_USER,
    )
    assert r3 == [], f"Single-char tokens should return [], got {r3}"


# ---------------------------------------------------------------------------
# Test 5: user_id scope isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_belief_view_respects_user_id_scope(
    pool: asyncpg.Pool,
) -> None:
    """Claims for user A must not appear in user B's query results.

    We directly INSERT claims for two users and then query as one of them
    to verify the user_id filter is enforced.
    """
    user_a = "scope-test-user-a"
    user_b = "scope-test-user-b"

    async def _insert_claim(user_id: str, destination: str) -> None:
        claim_id = f"belief-{uuid.uuid4().hex[:10]}"
        value_json = json.dumps({"destination": destination})
        async with pool.acquire() as conn:
            safe = user_id.replace("'", "''")
            await conn.execute(f"SET LOCAL app.user_id = '{safe}'")
            await conn.execute(
                """
                INSERT INTO belief_claims (
                    claim_id, user_id, attribute, value, scope,
                    evidence_turn_ids, status, occurred_at,
                    source_provenance, detector_confidence, detector_version
                ) VALUES (
                    $1, $2, 'trip.recent-family', $3::jsonb, 'global',
                    ARRAY['et-dummy']::text[], 'active', now(),
                    'user_stated', 0.9, $4
                )
                """,
                claim_id, user_id, value_json, DETECTOR_VERSION,
            )

    await _insert_claim(user_a, "Hawaii")
    await _insert_claim(user_b, "Paris")

    # Query as user_a — should see only Hawaii
    results_a = await search_belief_claims(
        pool, query="family trip destination", user_id=user_a,
    )
    assert len(results_a) == 1
    assert results_a[0].value.get("destination") == "Hawaii"

    # Query as user_b — should see only Paris
    results_b = await search_belief_claims(
        pool, query="family trip destination", user_id=user_b,
    )
    assert len(results_b) == 1
    assert results_b[0].value.get("destination") == "Paris"


# ---------------------------------------------------------------------------
# Test 6: Token-overlap ranking — higher-overlap result comes first
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_overlap_ranks_correctly(pool: asyncpg.Pool) -> None:
    """Claim with 2 matching tokens must rank above claim with 1 matching token.

    Query: "recent family trip"
      - Attribute "trip.recent-family" matches "trip", "recent", "family" → 3 tokens
      - Attribute "exercise.weekly" matches 0 tokens → not returned at all

    We also verify ordering between two attributes that each match different counts.
    """
    # Insert claim 1: "trip.recent-family" — matches "trip", "recent", "family"
    claim1_id = f"belief-{uuid.uuid4().hex[:10]}"
    claim2_id = f"belief-{uuid.uuid4().hex[:10]}"

    async with pool.acquire() as conn:
        await conn.execute(f"SET LOCAL app.user_id = '{TEST_USER}'")
        await conn.execute(
            """
            INSERT INTO belief_claims (
                claim_id, user_id, attribute, value, scope,
                evidence_turn_ids, status, occurred_at,
                source_provenance, detector_confidence, detector_version
            ) VALUES (
                $1, $2, 'trip.recent-family', '{"destination": "Tokyo"}'::jsonb, 'global',
                ARRAY['et-dummy1']::text[], 'active', now() - interval '10 minutes',
                'user_stated', 0.9, $3
            )
            """,
            claim1_id, TEST_USER, DETECTOR_VERSION,
        )
        await conn.execute(
            """
            INSERT INTO belief_claims (
                claim_id, user_id, attribute, value, scope,
                evidence_turn_ids, status, occurred_at,
                source_provenance, detector_confidence, detector_version
            ) VALUES (
                $1, $2, 'trip.vacation', '{"destination": "Rome"}'::jsonb, 'global',
                ARRAY['et-dummy2']::text[], 'active', now(),
                'user_stated', 0.85, $3
            )
            """,
            claim2_id, TEST_USER, DETECTOR_VERSION,
        )

    # Query: "recent family trip" — "trip.recent-family" matches more tokens
    results = await search_belief_claims(
        pool,
        query="recent family trip",
        user_id=TEST_USER,
    )

    # Both "trip.recent-family" and "trip.vacation" match "trip"
    # "trip.recent-family" also matches "recent" and "family" → higher overlap
    # So "trip.recent-family" must come first
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    assert results[0].attribute == "trip.recent-family", (
        f"Expected trip.recent-family first (higher overlap), got {results[0].attribute}"
    )
    assert results[0].overlap_score >= results[1].overlap_score, (
        "First result should have overlap_score >= second"
    )


# ---------------------------------------------------------------------------
# Test 7 (Finding 6): Non-transient errors propagate out of weft_recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_belief_view_propagates_non_transient_errors(
    pool: asyncpg.Pool,
) -> None:
    """Non-transient asyncpg errors must propagate, not be swallowed.

    The narrowed except clause only catches transient infrastructure errors
    (connection failure, timeout, interface error, import error).  A
    UniqueViolationError indicates a bug and must not be silently caught.
    """
    app = _make_app(pool)
    ctx = _make_ctx(app)

    with patch(
        "weft.views.belief_query.search_belief_claims",
        side_effect=asyncpg.UniqueViolationError(),
    ):
        with pytest.raises(asyncpg.UniqueViolationError):
            await weft_recall(
                ctx,
                query="x",
                tier="belief",
                limit=5,
            )
