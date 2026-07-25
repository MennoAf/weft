"""Turn-tier query planner + multi-anchor recall.

Two responsibilities:

1. ``route_query_to_tier`` — decide whether a query should hit the belief
   tier (semantic facts), the turn tier (raw dialogue trace), or both.
   Regex-first; cheap; deterministic. The driver: LongMemEval temporal-
   reasoning questions ("how many days between A and B", "when did I last
   ...") fail on belief-tier extraction because dates collapse during
   classification. Routing those to turns avoids the lossy step.

2. ``temporal_anchor`` — for queries with multiple named anchors
   ("between launch and demo", "after I started X but before Y"), split
   the query into per-anchor sub-queries and run ``recall_turns`` for
   each. Returns a mapping ``anchor → [turns]`` so the Reader can do
   anchored arithmetic without conflating the two halves.

This module is intentionally thin: heavy lifting lives in
``weft.episode_turns.recall_turns``. Adding a tiny LLM-based fallback
planner is on the roadmap but the cost/latency hit is real and the regex
covers the LongMemEval failure patterns we measured.

Spec: weft-d3a2ef78. Loom task: loom-d9ac7e18.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Literal

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.episode_turns import _RRF_K, list_recent_turns, recall_turns
from weft.models import EpisodeTurn, MemoryRecall, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)


Tier = Literal["belief", "turns", "both", "auto"]


# --- Routing ---


# Lowercased regexes — case folding happens before match. Each pattern
# captures a temporal-reasoning marker that belief-tier extraction tends
# to lose. Order doesn't matter; first hit returns.
_TURN_TIER_MARKERS: tuple[re.Pattern, ...] = (
    re.compile(r"\bhow many times\b"),
    re.compile(r"\bhow many (?:days|weeks|months|hours|minutes|years)\b"),
    re.compile(r"\bhow long (?:since|ago|before|after|between)\b"),
    re.compile(r"\bwhen (?:did|was)\b"),
    re.compile(r"\b(?:before|after|between|since|until)\b"),
    re.compile(r"\blast (?:time|week|month|year|tuesday|wednesday|thursday|friday|saturday|sunday|monday)\b"),
    re.compile(r"\b(?:earlier|later) (?:than|that)\b"),
    re.compile(r"\bwhat (?:day|date|time)\b"),
    re.compile(r"\bwhat (?:was|is) the chronology\b"),
)


# Queries that benefit from RRF across belief and turns: explicit
# episodic asks ("do you remember", "have we discussed"), self-referential
# fact queries that reach for prior dialogue ("what did I last say",
# "my decision on"), and remind-me prompts. The _TURN_TIER_MARKERS list
# above catches pure temporal-arithmetic queries that need raw turns;
# this list catches queries where the user wants the canonical fact
# (belief tier) AND the dialogue evidence (turn tier) fused together.
#
# These patterns are checked BEFORE _TURN_TIER_MARKERS in
# route_query_to_tier so a query like "what did I last say about X"
# routes to 'both' instead of falling through to 'turns'.
#
# Starter set is deliberately narrow (PRECISION over recall on first
# ship) — broaden based on Oracle data once Tier 1.5 RRF lands the
# 'both' dispatch path. Today the auto-path caller (weft_recall) only
# handles 'belief' and 'turns'; 'both' returns fall through to belief
# until P1.B2 wires the RRF fusion.
_BOTH_TIER_MARKERS: tuple[re.Pattern, ...] = (
    # Explicit episodic recall asks.
    re.compile(r"\b(?:do you|did we|have we) (?:recall|remember|discuss)"),
    # "have we / I (talked|discussed|mentioned)" + topic.
    re.compile(r"\bhave (?:we|i) (?:discussed|talked|mentioned)"),
    # Self-referential fact retrieval that reaches for prior dialogue.
    re.compile(
        r"\bwhat did i (?:last )?(?:say|tell|mention|decide|think|conclude)"
    ),
    # Possessive opinion / decision queries — fact + history together.
    re.compile(
        r"\b(?:my|our) (?:decision|opinion|view|stance|take|position) (?:on|about)\b"
    ),
    # Remind-me prompts — factual answer grounded in prior dialogue.
    re.compile(r"\bremind me (?:about|of|what|when)\b"),
    # Decision rationale is often omitted from concise handoffs/beliefs; retrieve
    # the quoted dialogue evidence rather than fabricating a reason.
    re.compile(r"\bwhy did (?:we|i) (?:reject|choose|pick|decide|change)\b"),
)


def route_query_to_tier(query: str) -> Tier:
    """Decide which retrieval tier a query should hit.

    Returns:
        ``'both'`` for queries that benefit from RRF across belief and
        turns (explicit episodic asks, self-referential fact queries
        that need dialogue evidence). ``'turns'`` for queries with
        explicit temporal markers (LongMemEval temporal-reasoning
        patterns). ``'belief'`` otherwise.

    Priority order is ``both`` → ``turns`` → ``belief``. A query like
    "what did I last say about the launch" matches the ``_BOTH_TIER_MARKERS``
    self-referential pattern AND would also match no turn marker today,
    but the ordering matters for queries that hit both lists (e.g.
    "do you recall when did I ship" hits ``do you recall`` first and
    routes to 'both' rather than falling through to 'turns').
    """
    q = query.lower()
    for pattern in _BOTH_TIER_MARKERS:
        if pattern.search(q):
            logger.debug("route_query_to_tier: %r matched %s → both",
                         query[:60], pattern.pattern)
            return "both"
    for pattern in _TURN_TIER_MARKERS:
        if pattern.search(q):
            logger.debug("route_query_to_tier: %r matched %s → turns",
                         query[:60], pattern.pattern)
            return "turns"
    return "belief"


# --- Anchor extraction ---


# Patterns that imply two named anchors. Capture groups carry the anchor
# strings so the caller can run a sub-query per anchor.
_ANCHOR_PATTERNS: tuple[re.Pattern, ...] = (
    # Three-stage chronology must be checked before the two-stage ``from``
    # pattern or the middle event is swallowed into the first anchor.
    re.compile(
        r"\bfrom\s+(.+?)\s+through\s+(.+?)\s+to\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
    # "between X and Y", "from X to Y"
    re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.,]|$)", re.IGNORECASE),
    re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[?.,]|$)", re.IGNORECASE),
    # "after X but before Y" / "before X and after Y"
    re.compile(
        r"\bafter\s+(.+?)\s+(?:but|and)\s+before\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbefore\s+(.+?)\s+(?:but|and)\s+after\s+(.+?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
)


def temporal_query_variants(query: str) -> list[str]:
    """Return the original query plus an event-focused temporal variant.

    Temporal scaffolding (``how many weeks ago did I`` / ``when did we``)
    often dominates keyword ranking while the event name carries the useful
    retrieval signal. Keep the original for semantic recall, and add a
    conservative lexical variant only when removing the scaffolding leaves a
    meaningful event phrase. Variants are ordered original-first and are
    deduplicated case-insensitively.
    """
    variants = [query.strip()]
    cleaned = re.sub(
        r"^\s*how\s+many\s+times\s+(?:have|has)\s+(?:i|we|you|we\s+all)\s+",
        "",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:how\s+many\s+(?:days|weeks|months|hours|minutes|years)\s+ago|"
        r"how\s+long\s+ago|when\s+did|when\s+was|"
        r"what\s+(?:day|date|time)\s+was)\s+",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:did\s+)?(?:i|we|you)\s+", "", cleaned,
        count=1, flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" ?.,")
    if len(cleaned) >= 4 and cleaned.casefold() != query.strip().casefold():
        variants.append(cleaned)

    chronology = re.sub(
        r"^\s*(?:what\s+is\s+the\s+order\s+of|order\s+of|what\s+was\s+the\s+order\s+of)\s+"
        r"(?:the\s+)?(?:three|four|five|six|several|multiple)\s+",
        "",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    chronology = re.sub(
        r"\s+from\s+earliest\s+to\s+latest\s*\??$", "", chronology,
        flags=re.IGNORECASE,
    ).strip(" ?.,")
    if (
        len(chronology) >= 4
        and chronology.casefold() not in {variant.casefold() for variant in variants}
        and chronology.casefold() != query.strip(" ?.,").casefold()
    ):
        variants.append(chronology)
    return variants


def extract_anchors(query: str) -> list[str]:
    """Extract named anchor phrases from a temporal query.

    Returns a list of anchor strings, longest-first. Empty list when no
    multi-anchor pattern fires — the caller should fall back to a single
    ``recall_turns(query)`` call in that case.

    Example:
        >>> extract_anchors("how many days between the demo and the retro?")
        ['the demo', 'the retro']
    """
    for pattern in _ANCHOR_PATTERNS:
        m = pattern.search(query)
        if m:
            anchors = [g.strip() for g in m.groups() if g and g.strip()]
            # Filter out trivially short anchors that are probably stop
            # words spilled in by greedy capture (``after I``, ``before he``).
            anchors = [a for a in anchors if len(a.split()) >= 1 and len(a) >= 2]
            if anchors:
                return anchors
    return []


# --- Multi-anchor recall ---


async def temporal_anchor(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    top_k_per_anchor: int = 5,
    embedder: EmbeddingProvider | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, list[EpisodeTurn]]:
    """Per-anchor turn recall for multi-anchor temporal questions.

    For queries like "how many days between the launch and the demo", we
    pull the top-K turns matching ``"the launch"`` separately from the
    top-K turns matching ``"the demo"``. The Reader then sees two
    grounded sets and can do the arithmetic without conflating evidence.

    When no multi-anchor pattern fires, falls back to a single
    ``recall_turns(query)`` keyed under the original query string. That
    way the caller always gets a populated dict and never has to special-
    case the no-anchor path.

    Args:
        embedder: required when callers want vector recall in the
            sub-queries. If None, anchor sub-queries are keyword-only
            (still useful — anchor strings tend to contain the rare
            keywords that BM25 hits cleanly).

    Returns:
        ``{anchor_text: [EpisodeTurn, ...]}`` ordered as anchors appear
        in the query. The original query is included as a key when
        anchor extraction failed.
    """
    anchors = extract_anchors(query)
    if not anchors:
        # No multi-anchor pattern — search the original question and, for
        # temporal scaffolding, an event-focused lexical variant. Unioning
        # these result sets prevents words like "how many weeks ago" from
        # crowding the actual event (e.g. "friends and family sale at
        # Nordstrom") out of the candidate pool.
        variants = temporal_query_variants(query)
        seen: set[str] = set()
        merged: list[EpisodeTurn] = []
        for variant in variants:
            embedding = await embedder.embed(variant) if embedder else None
            turns = await recall_turns(
                pool, variant,
                project_id=project_id, since=since, until=until,
                top_k=top_k_per_anchor, embedding=embedding,
            )
            for turn in turns:
                if turn.id in seen:
                    continue
                seen.add(turn.id)
                merged.append(turn)
                if len(merged) >= top_k_per_anchor:
                    break
            if len(merged) >= top_k_per_anchor:
                break
        if not merged:
            # Empty hybrid hit — try a temporal-only fallback so the
            # Reader at least sees recent dialogue.
            merged = await list_recent_turns(
                pool, project_id=project_id, since=since, until=until,
                limit=top_k_per_anchor,
            )
        return {query: merged[:top_k_per_anchor]}

    out: dict[str, list[EpisodeTurn]] = {}
    for anchor in anchors:
        embedding = await embedder.embed(anchor) if embedder else None
        out[anchor] = await recall_turns(
            pool, anchor,
            project_id=project_id, since=since, until=until,
            top_k=top_k_per_anchor, embedding=embedding,
        )
    return out


# --- Both-tier RRF fusion ---


async def recall_both(
    pool: asyncpg.Pool,
    query: str,
    *,
    project_id: str | None = None,
    top_k: int = 10,
    embedder: EmbeddingProvider | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
) -> list[dict[str, Any]]:
    """Run belief + turn recall in parallel, fuse via RRF, return unified list.

    The driver: ``route_query_to_tier`` returns 'both' for explicit episodic
    asks ("do you remember", "what did I last say") where the user wants
    the canonical fact (belief tier) AND the dialogue evidence (turn tier)
    fused together. This function runs both halves in parallel and fuses
    them via Reciprocal Rank Fusion using the same K=60 constant as
    ``weft.store.search_hybrid`` and ``weft.episode_turns.recall_turns``
    (imported as ``_RRF_K``).

    Heterogeneous fusion: belief items and turn items are disjoint — no
    single item appears in both lists — so RRF degenerates to
    ``rrf_score = 1 / (K + rank_in_originating_list)``. Sort descending
    and take ``top_k``.

    Scoping is applied to BOTH halves identically so cross-project context
    cannot bleed in:

    * ``project_id`` → both halves
    * ``since`` / ``until`` → turn half only (memories don't carry an
      occurred_at; their lifecycle is created_at/updated_at, not the
      semantic "when did this happen" the temporal filter implies)
    * ``agent_id`` / ``user_id`` → belief half only (turn-tier rows don't
      carry agent_id; user_id on episode_turns is RLS-enforced via the
      session GUC at acquire-time, not a query param)

    Each half's ``top_k`` is set to ``2 * top_k`` so the fuser has
    material — RRF on two N-item lists with no overlap is just sort by
    rank, but oversampling lets us pick a richer mix when one side dwarfs
    the other.

    Returns a list of unified entries:
        ``{"kind": "memory" | "turn", "payload": {...}, "rank": int,
           "rrf_score": float}``

    where ``payload`` is the same dict the existing belief / turn paths
    return (``MemoryRecall.to_dict()`` for belief, ``EpisodeTurn.to_dict()``
    for turns), ``rank`` is the 1-indexed position within the originating
    list, and ``rrf_score`` is the per-item RRF contribution.
    """
    from weft.store import search_hybrid

    half_k = max(1, top_k * 2)

    # Both halves need an embedding. Compute once; share across the gather.
    embedding: list[float] | None = None
    if embedder is not None:
        embedding = await embedder.embed(query)

    async def _belief_half() -> list[MemoryRecall]:
        if embedding is None:
            # Belief search_hybrid requires an embedding. Without one we
            # can't fuse a vector signal — fall back to keyword-only by
            # returning an empty list so RRF degrades gracefully.
            from weft.store import search_by_keyword
            return await search_by_keyword(
                pool, query,
                limit=half_k,
                status=status,
                memory_type=memory_type,
                topic=topic,
                project_id=project_id,
                agent_id=agent_id,
                user_id=user_id,
                sources=sources,
                include_agent_provenance=include_agent_provenance,
            )
        return await search_hybrid(
            pool, query, embedding,
            limit=half_k,
            status=status,
            memory_type=memory_type,
            topic=topic,
            project_id=project_id,
            agent_id=agent_id,
            user_id=user_id,
            sources=sources,
            include_agent_provenance=include_agent_provenance,
        )

    # WEFT_HIERARCHICAL=1 swaps the flat recall_turns for the
    # hierarchical episode→turn descent. Same env-var convention as
    # the MCP `_weft_recall_turns` dispatch site so a single env-var
    # toggle covers both turn-only and both-tier paths.
    import os as _os
    _hierarchical = _os.environ.get("WEFT_HIERARCHICAL") == "1"

    async def _turn_half() -> list[EpisodeTurn]:
        if _hierarchical:
            from weft.episode_turns import recall_turns_hierarchical
            return await recall_turns_hierarchical(
                pool, query,
                project_id=project_id,
                since=since,
                until=until,
                top_k_episodes=10,
                top_k_turns=half_k,
                embedding=embedding,
            )
        return await recall_turns(
            pool, query,
            project_id=project_id,
            since=since,
            until=until,
            top_k=half_k,
            embedding=embedding,
        )

    # Parallel by default. When the caller has activated an RLS-scoped
    # connection via ``acquire()`` (contextvar), both halves race on the
    # same single connection — asyncpg refuses concurrent queries, so we
    # serialize. The check is cheap and keeps both call paths correct:
    #
    #   * MCP tool path: outer ``async with acquire(pool)`` sets contextvar
    #     → run sequentially on the bound conn (preserves user_id RLS).
    #   * Direct call path (tests, internal callers): no contextvar →
    #     each half pulls its own pool connection, gather races them.
    from weft.db.connection import _current_conn
    if _current_conn.get(None) is not None:
        belief_results = await _belief_half()
        turn_results = await _turn_half()
    else:
        belief_results, turn_results = await asyncio.gather(
            _belief_half(), _turn_half(),
        )

    # Disjoint RRF: each item gets one term, 1 / (K + rank_in_its_list).
    fused: list[dict[str, Any]] = []
    for i, recall in enumerate(belief_results):
        rank = i + 1
        fused.append({
            "kind": "memory",
            "payload": recall.to_dict(),
            "rank": rank,
            "rrf_score": 1.0 / (_RRF_K + rank),
        })
    for i, turn in enumerate(turn_results):
        rank = i + 1
        fused.append({
            "kind": "turn",
            "payload": turn.to_dict(),
            "rank": rank,
            "rrf_score": 1.0 / (_RRF_K + rank),
        })

    fused.sort(key=lambda e: e["rrf_score"], reverse=True)
    return fused[:top_k]
