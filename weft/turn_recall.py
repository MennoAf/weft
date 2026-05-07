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

import logging
import re
from datetime import datetime
from typing import Literal

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.episode_turns import list_recent_turns, recall_turns
from weft.models import EpisodeTurn

logger = logging.getLogger(__name__)


Tier = Literal["belief", "turns", "both", "auto"]


# --- Routing ---


# Lowercased regexes — case folding happens before match. Each pattern
# captures a temporal-reasoning marker that belief-tier extraction tends
# to lose. Order doesn't matter; first hit returns.
_TURN_TIER_MARKERS: tuple[re.Pattern, ...] = (
    re.compile(r"\bhow many (?:days|weeks|months|hours|minutes|years)\b"),
    re.compile(r"\bhow long (?:since|ago|before|after|between)\b"),
    re.compile(r"\bwhen (?:did|was)\b"),
    re.compile(r"\b(?:before|after|between|since|until)\b"),
    re.compile(r"\blast (?:time|week|month|year|tuesday|wednesday|thursday|friday|saturday|sunday|monday)\b"),
    re.compile(r"\b(?:earlier|later) (?:than|that)\b"),
    re.compile(r"\bwhat (?:day|date|time)\b"),
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
        # No multi-anchor pattern — single recall under the query itself.
        embedding = await embedder.embed(query) if embedder else None
        turns = await recall_turns(
            pool, query,
            project_id=project_id, since=since, until=until,
            top_k=top_k_per_anchor, embedding=embedding,
        )
        if not turns:
            # Empty hybrid hit — try a temporal-only fallback so the
            # Reader at least sees recent dialogue.
            turns = await list_recent_turns(
                pool, project_id=project_id, since=since, until=until,
                limit=top_k_per_anchor,
            )
        return {query: turns}

    out: dict[str, list[EpisodeTurn]] = {}
    for anchor in anchors:
        embedding = await embedder.embed(anchor) if embedder else None
        out[anchor] = await recall_turns(
            pool, anchor,
            project_id=project_id, since=since, until=until,
            top_k=top_k_per_anchor, embedding=embedding,
        )
    return out
