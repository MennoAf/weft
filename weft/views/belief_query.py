"""Belief-view query helper — read active claims from belief_claims.

This module provides the PRIMARY lookup step that augments ``weft_recall``
when ``tier='belief'`` is requested.  The caller inserts this lookup BEFORE
the legacy ``memories`` table search; if this returns results the legacy
search is skipped.  If this returns ``[]``, the caller falls through to the
existing belief-tier search.

The resolution strategy is token-overlap against the attribute column:

1. Tokenise the query (lowercase, alphanumeric, drop stop-words).
2. For each token, ILIKE-match against ``attribute``.
3. Rank by overlap count DESC, then ``occurred_at`` DESC.
4. Return at most ``limit`` active claims for the given ``user_id`` + ``scope``.

This is v1: no embedding, no ``attribute_hint``, no ``as_of``, no
``include_history``.  Those deferred features are specified in
docs/architecture/belief_view.md §3 and will be layered on top here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from weft.db.connection import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-word set for v1 token extraction
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "how", "i", "in", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "so", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "do", "does", "did", "tell", "show",
    "give", "about", "any", "some", "most", "recent", "last", "ever",
}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class BeliefClaimResult:
    """One active belief claim returned from the belief_claims table."""

    claim_id: str
    attribute: str
    value: Any
    occurred_at: datetime
    evidence_turn_ids: list[str]
    source_provenance: str
    detector_confidence: float
    overlap_score: int  # number of query tokens matched against attribute

    def to_recall_dict(self) -> dict:
        """Project the claim into a dict compatible with the weft_recall results array.

        Callers receive a uniform structure regardless of whether results came
        from the belief-view or the legacy memories path.
        """
        return {
            "id": self.claim_id,
            "tier": "belief-view",
            "kind": "belief_claim",
            "attribute": self.attribute,
            "value": self.value,
            "content": _render_value(self.value, self.attribute),
            "evidence_turn_ids": self.evidence_turn_ids,
            "source_provenance": self.source_provenance,
            "occurred_at": self.occurred_at.isoformat(),
            "detector_confidence": self.detector_confidence,
            "score": self.overlap_score,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(query: str) -> list[str]:
    """Lowercase, alphanumeric-only, drop stop-words.

    The token list is used for ILIKE matching against attribute names, which
    are dot-namespaced kebab/snake-cased keys (e.g. "trip.recent-family").
    Single-character tokens are also dropped as they produce too many false
    positives against attribute fragments.
    """
    return [
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _render_value(value: Any, attribute: str) -> str:
    """Render the claim's JSONB value as a human-readable string for the
    ``content`` field.  Callers may render their own; this is the default."""
    if isinstance(value, dict):
        parts = [f"{k}={v}" for k, v in value.items()]
        return f"{attribute}: " + ", ".join(parts)
    return f"{attribute}: {value}"


def _load_jsonb(raw: Any) -> Any:
    """Accommodate asyncpg JSONB-as-string-or-dict behaviour.

    asyncpg may return JSONB columns as a plain Python object (dict/list/…)
    when a codec is registered, or as a raw JSON string otherwise.  Mirrors
    the pattern in ``weft/store.py:1208-1212``.
    """
    if isinstance(raw, str):
        return json.loads(raw)
    # Already decoded (dict, list, int, bool, None, …).
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search_belief_claims(
    pool: asyncpg.Pool,
    *,
    query: str,
    user_id: str,
    scope: str = "global",
    limit: int = 10,
) -> list[BeliefClaimResult]:
    """Match active claims by token-overlap against the attribute name.

    Tokenises the query, then for each token does a substring ILIKE match
    against the ``attribute`` column (dot-namespaced + kebab/snake-cased,
    so token overlap is meaningful — "family trip" matches attribute
    "trip.recent-family").  Ranks by overlap count DESC, then
    ``occurred_at`` DESC.  Returns at most ``limit`` rows.

    Falls back to an empty list (not None) when no tokens or no matches.
    The caller is responsible for triggering the legacy memories search
    when this returns ``[]``.

    Args:
        pool: asyncpg connection pool.
        query: Free-text query from the caller.
        user_id: Owner identity; filters to this user's claims only.
        scope: Claim partition, default ``"global"``.
        limit: Maximum rows to return.

    Returns:
        List of :class:`BeliefClaimResult`, possibly empty.
    """
    tokens = _tokenize(query)
    if not tokens:
        logger.debug(
            "search_belief_claims: no tokens after stop-word removal; returning []",
        )
        return []

    # Build a parametrised OR-of-ILIKE for each token.  Each token
    # contributes one CASE WHEN expression to the overlap count.
    # Using ILIKE on the indexed attribute column with kebab/snake keys.
    overlap_terms: list[str] = []
    where_terms: list[str] = []
    params: list[Any] = [user_id, scope]
    idx = 3  # $1=user_id, $2=scope
    for token in tokens:
        like_pattern = f"%{token}%"
        params.append(like_pattern)
        overlap_terms.append(
            f"(CASE WHEN attribute ILIKE ${idx} THEN 1 ELSE 0 END)"
        )
        where_terms.append(f"attribute ILIKE ${idx}")
        idx += 1

    params.append(limit)
    limit_param_idx = idx

    overlap_expr = " + ".join(overlap_terms)
    where_expr = " OR ".join(where_terms)

    sql = f"""
        SELECT claim_id, attribute, value, occurred_at, evidence_turn_ids,
               source_provenance, detector_confidence,
               ({overlap_expr}) AS overlap_score
        FROM belief_claims
        WHERE user_id = $1
          AND scope = $2
          AND status = 'active'
          AND ({where_expr})
        ORDER BY ({overlap_expr}) DESC, occurred_at DESC
        LIMIT ${limit_param_idx}
    """

    db = get_db(pool)
    rows = await db.fetch(sql, *params)

    results: list[BeliefClaimResult] = []
    for r in rows:
        results.append(
            BeliefClaimResult(
                claim_id=r["claim_id"],
                attribute=r["attribute"],
                value=_load_jsonb(r["value"]),
                occurred_at=r["occurred_at"],
                evidence_turn_ids=list(r["evidence_turn_ids"]),
                source_provenance=r["source_provenance"],
                detector_confidence=float(r["detector_confidence"]),
                overlap_score=int(r["overlap_score"]),
            )
        )

    logger.debug(
        "search_belief_claims: query=%r tokens=%r user_id=%s → %d result(s)",
        query[:60],
        tokens,
        user_id,
        len(results),
    )
    return results
