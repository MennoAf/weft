"""Replay linkage: map a missed/re-asked query to the implicated episode turns.

Given a query that produced a retrieval miss (or was re-asked), this module
resolves the turn_ids of the episode whose belief coverage was inadequate.
The turn_ids are what the downstream replay enqueuer (L3) needs to feed back
into the memory pipeline.

Resolution strategy (first successful path wins):

(a) Belief-claim path — PRIMARY
    Tokenise the query against ``belief_claims.attribute`` (the same token-
    overlap used by :func:`weft.views.belief_query.search_belief_claims`).
    The resulting active claims carry ``evidence_turn_ids`` — the exact turns
    that were cited as evidence when the claim was extracted.  These are the
    turns that were used to form the belief that *should* have answered the
    query; replaying them is the most targeted correction.

(b) Nearest-episode path — FALLBACK
    When no matching belief claims exist (e.g., the topic was never extracted
    into a belief at all), fall back to a keyword search over ``episode_turns``
    via :func:`weft.episode_turns.recall_turns`.  The top-K turns from the
    nearest episode constitute the implicated set.  This is broader than (a)
    but guarantees a non-empty result when the belief layer has no coverage.

Return value is always a deduplicated list of turn_id strings.  The list is
empty only when the database has no episode_turns at all (pathological case).
Callers that need to enqueue should check for an empty list and skip.

Spec: loom-c483a5dc (E1.L2 in the replay loop epic loom-8a9a0ff0).
Depends on: loom-a6fa27eb (replay_queue table, migration v53).

enqueue_replay_on_miss (E1.L3, loom-5d414368):
    Called from apply_reask_feedback (store.py) after a fresh claim.
    Resolves implicated turns, groups them by episode, and inserts one
    pending replay_queue row per distinct episode (idempotent).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import asyncpg

from weft.db.connection import get_db
from weft.views.belief_query import search_belief_claims

logger = logging.getLogger(__name__)


@dataclass
class ReaskPair:
    """A (original_query, reask_query) pair as produced by detect_reasked_queries.

    Callers may pass either a free-form ``missed_query`` string or a
    :class:`ReaskPair`; the resolution logic uses the ``reask_query`` text
    (the more-recent, more-refined formulation) to search for implicated turns.
    """
    original_query: str
    reask_query: str


async def resolve_implicated_turns(
    pool: asyncpg.Pool,
    query_or_pair: str | ReaskPair,
    *,
    user_id: str | None = None,
    scope: str = "global",
    top_k: int = 20,
    fallback_top_k: int = 10,
) -> list[str]:
    """Resolve implicated turn_ids for a retrieval miss or re-asked query.

    Given a missed query or a (original, reask) pair from
    :func:`weft.reask.detect_reasked_queries`, returns the turn_ids of the
    episode whose belief coverage was inadequate.

    Resolution strategy (first successful path wins):

    **(a) Belief-claim path (primary):** search active ``belief_claims`` by
    token-overlap on ``attribute`` (same tokenisation as
    :func:`weft.views.belief_query.search_belief_claims`).  Collect the
    ``evidence_turn_ids`` from all matching claims and return the union.
    This is the most targeted path: it names the exact turns that were
    cited when the belief was formed, so replaying them re-runs extraction
    from the primary source.

    **(b) Nearest-episode path (fallback):** when no belief claims match
    the query — topic never extracted, belief layer cold-start, etc. —
    run a keyword search over ``episode_turns`` using
    ``weft.episode_turns.recall_turns``.  The ``top_k`` highest-scoring
    turns (keyword path, no vector) are returned.  This is broader but
    guarantees coverage.

    Args:
        pool: asyncpg connection pool. RLS GUC (``app.user_id``) is expected
            to be set on the pool connection via the ``setup`` callback; callers
            operating outside the normal request path must set it explicitly.
        query_or_pair: Either a plain missed-query string or a
            :class:`ReaskPair`. When a ``ReaskPair`` is supplied the
            ``reask_query`` text is used for the search (the later, more-
            refined formulation).
        user_id: If supplied, scopes the belief-claim search to this user.
            Defaults to None (uses the pool-level RLS GUC).
        scope: belief_claims scope partition; default ``"global"``.
        top_k: candidate limit for the keyword search in the belief-claim path
            (passed as ``limit`` to :func:`search_belief_claims`).
        fallback_top_k: candidate limit for the nearest-episode fallback.

    Returns:
        Deduplicated list of turn_id strings (``et-*`` prefix), in the order
        they were encountered across matching claims (path a) or by relevance
        rank (path b).  Empty list iff the database has no matching turns at all.
    """
    # Normalise: extract the query text to search with.
    if isinstance(query_or_pair, ReaskPair):
        query_text = query_or_pair.reask_query
    else:
        query_text = query_or_pair

    # Resolve user_id from GUC when not passed explicitly.
    effective_user_id = user_id
    if effective_user_id is None:
        # Pull the GUC so the belief-claim path has an explicit user scope.
        # Falls back to "" (empty) which will produce no results from the
        # belief-claim lookup — the fallback path will then run.
        try:
            effective_user_id = await get_db(pool).fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
        except Exception as exc:
            logger.warning(
                "resolve_implicated_turns: could not read app.user_id GUC: %s", exc,
            )
            effective_user_id = ""

    # --- Path (a): belief-claim evidence_turn_ids ---
    if effective_user_id:
        try:
            claims = await search_belief_claims(
                pool,
                query=query_text,
                user_id=effective_user_id,
                scope=scope,
                limit=top_k,
            )
        except Exception as exc:
            logger.warning(
                "resolve_implicated_turns: belief-claim search failed, "
                "falling through to nearest-episode path: %s", exc,
            )
            claims = []
    else:
        claims = []

    if claims:
        seen: set[str] = set()
        turn_ids: list[str] = []
        for claim in claims:
            for tid in claim.evidence_turn_ids:
                if tid not in seen:
                    seen.add(tid)
                    turn_ids.append(tid)
        logger.debug(
            "resolve_implicated_turns: path=belief-claim query=%r "
            "claims=%d turn_ids=%d",
            query_text[:60],
            len(claims),
            len(turn_ids),
        )
        return turn_ids

    # --- Path (b): nearest-episode via keyword turn recall ---
    logger.debug(
        "resolve_implicated_turns: path=nearest-episode (belief-claim miss) query=%r",
        query_text[:60],
    )
    try:
        from weft.episode_turns import recall_turns  # late import to keep graph minimal

        turns = await recall_turns(
            pool,
            query_text,
            top_k=fallback_top_k,
            # No embedding — keyword-only path avoids needing an embedder here.
            embedding=None,
        )
        turn_ids = [t.id for t in turns]
        logger.debug(
            "resolve_implicated_turns: nearest-episode returned %d turns",
            len(turn_ids),
        )
        return turn_ids
    except Exception as exc:
        logger.warning(
            "resolve_implicated_turns: nearest-episode fallback failed: %s", exc,
        )
        return []


async def enqueue_replay_on_miss(
    pool: asyncpg.Pool,
    query_text: str,
    user_id: str,
    *,
    reason: str = "reask-miss",
    top_k: int = 20,
    fallback_top_k: int = 10,
) -> int:
    """Enqueue pending replay_queue rows for a detected retrieval miss.

    Called from :func:`weft.store.apply_reask_feedback` immediately after a
    fresh claim of a miss row (the atomically-flipped ``is_reask_miss`` UPDATE).
    Resolves the implicated episode turns via
    :func:`resolve_implicated_turns`, groups them by episode, and inserts ONE
    ``pending`` ``replay_queue`` row per distinct episode.

    Idempotency: uses a claim-first INSERT … WHERE NOT EXISTS pattern so that
    repeated calls for the same episode produce no duplicate ``pending`` row.
    If a pending row already exists for an episode, that episode is silently
    skipped.  A ``done`` row does NOT block a new ``pending`` row — replaying
    an episode twice when a second miss is detected is correct behaviour.

    Args:
        pool: asyncpg connection pool.  The RLS GUC (``app.user_id``) must
            already be set on the pool/connection to satisfy the NOT NULL
            constraint and RLS INSERT policy on ``replay_queue``.
        query_text: The original missed query text, used to resolve implicated
            turns (forwarded to :func:`resolve_implicated_turns`).
        user_id: The user ID for the ``replay_queue.user_id`` column and for
            scoping the turn/belief resolution.
        reason: Free-text label stored in ``replay_queue.reason``.
            Defaults to ``"reask-miss"``.
        top_k: Belief-claim path candidate limit (forwarded).
        fallback_top_k: Nearest-episode fallback candidate limit (forwarded).

    Returns:
        Number of new ``replay_queue`` rows inserted (0 when all episodes
        already had a pending row or when no turns could be resolved).

    Spec: loom-5d414368 (E1.L3 in replay loop epic loom-8a9a0ff0).
    """
    turn_ids = await resolve_implicated_turns(
        pool,
        query_text,
        user_id=user_id,
        top_k=top_k,
        fallback_top_k=fallback_top_k,
    )
    if not turn_ids:
        logger.debug(
            "enqueue_replay_on_miss: no implicated turns for query=%r user=%s",
            query_text[:60],
            user_id,
        )
        return 0

    # Map turn_ids → episode_id.  Each turn belongs to exactly one episode.
    # We query episode_turns for the subset of turn_ids that were resolved.
    rows = await get_db(pool).fetch(
        """
        SELECT id, episode_id
        FROM episode_turns
        WHERE id = ANY($1::text[])
        """,
        turn_ids,
    )
    if not rows:
        logger.debug(
            "enqueue_replay_on_miss: no episode_turns found for %d turn_ids",
            len(turn_ids),
        )
        return 0

    # Group turn_ids by episode_id, preserving the resolution order.
    episode_turns: dict[str, list[str]] = {}
    for row in rows:
        episode_id = row["episode_id"]
        turn_id = row["id"]
        if episode_id not in episode_turns:
            episode_turns[episode_id] = []
        episode_turns[episode_id].append(turn_id)

    inserted = 0
    for episode_id, ep_turn_ids in episode_turns.items():
        rq_id = f"rq-{uuid.uuid4().hex[:10]}"
        # Claim-first idempotency: only INSERT when no pending row exists for
        # this episode.  A 'done' row does not block a new 'pending' row.
        status = await get_db(pool).execute(
            """
            INSERT INTO replay_queue (id, episode_id, turn_ids, reason, status, user_id)
            SELECT $1, $2, $3, $4, 'pending', $5
            WHERE NOT EXISTS (
                SELECT 1 FROM replay_queue
                WHERE episode_id = $2
                  AND status = 'pending'
            )
            """,
            rq_id,
            episode_id,
            ep_turn_ids,
            reason,
            user_id,
        )
        if status == "INSERT 0 1":
            inserted += 1
            logger.debug(
                "enqueue_replay_on_miss: queued episode=%s turns=%d id=%s",
                episode_id,
                len(ep_turn_ids),
                rq_id,
            )
        else:
            logger.debug(
                "enqueue_replay_on_miss: episode=%s already has a pending row — skipped",
                episode_id,
            )

    return inserted
