"""Topic-anchored complete memory gather (Tier-1, V1).

Gathers ALL active memories under one or more topic tags for the calling user,
with NO limit cap — proving V1 completeness. Uses the ``= ANY(topic)`` predicate
already established in ``weft/store.py`` but issues its own unbounded query
rather than routing through the limit=10/50-capped search functions.

Secondary: merges entity-linked memories via an inline entity_mentions JOIN
(scoped to the caller's user-visibility predicate, capped at LIMIT 100). If
the entity set hits that 100-cap, ``truncated=True`` is returned to surface
the incompleteness.

RLS enforcement: sets ``current_user_id`` contextvar before calling ``acquire()``
so the RLS SELECT policy filters correctly. Also adds application-level user_id
WHERE filtering (mirroring ``store.list_memories``) for defense-in-depth.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import asyncpg

from weft.db.connection import acquire
from weft.models import Memory
from weft.schema.versioning import SYSTEM_GLOBAL_USER_ID
from weft.store import _row_to_memory

logger = logging.getLogger(__name__)

# The LIMIT on the entity-secondary query — if the entity-linked set reaches
# this number, we cannot know whether rows were dropped.
_ENTITY_MEMORIES_LIMIT = 100


class TopicGatherResult(TypedDict):
    memories: list[Memory]
    complete: bool
    truncated: bool


async def gather_topic_memories(
    pool: asyncpg.Pool,
    tags: list[str],
    user_id: str,
    budget_tokens: int = 2000,
) -> TopicGatherResult:
    """Gather ALL active memories under the given topic tags (V1 completeness).

    Issues an UNBOUNDED ``SELECT * FROM memories WHERE status='active' AND
    (<tag> = ANY(topic) OR ...) ORDER BY created_at ASC`` — no LIMIT —
    so the full membership set is returned regardless of count.

    ``budget_tokens`` governs downstream digest synthesis, NOT a row cap
    here.  Every matching active memory is returned; ``complete`` reflects
    whether the primary gather was unbounded (always True from the primary
    path).  ``truncated`` is set True when the entity-graph secondary merge
    would exceed the LIMIT 100 cap on the entity-secondary query.

    Args:
        pool: asyncpg pool.
        tags: One or more ``memories.topic[]`` tags to gather under.
        user_id: The calling user's ID — used to set RLS context via acquire()
            and for application-level WHERE filtering (defense-in-depth).
        budget_tokens: Token budget hint for downstream synthesis. Does NOT
            cap the primary gather.

    Returns:
        TopicGatherResult with keys:
            - memories: List of Memory objects ordered by created_at ASC.
            - complete: True when the primary gather returned all matching rows.
            - truncated: True when entity secondary set hit the LIMIT 100 cap.
    """
    from weft.auth import current_user_id

    token = current_user_id.set(user_id)
    try:
        return await _gather(pool, tags, user_id, budget_tokens)
    finally:
        current_user_id.reset(token)


async def _gather(
    pool: asyncpg.Pool,
    tags: list[str],
    user_id: str,
    budget_tokens: int,
) -> TopicGatherResult:
    """Inner gather — runs inside the user_id contextvar already set."""
    if not tags:
        return TopicGatherResult(memories=[], complete=True, truncated=False)

    async with acquire(pool) as conn:
        # Build unbounded WHERE clause using = ANY(topic) for each tag.
        # Multiple tags are OR'd so any tag match is included.
        # This is the same predicate idiom as store.py:186/303/414 but
        # WITHOUT the LIMIT cap from store.py:142/242/367/487.
        conditions: list[str] = []
        params: list = []
        idx = 1

        conditions.append("status = 'active'")

        tag_clauses = []
        for tag in tags:
            tag_clauses.append(f"${idx} = ANY(topic)")
            params.append(tag)
            idx += 1

        conditions.append("(" + " OR ".join(tag_clauses) + ")")

        # Application-level user_id filter — mirrors list_memories lines 206-219
        # (store.py) for defense-in-depth on top of RLS. Matches the calling
        # user's rows + system-global sentinel rows + workspace-member rows.
        conditions.append(
            f"(user_id = ${idx} OR user_id = ${idx + 1} OR ("
            f"workspace_id IS NOT NULL AND EXISTS ("
            f"SELECT 1 FROM workspace_members wm "
            f"WHERE wm.workspace_id = memories.workspace_id "
            f"AND wm.member_identity->>'user_id' = ${idx}"
            f")))"
        )
        params.append(user_id)
        params.append(SYSTEM_GLOBAL_USER_ID)
        idx += 2

        where = "WHERE " + " AND ".join(conditions)

        # Unbounded query — no LIMIT
        sql = f"""
            SELECT * FROM memories
            {where}
            ORDER BY created_at ASC
        """

        rows = await conn.fetch(sql, *params)
        primary_memories: list[Memory] = [_row_to_memory(r) for r in rows]

        # --- Secondary: entity-linked memories ---
        # Look up entities linked to memories in the primary set, then gather
        # additional entity-linked memories that may not be in the primary set.
        # Uses an inline query with LIMIT 100 — if the set hits the cap, truncated=True.
        #
        # Defense-in-depth: applies the same user-visibility predicate as the
        # primary query (caller-owned OR system-global OR workspace-member) so
        # that the secondary path cannot leak other users' memories even when
        # the pool runs as a superuser (e.g., in testcontainers environments
        # where RLS is bypassed entirely).
        truncated = False
        seen_ids: set[str] = {m.id for m in primary_memories}

        # Find entity IDs linked to any of our primary memories
        if seen_ids:
            entity_rows = await conn.fetch(
                """
                SELECT DISTINCT em.entity_id
                FROM entity_mentions em
                WHERE em.memory_id = ANY($1::text[])
                """,
                list(seen_ids),
            )
            entity_ids = [r["entity_id"] for r in entity_rows]

            # For each entity, gather its memories with user-visibility filter.
            # Mirrors the primary query's visibility predicate:
            #   user_id = caller OR user_id = SYSTEM_GLOBAL_USER_ID OR workspace-member
            for entity_id in entity_ids:
                entity_mem_rows = await conn.fetch(
                    """
                    SELECT m.* FROM memories m
                    JOIN entity_mentions em ON m.id = em.memory_id
                    WHERE em.entity_id = $1
                      AND m.status = 'active'
                      AND (
                        m.user_id = $2
                        OR m.user_id = $3
                        OR (
                          m.workspace_id IS NOT NULL
                          AND EXISTS (
                            SELECT 1 FROM workspace_members wm
                            WHERE wm.workspace_id = m.workspace_id
                              AND wm.member_identity->>'user_id' = $2
                          )
                        )
                      )
                    ORDER BY em.mentioned_at DESC
                    LIMIT $4
                    """,
                    entity_id,
                    user_id,
                    SYSTEM_GLOBAL_USER_ID,
                    _ENTITY_MEMORIES_LIMIT,
                )
                entity_mems = [_row_to_memory(r) for r in entity_mem_rows]

                # If we got exactly the limit cap, rows may have been dropped
                if len(entity_mems) >= _ENTITY_MEMORIES_LIMIT:
                    truncated = True

                # Merge in memories not already in the primary set
                for mem in entity_mems:
                    if mem.id not in seen_ids:
                        primary_memories.append(mem)
                        seen_ids.add(mem.id)

        # Re-sort merged set by (created_at, id) ASC for deterministic tiebreak
        primary_memories.sort(key=lambda m: (m.created_at, m.id))

        return TopicGatherResult(
            memories=primary_memories,
            complete=True,
            truncated=truncated,
        )
