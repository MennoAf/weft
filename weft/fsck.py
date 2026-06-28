"""Weft file system consistency check (fsck).

Lists vector-only-reachable orphan memories — memories that exist in the
embedding index but have no tag/entity/episode edges. This is the leading
indicator of future recall misses.

Orphan definition: an active memory with:
  * topic IS NULL OR topic = '{}' (empty array)
  * AND no entity_mentions row linking it to any entity
  * AND no episode_memories row linking it to any episode

The query uses NOT EXISTS subqueries to ensure all edge types are checked,
so no orphan is missed and trust in the tool is preserved.
"""

from __future__ import annotations

import asyncpg

from weft.db.connection import get_db
from weft.schema import SYSTEM_GLOBAL_USER_ID


async def list_orphan_memories(
    pool: asyncpg.Pool,
    user_id: str | None = None,
) -> list[dict]:
    """List memories reachable ONLY by vector cosine (orphan memories).

    An orphan memory is an active memory that:
      * has NO topic tags (topic IS NULL or topic = '{}')
      * has NO entity_mentions link
      * has NO episode_memories link

    All three edge types must be absent for a memory to be orphan, or the
    tool over-reports and kills user trust.

    Args:
        pool: Database connection pool.
        user_id: If provided, filters to memories owned by this user OR
                 globally-scoped memories (SYSTEM_GLOBAL_USER_ID sentinel).
                 If None, returns all orphans visible to the current RLS context.

    Returns:
        List of dicts with keys: memory_id, reason.
        reason is always "vector-only reachable" (the leading cause).
    """
    db = get_db(pool)

    # RLS-scoped query: respect user_id filtering + workspace membership.
    # Mirrors the scoping in store.list_memories() but with the orphan filter.
    if user_id is not None:
        # Match owner-scoped + system-global rows, plus rows in workspaces
        # where the caller is a member. Mirrors memories_select RLS (mig 36).
        rows = await db.fetch(
            """
            SELECT id AS memory_id, 'vector-only reachable' AS reason
            FROM memories
            WHERE status = 'active'
              AND review_status = 'active'
              AND (topic IS NULL OR topic = '{}')
              AND NOT EXISTS (
                SELECT 1 FROM entity_mentions em
                WHERE em.memory_id = memories.id
              )
              AND NOT EXISTS (
                SELECT 1 FROM episode_memories em_ep
                WHERE em_ep.memory_id = memories.id
              )
              AND (
                user_id = $1
                OR user_id = $2
                OR (
                  workspace_id IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM workspace_members wm
                    WHERE wm.workspace_id = memories.workspace_id
                    AND wm.member_identity->>'user_id' = $1
                  )
                )
              )
            ORDER BY created_at DESC
            """,
            user_id,
            SYSTEM_GLOBAL_USER_ID,
        )
    else:
        # No user filter — return orphans visible to current RLS context
        # (which may be empty if no user is authenticated).
        rows = await db.fetch(
            """
            SELECT id AS memory_id, 'vector-only reachable' AS reason
            FROM memories
            WHERE status = 'active'
              AND review_status = 'active'
              AND (topic IS NULL OR topic = '{}')
              AND NOT EXISTS (
                SELECT 1 FROM entity_mentions em
                WHERE em.memory_id = memories.id
              )
              AND NOT EXISTS (
                SELECT 1 FROM episode_memories em_ep
                WHERE em_ep.memory_id = memories.id
              )
            ORDER BY created_at DESC
            """,
        )

    return [{"memory_id": row["memory_id"], "reason": row["reason"]} for row in rows]
