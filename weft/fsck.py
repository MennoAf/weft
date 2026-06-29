"""Weft file system consistency check (fsck).

Lists vector-only-reachable orphan memories — memories that exist in the
embedding index but have no structural graph edges.

Corrected orphan definition (provenance-first spec):
  An active memory is an orphan only when ALL of the following hold:
  * review_status = 'active' (not pending a merge review)
  * topic IS NULL OR topic = '{}' (empty array)
  * AND no entity_mentions row linking it to any entity
  * AND no episode_memories row linking it to any episode
  * AND no memory_relationships row linking it to any other memory

Reading provenance/state (review_status) FIRST ensures fsck does not flag
edges that already have a structural explanation:

  (a) Edges pending a duplicate-belief merge — memories with
      review_status='pending_review' are cross-project merge candidates
      produced by the L2 dedup path (0.6 ≤ sim < 0.85 cross-project).
      They are excluded by the review_status = 'active' predicate.
      Flagging them as defects would misrepresent a planned merge as breakage.

  (b) Blessed dream-links between distinct memories — any memory that has
      at least one row in memory_relationships (source_id or target_id) is
      explicitly connected to another memory.  Such a link is the structural
      evidence that the memory is reachable via the graph, not purely by
      vector similarity.  Flagging it would generate false defect reports
      for every memory the consolidation pipeline has already linked.

Asymmetric scoping (weft-ab1e37c0) — the memory_relationships check carries
NO project_id filter.  This is deliberate and must not be changed to a
"uniform" project-scoped check:
  • Catalog half (loc_key): project_id is a hard wall.  memory_relationships
    has no project_id column, so catalog links are tested by their mere
    presence — that is sufficient for the catalog half.
  • Associative half (beliefs): project_id is a soft boost; beliefs leak
    across projects by design.  Adding a project_id filter here would cause
    cross-project dream-links to be silently ignored, marking legitimately
    linked belief memories as orphans.
  Add no project_id predicate to the memory_relationships EXISTS subquery.
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

    Provenance-first spec: reads review_status BEFORE judging a memory a
    defect.  A memory is orphan only when it has no structural graph edge that
    explains its existence:

      * review_status = 'active'
          → excludes edges pending a duplicate-belief merge (case a).
            L2 marks cross-project merge candidates (0.6 ≤ sim < 0.85) as
            'pending_review'; they are NOT defects.

      * no topic tags, entity_mentions, or episode_memories
          → the usual structural-edge checks.

      * no memory_relationships row (source_id or target_id)
          → excludes blessed dream-links between distinct memories (case b).
            Any explicit relationship is structural evidence of reachability.

    Asymmetric scoping: the memory_relationships subquery has no project_id
    filter.  Catalog memories (hard wall) are handled by presence alone;
    associative memories (soft boost) must recognise cross-project links.
    Do NOT unify scoping across the two halves.

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

    # Shared orphan predicate — provenance/state read first:
    #   1. review_status = 'active'  → exclude pending merge candidates (a)
    #   2. topic absent               → no tag edge
    #   3. entity_mentions absent     → no entity edge
    #   4. episode_memories absent    → no episode edge
    #   5. memory_relationships absent → no dream-link (b); NO project_id filter
    _ORPHAN_PREDICATE = """
        status = 'active'
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
        AND NOT EXISTS (
            -- Asymmetric scoping: no project_id filter here.
            -- Catalog half: presence alone is sufficient for the hard-wall test.
            -- Associative half (beliefs): cross-project links must not be ignored.
            SELECT 1 FROM memory_relationships mr
            WHERE mr.source_id = memories.id
               OR mr.target_id = memories.id
        )
    """

    # RLS-scoped query: respect user_id filtering + workspace membership.
    # Mirrors the scoping in store.list_memories() but with the orphan filter.
    if user_id is not None:
        # Match owner-scoped + system-global rows, plus rows in workspaces
        # where the caller is a member. Mirrors memories_select RLS (mig 36).
        rows = await db.fetch(
            f"""
            SELECT id AS memory_id, 'vector-only reachable' AS reason
            FROM memories
            WHERE {_ORPHAN_PREDICATE}
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
            f"""
            SELECT id AS memory_id, 'vector-only reachable' AS reason
            FROM memories
            WHERE {_ORPHAN_PREDICATE}
            ORDER BY created_at DESC
            """,
        )

    return [{"memory_id": row["memory_id"], "reason": row["reason"]} for row in rows]
