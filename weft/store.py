"""Postgres store — ONLY writer to the database for memory data.

Handles CRUD operations, relationship management, and vector similarity search.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import asyncpg

from weft.auth import current_user_id, get_caller_mode
from weft.db.connection import acquire, get_db
from weft.models import (
    Memory,
    MemoryCreate,
    MemoryRecall,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    RelationType,
    _weft_id,
)
from weft.quarantine import looks_like_instruction
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_UNSET = object()  # sentinel: distinguish "not provided" from explicit None


async def store_memory(
    pool: asyncpg.Pool,
    create: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    """Store a new memory. Returns the created Memory.

    Phase 2:
    * Layer 1 — stamps ``write_provenance`` from the request's caller-mode
      contextvar. Defaults to 'supervisor' when no HTTP middleware is in
      play (CLI, scheduler, internal callers). Agent-container callers
      that prepend ``X-Weft-Caller-Mode: agent`` get tagged 'agent'.
    * Layer 3 — agent-mode writes whose content reads as an instruction
      (URLs, git remotes, imperative verbs, "when X do Y" structure,
      absolute system paths, API endpoints) land in
      ``review_status='pending_review'`` and stay out of retrieval until
      the supervisor reviews them via :mod:`weft.quarantine`.
    """
    memory_id = _weft_id()
    now = datetime.now(timezone.utc)
    token_count = estimate_tokens(create.content)
    write_provenance = get_caller_mode()
    review_status = "active"
    if write_provenance == "agent" and looks_like_instruction(create.content):
        review_status = "pending_review"
        logger.info(
            "quarantine: agent-provenance write flagged as pending_review (id=%s)",
            memory_id,
        )

    db = get_db(pool)
    await db.execute(
        """
        INSERT INTO memories (
            id, type, topic, content, source, confidence,
            token_count, created_at, updated_at, accessed_at,
            access_count, project_id, agent_id, embedding, status, pinned,
            review_after, workspace_id, user_id, write_provenance, review_status
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $8, $8,
            0, $9, $10, $11::vector, 'active', $12,
            $13, $14, nullif(current_setting('app.user_id', true), ''), $15, $16
        )
        """,
        memory_id,
        create.type.value,
        create.topic,
        create.content,
        create.source.value,
        create.confidence,
        token_count,
        now,
        create.project_id,
        create.agent_id,
        embedding,
        create.pinned,
        create.review_after,
        create.workspace_id,
        write_provenance,
        review_status,
    )

    # V3 write-invalidation: flip any cached digest for this memory's topic tags
    # to stale=true so the next read regenerates a fresh digest.
    #
    # Design note — resolving "same transaction boundary" vs "best-effort non-raising":
    # We call mark_stale_for_tags via get_db(pool) (inside topic_digest_cache),
    # which reuses _current_conn when inside an acquire() context. This means the
    # stale flip is on the SAME connection/transaction as the memory INSERT (atomic
    # on the happy path). The try/except in mark_stale_for_tags ensures that any
    # DB error here is swallowed with a warning — it never aborts the memory write.
    if create.topic:
        try:
            from weft.topic_digest_cache import mark_stale_for_tags
            user_id_val = current_user_id.get()
            if user_id_val:
                await mark_stale_for_tags(pool, user_id_val, create.topic)
        except Exception as _exc:
            logger.warning(
                "store_memory: digest invalidation hook failed (id=%s): %s",
                memory_id,
                _exc,
            )

    return Memory(
        id=memory_id,
        type=create.type,
        topic=create.topic,
        content=create.content,
        source=create.source,
        confidence=create.confidence,
        token_count=token_count,
        created_at=now,
        updated_at=now,
        accessed_at=now,
        access_count=0,
        project_id=create.project_id,
        agent_id=create.agent_id,
        workspace_id=create.workspace_id,
        status=MemoryStatus.active,
        pinned=create.pinned,
        review_after=create.review_after,
        write_provenance=write_provenance,
        review_status=review_status,
    )


async def get_memory(pool: asyncpg.Pool, memory_id: str) -> Memory | None:
    """Fetch a single memory by ID. Returns None if not found."""
    row = await get_db(pool).fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
    if not row:
        return None
    return _row_to_memory(row)


async def list_memories(
    pool: asyncpg.Pool,
    *,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    pinned: bool | None = None,
    exact_scope: bool = False,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[Memory]:
    """List memories with optional filters.

    Scoping: pass project_id and/or agent_id to narrow results.
    Each axis uses OR-NULL logic (matches the value OR global memories)
    by default.  Pass exact_scope=True to match the exact project_id /
    agent_id without the OR-NULL fallback (useful for pruning operations
    that should not touch global or cross-project records).
    Omit both for brain-wide (unscoped) queries.

    user_id: If provided, filters to memories owned by this user OR globally-scoped
    memories (user_id = SYSTEM_GLOBAL_USER_ID sentinel). If None, returns all.

    include_agent_provenance: Phase 2 / Layer 2 filter. ``False`` excludes
    rows written in agent-mode — used by retrieval paths feeding agent
    system prompts (mode='code'). ``True`` (default) preserves legacy
    behavior; Face-context callers wrap agent rows at projection time
    via ``retrieval_modes.wrap_untrusted_for_face``.

    include_pending_review: Phase 2 / Layer 3 filter. ``False`` (default)
    excludes quarantined writes; only ``weft_quarantine_review`` opts in.
    """
    conditions = []
    params: list = []
    idx = 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if not include_agent_provenance:
        conditions.append("write_provenance != 'agent'")
    if not include_pending_review:
        conditions.append("review_status = 'active'")

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        if exact_scope:
            conditions.append(f"project_id = ${idx}")
        else:
            conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        if exact_scope:
            conditions.append(f"agent_id = ${idx}")
        else:
            conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        # Match owner-scoped + system-global rows, plus rows in workspaces
        # where the caller is a member. Mirrors memories_select RLS (mig 36).
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

    if pinned is not None:
        conditions.append(f"pinned = ${idx}")
        params.append(pinned)
        idx += 1

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT * FROM memories {where}
        ORDER BY updated_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    rows = await get_db(pool).fetch(query, *params)
    return [_row_to_memory(r) for r in rows]


async def search_by_vector(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    limit: int = 10,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    exclude_ids: list[str] | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
    facet_boost_project_id: str | None = None,
) -> list[MemoryRecall]:
    """Search memories by vector similarity (cosine distance).

    Scoping: pass project_id and/or agent_id to narrow results.
    Each axis uses OR-NULL logic (matches the value OR global memories).
    Omit both for brain-wide (unscoped) queries.

    user_id: If provided, filters to memories owned by this user OR globally-scoped
    memories (user_id = SYSTEM_GLOBAL_USER_ID sentinel). If None, returns all.

    exclude_ids: memory IDs to exclude from results (e.g., already surfaced
    by primer). Uses NOT id = ANY($N) for efficient filtering.

    sources: allowlist of MemorySource values. None = no filter. Passed pre-ANN
    so top-K stays meaningful when excluded sources dominate the pool.

    include_agent_provenance / include_pending_review: Phase 2 Layer 2/3
    filters. See ``list_memories`` for semantics. Both filters are pushed
    into the WHERE clause so the ANN top-K stays meaningful when agent /
    quarantined rows dominate the pool.

    facet_boost_project_id: When set, the hard project_id wall is dropped and
    a post-query ranking boost (_FACET_BOOST) is applied to memories whose
    project_facets contains this project.  Use for associative/belief recall
    where a belief shared across projects should surface in all of them.
    The catalog path (retrieval_mode='code') must NOT use this parameter —
    pass project_id for the hard wall instead.
    """
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    # Similarity threshold in SQL so the DB handles filtering atomically
    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if not include_agent_provenance:
        conditions.append("write_provenance != 'agent'")
    if not include_pending_review:
        conditions.append("review_status = 'active'")

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    # Facet-boost path: no project wall — beliefs surface across projects,
    # ranked up when project_facets contains the caller's project.
    # Catalog path: project_id wall is kept for hard scoping.
    if facet_boost_project_id is None and project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        # Match owner-scoped + system-global rows, plus rows in workspaces
        # where the caller is a member. Mirrors memories_select RLS (mig 36).
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

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    if sources:
        conditions.append(f"source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    # Over-fetch when facet boost is active so the re-rank has enough
    # candidates to surface boosted items that the raw distance order missed.
    fetch_limit = limit * 2 if facet_boost_project_id is not None else limit

    query = f"""
        SELECT *,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        {where}
        ORDER BY embedding <=> $1::vector, id
        LIMIT ${idx}
    """
    params.append(fetch_limit)

    rows = await get_db(pool).fetch(query, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        sim = float(row["similarity"])
        if facet_boost_project_id is not None and facet_boost_project_id in memory.project_facets:
            sim = sim * _FACET_BOOST
        results.append(MemoryRecall(memory=memory, similarity=sim))

    if facet_boost_project_id is not None:
        # Re-rank by boosted similarity descending; tie-break by id for determinism.
        results.sort(key=lambda r: (-r.similarity, r.memory.id))
        results = results[:limit]

    return results


async def search_by_keyword(
    pool: asyncpg.Pool,
    query: str,
    *,
    limit: int = 10,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    exclude_ids: list[str] | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
    facet_boost_project_id: str | None = None,
) -> list[MemoryRecall]:
    """Search memories by full-text keyword match (BM25 ranking via ts_rank).

    Uses the search_tsv tsvector column with plainto_tsquery for robust
    keyword matching including stemming and stop-word removal.

    user_id: If provided, filters to memories owned by this user OR globally-scoped
    memories (user_id = SYSTEM_GLOBAL_USER_ID sentinel). If None, returns all.

    include_agent_provenance / include_pending_review: see ``list_memories``.

    facet_boost_project_id: When set, drops the project_id wall and boosts
    memories whose project_facets contains this project.  Mirror of the
    same parameter on search_by_vector.
    """
    conditions = ["search_tsv IS NOT NULL"]
    params: list = []
    idx = 1

    # $1 = tsquery
    conditions.append(f"search_tsv @@ plainto_tsquery('english', ${idx})")
    params.append(query)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if not include_agent_provenance:
        conditions.append("write_provenance != 'agent'")
    if not include_pending_review:
        conditions.append("review_status = 'active'")

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    # Facet-boost path: no project wall — beliefs surface across projects.
    # Catalog path: keep the hard wall.
    if facet_boost_project_id is None and project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if user_id is not None:
        # Match owner-scoped + system-global rows, plus rows in workspaces
        # where the caller is a member. Mirrors memories_select RLS (mig 36).
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

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    if sources:
        conditions.append(f"source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    # Over-fetch when facet boost is active so the re-rank has candidates.
    fetch_limit = limit * 2 if facet_boost_project_id is not None else limit

    sql = f"""
        SELECT *,
               ts_rank(search_tsv, plainto_tsquery('english', $1)) AS rank
        FROM memories
        {where}
        ORDER BY rank DESC
        LIMIT ${idx}
    """
    params.append(fetch_limit)

    rows = await get_db(pool).fetch(sql, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        # Normalize ts_rank (typically 0–1 but can exceed 1) into 0–1 range
        # for compatibility with MemoryRecall.similarity
        raw_rank = float(row["rank"])
        similarity = min(1.0, raw_rank)
        if facet_boost_project_id is not None and facet_boost_project_id in memory.project_facets:
            similarity = similarity * _FACET_BOOST
        results.append(MemoryRecall(memory=memory, similarity=similarity))

    if facet_boost_project_id is not None:
        results.sort(key=lambda r: (-r.similarity, r.memory.id))
        results = results[:limit]

    return results


# Reciprocal Rank Fusion constant (standard default from the literature)
_RRF_K = 60


async def search_hybrid(
    pool: asyncpg.Pool,
    query: str,
    embedding: list[float],
    *,
    limit: int = 10,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    user_id: str | None = None,
    exclude_ids: list[str] | None = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
    facet_boost_project_id: str | None = None,
) -> list[MemoryRecall]:
    """Hybrid search combining vector similarity and BM25 keyword matching.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval
    methods. RRF is rank-based, so it handles the different score scales
    (cosine similarity vs ts_rank) naturally.

    vector_weight/keyword_weight control the relative importance of each
    signal in the RRF formula: score = w / (k + rank).

    user_id: If provided, filters to memories owned by this user OR globally-scoped
    memories (user_id = SYSTEM_GLOBAL_USER_ID sentinel). If None, returns all.

    facet_boost_project_id: Forwarded to both sub-searches.  See
    search_by_vector for semantics.
    """
    # Fetch broader candidate sets from both methods, then fuse
    candidate_limit = limit * 3  # over-fetch to ensure good fusion

    vector_results = await search_by_vector(
        pool,
        embedding,
        limit=candidate_limit,
        threshold=threshold,
        status=status,
        memory_type=memory_type,
        topic=topic,
        project_id=project_id,
        agent_id=agent_id,
        user_id=user_id,
        exclude_ids=exclude_ids,
        sources=sources,
        include_agent_provenance=include_agent_provenance,
        include_pending_review=include_pending_review,
        facet_boost_project_id=facet_boost_project_id,
    )

    keyword_results = await search_by_keyword(
        pool,
        query,
        limit=candidate_limit,
        status=status,
        memory_type=memory_type,
        topic=topic,
        project_id=project_id,
        agent_id=agent_id,
        user_id=user_id,
        exclude_ids=exclude_ids,
        sources=sources,
        include_agent_provenance=include_agent_provenance,
        include_pending_review=include_pending_review,
        facet_boost_project_id=facet_boost_project_id,
    )

    # Build rank maps (1-indexed)
    vector_ranks: dict[str, int] = {}
    for i, r in enumerate(vector_results):
        vector_ranks[r.memory.id] = i + 1

    keyword_ranks: dict[str, int] = {}
    for i, r in enumerate(keyword_results):
        keyword_ranks[r.memory.id] = i + 1

    # Collect all candidate memories
    all_memories: dict[str, MemoryRecall] = {}
    for r in vector_results:
        all_memories[r.memory.id] = r
    for r in keyword_results:
        if r.memory.id not in all_memories:
            all_memories[r.memory.id] = r

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    absent_rank = candidate_limit + 1  # penalty rank for missing results

    for mid in all_memories:
        v_rank = vector_ranks.get(mid, absent_rank)
        k_rank = keyword_ranks.get(mid, absent_rank)
        rrf_scores[mid] = (
            vector_weight / (_RRF_K + v_rank)
            + keyword_weight / (_RRF_K + k_rank)
        )

    # Sort by RRF score descending, take top `limit`
    sorted_ids = sorted(rrf_scores, key=lambda mid: rrf_scores[mid], reverse=True)[:limit]

    # Build results with RRF score as similarity (normalized to 0–1)
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    results = []
    for mid in sorted_ids:
        recall = all_memories[mid]
        normalized_score = rrf_scores[mid] / max_rrf if max_rrf > 0 else 0.0
        results.append(MemoryRecall(memory=recall.memory, similarity=normalized_score))

    return results


_CROSS_PROJECT_PENALTY = 0.8

# Facet-boost: ranking multiplier applied to beliefs whose project_facets
# contains the current project.  A 15% lift is enough to pull a same-project
# belief above a slightly-lower-similarity cross-project belief without
# drowning out clearly-more-relevant cross-project hits.
_FACET_BOOST = 1.15


async def search_cross_project(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    exclude_project_id: str | None,
    limit: int = 3,
    threshold: float = 0.5,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    exclude_ids: list[str] | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
) -> list[MemoryRecall]:
    """Search memories from OTHER projects (cross-project insights).

    Returns memories NOT in exclude_project_id, with a 0.8x relevance penalty
    applied to similarity scores so cross-project hits never outrank
    same-project hits.

    If exclude_project_id is None (global context), returns only project-scoped
    memories (project_id IS NOT NULL) to avoid returning all globals.
    """
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    # Similarity threshold
    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    # Cross-project exclusion (NULL-safe)
    if exclude_project_id is not None:
        conditions.append(f"(project_id != ${idx} OR project_id IS NULL)")
        params.append(exclude_project_id)
        idx += 1
    else:
        # Global context: only show project-scoped memories
        conditions.append("project_id IS NOT NULL")

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if not include_agent_provenance:
        conditions.append("write_provenance != 'agent'")
    if not include_pending_review:
        conditions.append("review_status = 'active'")

    if exclude_ids:
        conditions.append(f"NOT (id = ANY(${idx}::text[]))")
        params.append(exclude_ids)
        idx += 1

    if sources:
        conditions.append(f"source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT *,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        {where}
        ORDER BY embedding <=> $1::vector, id
        LIMIT ${idx}
    """
    params.append(limit)

    rows = await get_db(pool).fetch(query, *params)

    results = []
    for row in rows:
        memory = _row_to_memory(row)
        # Apply 0.8x penalty for cross-project results
        raw_sim = float(row["similarity"])
        penalized_sim = raw_sim * _CROSS_PROJECT_PENALTY
        results.append(MemoryRecall(memory=memory, similarity=penalized_sim))
    return results


async def count_by_vector(
    pool: asyncpg.Pool,
    embedding: list[float],
    *,
    threshold: float = 0.0,
    status: MemoryStatus | None = MemoryStatus.active,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
    sources: list[str] | None = None,
    include_agent_provenance: bool = True,
    include_pending_review: bool = False,
) -> int:
    """Count total memories matching a vector search (same filters as search_by_vector, no LIMIT)."""
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    idx = 1

    params.append(embedding)
    idx += 1  # $1 = embedding

    conditions.append(f"1 - (embedding <=> $1::vector) >= ${idx}")
    params.append(threshold)
    idx += 1

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if not include_agent_provenance:
        conditions.append("write_provenance != 'agent'")
    if not include_pending_review:
        conditions.append("review_status = 'active'")

    if memory_type:
        conditions.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if topic:
        conditions.append(f"${idx} = ANY(topic)")
        params.append(topic)
        idx += 1

    if project_id is not None:
        conditions.append(f"(project_id = ${idx} OR project_id IS NULL)")
        params.append(project_id)
        idx += 1

    if agent_id is not None:
        conditions.append(f"(agent_id = ${idx} OR agent_id IS NULL)")
        params.append(agent_id)
        idx += 1

    if sources:
        conditions.append(f"source = ANY(${idx}::text[])")
        params.append(sources)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)
    query = f"SELECT COUNT(*) FROM memories {where}"

    return await get_db(pool).fetchval(query, *params)


async def get_recent_writes(
    pool: asyncpg.Pool,
    *,
    limit: int = 10,
    project_id: str | None = None,
) -> list[dict]:
    """Return the most recently created memories with provenance info."""
    db = get_db(pool)
    if project_id is not None:
        rows = await db.fetch(
            """SELECT id, type, content, source, agent_id, project_id, created_at
               FROM memories
               WHERE status = 'active' AND (project_id = $1 OR project_id IS NULL)
               ORDER BY created_at DESC LIMIT $2""",
            project_id, limit,
        )
    else:
        rows = await db.fetch(
            """SELECT id, type, content, source, agent_id, project_id, created_at
               FROM memories
               WHERE status = 'active'
               ORDER BY created_at DESC LIMIT $1""",
            limit,
        )
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "content": r["content"][:80] + ("..." if len(r["content"]) > 80 else ""),
            "source": r["source"],
            "agent_id": r["agent_id"],
            "project_id": r["project_id"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


async def update_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    content: str | None = None,
    confidence: float | None = None,
    status: MemoryStatus | None = None,
    memory_type: MemoryType | None = None,
    topic: list[str] | None = None,
    embedding: list[float] | None = None,
    pinned: bool | None = None,
    project_id: str | None = _UNSET,
    review_after: datetime | None = _UNSET,
) -> Memory | None:
    """Update mutable fields of a memory. Returns updated Memory or None."""
    sets = ["updated_at = now()"]
    params: list = []
    idx = 1

    if content is not None:
        sets.append(f"content = ${idx}")
        params.append(content)
        idx += 1
        sets.append(f"token_count = ${idx}")
        params.append(estimate_tokens(content))
        idx += 1

    if confidence is not None:
        sets.append(f"confidence = ${idx}")
        params.append(confidence)
        idx += 1

    if memory_type is not None:
        sets.append(f"type = ${idx}")
        params.append(memory_type.value)
        idx += 1

    if status is not None:
        sets.append(f"status = ${idx}")
        params.append(status.value)
        idx += 1

    if topic is not None:
        sets.append(f"topic = ${idx}")
        params.append(topic)
        idx += 1

    if embedding is not None:
        sets.append(f"embedding = ${idx}::vector")
        params.append(embedding)
        idx += 1

    if pinned is not None:
        sets.append(f"pinned = ${idx}")
        params.append(pinned)
        idx += 1

    if project_id is not _UNSET:
        sets.append(f"project_id = ${idx}")
        params.append(project_id)
        idx += 1

    if review_after is not _UNSET:
        sets.append(f"review_after = ${idx}")
        params.append(review_after)
        idx += 1

    set_clause = ", ".join(sets)
    params.append(memory_id)

    row = await get_db(pool).fetchrow(
        f"UPDATE memories SET {set_clause} WHERE id = ${idx} RETURNING *",
        *params,
    )
    return _row_to_memory(row) if row else None


async def upsert_by_topic(
    pool: asyncpg.Pool,
    *,
    topic: list[str],
    project_id: str | None,
    content: str,
    memory_type: MemoryType,
    source: MemorySource,
    confidence: float,
    review_after: datetime | None = None,
    embedding: list[float] | None = None,
) -> Memory:
    """Find a memory by exact topic array + project_id match; update or create.

    Uses a transaction to ensure atomicity. If a matching active memory exists,
    updates its content, confidence, updated_at, and optionally embedding.
    Otherwise creates a new memory with all provided fields.

    Returns the updated or newly created Memory.
    """
    async with acquire(pool) as conn:
        async with conn.transaction():
            # Look for an existing active memory with exact topic match
            if project_id is not None:
                existing = await conn.fetchrow(
                    """
                    SELECT * FROM memories
                    WHERE topic @> $1 AND topic <@ $1
                      AND project_id = $2
                      AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    topic,
                    project_id,
                )
            else:
                existing = await conn.fetchrow(
                    """
                    SELECT * FROM memories
                    WHERE topic @> $1 AND topic <@ $1
                      AND project_id IS NULL
                      AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    topic,
                )

            if existing:
                # Update the existing memory
                sets = ["content = $1", "confidence = $2", "updated_at = now()"]
                params: list = [content, confidence]
                idx = 3

                sets.append(f"token_count = ${idx}")
                params.append(estimate_tokens(content))
                idx += 1

                if embedding is not None:
                    sets.append(f"embedding = ${idx}::vector")
                    params.append(embedding)
                    idx += 1

                set_clause = ", ".join(sets)
                params.append(existing["id"])

                row = await conn.fetchrow(
                    f"UPDATE memories SET {set_clause} WHERE id = ${idx} RETURNING *",
                    *params,
                )
                return _row_to_memory(row)

            # Create a new memory
            memory_id = _weft_id()
            now = datetime.now(timezone.utc)
            token_count = estimate_tokens(content)
            await conn.execute(
                """
                INSERT INTO memories (
                    id, type, topic, content, source, confidence,
                    token_count, created_at, updated_at, accessed_at,
                    access_count, project_id, embedding, status,
                    pinned, review_after, user_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $8, $8,
                    0, $9, $10::vector, 'active',
                    false, $11, nullif(current_setting('app.user_id', true), '')
                )
                """,
                memory_id,
                memory_type.value,
                topic,
                content,
                source.value,
                confidence,
                token_count,
                now,
                project_id,
                embedding,
                review_after,
            )

            return Memory(
                id=memory_id,
                type=memory_type,
                topic=topic,
                content=content,
                source=source,
                confidence=confidence,
                token_count=token_count,
                created_at=now,
                updated_at=now,
                accessed_at=now,
                access_count=0,
                project_id=project_id,
                status=MemoryStatus.active,
                pinned=False,
                review_after=review_after,
            )


async def delete_memory(pool: asyncpg.Pool, memory_id: str, *, hard: bool = False) -> bool:
    """Delete a memory. Soft-delete (archive) by default, hard-delete if specified.

    When the operation succeeds, any associated recall-canary probes are disabled
    (``enabled = FALSE``) so orphan probes for non-active memories cannot pollute
    the canary miss-rate counter or grow unbounded.
    """
    db = get_db(pool)
    if hard:
        result = await db.execute("DELETE FROM memories WHERE id = $1", memory_id)
    else:
        result = await db.execute(
            "UPDATE memories SET status = 'archived', updated_at = now() WHERE id = $1",
            memory_id,
        )
    deleted = result.split()[-1] != "0"
    if deleted:
        await db.execute(
            "UPDATE recall_canary SET enabled = FALSE WHERE memory_id = $1",
            memory_id,
        )
        logger.debug("delete_memory: disabled canary probes for memory_id=%s", memory_id)
    return deleted


async def touch_memory(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    implicit_alpha: float = 0.05,
) -> None:
    """Update accessed_at, increment access_count, and apply mild usefulness bump.

    The implicit bump treats retrieval as weak positive evidence of usefulness.
    Uses EMA with a small alpha (default 0.05) — much weaker than explicit
    feedback (alpha=0.3) so it takes many accesses to move the needle.

    Formula: new_score = (1 - alpha) * old_score + alpha * 1.0
    """
    await get_db(pool).execute(
        """
        UPDATE memories
        SET accessed_at = now(),
            access_count = access_count + 1,
            usefulness_score = LEAST(1.0, (1.0 - $2) * usefulness_score + $2)
        WHERE id = $1
        """,
        memory_id,
        implicit_alpha,
    )


async def bump_retrieval_telemetry(
    pool: asyncpg.Pool,
    memory_ids: list[str],
) -> None:
    """Record that ``memory_ids`` were returned by a retrieval surface.

    Bumps ``last_retrieved_at`` to now() and increments ``retrieval_count`` for
    every id in a single statement. Side-effect-free at the ranking layer —
    deliberately distinct from ``touch_memory``'s usefulness-EMA bump, so the
    raw retrieval signal stays unconfounded for the 2-week observation window
    (Anvil-reframed Step 1; see migration v49 for provenance).

    Empty input is a no-op so callers can pass result lists unconditionally.
    """
    if not memory_ids:
        return
    await get_db(pool).execute(
        """
        UPDATE memories
        SET last_retrieved_at = now(),
            retrieval_count = retrieval_count + 1
        WHERE id = ANY($1::text[])
        """,
        memory_ids,
    )


async def log_recall_query(
    pool: asyncpg.Pool,
    *,
    tool_name: str,
    query_text: str,
    project_id: str | None = None,
    tier: str | None = None,
    mode: str | None = None,
    retrieval_mode: str | None = None,
    result_count: int | None = None,
) -> None:
    """Record one weft_recall / weft_search_all invocation in weft_recall_queries.

    Step 1.5 of the compounding loop (v50). The 2-week observation window
    measures three baseline metrics — calls/week, repeat-query %, and
    consecutive-query semantic similarity — all of which need a per-call
    query log. Embeddings are NOT stored here; metric (c) re-embeds the
    text at analysis time so model choice is deferred.

    Errors are caught and logged but never raised: this is observation
    telemetry on the hot recall path, and a logging failure must never
    break a user-facing query. Callers should treat this as fire-and-forget.
    """
    import uuid
    query_id = f"rq-{uuid.uuid4().hex[:8]}"
    try:
        await get_db(pool).execute(
            """
            INSERT INTO weft_recall_queries
                (query_id, project_id, tool_name, query_text,
                 tier, mode, retrieval_mode, result_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            query_id,
            project_id,
            tool_name,
            query_text,
            tier,
            mode,
            retrieval_mode,
            result_count,
        )
    except (asyncpg.PostgresError, OSError, ConnectionError) as exc:
        logger.warning("log_recall_query failed (tool=%s): %s", tool_name, exc)


async def record_feedback(
    pool: asyncpg.Pool,
    memory_id: str,
    helpful: bool,
    alpha: float = 0.3,
) -> dict:
    """Record usefulness feedback using exponential moving average.

    new_score = alpha * signal + (1 - alpha) * old_score
    where signal = 1.0 for helpful, 0.0 for not helpful.
    """
    async with acquire(pool) as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT usefulness_score, usefulness_count FROM memories WHERE id = $1 FOR UPDATE",
                memory_id,
            )
            if row is None:
                raise ValueError(f"Memory {memory_id} not found")

            old_score = float(row["usefulness_score"]) if row["usefulness_score"] is not None else 1.0
            signal = 1.0 if helpful else 0.0
            new_score = alpha * signal + (1 - alpha) * old_score
            new_score = max(0.0, min(1.0, new_score))
            new_count = (row["usefulness_count"] or 0) + 1

            await conn.execute(
                """
                UPDATE memories
                SET usefulness_score = $1, usefulness_count = $2, updated_at = now()
                WHERE id = $3
                """,
                new_score,
                new_count,
                memory_id,
            )

    return {
        "memory_id": memory_id,
        "usefulness_score": new_score,
        "usefulness_count": new_count,
    }


async def get_recent_recall_queries(
    pool: asyncpg.Pool,
    *,
    window_minutes: int = 30,
    limit: int = 200,
    user_id: str | None = None,
    scope_to_user: bool = False,
) -> list[dict]:
    """Fetch recent weft_recall_queries rows for re-ask detection.

    Returns rows ordered by created_at ASC (oldest first) within the last
    ``window_minutes``. Only rows NOT already marked as re-ask misses are
    returned (avoids double-processing).

    User scoping: when ``scope_to_user`` is True the query adds an explicit
    ``user_id IS NOT DISTINCT FROM $user_id`` predicate (NULL-safe: pass
    user_id=None to scope to the unauthenticated single-user rows). This is
    defense-in-depth that does NOT rely on RLS — required when the caller is a
    system/scheduler context that may bypass RLS (see scheduler's per-user
    fan-out, loom-fdd9282a). When ``scope_to_user`` is False (default) the
    behavior is unchanged: rows are filtered by RLS / the caller's app.user_id
    only — correct for an authenticated per-request caller.

    Returned dicts have the same field shape as ``QueryRow.from_dict``
    expects: query_id, query_text, created_at, tool_name, project_id,
    tier, mode, retrieval_mode, result_count.
    """
    conditions = [
        "created_at >= now() - make_interval(mins => $1)",
        "is_reask_miss = FALSE",
    ]
    params: list = [window_minutes]
    idx = 2
    if scope_to_user:
        conditions.append(f"user_id IS NOT DISTINCT FROM ${idx}")
        params.append(user_id)
        idx += 1
    where = " AND ".join(conditions)
    params.append(limit)
    rows = await get_db(pool).fetch(
        f"""
        SELECT query_id, query_text, created_at, tool_name, project_id,
               tier, mode, retrieval_mode, result_count
        FROM weft_recall_queries
        WHERE {where}
        ORDER BY created_at ASC
        LIMIT ${idx}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def get_distinct_reask_user_ids(
    pool: asyncpg.Pool,
    *,
    window_minutes: int = 30,
) -> list[str | None]:
    """List distinct user_ids with unprocessed recall queries in the window.

    SYSTEM / ADMIN enumerator for the scheduler's per-user re-ask fan-out
    (loom-fdd9282a). It deliberately reads across users, so it MUST run only
    from a trusted system context (the scheduler), never on a user-facing
    request path. The returned ids are then each processed under their own
    per-user scope so no user's data bleeds into another's pass.

    Returns one ``None`` entry for the unauthenticated single-user deployment
    (recall queries written with a NULL user_id).
    """
    rows = await get_db(pool).fetch(
        """
        SELECT DISTINCT user_id
        FROM weft_recall_queries
        WHERE created_at >= now() - make_interval(mins => $1)
          AND is_reask_miss = FALSE
        """,
        window_minutes,
    )
    return [r["user_id"] for r in rows]


async def apply_reask_feedback(
    pool: asyncpg.Pool,
    missed_query_id: str,
    satisfying_memory_id: str,
) -> dict | None:
    """Apply usefulness feedback for a detected re-ask miss.

    When the re-ask detector identifies (original_query, reask_query) pairs,
    call this function with:
      * ``missed_query_id`` — the query_id of the ORIGINAL (missed) query.
      * ``satisfying_memory_id`` — the memory.id that answered the SECOND
        (successful) re-ask, which we want to boost since it was the right
        answer all along.

    This function is IDEMPOTENT: it claims the miss row atomically before
    boosting the EMA, so scheduler retries and partial failures cannot
    double-boost the usefulness score.

    Steps:
    1. Atomically claim the row by flipping ``is_reask_miss`` from FALSE to
       TRUE in a single UPDATE.  If 0 rows are affected (already processed,
       or concurrent claim won), return ``None`` immediately — no EMA boost.
    2. Only when the claim affects exactly 1 row: call ``record_feedback`` to
       apply the EMA boost to ``satisfying_memory_id``.

    Returns:
      * The ``record_feedback`` result dict (with ``usefulness_score`` after
        the boost) on a fresh claim.
      * ``None`` when the row was already processed (idempotent no-op).

    Errors are NOT swallowed — this runs in a scheduler context where the
    caller wraps in try/except per-pair.
    """
    # 1. Atomically claim the miss row.  The WHERE clause guarantees this
    #    only affects an unprocessed row; asyncpg returns a status string
    #    like 'UPDATE 1' or 'UPDATE 0'.
    status = await get_db(pool).execute(
        """
        UPDATE weft_recall_queries
        SET is_reask_miss = TRUE,
            reask_satisfying_memory_id = $2
        WHERE query_id = $1
          AND is_reask_miss = FALSE
        """,
        missed_query_id,
        satisfying_memory_id,
    )

    if status != "UPDATE 1":
        # Already processed or row not found — idempotent no-op.
        return None

    # 2. Fetch the query_text and user_id from the now-claimed row so we can
    #    enqueue a replay for the implicated episode(s).  The row is guaranteed
    #    to exist because we just updated it; a None result would be a bug.
    claimed_row = await get_db(pool).fetchrow(
        "SELECT query_text, user_id FROM weft_recall_queries WHERE query_id = $1",
        missed_query_id,
    )
    if claimed_row is not None:
        from weft.replay import enqueue_replay_on_miss  # late import: avoids circular dep

        try:
            await enqueue_replay_on_miss(
                pool,
                query_text=claimed_row["query_text"],
                user_id=claimed_row["user_id"],
            )
        except Exception as exc:
            # Enqueue failures must not abort the EMA boost — the two operations
            # are independent.  Log at warning level so observability is preserved,
            # and bump the aggregate counter so a persistently-broken enqueue is
            # visible in weft_check_health, not just buried per-occurrence in logs.
            logger.warning(
                "apply_reask_feedback: enqueue_replay_on_miss failed "
                "(query_id=%s): %s",
                missed_query_id,
                exc,
            )
            from weft.counters import COUNTER_REPLAY_ENQUEUE_FAILED, increment_counter

            await increment_counter(pool, COUNTER_REPLAY_ENQUEUE_FAILED)

    # 3. Fresh claim: boost the satisfying memory via the existing EMA path.
    return await record_feedback(pool, satisfying_memory_id, helpful=True)


# --- Relationships ---


async def add_relationship(
    pool: asyncpg.Pool,
    source_id: str,
    target_id: str,
    relation: RelationType,
) -> MemoryRelationship:
    """Create a relationship between two memories."""
    now = datetime.now(timezone.utc)
    await get_db(pool).execute(
        """
        INSERT INTO memory_relationships (source_id, target_id, relation, created_at, user_id)
        VALUES ($1, $2, $3, $4, nullif(current_setting('app.user_id', true), ''))
        ON CONFLICT (source_id, target_id, relation) DO NOTHING
        """,
        source_id,
        target_id,
        relation.value,
        now,
    )
    return MemoryRelationship(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        created_at=now,
    )


async def get_relationships(
    pool: asyncpg.Pool,
    memory_id: str,
    *,
    relation: RelationType | None = None,
) -> list[MemoryRelationship]:
    """Get all relationships for a memory (as source or target)."""
    db = get_db(pool)
    if relation:
        rows = await db.fetch(
            """
            SELECT * FROM memory_relationships
            WHERE (source_id = $1 OR target_id = $1) AND relation = $2
            """,
            memory_id,
            relation.value,
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM memory_relationships WHERE source_id = $1 OR target_id = $1",
            memory_id,
        )
    return [
        MemoryRelationship(
            source_id=r["source_id"],
            target_id=r["target_id"],
            relation=RelationType(r["relation"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


async def remove_relationship(
    pool: asyncpg.Pool,
    source_id: str,
    target_id: str,
    relation: RelationType,
) -> bool:
    """Remove a specific relationship. Returns True if deleted."""
    result = await get_db(pool).execute(
        """
        DELETE FROM memory_relationships
        WHERE source_id = $1 AND target_id = $2 AND relation = $3
        """,
        source_id,
        target_id,
        relation.value,
    )
    return result.split()[-1] != "0"


# --- Stats ---


async def get_stats(pool: asyncpg.Pool) -> dict:
    """Get memory statistics."""
    db = get_db(pool)
    total = await db.fetchval("SELECT COUNT(*) FROM memories")
    by_status = await db.fetch(
        "SELECT status, COUNT(*) as count FROM memories GROUP BY status"
    )
    by_type = await db.fetch(
        "SELECT type, COUNT(*) as count FROM memories WHERE status = 'active' GROUP BY type"
    )
    by_topic = await db.fetch(
        """
        SELECT unnest(topic) as topic, COUNT(*) as count
        FROM memories WHERE status = 'active'
        GROUP BY topic ORDER BY count DESC LIMIT 20
        """
    )
    recent = await db.fetch(
        "SELECT id, content, accessed_at FROM memories WHERE status = 'active' ORDER BY accessed_at DESC LIMIT 5"
    )

    return {
        "total": total,
        "by_status": {r["status"]: r["count"] for r in by_status},
        "by_type": {r["type"]: r["count"] for r in by_type},
        "top_topics": {r["topic"]: r["count"] for r in by_topic},
        "recently_accessed": [
            {"id": r["id"], "content": r["content"][:80], "accessed_at": r["accessed_at"].isoformat()}
            for r in recent
        ],
    }


# --- Metadata (system-level key-value) ---


async def get_metadata(pool: asyncpg.Pool, key: str) -> dict | None:
    """Get a metadata value by key. Returns None if not found or table missing."""
    try:
        row = await get_db(pool).fetchrow(
            "SELECT value FROM weft_metadata WHERE key = $1", key,
        )
        if row is None:
            return None
        val = row["value"]
        # asyncpg returns JSONB as a string or dict depending on codec
        if isinstance(val, str):
            return json.loads(val)
        return dict(val)
    except Exception:
        # Table may not exist if migration hasn't run yet
        return None


async def set_metadata(pool: asyncpg.Pool, key: str, value: dict) -> None:
    """Upsert a metadata value (idempotent)."""
    await get_db(pool).execute(
        """INSERT INTO weft_metadata (key, value, updated_at)
           VALUES ($1, $2::jsonb, NOW())
           ON CONFLICT (key) DO UPDATE
           SET value = EXCLUDED.value, updated_at = NOW()""",
        key, json.dumps(value),
    )


# --- Changes since ---


async def get_last_handoff_timestamp(
    pool: asyncpg.Pool,
    project_id: str | None = None,
) -> datetime | None:
    """Return the created_at of the most recent handoff memory, or None.

    Scoping: project_id=None queries global handoffs (project_id IS NULL).
    A non-None project_id matches that exact project.
    """
    db = get_db(pool)
    if project_id is not None:
        row = await db.fetchrow(
            """
            SELECT MAX(created_at) AS ts
            FROM memories
            WHERE type = 'handoff' AND status = 'active'
              AND project_id = $1
            """,
            project_id,
        )
    else:
        row = await db.fetchrow(
            """
            SELECT MAX(created_at) AS ts
            FROM memories
            WHERE type = 'handoff' AND status = 'active'
              AND project_id IS NULL
            """,
        )
    return row["ts"] if row and row["ts"] else None


async def get_memory_changes_since(
    pool: asyncpg.Pool,
    *,
    since: datetime,
    project_id: str | None = None,
) -> dict:
    """Count memory mutations since a timestamp using conditional aggregation.

    Returns dict with keys: memories_created, memories_archived, memories_revised, since.
    - created: active memories with created_at > since
    - archived: archived memories with updated_at > since
    - revised: active memories updated since the timestamp but created before it
    """
    db = get_db(pool)
    if project_id is not None:
        row = await db.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'active' AND created_at > $1
                ) AS memories_created,
                COUNT(*) FILTER (
                    WHERE status = 'archived' AND updated_at > $1
                ) AS memories_archived,
                COUNT(*) FILTER (
                    WHERE status = 'active' AND updated_at > $1 AND created_at <= $1
                ) AS memories_revised
            FROM memories
            WHERE project_id = $2
            """,
            since, project_id,
        )
    else:
        row = await db.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'active' AND created_at > $1
                ) AS memories_created,
                COUNT(*) FILTER (
                    WHERE status = 'archived' AND updated_at > $1
                ) AS memories_archived,
                COUNT(*) FILTER (
                    WHERE status = 'active' AND updated_at > $1 AND created_at <= $1
                ) AS memories_revised
            FROM memories
            WHERE project_id IS NULL
            """,
            since,
        )
    return {
        "memories_created": row["memories_created"],
        "memories_archived": row["memories_archived"],
        "memories_revised": row["memories_revised"],
        "since": since.isoformat(),
    }


# --- Helpers ---


def _row_to_memory(row: asyncpg.Record) -> Memory:
    """Convert a database row to a Memory model."""
    return Memory(
        id=row["id"],
        type=MemoryType(row["type"]),
        topic=list(row["topic"]) if row["topic"] else [],
        content=row["content"],
        source=row["source"],
        confidence=row["confidence"],
        token_count=row["token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        accessed_at=row["accessed_at"],
        access_count=row["access_count"],
        project_id=row["project_id"],
        agent_id=row["agent_id"],
        workspace_id=row["workspace_id"] if row.get("workspace_id") is not None else None,
        status=MemoryStatus(row["status"]),
        pinned=bool(row["pinned"]) if row.get("pinned") is not None else False,
        usefulness_score=float(row["usefulness_score"]) if row["usefulness_score"] is not None else 1.0,
        usefulness_count=row["usefulness_count"] if row["usefulness_count"] is not None else 0,
        last_boosted_at=row.get("last_boosted_at"),
        review_after=row["review_after"] if row.get("review_after") is not None else None,
        write_provenance=row["write_provenance"] if row.get("write_provenance") is not None else "supervisor",
        review_status=row["review_status"] if row.get("review_status") is not None else "active",
        project_facets=list(row["project_facets"]) if row.get("project_facets") else [],
    )
