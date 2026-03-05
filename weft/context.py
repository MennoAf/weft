"""Budget-aware context loading — pack the best memories within a token budget.

Pipeline:
1. Retrieve candidates via vector search
2. Score with relevance engine
3. Deduplicate by topic (max N per topic)
4. Greedily pack by score until budget exhausted
"""

from __future__ import annotations

import asyncpg

from weft.models import MemoryRecall, MemoryStatus, MemoryType
from weft.relevance import RelevanceWeights, ScoredMemory, rank_memories
from weft.store import list_memories, search_by_vector


def _deduplicate_by_topic(
    scored: list[ScoredMemory],
    max_per_topic: int = 3,
) -> list[ScoredMemory]:
    """Limit results per topic to ensure diversity.

    Memories without topics pass through uncapped.
    """
    topic_counts: dict[str, int] = {}
    result: list[ScoredMemory] = []

    for sm in scored:
        topics = sm.memory.topic
        if not topics:
            result.append(sm)
            continue

        # Check if any of this memory's topics have hit the cap
        if all(topic_counts.get(t, 0) < max_per_topic for t in topics):
            result.append(sm)
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1

    return result


def _pack_by_budget(
    scored: list[ScoredMemory],
    budget_tokens: int,
) -> list[ScoredMemory]:
    """Greedily pack memories by score until token budget is exhausted."""
    packed: list[ScoredMemory] = []
    used = 0

    for sm in scored:
        cost = sm.memory.token_count or 1
        if used + cost <= budget_tokens:
            packed.append(sm)
            used += cost

    return packed


async def build_context(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    *,
    budget_tokens: int = 4000,
    candidate_limit: int = 50,
    max_per_topic: int = 3,
    threshold: float = 0.1,
    weights: RelevanceWeights | None = None,
    memory_type: MemoryType | None = None,
    topic: str | None = None,
    project_id: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Build a context-optimized set of memories within a token budget.

    Returns:
        {
            "memories": [ScoredMemory.to_dict(), ...],
            "total_tokens": int,
            "remaining_budget": int,
            "count": int,
            "budget_tokens": int,
        }
    """
    # 0. Load pinned memories first (always included, highest usefulness first)
    pinned_mems = await list_memories(
        pool, status=MemoryStatus.active, pinned=True,
        project_id=project_id, agent_id=agent_id, limit=100,
    )
    pinned_mems.sort(key=lambda m: m.usefulness_score, reverse=True)
    pinned_packed: list[dict] = []
    pinned_ids: set[str] = set()
    pinned_tokens = 0
    for mem in pinned_mems:
        cost = mem.token_count or 1
        if pinned_tokens + cost <= budget_tokens:
            pinned_packed.append(mem.to_dict())
            pinned_ids.add(mem.id)
            pinned_tokens += cost

    remaining_budget = budget_tokens - pinned_tokens

    # 1. Retrieve candidates via vector search
    recalls: list[MemoryRecall] = await search_by_vector(
        pool,
        query_embedding,
        limit=candidate_limit,
        threshold=threshold,
        status=MemoryStatus.active,
        memory_type=memory_type,
        topic=topic,
        project_id=project_id,
        agent_id=agent_id,
    )

    # Filter out already-included pinned memories
    recalls = [r for r in recalls if r.memory.id not in pinned_ids]

    if not recalls and not pinned_packed:
        return {
            "memories": [],
            "total_tokens": 0,
            "remaining_budget": budget_tokens,
            "count": 0,
            "budget_tokens": budget_tokens,
        }

    # 2. Score with relevance engine
    scored = rank_memories(recalls, weights=weights) if recalls else []

    # 3. Deduplicate by topic
    diverse = _deduplicate_by_topic(scored, max_per_topic=max_per_topic)

    # 4. Pack within remaining budget (after pinned)
    packed = _pack_by_budget(diverse, remaining_budget)

    total_tokens = pinned_tokens + sum(sm.memory.token_count or 1 for sm in packed)
    all_memories = pinned_packed + [sm.to_dict() for sm in packed]

    return {
        "memories": all_memories,
        "total_tokens": total_tokens,
        "remaining_budget": budget_tokens - total_tokens,
        "count": len(all_memories),
        "budget_tokens": budget_tokens,
    }
