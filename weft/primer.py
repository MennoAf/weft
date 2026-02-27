"""Session primer — assemble structured context for session startup.

Builds a payload with four priority layers:
0. Pinned memories (always included first)
1. Preferences & user_model (immortal, always included)
2. Recent work (accessed within N days)
3. Project-relevant memories (fill remaining budget)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import asyncpg

from weft.models import Memory, MemoryStatus, MemoryType
from weft.store import list_memories
from weft.tokens import estimate_tokens


async def _fetch_with_globals(
    pool: asyncpg.Pool,
    project_id: str | None,
    **kwargs,
) -> list[Memory]:
    """Fetch memories matching filters, including global memories when project-scoped.

    When project_id is given, we fetch both project-specific and global
    (project_id=None) memories and merge them, deduplicating by id.
    """
    if project_id is None:
        # No project filter — returns all memories regardless of project
        return await list_memories(pool, **kwargs)

    # Fetch project-scoped memories
    project_mems = await list_memories(pool, project_id=project_id, **kwargs)

    # Also fetch global memories (project_id IS NULL)
    # list_memories only filters when project_id is not None,
    # so we need a separate query for globals. We pass project_id=None
    # but list_memories won't filter on it. Instead, we fetch all and
    # filter client-side.
    all_mems = await list_memories(pool, **kwargs)
    global_mems = [m for m in all_mems if m.project_id is None]

    # Merge and deduplicate
    seen: set[str] = set()
    merged: list[Memory] = []
    for m in project_mems + global_mems:
        if m.id not in seen:
            seen.add(m.id)
            merged.append(m)
    return merged


async def build_primer(
    pool: asyncpg.Pool,
    *,
    project_id: str | None = None,
    budget_tokens: int = 4000,
    recent_days: int = 7,
) -> dict:
    """Assemble a structured context payload from memories.

    Sections are filled in priority order within the token budget:
    0. pinned memories (always first)
    1. preferences + user_model (always second)
    2. recently accessed memories
    3. project-relevant memories

    Returns a dict with sections and budget info.
    """
    used_tokens = 0
    seen_ids: set[str] = set()

    # --- Section 0: Pinned memories (highest priority) ---
    pinned_raw = await _fetch_with_globals(
        pool, project_id,
        status=MemoryStatus.active, pinned=True, limit=100,
    )
    pinned_raw.sort(key=lambda m: m.confidence, reverse=True)

    pinned_section: list[dict] = []
    for mem in pinned_raw:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            pinned_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 1: Preferences & user_model (immortal) ---
    prefs_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.preference, status=MemoryStatus.active,
        limit=100,
    )
    user_models = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.user_model, status=MemoryStatus.active,
        limit=100,
    )
    immortals = sorted(
        prefs_raw + user_models,
        key=lambda m: m.confidence, reverse=True,
    )

    preferences_section: list[dict] = []
    for mem in immortals:
        if mem.id in seen_ids:
            continue
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            preferences_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 2: Recent work ---
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    recent_raw = await _fetch_with_globals(
        pool, project_id,
        status=MemoryStatus.active, limit=50,
    )
    # Filter to recently accessed, exclude already-seen
    recent_candidates = [
        m for m in recent_raw
        if m.id not in seen_ids and m.accessed_at >= cutoff
    ]
    # Sort by accessed_at descending
    recent_candidates.sort(key=lambda m: m.accessed_at, reverse=True)

    recent_section: list[dict] = []
    for mem in recent_candidates:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            recent_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 3: Relevant (fill remaining budget) ---
    relevant_section: list[dict] = []
    if project_id and used_tokens < budget_tokens:
        relevant_raw = await list_memories(
            pool, status=MemoryStatus.active, project_id=project_id, limit=50,
        )
        relevant_candidates = [
            m for m in relevant_raw if m.id not in seen_ids
        ]
        for mem in relevant_candidates:
            cost = mem.token_count or estimate_tokens(mem.content)
            if used_tokens + cost <= budget_tokens:
                relevant_section.append(mem.to_dict())
                seen_ids.add(mem.id)
                used_tokens += cost

    return {
        "pinned": pinned_section,
        "preferences": preferences_section,
        "recent_work": recent_section,
        "relevant": relevant_section,
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
    }
