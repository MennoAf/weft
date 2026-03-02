"""Session primer — assemble structured context for session startup.

Builds a payload with priority layers:
0. Pinned memories (always included first)
1. Last session handoff (most recent only — continuity)
2. Preferences & user_model (immortal, always included)
3. Recent work (accessed within N days)
4. Project-relevant memories (fill remaining budget)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import asyncpg

from weft.models import Memory, MemoryStatus, MemoryType
from weft.store import list_memories
from weft.tokens import estimate_tokens


# Topics that signal aspirational/planned items rather than concrete work
_IDEA_TOPICS = frozenset({"improvement", "idea", "issue-log", "backlog", "wishlist"})


def _is_idea(mem: Memory) -> bool:
    """Return True if the memory looks aspirational rather than concrete work."""
    topics = {t.lower() for t in (mem.topic or [])}
    return bool(topics & _IDEA_TOPICS)


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

    # --- Section 1: Last session handoff (most recent only) ---
    handoff_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.handoff, status=MemoryStatus.active, limit=5,
    )
    # Fallback: if no typed handoffs found, check for topic "session-handoff"
    # (handles memories created before the handoff type existed or mistyped)
    if not handoff_raw:
        topic_raw = await _fetch_with_globals(
            pool, project_id,
            topic="session-handoff", status=MemoryStatus.active, limit=5,
        )
        handoff_raw = [m for m in topic_raw if "Session Handoff" in m.content]
    # Filter out already-seen (e.g., if a handoff was also pinned)
    handoff_candidates = [m for m in handoff_raw if m.id not in seen_ids]
    # Take only the most recent handoff
    handoff_candidates.sort(key=lambda m: m.created_at, reverse=True)

    handoff_section: list[dict] = []
    if handoff_candidates:
        mem = handoff_candidates[0]
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            handoff_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 2: Preferences & user_model (immortal) ---
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

    # --- Section 3: Recent work + ideas ---
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    recent_raw = await _fetch_with_globals(
        pool, project_id,
        status=MemoryStatus.active, limit=50,
    )
    # Filter to recently accessed, exclude already-seen and handoffs
    # (handoffs have their own section)
    recent_candidates = [
        m for m in recent_raw
        if m.id not in seen_ids and m.accessed_at >= cutoff
        and m.type != MemoryType.handoff
    ]
    # Sort: project-scoped first (when project_id is set), then by accessed_at
    recent_candidates.sort(
        key=lambda m: (
            0 if project_id and m.project_id == project_id else 1,
            -(m.accessed_at.timestamp()),
        ),
    )

    # Split into concrete work vs aspirational ideas
    recent_section: list[dict] = []
    ideas_section: list[dict] = []
    for mem in recent_candidates:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            if _is_idea(mem):
                ideas_section.append(mem.to_dict())
            else:
                recent_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 4: Relevant (fill remaining budget) ---
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
        "handoff": handoff_section,
        "preferences": preferences_section,
        "recent_work": recent_section,
        "ideas": ideas_section,
        "relevant": relevant_section,
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
    }
