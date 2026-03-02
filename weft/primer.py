"""Session primer — assemble structured context for session startup.

Builds a payload with priority layers:
0. Pinned memories (always included first)
1. Last session handoff (most recent only — continuity)
2. Preferences & user_model (immortal, always included)
3. Recent work (accessed within N days, excluding ideas/aspirational items)
4. Active issues (type=issue, lightweight summary of what's broken)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncpg

from weft.models import Memory, MemoryStatus, MemoryType
from weft.store import list_memories
from weft.tokens import estimate_tokens


# Topics that signal aspirational/planned items rather than concrete work.
# These are excluded from the primer — ideas are pull-not-push context.
_IDEA_TOPICS = frozenset({"improvement", "idea", "issue-log", "backlog", "wishlist"})

# Content markers that indicate a memory describes completed/resolved work.
# These are excluded from the primer entirely — completed items aren't
# actionable and waste budget that should go to live context.
_COMPLETED_MARKERS = ("(DONE)", "(FIXED)", "Improvement (DONE)", "Bug (FIXED)",
                      "Feedback (FIXED)")

# When priming for a specific project, limit how many non-project items
# can appear in each section. This prevents cross-project noise from
# consuming budget that should go to project-relevant context.
_MAX_GLOBAL_RECENT = 3
_MAX_GLOBAL_PREFS = 3


def _is_idea(mem: Memory) -> bool:
    """Return True if the memory looks aspirational rather than concrete work."""
    topics = {t.lower() for t in (mem.topic or [])}
    return bool(topics & _IDEA_TOPICS)


def _is_completed(mem: Memory) -> bool:
    """Return True if the memory describes work that's already done/fixed."""
    content = mem.content
    topics = {t.lower() for t in (mem.topic or [])}
    if any(marker in content for marker in _COMPLETED_MARKERS):
        return True
    if "done" in topics and "improvement" in topics:
        return True
    if "fixed" in topics:
        return True
    return False


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
    1. last session handoff (continuity)
    2. preferences + user_model (immortal)
    3. recent work (concrete, non-aspirational)
    4. active issues (what's broken right now)

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
        key=lambda m: (
            # Project-scoped first when project_id is set
            0 if project_id and m.project_id == project_id else 1,
            -m.confidence,
        ),
    )

    preferences_section: list[dict] = []
    global_pref_count = 0
    for mem in immortals:
        if mem.id in seen_ids:
            continue
        # Cap non-project preferences when working in a specific project
        is_global = project_id and mem.project_id != project_id
        if is_global and global_pref_count >= _MAX_GLOBAL_PREFS:
            continue
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            preferences_section.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost
            if is_global:
                global_pref_count += 1

    # --- Section 3: Recent work (excludes ideas and completed items) ---
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    recent_raw = await _fetch_with_globals(
        pool, project_id,
        status=MemoryStatus.active, limit=50,
    )
    # Filter to recently accessed, exclude already-seen, handoffs, issues,
    # ideas (aspirational — pull not push), and completed items (DONE/FIXED).
    recent_candidates = [
        m for m in recent_raw
        if m.id not in seen_ids and m.accessed_at >= cutoff
        and m.type not in (MemoryType.handoff, MemoryType.issue)
        and not _is_completed(m)
        and not _is_idea(m)
    ]
    # Sort: project-scoped first, then by accessed_at.
    recent_candidates.sort(
        key=lambda m: (
            0 if project_id and m.project_id == project_id else 1,
            -(m.accessed_at.timestamp()),
        ),
    )

    recent_section: list[dict] = []
    global_recent_count = 0
    for mem in recent_candidates:
        is_global = project_id and mem.project_id != project_id
        if is_global and global_recent_count >= _MAX_GLOBAL_RECENT:
            continue
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            recent_section.append(mem.to_dict())
            if is_global:
                global_recent_count += 1
            seen_ids.add(mem.id)
            used_tokens += cost

    # --- Section 4: Active issues (lightweight "what's broken" summary) ---
    issues_raw = await _fetch_with_globals(
        pool, project_id,
        memory_type=MemoryType.issue, status=MemoryStatus.active, limit=20,
    )
    issue_candidates = [m for m in issues_raw if m.id not in seen_ids]
    # Most recently created issues first
    issue_candidates.sort(key=lambda m: m.created_at, reverse=True)

    issue_items: list[dict] = []
    for mem in issue_candidates:
        cost = mem.token_count or estimate_tokens(mem.content)
        if used_tokens + cost <= budget_tokens:
            issue_items.append(mem.to_dict())
            seen_ids.add(mem.id)
            used_tokens += cost

    return {
        "pinned": pinned_section,
        "handoff": handoff_section,
        "preferences": preferences_section,
        "recent_work": recent_section,
        "active_issues": {"count": len(issue_items), "items": issue_items},
        "total_tokens": used_tokens,
        "budget_tokens": budget_tokens,
        "budget_remaining": budget_tokens - used_tokens,
    }
