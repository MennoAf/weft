"""Working memory section — open episodes with linked memories.

Shows active episodes (working memory) so the agent knows what's
currently being tracked.  Omitted when no open episodes exist.

Tier-2 in progressive disclosure (count only in progressive mode,
full content in full mode).
"""

from __future__ import annotations

import logging

from weft.episodes import get_episode_memories, list_episodes
from weft.models import EpisodeStatus
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    PrimerContext,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS.get("working_memory", 200)
_MAX_EPISODES = 5
_MAX_MEMORIES_PER_EPISODE = 3


async def build_working_memory_section(ctx: PrimerContext) -> SectionResult:
    """Fetch open episodes and their linked memories for the primer."""
    episodes = await list_episodes(
        ctx.pool,
        status=EpisodeStatus.open,
        project_id=ctx.project_id,
        limit=_MAX_EPISODES,
    )

    if not episodes:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason="no open episodes")

    items: list[dict] = []
    section_used = 0

    for ep in episodes:
        # Build episode summary entry.
        memories = await get_episode_memories(ctx.pool, ep.id, limit=_MAX_MEMORIES_PER_EPISODE)
        memory_summaries = [
            {"id": m.id, "content": m.content[:120]} for m in memories
        ]

        age_hours = (ctx.now - ep.started_at).total_seconds() / 3600
        entry: dict = {
            "id": ep.id,
            "title": ep.title,
            "memory_count": len(memories),
            "age_hours": round(age_hours, 1),
        }
        if ep.summary:
            entry["summary"] = ep.summary
        if memory_summaries:
            entry["memories"] = memory_summaries

        # Estimate cost.
        text = ep.title + (ep.summary or "")
        for ms in memory_summaries:
            text += ms["content"]
        cost = estimate_tokens(text) + 20  # overhead for dict keys

        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            items.append(entry)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["working_memory"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
