"""Handoff section — most recent session handoff for continuity.

Always included (tier 1).  Fetches by type=handoff with a deprecated
topic="session-handoff" fallback.  Oversized handoffs are truncated
(not dropped) via truncate_to_token_budget.

Reference: weft/primer.py lines 251-254 (fetch), 408-453 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus, MemoryType
from weft.primer_sections.context import (
    DICT_OVERHEAD_TOKENS,
    SECTION_BUDGETS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.store import list_memories
from weft.tokens import estimate_tokens, truncate_to_token_budget

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["handoff"]


async def fetch_handoff_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch most recent handoff (parallel-safe, no ctx mutation).

    Tries typed (type=handoff) first; falls back to deprecated
    topic="session-handoff" only when typed returns empty. seen_ids
    filtering is deferred to pack.
    """
    raw = await list_memories(
        ctx.pool, memory_type=MemoryType.handoff, status=MemoryStatus.active,
        limit=5, **ctx.scope,
    )
    used_fallback = False
    if not raw:
        raw = await list_memories(
            ctx.pool, topic="session-handoff", status=MemoryStatus.active,
            limit=5, **ctx.scope,
        )
        used_fallback = bool(raw)
    return SectionFetch(payload={"raw": raw, "used_fallback": used_fallback})


def pack_handoff_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack the most recent handoff against the budget (sequential, mutates ctx)."""
    payload = fetched.payload or {"raw": [], "used_fallback": False}
    raw = payload["raw"]
    candidates = [m for m in raw if m.id not in ctx.seen_ids]

    if payload["used_fallback"] and candidates:
        logger.warning(
            "Handoff found via topic fallback — re-store with type=handoff "
            "to silence this warning (fallback will be removed in v0.3)",
        )

    # Prefer project-scoped handoffs over global (NULL) ones, then most recent.
    candidates.sort(
        key=lambda m: (m.project_id is not None, m.created_at), reverse=True,
    )

    items: list[dict] = []
    section_used = 0

    if candidates:
        mem = candidates[0]
        # Always re-estimate from content — stored token_count may be stale.
        cost = estimate_tokens(mem.content) + DICT_OVERHEAD_TOKENS
        content = mem.content
        # Truncate oversized handoffs instead of dropping them.
        cap = min(_CAP, ctx.budget_tokens - ctx.used_tokens)
        if cost > cap and cap > 0:
            content, cost = truncate_to_token_budget(content, cap)
            cost += DICT_OVERHEAD_TOKENS
        if ctx.used_tokens + cost <= ctx.budget_tokens and cap > 0:
            age_hours = (ctx.now - mem.created_at).total_seconds() / 3600
            entry = {
                "id": mem.id,
                "type": mem.type.value,
                "content": content,
                "confidence": mem.confidence,
                "created_at": mem.created_at.isoformat() if mem.created_at else None,
                "age_hours": round(age_hours, 1),
            }
            items.append(entry)
            ctx.seen_ids.add(mem.id)
            ctx.used_tokens += cost
            section_used = cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["handoff"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_handoff_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack handoff (single-shot wrapper)."""
    return pack_handoff_section(ctx, await fetch_handoff_section(ctx))
