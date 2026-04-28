"""Triggers section — active proactive triggers for the primer.

Tier 2 (deferred under progressive disclosure).  Shows enabled triggers
so the agent knows what proactive actions are scheduled or waiting for
conditions to be met.
"""

from __future__ import annotations

import logging

from weft.models import TriggerStatus
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.tokens import estimate_tokens
from weft.triggers import list_triggers

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["triggers"]
_MAX = SECTION_MAX_ITEMS["triggers"]


async def fetch_triggers_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch enabled proactive triggers (parallel-safe, no ctx mutation)."""
    raw = await list_triggers(
        ctx.pool, status=TriggerStatus.enabled,
        project_id=ctx.project_id, limit=_MAX * 2,
    )
    if not raw:
        return SectionFetch(skipped=True, skip_reason="No enabled triggers")
    return SectionFetch(payload=raw)


def pack_triggers_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack triggers against the budget (sequential, mutates ctx)."""
    if fetched.skipped:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason=fetched.skip_reason)

    raw = fetched.payload or []
    items: list[dict] = []
    section_used = 0

    for trigger in raw:
        if len(items) >= _MAX:
            break
        text = f"{trigger.name}: {trigger.condition_type.value} → {trigger.action}"
        if trigger.fire_count > 0:
            text += f" (fired {trigger.fire_count}x)"
        cost = estimate_tokens(text)
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            entry = {
                "id": trigger.id,
                "name": trigger.name,
                "condition_type": trigger.condition_type.value,
                "action": trigger.action,
                "fire_count": trigger.fire_count,
            }
            if trigger.cooldown_hours is not None:
                entry["cooldown_hours"] = trigger.cooldown_hours
            if trigger.max_fires is not None:
                entry["max_fires"] = trigger.max_fires
            items.append(entry)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["triggers"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_triggers_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack triggers (single-shot wrapper)."""
    return pack_triggers_section(ctx, await fetch_triggers_section(ctx))
