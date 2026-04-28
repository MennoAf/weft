"""Autonomy section — active autonomy policies for the agent.

Tier 2 (deferred under progressive disclosure).  Shows the agent's
current permission boundaries: what's blocked (never), what needs
approval (earned), and what's safe to do (always).
"""

from __future__ import annotations

import logging

from weft.autonomy import AutonomyTier, list_policies
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["autonomy"]
_MAX = SECTION_MAX_ITEMS["autonomy"]


async def fetch_autonomy_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch autonomy policies (parallel-safe, no ctx mutation)."""
    raw = await list_policies(ctx.pool, enabled_only=True, limit=_MAX * 2)
    return SectionFetch(payload=raw)


def pack_autonomy_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack autonomy policies against the budget (sequential, mutates ctx)."""
    raw = fetched.payload or []
    items: list[dict] = []
    section_used = 0
    for policy in raw:
        if len(items) >= _MAX:
            break
        text = f"{policy.action}: {policy.tier.value}"
        if policy.description:
            text += f" — {policy.description}"
        cost = estimate_tokens(text)
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            items.append({
                "action": policy.action,
                "tier": policy.tier.value,
                "description": policy.description,
                "conditions": policy.conditions if policy.conditions else None,
                "id": policy.id,
            })
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["autonomy"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)


async def build_autonomy_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack autonomy policies (single-shot wrapper)."""
    return pack_autonomy_section(ctx, await fetch_autonomy_section(ctx))
