"""Degradation section — active degradation policies for the primer.

Tier 2 (deferred under progressive disclosure).  Shows active degradation
policies so the agent knows what guardrails are in place and which have
recently fired.
"""

from __future__ import annotations

import logging

from weft.degradation import list_policies
from weft.models import DegradationPolicyStatus
from weft.primer_sections.context import (
    SECTION_BUDGETS,
    SECTION_MAX_ITEMS,
    PrimerContext,
    SectionResult,
)
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["degradation"]
_MAX = SECTION_MAX_ITEMS["degradation"]


async def build_degradation_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack active degradation policies."""
    raw = await list_policies(
        ctx.pool,
        status=DegradationPolicyStatus.active,
        project_id=ctx.project_id,
        limit=_MAX * 2,
    )

    # Also include recently-fired policies (still relevant context)
    fired = await list_policies(
        ctx.pool,
        status=DegradationPolicyStatus.fired,
        project_id=ctx.project_id,
        limit=5,
    )
    raw.extend(fired)

    if not raw:
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason="No active degradation policies")

    items: list[dict] = []
    section_used = 0

    for policy in raw:
        if len(items) >= _MAX:
            break
        text = (
            f"{policy.name}: {policy.trigger_type.value} → {policy.action.value}"
            f" [{policy.status.value}]"
        )
        if policy.fire_count > 0:
            text += f" (fired {policy.fire_count}x)"
        cost = estimate_tokens(text)
        if ctx.fits_or_guarantee(cost, section_used, _CAP):
            entry = {
                "id": policy.id,
                "name": policy.name,
                "trigger_type": policy.trigger_type.value,
                "action": policy.action.value,
                "status": policy.status.value,
                "fire_count": policy.fire_count,
            }
            if policy.description:
                entry["description"] = policy.description
            items.append(entry)
            ctx.used_tokens += cost
            section_used += cost
        else:
            ctx.excluded += 1

    ctx.section_tokens["degradation"] = section_used
    return SectionResult(items=items, tokens_used=section_used, skipped=False)
