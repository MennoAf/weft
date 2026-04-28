"""Grounding section — one-line project description for orientation.

Renders a single string (not a list) from the first memory with
topic="project-grounding" for the current project_id.  Skipped when
no project_id is set.

Reference: weft/primer.py lines 227-234 (fetch), 332-345 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.models import MemoryStatus
from weft.primer_sections.context import (
    GROUNDING_TOPIC,
    SECTION_BUDGETS,
    PrimerContext,
    SectionFetch,
    SectionResult,
)
from weft.store import list_memories
from weft.tokens import estimate_tokens

logger = logging.getLogger(__name__)

_CAP = SECTION_BUDGETS["grounding"]


async def fetch_grounding_section(ctx: PrimerContext) -> SectionFetch:
    """Fetch grounding memory (parallel-safe, no ctx mutation)."""
    if not ctx.project_id:
        return SectionFetch(skipped=True, skip_reason="no project_id")
    raw = await list_memories(
        ctx.pool, project_id=ctx.project_id,
        topic=GROUNDING_TOPIC, status=MemoryStatus.active, limit=1,
    )
    return SectionFetch(payload=raw)


def pack_grounding_section(ctx: PrimerContext, fetched: SectionFetch) -> SectionResult:
    """Pack grounding line against the budget (sequential, mutates ctx)."""
    if fetched.skipped:
        ctx.section_tokens["grounding"] = 0
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason=fetched.skip_reason)

    raw = fetched.payload or []
    if not raw:
        ctx.section_tokens["grounding"] = 0
        return SectionResult(items=[], tokens_used=0, skipped=False)

    mem = raw[0]
    cost = mem.token_count or estimate_tokens(mem.content)

    if not ctx.fits(cost, 0, _CAP):
        ctx.excluded += 1
        ctx.section_tokens["grounding"] = 0
        return SectionResult(items=[], tokens_used=0, skipped=False)

    ctx.seen_ids.add(mem.id)
    ctx.used_tokens += cost
    ctx.section_tokens["grounding"] = cost

    return SectionResult(
        items=[{"grounding_line": mem.content}],
        tokens_used=cost,
        skipped=False,
    )


async def build_grounding_section(ctx: PrimerContext) -> SectionResult:
    """Fetch and pack grounding (single-shot wrapper)."""
    return pack_grounding_section(ctx, await fetch_grounding_section(ctx))
