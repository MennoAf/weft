"""Entities section — known people, projects, and tools.

Tier 2 (deferred under progressive disclosure).  Cap is scaled by
ctx.entity_boost from mode weights.

Reference: weft/primer.py line 296 (fetch), 659-682 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_entities_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_entities_section not yet implemented — see PRIMER_REFACTOR.md §Entities"
    )
