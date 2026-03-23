"""Decisions section — closed/vetoed decisions (what NOT to suggest).

Tier 2 (deferred under progressive disclosure).  Project-scoped decisions
sort before global ones.  Includes review_after annotation.

Reference: weft/primer.py lines 268-272 (fetch), 604-657 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_decisions_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_decisions_section not yet implemented — see PRIMER_REFACTOR.md §Decisions"
    )
