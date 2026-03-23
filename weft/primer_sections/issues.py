"""Issues section — active bugs and blockers.

Always included (tier 1).  Filters out unscoped ingested memories.
When query-biased, uses blended similarity + usefulness ranking.

Reference: weft/primer.py lines 263-266 (fetch), 513-557 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_issues_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_issues_section not yet implemented — see PRIMER_REFACTOR.md §Issues"
    )
