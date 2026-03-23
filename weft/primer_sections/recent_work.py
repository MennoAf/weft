"""Recent work section — milestone breadcrumbs from the last 72 hours.

When query-biased, uses blended similarity + recency ranking (with
effective_sim_weight adjusted by recency_bias from mode weights).
Filters out unscoped ingested memories.

Reference: weft/primer.py lines 258-261 (fetch), 455-511 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_recent_work_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_recent_work_section not yet implemented — see PRIMER_REFACTOR.md §RecentWork"
    )
