"""Rules section — pinned memories that override default agent behavior.

Always included (tier 1, never deferred).  Sorted by confidence ×
usefulness × recency.  Each entry includes review_after annotation.

Reference: weft/primer.py lines 236-238 (fetch), 347-373 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_rules_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_rules_section not yet implemented — see PRIMER_REFACTOR.md §Rules"
    )
