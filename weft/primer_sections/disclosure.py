"""Progressive disclosure — post-processing that wraps tier 2 sections.

This is NOT a peer section.  It is a post-processing step that runs AFTER
all sections are packed and converts tier 2 sections (decisions, recent_work,
behaviors, entities) into deferred summaries (count + hint).  It also
recalculates used_tokens to reflect only tier 1 content.

Architecturally, this function transforms a fully-packed result dict into
the progressive-disclosure variant.  The orchestrator calls it conditionally
when ctx.disclosure == "progressive".

Reference: weft/primer.py lines 794-855.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def apply_progressive_disclosure(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "apply_progressive_disclosure not yet implemented — see PRIMER_REFACTOR.md §Disclosure"
    )
