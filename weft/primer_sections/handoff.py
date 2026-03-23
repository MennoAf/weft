"""Handoff section — most recent session handoff for continuity.

Always included (tier 1).  Fetches by type=handoff with a deprecated
topic="session-handoff" fallback.  Oversized handoffs are truncated
(not dropped) via truncate_to_token_budget.

Reference: weft/primer.py lines 251-254 (fetch), 408-453 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_handoff_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_handoff_section not yet implemented — see PRIMER_REFACTOR.md §Handoff"
    )
