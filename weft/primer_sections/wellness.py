"""Wellness snapshot section — check-in pattern summary.

Runs independently of budget packing (not token-budgeted).  Wraps the
check_in_patterns.analyze_all() output into a concise snapshot of trends,
streaks, and rolling averages.  Failure-tolerant: never blocks the primer.

This section is the canonical example of the "tack stuff onto the end"
anti-pattern that this refactor is designed to prevent.  In the refactored
version, it is a proper section with explicit error handling.

Reference: weft/primer.py lines 701-736.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_wellness_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_wellness_section not yet implemented — see PRIMER_REFACTOR.md §Wellness"
    )
