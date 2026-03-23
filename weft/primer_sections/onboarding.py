"""Onboarding section — hints and cold-start detection.

Post-processing step that runs AFTER all sections are packed.  Generates
empty-section hints, detects cold starts (no handoff + few items), and
runs an RLS diagnostic if all sections are empty but the DB has rows.

Reference: weft/primer.py lines 744-792.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_onboarding_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_onboarding_section not yet implemented — see PRIMER_REFACTOR.md §Onboarding"
    )
