"""Anti-patterns section — pitfalls the agent should avoid.

Always included (tier 1).  Same ranking and filtering logic as issues.
Capped at 3 items.

Reference: weft/primer.py lines 273-277 (fetch), 559-602 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_anti_patterns_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_anti_patterns_section not yet implemented — see PRIMER_REFACTOR.md §AntiPatterns"
    )
