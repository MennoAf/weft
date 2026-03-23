"""Grounding section — one-line project description for orientation.

Renders a single string (not a list) from the first memory with
topic="project-grounding" for the current project_id.  Skipped when
no project_id is set.

Reference: weft/primer.py lines 227-234 (fetch), 332-345 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_grounding_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_grounding_section not yet implemented — see PRIMER_REFACTOR.md §Grounding"
    )
