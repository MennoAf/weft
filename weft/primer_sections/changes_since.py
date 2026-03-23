"""Changes-since section — what changed since the last session handoff.

Runs independently of budget packing.  Fetches memory changes and
recent git commits since the last handoff timestamp.

Reference: weft/primer.py lines 684-699.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_changes_since_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_changes_since_section not yet implemented — see PRIMER_REFACTOR.md §ChangesSince"
    )
