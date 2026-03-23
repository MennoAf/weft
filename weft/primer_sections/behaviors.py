"""Behaviors section — persistent agent rules and strategies.

When query-biased (ctx.biased), uses match_behaviors with vector similarity.
Otherwise, lists by priority.  Cap is scaled by ctx.behavior_boost (from
mode weights).

Reference: weft/primer.py lines 241-249 (fetch), 375-406 (pack).
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.primer_sections.context import PrimerContext, SectionResult

logger = logging.getLogger(__name__)


async def build_behaviors_section(ctx: PrimerContext) -> SectionResult:
    raise NotImplementedError(
        "build_behaviors_section not yet implemented — see PRIMER_REFACTOR.md §Behaviors"
    )
