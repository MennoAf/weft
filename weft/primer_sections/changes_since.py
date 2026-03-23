"""Changes-since section — what changed since the last session handoff.

Runs independently of budget packing.  Fetches memory changes and
recent git commits since the last handoff timestamp.

Reference: weft/primer.py lines 684-699.
Line numbers as-of commit 2955aed.
"""

from __future__ import annotations

import logging

from weft.git_utils import get_recent_commits
from weft.primer_sections.context import (
    MAX_CHANGES_SINCE_COMMITS,
    PrimerContext,
    SectionResult,
)
from weft.store import get_last_handoff_timestamp, get_memory_changes_since

logger = logging.getLogger(__name__)


async def build_changes_since_section(ctx: PrimerContext) -> SectionResult:
    """Compute changes since the last session handoff."""
    try:
        handoff_ts = await get_last_handoff_timestamp(
            ctx.pool, project_id=ctx.project_id,
        )
        if handoff_ts is None:
            return SectionResult(items=[], tokens_used=0, skipped=True,
                                 skip_reason="no handoff timestamp")

        changes = await get_memory_changes_since(
            ctx.pool, since=handoff_ts, project_id=ctx.project_id,
        )
        try:
            commits = await get_recent_commits(since=handoff_ts)
        except Exception:
            commits = []
        changes["recent_commits"] = commits[:MAX_CHANGES_SINCE_COMMITS]

        return SectionResult(
            items=[changes], tokens_used=0, skipped=False,
        )
    except Exception as exc:
        logger.warning("Failed to compute changes_since: %s", exc)
        return SectionResult(items=[], tokens_used=0, skipped=True,
                             skip_reason=f"error: {exc}")
