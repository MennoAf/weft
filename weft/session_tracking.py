"""Session-scoped memory access tracking and implicit usefulness signals.

Tracks which memories are surfaced during each MCP session. When a session
concludes successfully (via weft_handoff or weft_learn), all accessed
memories receive a small, session-deduplicated usefulness boost.

Session ID is managed via contextvars so it threads through without
polluting function signatures.

Phase 2 follow-on (mig 41): the same write also records reader_user_id,
reader_caller_mode, and retrieval_mode pulled from the auth contextvars,
turning this table into the read-side audit log. The (session_id,
memory_id) PK preserves session-deduplicated semantics for both the
usefulness boost and the audit trail — first access wins, subsequent
accesses in the same session add no forensic information.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

import asyncpg

from weft.auth import current_user_id, get_caller_mode

logger = logging.getLogger(__name__)

# Session ID for the current MCP session
_session_id: ContextVar[str] = ContextVar("weft_session_id")

# Boost constants — conservative defaults per design discussion:
# - Implicit access: +0.01 per session (much weaker than explicit feedback)
# - Explicit helpful:  +0.05 (via record_feedback, unchanged)
# - Explicit unhelpful: -0.05 (via record_feedback, unchanged)
IMPLICIT_ACCESS_BOOST = 0.01
USEFULNESS_CAP = 1.0


def get_session_id() -> str:
    """Get or create a session ID for the current context."""
    try:
        return _session_id.get()
    except LookupError:
        sid = f"ses-{uuid.uuid4().hex[:12]}"
        _session_id.set(sid)
        return sid


def set_session_id(session_id: str) -> None:
    """Explicitly set the session ID (e.g., from MCP framework)."""
    _session_id.set(session_id)


async def log_memory_access(
    pool: asyncpg.Pool,
    memory_ids: list[str],
    tool_name: str,
    session_id: str | None = None,
    retrieval_mode: str | None = None,
) -> None:
    """Log that memories were accessed in the current session.

    Fire-and-forget safe — never raises, logs warnings on failure.
    Uses INSERT ... ON CONFLICT DO NOTHING for session-level deduplication:
    accessing the same memory 5 times in one session records one row.

    Phase 2 follow-on (mig 41): also stamps reader_user_id, reader_caller_mode,
    and retrieval_mode from the auth contextvars so a supervisor can answer
    "who read this poisoned memory and when?" during incident response.
    Pulled from contextvars rather than passed explicitly because every
    retrieval path goes through the same auth middleware — explicit args
    would just be re-reading the same contextvars at every call site.
    """
    if not memory_ids:
        return

    sid = session_id or get_session_id()
    reader_uid = current_user_id.get()
    reader_mode = get_caller_mode()

    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memory_access_log (
                    session_id, memory_id, tool_name,
                    reader_user_id, reader_caller_mode, retrieval_mode
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (session_id, memory_id) DO NOTHING
                """,
                [
                    (sid, mid, tool_name, reader_uid, reader_mode, retrieval_mode)
                    for mid in memory_ids
                ],
            )
    except Exception as e:
        logger.warning("memory_access_log write failed: %s", e)


async def boost_session_memories(
    pool: asyncpg.Pool,
    session_id: str | None = None,
    boost: float = IMPLICIT_ACCESS_BOOST,
) -> dict:
    """Boost usefulness_score for all memories accessed in a session.

    Called at session end (handoff/learn). Session-deduplicated — each
    memory gets at most one boost per session regardless of how many
    times it was accessed. Returns summary dict for logging.
    """
    sid = session_id or get_session_id()

    try:
        async with pool.acquire() as conn:
            # Single UPDATE using the access log for deduplication
            result = await conn.execute(
                """
                UPDATE memories m
                SET usefulness_score = LEAST(m.usefulness_score + $1, $2),
                    last_boosted_at = now()
                FROM (
                    SELECT DISTINCT memory_id
                    FROM memory_access_log
                    WHERE session_id = $3
                ) AS accessed
                WHERE m.id = accessed.memory_id
                """,
                boost,
                USEFULNESS_CAP,
                sid,
            )

            count = int(result.split()[-1])  # "UPDATE N"
            if count > 0:
                logger.info(
                    "Boosted %d memories by +%.3f for session %s",
                    count, boost, sid,
                )
            return {"boosted": count, "session_id": sid, "boost": boost}
    except Exception as e:
        logger.warning("boost_session_memories failed: %s", e)
        return {"boosted": 0, "session_id": sid, "error": str(e)}


async def prune_old_access_logs(
    pool: asyncpg.Pool,
    cutoff_days: int = 90,
) -> int:
    """Delete access log entries older than cutoff_days. Returns count deleted.

    Safe to call from consolidation — logs failures without raising.
    """
    try:
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        result = await pool.execute(
            "DELETE FROM memory_access_log WHERE accessed_at < $1",
            cutoff,
        )
        count = int(result.split()[-1])
        if count > 0:
            logger.info("Pruned %d access log entries older than %d days", count, cutoff_days)
        return count
    except Exception as e:
        logger.warning("prune_old_access_logs failed: %s", e)
        return 0


async def get_session_memory_ids(
    pool: asyncpg.Pool,
    session_id: str | None = None,
) -> list[str]:
    """Return distinct memory IDs accessed in a session.

    Useful for weft_focus's exclude_memory_ids (differential output).
    """
    sid = session_id or get_session_id()

    try:
        rows = await pool.fetch(
            "SELECT memory_id FROM memory_access_log WHERE session_id = $1",
            sid,
        )
        return [r["memory_id"] for r in rows]
    except Exception as e:
        logger.warning("get_session_memory_ids failed: %s", e)
        return []
