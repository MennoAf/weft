"""Session-scoped memory access tracking and implicit usefulness signals.

Tracks which memories are surfaced during each MCP session. When a session
concludes successfully (via weft_handoff or weft_learn), all accessed
memories receive a small, session-deduplicated usefulness boost.

Session ID is managed via contextvars so it threads through without
polluting function signatures.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

import asyncpg

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
) -> None:
    """Log that memories were accessed in the current session.

    Fire-and-forget safe — never raises, logs warnings on failure.
    Uses INSERT ... ON CONFLICT DO NOTHING for session-level deduplication:
    accessing the same memory 5 times in one session records one row.
    """
    if not memory_ids:
        return

    sid = session_id or get_session_id()

    try:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memory_access_log (session_id, memory_id, tool_name)
                VALUES ($1, $2, $3)
                ON CONFLICT (session_id, memory_id) DO NOTHING
                """,
                [(sid, mid, tool_name) for mid in memory_ids],
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
                SET usefulness_score = LEAST(m.usefulness_score + $1, $2)
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
