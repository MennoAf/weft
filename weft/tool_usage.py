"""Durable aggregate telemetry for MCP tool invocations.

Tool usage is system telemetry rather than user data.  It is intentionally
daily-aggregated so the server can answer lifecycle questions (for example,
whether a deprecated tool is still in use) without retaining request payloads
or creating one database row per invocation.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import asyncpg

from weft.db.connection import get_db

logger = logging.getLogger(__name__)


async def record_tool_usage(
    pool: asyncpg.Pool,
    tool_name: str,
    *,
    called_at: datetime | None = None,
) -> None:
    """Record one invocation, without allowing telemetry to affect the call."""
    if not tool_name:
        return

    observed_at = called_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)

    try:
        await get_db(pool).execute(
            """
            INSERT INTO weft_tool_usage_daily (
                usage_date, tool_name, call_count, first_called_at, last_called_at
            )
            VALUES ($1, $2, 1, $3, $3)
            ON CONFLICT (usage_date, tool_name) DO UPDATE
                SET call_count = weft_tool_usage_daily.call_count + 1,
                    last_called_at = GREATEST(
                        weft_tool_usage_daily.last_called_at,
                        EXCLUDED.last_called_at
                    )
            """,
            observed_at.date(),
            tool_name,
            observed_at,
        )
    except (OSError, asyncpg.PostgresError, RuntimeError) as exc:
        # Usage telemetry is deliberately best-effort.  The MCP call itself
        # must not fail because the observability table is unavailable.
        logger.warning("record_tool_usage failed for %s: %s", tool_name, exc)


async def get_tool_usage_summary(
    pool: asyncpg.Pool,
    *,
    days: int = 30,
    today: date | None = None,
) -> dict:
    """Return aggregate tool usage for the trailing *days* calendar days."""
    if days <= 0:
        raise ValueError("days must be positive")

    end_date = today or date.today()
    since = end_date - timedelta(days=days - 1)
    rows = await get_db(pool).fetch(
        """
        SELECT tool_name,
               SUM(call_count)::BIGINT AS call_count,
               MIN(first_called_at) AS first_called_at,
               MAX(last_called_at) AS last_called_at
        FROM weft_tool_usage_daily
        WHERE usage_date BETWEEN $1 AND $2
        GROUP BY tool_name
        ORDER BY call_count DESC, tool_name
        """,
        since,
        end_date,
    )

    tools = [
        {
            "tool_name": row["tool_name"],
            "call_count": int(row["call_count"]),
            "first_called_at": row["first_called_at"].isoformat(),
            "last_called_at": row["last_called_at"].isoformat(),
        }
        for row in rows
    ]
    return {
        "window_days": days,
        "since": since.isoformat(),
        "through": end_date.isoformat(),
        "tools_used": len(tools),
        "total_calls": sum(tool["call_count"] for tool in tools),
        "tools": tools,
    }
