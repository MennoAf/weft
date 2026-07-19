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

RECORDER_VERSION = "2"
MIN_DEPRECATION_COVERAGE_DAYS = 30


async def record_tool_usage_heartbeat(
    pool: asyncpg.Pool,
    *,
    observed_at: datetime | None = None,
    successful_writes: int = 0,
    failure_count: int = 0,
    shutdown_drained: bool | None = None,
) -> None:
    """Upsert one versioned recorder-health marker for a UTC day."""
    if successful_writes < 0 or failure_count < 0:
        raise ValueError("coverage counters cannot be negative")
    observed = observed_at or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    else:
        observed = observed.astimezone(timezone.utc)
    await get_db(pool).execute(
        """
        INSERT INTO weft_tool_usage_coverage (
            coverage_date, recorder_version, first_heartbeat_at,
            last_heartbeat_at, successful_writes, failure_count,
            shutdown_drained
        ) VALUES ($1, $2, $3, $3, $4, $5, $6)
        ON CONFLICT (coverage_date) DO UPDATE
            SET recorder_version = EXCLUDED.recorder_version,
                last_heartbeat_at = GREATEST(
                    weft_tool_usage_coverage.last_heartbeat_at,
                    EXCLUDED.last_heartbeat_at
                ),
                successful_writes = (
                    weft_tool_usage_coverage.successful_writes
                    + EXCLUDED.successful_writes
                ),
                failure_count = (
                    weft_tool_usage_coverage.failure_count
                    + EXCLUDED.failure_count
                ),
                shutdown_drained = CASE
                    WHEN weft_tool_usage_coverage.shutdown_drained IS FALSE
                      OR EXCLUDED.shutdown_drained IS FALSE THEN FALSE
                    ELSE COALESCE(
                        EXCLUDED.shutdown_drained,
                        weft_tool_usage_coverage.shutdown_drained
                    )
                END
        """,
        observed.date(),
        RECORDER_VERSION,
        observed,
        successful_writes,
        failure_count,
        shutdown_drained,
    )


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
    else:
        observed_at = observed_at.astimezone(timezone.utc)

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
    await record_tool_usage_heartbeat(
        pool,
        observed_at=observed_at,
        successful_writes=1,
    )


async def get_tool_usage_summary(
    pool: asyncpg.Pool,
    *,
    days: int = 30,
    today: date | None = None,
) -> dict:
    """Return aggregate tool usage for the trailing *days* calendar days."""
    if days <= 0:
        raise ValueError("days must be positive")

    end_date = today or datetime.now(timezone.utc).date()
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

    coverage_rows = await get_db(pool).fetch(
        """
        SELECT coverage_date, recorder_version, first_heartbeat_at,
               last_heartbeat_at, successful_writes, failure_count,
               shutdown_drained
        FROM weft_tool_usage_coverage
        WHERE coverage_date BETWEEN $1 AND $2
        ORDER BY coverage_date
        """,
        since,
        end_date,
    )
    coverage_by_date = {row["coverage_date"]: row for row in coverage_rows}
    expected_dates = [since + timedelta(days=offset) for offset in range(days)]
    valid_dates = [
        day
        for day in expected_dates
        if day in coverage_by_date
        and coverage_by_date[day]["recorder_version"] == RECORDER_VERSION
        and int(coverage_by_date[day]["failure_count"]) == 0
        and coverage_by_date[day]["shutdown_drained"] is not False
    ]
    gap_dates = [day for day in expected_dates if day not in valid_dates]

    tools = [
        {
            "tool_name": row["tool_name"],
            "call_count": int(row["call_count"]),
            "first_called_at": row["first_called_at"].isoformat(),
            "last_called_at": row["last_called_at"].isoformat(),
        }
        for row in rows
    ]
    valid_days = len(valid_dates)
    return {
        "window_days": days,
        "since": since.isoformat(),
        "through": end_date.isoformat(),
        "tools_used": len(tools),
        "total_calls": sum(tool["call_count"] for tool in tools),
        "tools": tools,
        "coverage": {
            "recorder_version": RECORDER_VERSION,
            "expected_days": days,
            "valid_days": valid_days,
            "gap_days": len(gap_dates),
            "gap_dates": [day.isoformat() for day in gap_dates],
            "failure_total": sum(
                int(row["failure_count"]) for row in coverage_rows
            ),
            "successful_writes": sum(
                int(row["successful_writes"]) for row in coverage_rows
            ),
            "shutdown_not_drained_dates": [
                row["coverage_date"].isoformat()
                for row in coverage_rows
                if row["shutdown_drained"] is False
            ],
            "complete": valid_days == days,
        },
        "deprecation_eligible": (
            valid_days >= MIN_DEPRECATION_COVERAGE_DAYS
            and len(gap_dates) == 0
        ),
        "zero_use_classification": (
            "observed-zero" if valid_days == days else "coverage-incomplete"
        ),
    }
