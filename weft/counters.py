"""Global named telemetry counters — aggregate signal for swallowed failures.

Several code paths catch-and-continue by design: an enqueue failure must not
abort the EMA boost it accompanies, an auto-promotion failure must not abort
the calibration tick. Each logs a warning, but a warning-per-occurrence gives
no aggregate signal — a *persistently* broken site looks identical to a
one-off blip in the logs. These counters close that gap: every swallow site
increments a named counter, and ``weft_check_health`` surfaces the totals so a
rising ``replay.enqueue.failed`` (while ``replay_queue_depth`` stays pinned at
0) is a visible "enqueue is silently broken" tell rather than buried in logs.

Storage is the global ``weft_counters`` table (migration v54): a system table
with no user scoping, since these are operational health counts, not user
data, and they are incremented from paths that may run outside a user context.

The increment helper is **best-effort**: it never raises. Counters instrument
failure-handling paths — a counter that could itself throw would defeat the
"must not abort the path" contract it exists to support. A counter-write
failure is logged and swallowed.

Spec: Loom task loom-3c4a0be3.
"""

from __future__ import annotations

import logging

import asyncpg

from weft.db.connection import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Counter names — the canonical dotted identifiers. Import these rather than
# hardcoding the strings so the increment site and the health surface agree.
# ---------------------------------------------------------------------------

# store.apply_reask_feedback — enqueue_replay_on_miss try/except (weft-ee2d3cdf)
COUNTER_REPLAY_ENQUEUE_FAILED = "replay.enqueue.failed"

# calibration._maybe_auto_promote — evaluate_tier_change / update_policy_tier
# swallow (weft-f9f75bf9)
COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED = "calibration.auto_promote.failed"

# Reserved for the E2.L7 replay executor — surfaced now (reads 0 until the
# executor lands and increments it).
COUNTER_REPLAY_EXECUTOR_FAILED = "replay.executor.failed"

# Terminal replay status persistence failed after a bounded retry. The row stays
# pending for stale reaping, but this counter makes the operational seam visible.
COUNTER_REPLAY_TERMINAL_STATUS_FAILED = "replay.terminal_status.failed"

# reap_stale_pending_replays — orphaned 'pending' rows reaped to 'failed' past
# REPLAY_QUEUE_STALENESS_DAYS. Not a failure counter (reaping is remediation, not
# an error): a rising value tracks how many orphaned rows the sweep has drained.
# The live backlog is the replay_queue_stale_pending gauge (count_stale_pending_replays).
COUNTER_REPLAY_STALE_REAPED = "replay.stale.reaped"

# The set surfaced by weft_check_health, in display order.
FAILURE_COUNTERS: tuple[str, ...] = (
    COUNTER_REPLAY_ENQUEUE_FAILED,
    COUNTER_CALIBRATION_AUTO_PROMOTE_FAILED,
    COUNTER_REPLAY_EXECUTOR_FAILED,
    COUNTER_REPLAY_TERMINAL_STATUS_FAILED,
)


async def increment_counter(pool: asyncpg.Pool, name: str, *, by: int = 1) -> None:
    """Atomically add *by* to the named counter (creating it at *by* if absent).

    Best-effort: a counter-write failure is logged and swallowed, never raised.
    Callers instrument failure-handling paths and must not be aborted by the
    instrumentation itself.
    """
    try:
        await get_db(pool).execute(
            """
            INSERT INTO weft_counters (name, count, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (name) DO UPDATE
                SET count = weft_counters.count + EXCLUDED.count,
                    updated_at = now()
            """,
            name,
            by,
        )
    except Exception:  # noqa: BLE001 — telemetry must never break its caller
        logger.warning("increment_counter failed for %r", name, exc_info=True)


async def get_counter(pool: asyncpg.Pool, name: str) -> int:
    """Return the current value of a named counter (0 if it has never fired)."""
    value = await get_db(pool).fetchval(
        "SELECT count FROM weft_counters WHERE name = $1", name
    )
    return int(value) if value is not None else 0


async def get_counters(pool: asyncpg.Pool, names: tuple[str, ...]) -> dict[str, int]:
    """Return values for *names* as a dict, defaulting missing counters to 0."""
    rows = await get_db(pool).fetch(
        "SELECT name, count FROM weft_counters WHERE name = ANY($1::text[])",
        list(names),
    )
    found = {r["name"]: int(r["count"]) for r in rows}
    return {name: found.get(name, 0) for name in names}
