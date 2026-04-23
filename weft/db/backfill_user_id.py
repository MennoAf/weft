"""One-time backfill: assign a canonical user_id to every row where user_id IS NULL
across all 7 user-scoped tables. Idempotent — safe to re-run.

Admin safety: accepts an explicit ``user_id`` override so operators can pass a
deliberate identity (e.g. the JWT ``sub`` used by the hosted server) instead
of whatever ``get_user_id()`` happens to resolve in the current environment.
A dry-run mode reports the landscape (distinct existing user_ids, NULL counts
per table) without mutating anything — always inspect this first before
running against production.

Audit table ``audit_backfill_user_id`` is created by migration 32 and logs
one row per backfilled source row (table name, row id, old_scope, new_scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import asyncpg

from weft.config.user_identity import get_user_id

logger = logging.getLogger(__name__)

# (table_name, pk_column, has_project_id)
_TABLES: list[tuple[str, str, bool]] = [
    ("behaviors", "id", True),
    ("entities", "id", True),
    ("episodes", "id", True),
    ("modes", "id", True),
    ("autonomy_policies", "id", True),
    ("calibration_records", "id", True),
    ("degradation_policies", "id", True),
]


@dataclass
class BackfillDryRun:
    """Pre-mutation snapshot of the backfill landscape.

    per_table maps ``table_name -> {"null_count": int, "distinct_user_ids":
    [str], "total_rows": int}``. proposed_user_id is what the backfill would
    stamp onto the NULL rows if invoked for real. total_null_rows is the sum
    across all scoped tables — the expected ``return`` value of a real run.
    """

    proposed_user_id: str
    per_table: dict[str, dict] = field(default_factory=dict)
    total_null_rows: int = 0


async def dry_run_backfill_user_id(
    conn: asyncpg.Pool,
    *,
    user_id: str | None = None,
) -> BackfillDryRun:
    """Report the backfill landscape without mutating anything.

    For each scoped table: counts NULL user_id rows, lists distinct non-NULL
    user_ids that already exist (so operators can see whether multiple
    identities are in play before unifying), and returns the total row count.
    """
    proposed_uid = user_id or get_user_id()
    report = BackfillDryRun(proposed_user_id=proposed_uid)

    async with conn.acquire() as c:
        for table, _pk, _has_project in _TABLES:
            null_count = await c.fetchval(
                f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL"
            )
            distinct_rows = await c.fetch(
                f"SELECT DISTINCT user_id FROM {table} "
                f"WHERE user_id IS NOT NULL ORDER BY user_id"
            )
            total = await c.fetchval(f"SELECT COUNT(*) FROM {table}")

            report.per_table[table] = {
                "null_count": int(null_count),
                "distinct_user_ids": [r["user_id"] for r in distinct_rows],
                "total_rows": int(total),
            }
            report.total_null_rows += int(null_count)

    return report


async def backfill_user_id(
    conn: asyncpg.Pool,
    *,
    user_id: str | None = None,
) -> int:
    """Assign a canonical user_id to every NULL-user_id row across scoped tables.

    Idempotent: re-runs stamp nothing and return 0 because NULL rows are gone.
    Logs each update to ``audit_backfill_user_id``.

    Args:
        conn: Database pool.
        user_id: Explicit identity to stamp. When None, resolves via
            ``get_user_id()`` — which in turn honors ``WEFT_USER_ID`` env var
            first, then ``~/.weft/user_id.json``, then a generated fallback.
            Pass an explicit value for admin operations against shared DBs.

    Returns:
        Total number of rows migrated across all scoped tables.
    """
    uid = user_id or get_user_id()
    total = 0

    async with conn.acquire() as c:
        async with c.transaction():
            for table, pk_col, has_project_id in _TABLES:
                if has_project_id:
                    rows = await c.fetch(
                        f"SELECT {pk_col}, project_id FROM {table} WHERE user_id IS NULL"
                    )
                else:
                    rows = await c.fetch(
                        f"SELECT {pk_col} FROM {table} WHERE user_id IS NULL"
                    )

                if not rows:
                    continue

                for row in rows:
                    row_id = str(row[pk_col])

                    if has_project_id and row["project_id"] is not None:
                        old_scope = "dual-scoped"
                    else:
                        old_scope = "pure-user"

                    await c.execute(
                        f"UPDATE {table} SET user_id = $1 WHERE {pk_col} = $2",
                        uid, row[pk_col],
                    )

                    await c.execute(
                        """
                        INSERT INTO audit_backfill_user_id
                            (source_table, row_id, old_scope, new_scope)
                        VALUES ($1, $2, $3, $4)
                        """,
                        table, row_id, old_scope, "user-scoped",
                    )

                    total += 1

    logger.info(
        "backfill_user_id_complete",
        extra={"rows_migrated": total, "user_id": uid},
    )
    return total
