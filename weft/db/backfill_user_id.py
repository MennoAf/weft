# Audit table created by migration loom-c0495cf2 (landing after this script ships).

"""One-time backfill: assign config-sourced user_id to every row where user_id IS NULL
across all 7 user-scoped tables. Idempotent — safe to re-run.
"""

from __future__ import annotations

import logging

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


async def backfill_user_id(conn: asyncpg.Pool) -> int:
    """One-time migration: assign config-sourced user_id to every row where user_id IS NULL
    across all 7 user-scoped tables. Idempotent: safe to re-run. Logs each update to
    audit_backfill_user_id. Returns the number of rows migrated.
    """
    uid = get_user_id()
    total = 0

    async with conn.acquire() as c:
        async with c.transaction():
            for table, pk_col, has_project_id in _TABLES:
                # Fetch all rows with user_id IS NULL
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

                    # Determine scope class
                    if has_project_id and row["project_id"] is not None:
                        old_scope = "dual-scoped"
                    else:
                        old_scope = "pure-user"

                    # Update the row
                    await c.execute(
                        f"UPDATE {table} SET user_id = $1 WHERE {pk_col} = $2",
                        uid, row[pk_col],
                    )

                    # Write audit row
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
