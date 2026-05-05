"""Repair behaviors whose stored scope contradicts their stored project_id.

Background: prior to the fix in commit-pending, ``weft_behavior_add``
auto-resolved ``project_id`` from the caller's CWD even when
``scope='global'`` was supplied. The result was a row with
``scope='global'`` AND ``project_id='<caller_cwd>'`` — which the
``match_behaviors`` and ``list_behaviors`` filters interpret as
project-scoped (the OR-NULL filter requires ``project_id IS NULL`` for
truly cross-project matching). So globally-intended rules silently
fired only inside their birth-project.

This script finds those rows and (optionally) sets ``project_id=NULL``
to honor the declared scope.

Usage:
    uv run python scripts/fix_global_behavior_scope.py            # dry-run
    uv run python scripts/fix_global_behavior_scope.py --apply    # repair

Reversible: run the inverse UPDATE if you need to put project_id back.

Also surfaces — but does NOT auto-fix — any behaviors with
``user_id IS NULL``. Migration 36 backfilled all NULLs to the
SYSTEM_GLOBAL sentinel; a NULL here would mean a row written by a
caller that bypassed the DB DEFAULT (the column is NOT NULL, so this
should be impossible — but the audit is cheap, so we run it).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()
load_dotenv(Path.home() / ".weft" / ".env")


@dataclass
class Finding:
    row_id: str
    scope: str
    project_id: str | None
    user_id: str | None
    trigger_preview: str
    reason: str


def _ssl_ctx_for(dsn: str) -> ssl.SSLContext | None:
    if "supabase.co" in dsn or "supabase.com" in dsn or "sslmode=require" in dsn:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


async def _connect() -> asyncpg.Connection:
    from weft.config import _encode_dsn_password
    dsn = os.environ.get("WEFT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("ERROR: set WEFT_DATABASE_URL or DATABASE_URL")
    dsn = _encode_dsn_password(dsn)
    if "+psycopg2" in dsn:
        dsn = dsn.replace("+psycopg2", "")
    kwargs: dict = {}
    if ":6543/" in dsn or "pooler.supabase.com" in dsn:
        kwargs["statement_cache_size"] = 0
    ctx = _ssl_ctx_for(dsn)
    if ctx is not None:
        kwargs["ssl"] = ctx
    return await asyncpg.connect(dsn, **kwargs)


async def find_misscoped_global(conn: asyncpg.Connection) -> list[Finding]:
    rows = await conn.fetch(
        """
        SELECT id, scope, project_id, user_id, trigger_pattern
          FROM behaviors
         WHERE status = 'active'
           AND scope = 'global'
           AND project_id IS NOT NULL
         ORDER BY created_at DESC
        """,
    )
    return [
        Finding(
            row_id=str(r["id"]),
            scope=r["scope"],
            project_id=r["project_id"],
            user_id=r["user_id"],
            trigger_preview=(r["trigger_pattern"] or "")[:90],
            reason="scope_global_with_project_id",
        )
        for r in rows
    ]


async def find_null_user_id(conn: asyncpg.Connection) -> list[Finding]:
    """Diagnostic only — should be empty post-mig-36."""
    rows = await conn.fetch(
        """
        SELECT id, scope, project_id, user_id, trigger_pattern
          FROM behaviors
         WHERE status = 'active'
           AND user_id IS NULL
         ORDER BY created_at DESC
        """,
    )
    return [
        Finding(
            row_id=str(r["id"]),
            scope=r["scope"],
            project_id=r["project_id"],
            user_id=r["user_id"],
            trigger_preview=(r["trigger_pattern"] or "")[:90],
            reason="user_id_is_null_post_mig_36",
        )
        for r in rows
    ]


def _print_findings(label: str, findings: list[Finding]) -> None:
    print(f"\n▸ {label}: {len(findings)} row(s)")
    if not findings:
        print("  (none)")
        return
    for f in findings[:25]:
        proj = f.project_id or "(NULL)"
        user = f.user_id or "(NULL)"
        print(f"  {f.row_id}  scope={f.scope}  project_id={proj}  user_id={user}")
        print(f"    trigger: {f.trigger_preview}")
    if len(findings) > 25:
        print(f"  ... and {len(findings) - 25} more")


async def _apply_global_scope_fix(
    conn: asyncpg.Connection, findings: list[Finding],
) -> int:
    if not findings:
        return 0
    ids = [f.row_id for f in findings]
    result = await conn.execute(
        """
        UPDATE behaviors
           SET project_id = NULL,
               updated_at = now()
         WHERE id = ANY($1::text[])
           AND scope = 'global'
           AND project_id IS NOT NULL
        """,
        ids,
    )
    # asyncpg returns "UPDATE n"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the UPDATE. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    conn = await _connect()
    try:
        misscoped = await find_misscoped_global(conn)
        null_users = await find_null_user_id(conn)

        _print_findings("scope='global' with non-NULL project_id", misscoped)
        _print_findings(
            "user_id IS NULL (should be empty post-mig-36)", null_users,
        )

        if null_users:
            print(
                "\n⚠ user_id IS NULL rows present — NOT auto-fixing here. "
                "Investigate how they bypassed the NOT NULL default before "
                "deciding whether to set them to '__system_global_zathras__' "
                "or to the row owner's user_id.",
            )

        if not args.apply:
            print("\nDry-run only. Re-run with --apply to repair scope='global' rows.")
            return 0

        if not misscoped:
            print("\nNothing to repair. Exiting.")
            return 0

        n = await _apply_global_scope_fix(conn, misscoped)
        print(f"\n✓ Updated {n} row(s): project_id set to NULL.")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
