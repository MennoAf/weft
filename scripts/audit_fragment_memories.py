"""Find and (optionally) archive fragment-shaped memories and behaviors.

Background: the 2026-05-04 audit found two recall-pollution sources —
the `before/after` regex in `weft/extract.py` produced behaviors with
truncated triggers ("after a", "after every"), and `weft_remember`
accepted bare markdown headings and trailing-colon fragments.

extract.py and weft_remember are now patched. This script finds the
historical pollution that already lives in the DB.

Usage:
    uv run python scripts/audit_fragment_memories.py            # dry-run
    uv run python scripts/audit_fragment_memories.py --apply    # archive

Archive is reversible: rows get status='archived' (not deleted). Restore
with `UPDATE memories SET status = 'active' WHERE id = ...`.
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

# Project root on sys.path so `weft.*` imports resolve when this script
# is invoked as a path rather than via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env files the same way weft.config does, so DATABASE_URL is
# available without shell sourcing (which mangles special chars in
# the Supabase password).
load_dotenv()
load_dotenv(Path.home() / ".weft" / ".env")


# Conservative determiner set for the audit. Articles, possessives, and
# "every" almost never legitimately close a clause; "every"/"each" are
# determiners; demonstratives ("this", "that", "these", "those") and
# quantifiers ("all", "any", "some") DO end legitimate prose triggers
# (e.g. "I noticed this", "do all of these") so we exclude them here
# to keep audit precision high.
_TRIGGER_TRAIL_DETERMINERS = (
    "a", "an", "the", "every", "my", "our", "your", "their",
)

# Heading-only flag: only fire when the entire stripped content is a
# single line starting with `#` AND short enough to plausibly be a bare
# title rather than a one-paragraph summary. Above this length, the
# heading marker may be intentional formatting on a real summary.
_MAX_HEADING_ONLY_LEN = 80


@dataclass
class Finding:
    table: str
    row_id: str
    project_id: str | None
    reason: str
    preview: str


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


async def find_fragment_memories(conn: asyncpg.Connection) -> list[Finding]:
    """Memories that look like extraction artifacts: bare headings,
    trailing-colon truncations, sub-15-char ghost writes."""
    rows = await conn.fetch(
        """
        SELECT id, project_id, content
          FROM memories
         WHERE status = 'active'
           AND (
                length(trim(content)) < 15
             OR trim(content) ~ '^#{1,6}\\s+\\S[^\\n]*$'
             OR trim(content) ~ ':\\s*$'
           )
         ORDER BY created_at DESC
        """,
    )
    findings: list[Finding] = []
    for r in rows:
        content = (r["content"] or "").strip()
        if len(content) < 15:
            reason = "content_too_short"
        elif content.endswith(":"):
            reason = "trailing_colon"
        elif (
            "\n" not in content
            and len(content) <= _MAX_HEADING_ONLY_LEN
            and content.lstrip().startswith("#")
        ):
            reason = "heading_only"
        else:
            # Long single-line memory starting with `#` — likely an
            # intentional summary, not a bare title. Skip.
            continue
        findings.append(Finding(
            table="memories",
            row_id=str(r["id"]),
            project_id=r["project_id"],
            reason=reason,
            preview=content[:120],
        ))
    return findings


async def find_fragment_behaviors(conn: asyncpg.Connection) -> list[Finding]:
    """Behaviors with triggers that look like regex over-match artifacts —
    triggers ending in a bare determiner, or sub-5-char triggers/actions."""
    rows = await conn.fetch(
        """
        SELECT id, project_id, trigger_pattern, action
          FROM behaviors
         WHERE status = 'active'
        """,
    )
    findings: list[Finding] = []
    for r in rows:
        trigger = (r["trigger_pattern"] or "").strip()
        action = (r["action"] or "").strip()
        last = trigger.split()[-1].lower().strip(".,;:") if trigger.split() else ""
        if last in _TRIGGER_TRAIL_DETERMINERS:
            reason = "trigger_ends_in_determiner"
        elif len(trigger) < 5 or len(action) < 5:
            reason = "trigger_or_action_too_short"
        else:
            continue
        preview = f"trigger={trigger!r} → action={action[:80]!r}"
        findings.append(Finding(
            table="behaviors",
            row_id=str(r["id"]),
            project_id=r["project_id"],
            reason=reason,
            preview=preview,
        ))
    return findings


def _print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("  (none)")
        return
    by_reason: dict[str, list[Finding]] = {}
    for f in findings:
        by_reason.setdefault(f.reason, []).append(f)
    for reason, items in sorted(by_reason.items()):
        print(f"\n  [{reason}] — {len(items)} row(s)")
        for f in items[:10]:
            proj = f.project_id or "(no project)"
            print(f"    {f.row_id}  proj={proj}")
            print(f"      {f.preview}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")


async def _archive(conn: asyncpg.Connection, findings: list[Finding]) -> int:
    n = 0
    for f in findings:
        await conn.execute(
            f"UPDATE {f.table} SET status = 'archived', updated_at = now() WHERE id = $1",
            f.row_id,
        )
        n += 1
    return n


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Archive flagged rows (status='archived'). Default is dry-run.",
    )
    args = parser.parse_args()

    conn = await _connect()
    try:
        print("=== Audit: fragment-shaped memories ===")
        mem_findings = await find_fragment_memories(conn)
        print(f"\nMEMORIES — {len(mem_findings)} candidate row(s)")
        _print_findings(mem_findings)

        print("\n\n=== Audit: fragment-shaped behaviors ===")
        beh_findings = await find_fragment_behaviors(conn)
        print(f"\nBEHAVIORS — {len(beh_findings)} candidate row(s)")
        _print_findings(beh_findings)

        all_findings = mem_findings + beh_findings
        print(f"\n\nTOTAL: {len(all_findings)} row(s) flagged.")

        if args.apply:
            if not all_findings:
                print("Nothing to archive.")
                return 0
            confirm = input(f"\nArchive all {len(all_findings)} rows? Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                return 1
            n = await _archive(conn, all_findings)
            print(f"Archived {n} row(s). Reverse with status='active' on the affected ids.")
        else:
            print("\n(dry-run — pass --apply to archive)")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
