"""Repair cross-tenant recall-canary outcomes from the owner-scoping bug.

The affected scheduler selected every owner's probes through a BYPASSRLS
service role, then searched only the deployment owner's memory corpus. Outcomes
for non-default owners were therefore deterministic false misses.

This command is dry-run by default. Applying a repair requires exact expected
counts and writes a restore journal before the transaction mutates anything.

Examples:
    uv run python scripts/repair_canary_cross_tenant.py \
        --token-label codex-desktop --before 2026-07-17T12:00:00Z

    uv run python scripts/repair_canary_cross_tenant.py \
        --token-label codex-desktop --before 2026-07-17T12:00:00Z \
        --expected-events 327 --expected-misses 327 \
        --journal /secure/canary-repair-20260717.json --apply

    uv run python scripts/repair_canary_cross_tenant.py \
        --restore /secure/canary-repair-20260717.json --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg

from weft.config import load_config
from weft.db.connection import create_pool


JOURNAL_VERSION = 1
COUNTER_CANARY_MISS = "canary.miss"


class RepairMismatch(RuntimeError):
    """Raised when live state does not match the operator's apply gate."""


@dataclass(frozen=True)
class AuditEvent:
    id: int
    probe_id: str
    user_id: str
    audited_at: datetime
    hit: bool

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["audited_at"] = self.audited_at.isoformat()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AuditEvent:
        return cls(
            id=int(payload["id"]),
            probe_id=str(payload["probe_id"]),
            user_id=str(payload["user_id"]),
            audited_at=parse_timestamp(str(payload["audited_at"])),
            hit=bool(payload["hit"]),
        )


@dataclass(frozen=True)
class ProbeState:
    probe_id: str
    audit_count: int
    miss_count: int
    last_audit_at: datetime | None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["last_audit_at"] = (
            self.last_audit_at.isoformat() if self.last_audit_at else None
        )
        return payload


@dataclass(frozen=True)
class RepairPreview:
    user_id: str
    before: datetime
    events: tuple[AuditEvent, ...]
    probes: tuple[ProbeState, ...]
    global_counter_before: int

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def miss_count(self) -> int:
        return sum(not event.hit for event in self.events)

    @property
    def hit_count(self) -> int:
        return self.event_count - self.miss_count

    def summary(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "before": self.before.isoformat(),
            "events": self.event_count,
            "misses": self.miss_count,
            "hits": self.hit_count,
            "affected_probes": len(self.probes),
            "global_counter_before": self.global_counter_before,
            "global_counter_after": self.global_counter_before - self.miss_count,
        }


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and require an explicit timezone."""
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone (for example, Z or +00:00)")
    return parsed.astimezone(timezone.utc)


async def resolve_user_id(conn: asyncpg.Connection, token_label: str) -> str:
    """Resolve a token label to exactly one owner, including revoked tokens."""
    rows = await conn.fetch(
        "SELECT DISTINCT user_id FROM weft_tokens WHERE label = $1 ORDER BY user_id",
        token_label,
    )
    if len(rows) != 1:
        raise RepairMismatch(
            f"token label {token_label!r} resolved to {len(rows)} owners; expected 1"
        )
    return str(rows[0]["user_id"])


async def inspect_repair(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    before: datetime,
    lock: bool = False,
) -> RepairPreview:
    """Read the exact events and counters a repair would change."""
    lock_clause = " FOR UPDATE OF a" if lock else ""
    event_rows = await conn.fetch(
        """
        SELECT a.id, a.probe_id, a.user_id, a.audited_at, a.hit
        FROM recall_canary_audit a
        JOIN recall_canary c ON c.probe_id = a.probe_id
        WHERE a.user_id = $1 AND c.user_id = $1 AND a.audited_at < $2
        ORDER BY a.id
        """ + lock_clause,
        user_id,
        before,
    )
    events = tuple(
        AuditEvent(
            id=int(row["id"]),
            probe_id=str(row["probe_id"]),
            user_id=str(row["user_id"]),
            audited_at=row["audited_at"],
            hit=bool(row["hit"]),
        )
        for row in event_rows
    )
    probe_ids = sorted({event.probe_id for event in events})
    probe_rows = []
    if probe_ids:
        probe_rows = await conn.fetch(
            """
            SELECT probe_id, audit_count, miss_count, last_audit_at
            FROM recall_canary
            WHERE user_id = $1 AND probe_id = ANY($2::text[])
            ORDER BY probe_id
            """ + (" FOR UPDATE" if lock else ""),
            user_id,
            probe_ids,
        )
    probes = tuple(
        ProbeState(
            probe_id=str(row["probe_id"]),
            audit_count=int(row["audit_count"]),
            miss_count=int(row["miss_count"]),
            last_audit_at=row["last_audit_at"],
        )
        for row in probe_rows
    )
    counter = await conn.fetchval(
        "SELECT count FROM weft_counters WHERE name = $1"
        + (" FOR UPDATE" if lock else ""),
        COUNTER_CANARY_MISS,
    )
    return RepairPreview(
        user_id=user_id,
        before=before,
        events=events,
        probes=probes,
        global_counter_before=int(counter or 0),
    )


def validate_apply_gate(
    preview: RepairPreview,
    *,
    expected_events: int,
    expected_misses: int,
) -> None:
    """Refuse ambiguous or changed repair scopes before writing a journal."""
    if preview.event_count != expected_events:
        raise RepairMismatch(
            f"event count changed: expected {expected_events}, found {preview.event_count}"
        )
    if preview.miss_count != expected_misses:
        raise RepairMismatch(
            f"miss count changed: expected {expected_misses}, found {preview.miss_count}"
        )
    if preview.hit_count:
        raise RepairMismatch(
            f"scope contains {preview.hit_count} hit event(s); false-scope repair expects misses only"
        )
    if not preview.events:
        raise RepairMismatch("repair scope is empty")
    if len(preview.probes) != len({event.probe_id for event in preview.events}):
        raise RepairMismatch("one or more affected probes are missing")

    deltas = _event_deltas(preview.events)
    for probe in preview.probes:
        checks, misses = deltas[probe.probe_id]
        if probe.audit_count < checks or probe.miss_count < misses:
            raise RepairMismatch(
                f"probe {probe.probe_id} counters are smaller than repair delta"
            )
    if preview.global_counter_before < preview.miss_count:
        raise RepairMismatch("global canary.miss counter is smaller than repair delta")


def _event_deltas(events: tuple[AuditEvent, ...]) -> dict[str, tuple[int, int]]:
    checks: dict[str, int] = defaultdict(int)
    misses: dict[str, int] = defaultdict(int)
    for event in events:
        checks[event.probe_id] += 1
        misses[event.probe_id] += int(not event.hit)
    return {probe_id: (checks[probe_id], misses[probe_id]) for probe_id in checks}


def _journal_payload(preview: RepairPreview) -> dict[str, Any]:
    return {
        "version": JOURNAL_VERSION,
        "kind": "recall_canary_cross_tenant_repair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied_at": None,
        "restored_at": None,
        "summary": preview.summary(),
        "events": [event.to_json() for event in preview.events],
        "probes_before": [probe.to_json() for probe in preview.probes],
        "global_counter_before": preview.global_counter_before,
    }


def _write_journal_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _update_journal(path: Path, field: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = datetime.now(timezone.utc).isoformat()
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


async def apply_repair(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    before: datetime,
    expected_events: int,
    expected_misses: int,
    journal_path: Path,
) -> RepairPreview:
    """Apply the gated repair atomically and persist a restore journal."""
    preview: RepairPreview
    async with conn.transaction():
        preview = await inspect_repair(
            conn, user_id=user_id, before=before, lock=True
        )
        validate_apply_gate(
            preview,
            expected_events=expected_events,
            expected_misses=expected_misses,
        )
        _write_journal_exclusive(journal_path, _journal_payload(preview))

        event_ids = [event.id for event in preview.events]
        deleted = await conn.fetchval(
            """
            WITH removed AS (
                DELETE FROM recall_canary_audit
                WHERE id = ANY($1::bigint[]) AND user_id = $2
                RETURNING 1
            )
            SELECT count(*) FROM removed
            """,
            event_ids,
            user_id,
        )
        if int(deleted or 0) != preview.event_count:
            raise RepairMismatch("deleted event count changed inside transaction")

        for probe_id, (checks, misses) in _event_deltas(preview.events).items():
            row = await conn.fetchrow(
                """
                UPDATE recall_canary c
                SET audit_count = audit_count - $3,
                    miss_count = miss_count - $4,
                    last_audit_at = (
                        SELECT max(a.audited_at)
                        FROM recall_canary_audit a
                        WHERE a.probe_id = c.probe_id AND a.user_id = c.user_id
                    )
                WHERE c.probe_id = $1 AND c.user_id = $2
                  AND c.audit_count >= $3 AND c.miss_count >= $4
                RETURNING c.probe_id
                """,
                probe_id,
                user_id,
                checks,
                misses,
            )
            if row is None:
                raise RepairMismatch(f"probe counter update failed for {probe_id}")

        counter = await conn.fetchval(
            """
            UPDATE weft_counters
            SET count = count - $2, updated_at = now()
            WHERE name = $1 AND count >= $2
            RETURNING count
            """,
            COUNTER_CANARY_MISS,
            preview.miss_count,
        )
        if counter is None:
            raise RepairMismatch("global canary.miss correction failed")

    _update_journal(journal_path, "applied_at")
    return preview


def load_journal(path: Path) -> tuple[dict[str, Any], tuple[AuditEvent, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != JOURNAL_VERSION:
        raise RepairMismatch("unsupported journal version")
    if payload.get("kind") != "recall_canary_cross_tenant_repair":
        raise RepairMismatch("journal kind does not match this repair")
    events_payload = payload.get("events")
    if not isinstance(events_payload, list) or not events_payload:
        raise RepairMismatch("journal has no events")
    events = tuple(AuditEvent.from_json(item) for item in events_payload)
    event_ids = [event.id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise RepairMismatch("journal contains duplicate event ids")
    user_ids = {event.user_id for event in events}
    if len(user_ids) != 1:
        raise RepairMismatch("journal contains events from multiple owners")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("user_id") not in user_ids:
        raise RepairMismatch("journal owner does not match its events")
    return payload, events


async def restore_repair(conn: asyncpg.Connection, journal_path: Path) -> int:
    """Restore deleted events and add their exact counter deltas."""
    payload, events = load_journal(journal_path)
    if payload.get("restored_at"):
        raise RepairMismatch("journal has already been restored")

    event_ids = [event.id for event in events]
    user_id = events[0].user_id
    async with conn.transaction():
        existing = await conn.fetchval(
            "SELECT count(*) FROM recall_canary_audit WHERE id = ANY($1::bigint[])",
            event_ids,
        )
        existing_count = int(existing or 0)
        if existing_count == len(events) and not payload.get("applied_at"):
            raise RepairMismatch("journal repair did not commit; all events still exist")
        if existing_count:
            raise RepairMismatch("one or more journal events already exist")

        for event in events:
            await conn.execute(
                """
                INSERT INTO recall_canary_audit
                    (id, probe_id, user_id, audited_at, hit)
                VALUES ($1, $2, $3, $4, $5)
                """,
                event.id,
                event.probe_id,
                event.user_id,
                event.audited_at,
                event.hit,
            )

        latest_by_probe: dict[str, datetime] = {}
        for event in events:
            latest = latest_by_probe.get(event.probe_id)
            if latest is None or event.audited_at > latest:
                latest_by_probe[event.probe_id] = event.audited_at
        for probe_id, (checks, misses) in _event_deltas(events).items():
            row = await conn.fetchrow(
                """
                UPDATE recall_canary
                SET audit_count = audit_count + $3,
                    miss_count = miss_count + $4,
                    last_audit_at = CASE
                        WHEN last_audit_at IS NULL OR last_audit_at < $5 THEN $5
                        ELSE last_audit_at
                    END
                WHERE probe_id = $1 AND user_id = $2
                RETURNING probe_id
                """,
                probe_id,
                user_id,
                checks,
                misses,
                latest_by_probe[probe_id],
            )
            if row is None:
                raise RepairMismatch(f"journal probe missing: {probe_id}")

        await conn.execute(
            """
            INSERT INTO weft_counters (name, count, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (name) DO UPDATE
            SET count = weft_counters.count + EXCLUDED.count,
                updated_at = now()
            """,
            COUNTER_CANARY_MISS,
            sum(not event.hit for event in events),
        )

    _update_journal(journal_path, "restored_at")
    return len(events)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-label", help="Unique token label whose owner is repaired")
    parser.add_argument("--before", help="Exclusive ISO-8601 cutoff for false events")
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--expected-misses", type=int)
    parser.add_argument("--journal", type=Path, help="New restore journal path")
    parser.add_argument("--restore", type=Path, help="Restore an applied journal")
    parser.add_argument("--apply", action="store_true", help="Perform the gated mutation")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    if args.restore:
        if not args.apply:
            raise RepairMismatch("--restore requires --apply")
    elif not args.token_label or not args.before:
        raise RepairMismatch("dry-run/apply requires --token-label and --before")
    elif args.apply and (
        args.expected_events is None
        or args.expected_misses is None
        or args.journal is None
    ):
        raise RepairMismatch(
            "--apply requires --expected-events, --expected-misses, and --journal"
        )

    pool = await create_pool(load_config())
    try:
        async with pool.acquire() as conn:
            if args.restore:
                restored = await restore_repair(conn, args.restore)
                print(json.dumps({"restored_events": restored}, indent=2))
                return 0

            user_id = await resolve_user_id(conn, args.token_label)
            before = parse_timestamp(args.before)
            if not args.apply:
                preview = await inspect_repair(
                    conn, user_id=user_id, before=before
                )
                print(json.dumps(preview.summary(), indent=2, sort_keys=True))
                print("\nDry-run only. No database rows changed.")
                return 0

            preview = await apply_repair(
                conn,
                user_id=user_id,
                before=before,
                expected_events=args.expected_events,
                expected_misses=args.expected_misses,
                journal_path=args.journal,
            )
            print(json.dumps(preview.summary(), indent=2, sort_keys=True))
            print(f"\nApplied. Restore journal: {args.journal}")
            return 0
    finally:
        await pool.close()


def main() -> int:
    args = _build_parser().parse_args()
    try:
        return asyncio.run(_run_cli(args))
    except (RepairMismatch, ValueError, OSError, asyncpg.PostgresError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
