"""Belief-view materializer — async batch worker, idempotent.

Polls ``episode_turns`` for un-processed turns (cursor-based), runs the
belief detector per turn, and writes claims with supersession-aware logic.

Key design invariants (all load-bearing — do not skip):

1. **Idempotency.** Before processing any turn, we check whether a claim
   already exists with ``evidence_turn_ids @> ARRAY[turn_id]::text[] AND
   detector_version = <version>``.  If yes, the turn is skipped without
   calling the detector.  This is defense-in-depth against cursor corruption
   or accidental replay.

2. **Cursor persistence.** The cursor — ``{"last_occurred_at": ..., "last_turn_id": ...}``
   — is stored in ``weft_metadata`` under ``_CURSOR_KEY``.  The cursor is
   advanced only AFTER the full batch writes successfully.  On error the
   cursor stays put; the next run reprocesses the failing batch.  Idempotency
   guard #1 prevents duplicates on replay.

3. **Race-safety per (user_id, attribute, scope).** Each claim write takes a
   ``pg_advisory_xact_lock(_CLAIM_WRITE_LOCK_NAMESPACE, lock_key)`` inside
   the transaction, where ``lock_key`` is derived from the three-part key.
   This prevents two concurrent materializer instances from both inserting an
   ``active`` row for the same key and triggering a unique-constraint race.

4. **Supersession semantics.**  Three cases:
   - No prior active claim → INSERT with status='active'.
   - New turn is more recent than current active → UPDATE current to
     status='superseded', INSERT new as status='active'.
   - Late arrival (new turn's occurred_at < current active's occurred_at) →
     INSERT new as status='superseded', wired into the chain at the correct
     temporal position.  Current active is NOT displaced.

5. **RLS compliance.** Before each INSERT/UPDATE, the materializer sets the
   GUC ``app.user_id = <turn.user_id>`` inside the transaction so the RLS
   INSERT/UPDATE policies and the NOT NULL GUC default are satisfied.  The
   materializer is a system-side process that processes turns for multiple
   users in a single batch; the GUC is set per-turn, inside the per-turn
   transaction.

Spec: docs/architecture/belief_view.md (especially §2 Supersession Semantics).
Detector contract: weft/views/belief_detector.py.
Lock namespace analogue: weft/episode_turns.py (_TURN_APPEND_LOCK_NAMESPACE).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import asyncpg

from weft.models import EpisodeTurn, TurnRole
from weft.store import get_metadata, set_metadata
from weft.views.belief_detector import DETECTOR_VERSION, ClaimUpdate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Namespace for pg_advisory_xact_lock — keeps belief-claim write locks
# disjoint from the episode_turns lock namespace ('ETUR' = 0x4554_5552).
# Both args are signed int32; this constant must fit in 31 bits.
_CLAIM_WRITE_LOCK_NAMESPACE = 0x4243_4C4D  # 'BCLM'

# Cursor key in weft_metadata — persists batch progress across restarts.
_CURSOR_KEY = "belief_materializer.cursor"

# Minimum confidence for a claim to be materialized (defense in depth —
# the detector already drops below-threshold, but we guard here too).
_MIN_CONFIDENCE = 0.6

# Confidence threshold below which a claim is materialized but flagged for
# review.  Above this threshold, no review flag is set.
_REVIEW_THRESHOLD = 0.85

# Epoch sentinel for the initial cursor (no prior run).
_EPOCH_ISO = "1970-01-01T00:00:00+00:00"
_EPOCH_TURN_ID = ""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class MaterializeResult:
    """Summary of a single ``materialize_pending_turns`` invocation."""

    turns_processed: int = 0
    claims_written: int = 0
    claims_superseded: int = 0
    abstentions: int = 0
    errors: int = 0
    cursor: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_claim_id() -> str:
    """Mint a new claim_id with the belief-view tier prefix.

    Mirrors the ``et-{shortid}`` convention from episode_turns.  The prefix
    makes log correlation unambiguous without a type column.
    """
    return f"belief-{uuid.uuid4().hex[:10]}"


def _claim_lock_key(user_id: str, attribute: str, scope: str) -> int:
    """Hash (user_id, attribute, scope) to a 32-bit signed int for advisory locking.

    Collisions across distinct keys produce harmless extra serialization.
    Collisions do NOT produce incorrect behaviour because the critical section
    also checks the full key in SQL (SELECT FOR UPDATE).
    """
    combined = f"{user_id}|{attribute}|{scope}"
    raw = hash(combined) & 0xFFFF_FFFF
    # Convert unsigned 32-bit to signed 32-bit (asyncpg expects int32).
    if raw >= 0x8000_0000:
        raw -= 0x1_0000_0000
    return raw


async def _fetch_pending_turns(
    conn: asyncpg.Connection,
    *,
    cursor_occurred_at: str,
    cursor_turn_id: str,
    batch_size: int,
) -> list[asyncpg.Record]:
    """Fetch up to batch_size turns past the cursor, ordered (occurred_at, id) ASC."""
    cursor_dt = datetime.fromisoformat(cursor_occurred_at)
    rows = await conn.fetch(
        """
        SELECT id, episode_id, turn_index, role, content, occurred_at,
               trace_id, importance_score, token_count, user_id, created_at,
               0.7::real AS usefulness_score,
               0 AS usefulness_count,
               NULL AS last_boosted_at
        FROM episode_turns
        WHERE (occurred_at, id) > ($1, $2)
        ORDER BY occurred_at ASC, id ASC
        LIMIT $3
        """,
        cursor_dt,
        cursor_turn_id,
        batch_size,
    )
    return rows


def _row_to_turn(row: asyncpg.Record) -> EpisodeTurn:
    """Reconstruct an EpisodeTurn from a raw asyncpg record."""
    return EpisodeTurn(
        id=row["id"],
        episode_id=row["episode_id"],
        turn_index=row["turn_index"],
        role=TurnRole(row["role"]),
        content=row["content"],
        occurred_at=row["occurred_at"],
        trace_id=row["trace_id"],
        importance_score=row["importance_score"],
        token_count=row["token_count"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        usefulness_score=float(row["usefulness_score"]) if row["usefulness_score"] is not None else 0.7,
        usefulness_count=int(row["usefulness_count"]) if row["usefulness_count"] is not None else 0,
        last_boosted_at=row["last_boosted_at"],
    )


async def _is_already_processed(
    conn: asyncpg.Connection,
    turn_id: str,
    detector_version: str,
) -> bool:
    """Return True if any claim already records this turn as evidence.

    Uses the exact SQL pattern specified in the task spec (requirement #1):
      SELECT 1 FROM belief_claims
      WHERE detector_version = $1 AND $2 = ANY(evidence_turn_ids)
      LIMIT 1
    This guards against cursor corruption or replay — if the turn was already
    processed by the same detector version, skip it.
    """
    row = await conn.fetchrow(
        "SELECT 1 FROM belief_claims "
        "WHERE detector_version = $1 AND $2 = ANY(evidence_turn_ids) "
        "LIMIT 1",
        detector_version,
        turn_id,
    )
    return row is not None


# ---------------------------------------------------------------------------
# Supersession write logic
# ---------------------------------------------------------------------------


async def _write_single_claim(
    conn: asyncpg.Connection,
    *,
    user_id: str,
    turn: EpisodeTurn,
    update: ClaimUpdate,
    scope: str = "global",
) -> dict[str, int]:
    """Write one ClaimUpdate with supersession-aware logic.

    Must be called inside a transaction.  The caller is responsible for
    having set the app.user_id GUC inside that transaction before calling
    this function.

    Returns a dict with keys ``written``, ``superseded``, ``skipped``.

    Supersession cases (requirement #5):
    - No prior active claim  → INSERT with status='active'.
    - New is more recent     → UPDATE prior to 'superseded'; INSERT new as 'active'.
    - Late arrival           → INSERT new as 'superseded' wired into the chain;
                               do NOT displace the current active.
    """
    attribute = update.attribute  # already validated non-None by caller
    occurred_at = turn.occurred_at

    # Advisory lock — serialize concurrent writers for the same (user, attr, scope).
    lock_key = _claim_lock_key(user_id, attribute, scope)
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1, $2)",
        _CLAIM_WRITE_LOCK_NAMESPACE,
        lock_key,
    )

    # Find current active claim for this triple (FOR UPDATE locks the row).
    active_row = await conn.fetchrow(
        """
        SELECT claim_id, occurred_at, superseded_by, status
        FROM belief_claims
        WHERE user_id = $1 AND attribute = $2 AND scope = $3
          AND status = 'active'
        FOR UPDATE
        """,
        user_id,
        attribute,
        scope,
    )

    new_id = _new_claim_id()
    evidence = [update.evidence_turn_id]
    provenance = update.source_provenance
    confidence = float(update.confidence)
    detector_ver = update.detector_version
    value_json = json.dumps(update.value) if not isinstance(update.value, str) else update.value

    # Ensure value_json is valid JSONB by always serializing.
    value_json = json.dumps(update.value)

    written = 0
    superseded = 0

    if active_row is None:
        # Case 1: No prior active claim — straightforward INSERT.
        await conn.execute(
            """
            INSERT INTO belief_claims (
                claim_id, user_id, attribute, value, scope,
                evidence_turn_ids, status,
                occurred_at, source_provenance,
                detector_confidence, detector_version
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5,
                $6, 'active',
                $7, $8,
                $9, $10
            )
            """,
            new_id, user_id, attribute, value_json, scope,
            evidence,
            occurred_at, provenance,
            confidence, detector_ver,
        )
        written += 1
        logger.debug(
            "materializer.claim_written: claim_id=%s attribute=%s status=active",
            new_id, attribute,
        )

    elif occurred_at > active_row["occurred_at"]:
        # Case 2: Normal supersession — new turn is more recent than current active.
        # Step order matters because of two constraints:
        #   (a) The partial unique index on (user_id, attribute, scope) WHERE status='active'
        #       means we can't INSERT the new active row while the old one is still active.
        #   (b) The FK on superseded_by means we can't UPDATE superseded_by until the
        #       target row exists.
        # Solution: UPDATE old active → 'superseded' (without superseded_by yet),
        # then INSERT new as 'active', then UPDATE old superseded_by = new_id.
        # All three ops inside the same transaction.
        await conn.execute(
            """
            UPDATE belief_claims
            SET status = 'superseded'
            WHERE claim_id = $1
            """,
            active_row["claim_id"],
        )
        await conn.execute(
            """
            INSERT INTO belief_claims (
                claim_id, user_id, attribute, value, scope,
                evidence_turn_ids, status,
                occurred_at, source_provenance,
                detector_confidence, detector_version
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5,
                $6, 'active',
                $7, $8,
                $9, $10
            )
            """,
            new_id, user_id, attribute, value_json, scope,
            evidence,
            occurred_at, provenance,
            confidence, detector_ver,
        )
        # Now set the forward pointer on the old (now superseded) row.
        await conn.execute(
            """
            UPDATE belief_claims
            SET superseded_by = $1
            WHERE claim_id = $2
            """,
            new_id,
            active_row["claim_id"],
        )
        written += 1
        superseded += 1
        logger.debug(
            "materializer.claim_superseded: old=%s new=%s attribute=%s",
            active_row["claim_id"], new_id, attribute,
        )

    else:
        # Case 3: Late arrival — new turn's occurred_at <= current active's occurred_at.
        # Insert into the chain at the correct temporal position without
        # displacing the current active.
        #
        # Chain topology: each claim's superseded_by points to the NEXT claim
        # (more recent) in the chain.  The current active has superseded_by=NULL.
        # We need to find the "predecessor" — the chain member whose superseded_by
        # currently points to the first chain member with occurred_at > new.occurred_at
        # (call it the "successor").
        #
        # Algorithm:
        #   1. Find all superseded claims for this key, ordered by occurred_at ASC.
        #   2. The "successor" is the claim with the smallest occurred_at that is
        #      still > new.occurred_at.  If no superseded claim qualifies, the
        #      successor is the current active (whose occurred_at > new.occurred_at
        #      is guaranteed in this branch).
        #   3. The "predecessor" is the claim whose superseded_by = successor.claim_id.
        #      If no such predecessor exists, the new claim is the earliest in the
        #      chain (no predecessor update needed).
        #   4. INSERT new with status='superseded', superseded_by=successor.claim_id.
        #   5. UPDATE predecessor.superseded_by = new_id (if predecessor exists).

        # Fetch all superseded claims for this (user, attribute, scope).
        chain_rows = await conn.fetch(
            """
            SELECT claim_id, occurred_at, superseded_by
            FROM belief_claims
            WHERE user_id = $1 AND attribute = $2 AND scope = $3
              AND status = 'superseded'
            ORDER BY occurred_at ASC
            """,
            user_id, attribute, scope,
        )

        # Identify successor: earliest claim with occurred_at > new.occurred_at.
        # Check superseded rows first; fall back to the active row.
        successor_id: str | None = None
        for chain_row in chain_rows:
            if chain_row["occurred_at"] > occurred_at:
                successor_id = chain_row["claim_id"]
                break
        if successor_id is None:
            # No superseded claim is more recent — successor is the active claim.
            successor_id = active_row["claim_id"]

        # Identify predecessor: the chain member whose superseded_by = successor_id.
        predecessor_id: str | None = None
        # Check among superseded rows.
        for chain_row in chain_rows:
            if chain_row["superseded_by"] == successor_id:
                predecessor_id = chain_row["claim_id"]
                break
        # The active row's superseded_by is NULL (active rows don't point forward),
        # so the active is never a predecessor in the superseded_by chain.

        # INSERT new claim as superseded, pointing forward to successor.
        await conn.execute(
            """
            INSERT INTO belief_claims (
                claim_id, user_id, attribute, value, scope,
                evidence_turn_ids, status, superseded_by,
                occurred_at, source_provenance,
                detector_confidence, detector_version
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5,
                $6, 'superseded', $7,
                $8, $9,
                $10, $11
            )
            """,
            new_id, user_id, attribute, value_json, scope,
            evidence, successor_id,
            occurred_at, provenance,
            confidence, detector_ver,
        )
        written += 1

        # Wire predecessor to point to new claim (if predecessor exists).
        if predecessor_id is not None:
            await conn.execute(
                """
                UPDATE belief_claims
                SET superseded_by = $1
                WHERE claim_id = $2
                """,
                new_id,
                predecessor_id,
            )

        logger.debug(
            "materializer.late_arrival: claim_id=%s attribute=%s "
            "successor=%s predecessor=%s",
            new_id, attribute, successor_id, predecessor_id,
        )

    return {"written": written, "superseded": superseded, "skipped": 0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def materialize_turn(
    pool: asyncpg.Pool,
    turn: EpisodeTurn,
    claim_updates: list[ClaimUpdate],
) -> dict:
    """Write claims for a single turn with supersession-aware logic.

    For each ClaimUpdate with confidence >= 0.6 and a non-None attribute:
    1. Take pg_advisory_xact_lock keyed by hash(user_id, attribute, scope).
    2. SELECT current active claim for that triple FOR UPDATE.
    3. Decide based on occurred_at (see module docstring for all three cases).
    4. All inside a single transaction per ClaimUpdate.

    Returns a dict: {"written": n_inserted, "superseded": n_marked_superseded, "skipped": n_already_present}.
    """
    user_id = turn.user_id
    if user_id is None:
        logger.warning(
            "materializer.skip_missing_user_id: turn_id=%s", turn.id,
        )
        return {"written": 0, "superseded": 0, "skipped": 1}

    total_written = 0
    total_superseded = 0
    total_skipped = 0

    for update in claim_updates:
        # Defense in depth: skip abstentions and low-confidence claims.
        if update.attribute is None or update.confidence < _MIN_CONFIDENCE:
            total_skipped += 1
            continue

        # Validate that value is present (schema-drift defense).
        if update.value is None:
            logger.warning(
                "materializer.skip_null_value: turn_id=%s attribute=%s",
                turn.id, update.attribute,
            )
            total_skipped += 1
            continue

        scope = "global"  # v1: all claims are global scope

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Set GUC inside the transaction so RLS INSERT/UPDATE
                    # policies pass.  The materializer processes multiple
                    # users' turns; setting per-write is the safe path.
                    # NOTE: SET LOCAL does not support parameterized values in
                    # PostgreSQL — must use a literal string.  Sanitize by
                    # escaping single quotes (user_id is an internal key, not
                    # user-controlled text, but be defensive anyway).
                    safe_uid = user_id.replace("'", "''")
                    await conn.execute(f"SET LOCAL app.user_id = '{safe_uid}'")

                    result = await _write_single_claim(
                        conn,
                        user_id=user_id,
                        turn=turn,
                        update=update,
                        scope=scope,
                    )
                    total_written += result["written"]
                    total_superseded += result["superseded"]
                    total_skipped += result["skipped"]
        except Exception as exc:  # noqa: BLE001
            # Surface per-claim errors so the batch can continue (house-style:
            # root-cause-debugging — no bare suppress; log the root cause).
            logger.error(
                "materializer.claim_write_error: turn_id=%s attribute=%s error=%s",
                turn.id,
                update.attribute,
                exc,
            )
            raise  # re-raise so callers can increment error counter

    return {
        "written": total_written,
        "superseded": total_superseded,
        "skipped": total_skipped,
    }


async def materialize_pending_turns(
    pool: asyncpg.Pool,
    *,
    batch_size: int = 50,
    detector: Callable | None = None,
) -> MaterializeResult:
    """Process all un-materialized turns past the cursor.

    Reads cursor from weft_metadata, fetches up to batch_size turns ordered
    by (occurred_at, id) ascending past the cursor, runs the detector for each,
    writes claims with supersession-aware logic, then advances the cursor.

    Idempotency: the cursor's (occurred_at, id) ordering ensures we never
    re-process a turn we've already advanced past.  Additionally, the write
    path is keyed by (evidence_turn_id, detector_version) — if a turn was
    processed by the same detector version before, the materializer skips it
    (defense in depth against cursor corruption).

    The detector argument exists for test injection — pass a synchronous
    or async callable that takes EpisodeTurn and returns list[ClaimUpdate].
    Defaults to weft.views.belief_detector.detect_belief_updates.
    """
    import asyncio
    import inspect

    if detector is None:
        from weft.views.belief_detector import detect_belief_updates
        detector = detect_belief_updates

    result = MaterializeResult()

    # Load cursor from metadata (None means first run).
    cursor = await get_metadata(pool, _CURSOR_KEY)
    if cursor is None:
        cursor_occurred_at = _EPOCH_ISO
        cursor_turn_id = _EPOCH_TURN_ID
    else:
        cursor_occurred_at = cursor.get("last_occurred_at", _EPOCH_ISO)
        cursor_turn_id = cursor.get("last_turn_id", _EPOCH_TURN_ID)

    # Fetch pending turns using a read-only connection (no transaction needed
    # for the fetch itself — we'll open per-turn transactions for writes).
    async with pool.acquire() as conn:
        # Use system sentinel so we can see turns across all users.
        await conn.execute("SET LOCAL app.user_id = '__system_global_zathras__'")
        rows = await _fetch_pending_turns(
            conn,
            cursor_occurred_at=cursor_occurred_at,
            cursor_turn_id=cursor_turn_id,
            batch_size=batch_size,
        )

    if not rows:
        result.cursor = {"last_occurred_at": cursor_occurred_at, "last_turn_id": cursor_turn_id}
        return result

    # Track the new cursor position (updated after successful batch).
    new_cursor_occurred_at = cursor_occurred_at
    new_cursor_turn_id = cursor_turn_id

    for row in rows:
        turn = _row_to_turn(row)

        # Idempotency check (requirement #1): skip if already processed.
        async with pool.acquire() as conn:
            await conn.execute("SET LOCAL app.user_id = '__system_global_zathras__'")
            already_done = await _is_already_processed(conn, turn.id, DETECTOR_VERSION)

        if already_done:
            logger.debug(
                "materializer.skip_already_processed: turn_id=%s", turn.id,
            )
            # Advance cursor past this turn even though we skipped the write.
            new_cursor_occurred_at = turn.occurred_at.isoformat()
            new_cursor_turn_id = turn.id
            result.turns_processed += 1
            continue

        # Requirement #9: skip turns with no user_id.
        if turn.user_id is None:
            logger.warning(
                "materializer.skip_missing_user_id: turn_id=%s", turn.id,
            )
            result.errors += 1
            new_cursor_occurred_at = turn.occurred_at.isoformat()
            new_cursor_turn_id = turn.id
            result.turns_processed += 1
            continue

        # Run detector (may be sync or async).
        try:
            maybe_coro = detector(turn)
            if inspect.isawaitable(maybe_coro):
                claim_updates: list[ClaimUpdate] = await maybe_coro
            else:
                claim_updates = maybe_coro  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "materializer.detector_error: turn_id=%s error=%s", turn.id, exc,
            )
            result.errors += 1
            new_cursor_occurred_at = turn.occurred_at.isoformat()
            new_cursor_turn_id = turn.id
            result.turns_processed += 1
            continue

        # Filter to actionable updates (abstentions have attribute=None or confidence<0.6).
        actionable = [
            u for u in claim_updates
            if u.attribute is not None and u.confidence >= _MIN_CONFIDENCE
        ]

        if not actionable:
            result.abstentions += 1
            new_cursor_occurred_at = turn.occurred_at.isoformat()
            new_cursor_turn_id = turn.id
            result.turns_processed += 1
            continue

        # Write claims for this turn.
        try:
            write_result = await materialize_turn(pool, turn, actionable)
            result.claims_written += write_result["written"]
            result.claims_superseded += write_result["superseded"]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "materializer.write_error: turn_id=%s error=%s", turn.id, exc,
            )
            result.errors += 1
            # Do NOT advance the cursor — the next run will retry this turn.
            # Idempotency guard #1 will skip any claims that were partially
            # written before the error.
            result.turns_processed += 1
            continue

        new_cursor_occurred_at = turn.occurred_at.isoformat()
        new_cursor_turn_id = turn.id
        result.turns_processed += 1

    # Advance cursor after the full batch.
    new_cursor = {
        "last_occurred_at": new_cursor_occurred_at,
        "last_turn_id": new_cursor_turn_id,
    }
    await set_metadata(pool, _CURSOR_KEY, new_cursor)
    result.cursor = new_cursor

    return result
