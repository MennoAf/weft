"""Aggregation replay executor — drain replay_queue → detect → write claims.

This is the consumer half of the recall-gap replay loop (E2.L7). The re-ask
feedback path (E1.L3, enqueue_replay_on_miss) enqueues a ``replay_queue`` row
whenever a recall miss is corrected; this executor drains those rows, hydrates
each row's turn-set, runs the multi-turn :func:`detect_aggregate_claims`
detector (E2.L6), and writes the resulting enumeration claims back into
``belief_claims`` through the materializer's supersession-aware write path —
the claims the single-turn detector could never see.

Design decisions (see loom-ebef8ec1):

* **Bypasses the per-turn idempotency skip.** The materializer's
  ``materialize_pending_turns`` skips turns that already have a claim for the
  same ``detector_version``. We deliberately call ``materialize_turn`` directly
  — it carries no such skip — because re-processing flagged turns is the entire
  point. Re-processing is instead bounded by the queue-row terminal status.

* **PROOF prefix.** Claims are re-stamped with detector_version
  ``replay-aggregate-detector-v1.0`` (REPLAY_DETECTOR_VERSION_PREFIX +
  AGGREGATE_DETECTOR_VERSION). The 'replay-' prefix is load-bearing: the
  ``replay_claims_30d`` health metric counts ``belief_claims WHERE
  detector_version LIKE 'replay-%'``. Any other prefix leaves that PROOF metric
  pinned at 0 even while the loop produces claims (see weft-99cac4e5). The
  '-aggregate-detector-v1.0' suffix keeps the claim attributable.

* **Terminal status, always.** Every claimed row is driven to a terminal
  status — 'done' on success, 'failed' on unrecoverable error — and the
  ``replay.executor.failed`` counter is bumped on failure. A row left 'pending'
  forever would pin its turns against the L4 retention guard permanently (an
  unbounded retention leak). 'failed' rows are not retained, so a poison row
  forfeits its turns rather than leaking; the re-ask loop re-enqueues a fresh
  row if recall still misses.

* **RLS context.** Reads (pending rows + turn hydration) run under the system
  sentinel so the executor sees all users' rows. The ``replay_queue`` UPDATE
  policy does NOT admit the sentinel, so status writes happen per-row under the
  row's own ``user_id`` (mirrored by the materializer's per-write GUC).

Spec: Loom task loom-ebef8ec1 (E2.L7 in the recall-gap epic loom-dcfaf656).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

import asyncpg

from weft.auth import current_user_id
from weft.counters import (
    COUNTER_REPLAY_EXECUTOR_FAILED,
    COUNTER_REPLAY_STALE_REAPED,
    COUNTER_REPLAY_TERMINAL_STATUS_FAILED,
    increment_counter,
)
from weft.db.connection import acquire, get_db
from weft.episode_turns import get_turns_by_ids
from weft.replay import (
    REPLAY_DETECTOR_VERSION_PREFIX,
    REPLAY_QUEUE_STALENESS_DAYS,
    REPLAY_QUEUE_STATUS_DONE,
    REPLAY_QUEUE_STATUS_FAILED,
    REPLAY_QUEUE_STATUS_PENDING,
)
from weft.schema import SYSTEM_GLOBAL_USER_ID
from weft.views import belief_detector
from weft.views.aggregate_detector import (
    AGGREGATE_DETECTOR_VERSION,
    AGGREGATE_OUTPUT_SCHEMA,
    REVIEW_CONFIDENCE,
    _AGGREGATE_SYSTEM_PROMPT,
    _MAX_TOKENS,
    AggregateRequest,
    _parse_aggregate_claims,
    build_aggregate_request,
    detect_aggregate_claims,
)
from weft.views.belief_detector import ClaimUpdate
from weft.views.materializer import _validate_user_id_for_write, materialize_turn

logger = logging.getLogger(__name__)

# Batch-API poll cadence for the consolidation replay pass. The pass submits one
# batch and waits for it to finish so a single consolidate() run drives queued
# rows to a terminal status. Most batches finish in minutes; on timeout the rows
# are left pending (NOT failed) and re-enqueued work is picked up next pass.
_BATCH_POLL_INTERVAL_S = 2.0
_BATCH_POLL_TIMEOUT_S = 300.0
_BATCH_ENDED_STATUS = "ended"

# Escalation tier (E3.L9): when the cheap Haiku pass abstains or scores every
# claim below REVIEW_CONFIDENCE, the inline executor retries the turn-set exactly
# once at Sonnet 4.6. Opus is deliberately never used (Pinch routing constraint
# weft-82b2860a) — the no-opus invariant is asserted by a test over this module.
SONNET_ESCALATION_MODEL = "claude-sonnet-4-6"

# detector_version stamped on Sonnet-escalated replay claims. Keeps the 'replay-'
# PROOF prefix (so replay_claims_30d still counts it) and the aggregate-detector
# suffix, with a tier marker so the escalated claim is attributable to Sonnet.
REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION = (
    REPLAY_DETECTOR_VERSION_PREFIX + "aggregate-detector-sonnet46-v1.0"
)

# detector_version stamped on replay-origin aggregate claims. Begins with the
# 'replay-' PROOF prefix (so replay_claims_30d counts it) and keeps the
# aggregate-detector suffix (so the claim stays attributable).
REPLAY_AGGREGATE_DETECTOR_VERSION = (
    REPLAY_DETECTOR_VERSION_PREFIX + AGGREGATE_DETECTOR_VERSION
)


@dataclass
class ReplayExecutorResult:
    """Summary of one drain pass."""

    rows_processed: int = 0
    rows_done: int = 0
    rows_failed: int = 0
    rows_reaped: int = 0
    claims_written: int = 0
    claims_superseded: int = 0


async def _fetch_pending_rows(pool: asyncpg.Pool, batch_size: int) -> list[asyncpg.Record]:
    """Read up to *batch_size* pending replay_queue rows (oldest first).

    Must be called inside a system-sentinel context so the SELECT policy admits
    rows across all users.
    """
    return await get_db(pool).fetch(
        """
        SELECT id, episode_id, turn_ids, user_id
        FROM replay_queue
        WHERE status = $1
        ORDER BY created_at ASC
        LIMIT $2
        """,
        REPLAY_QUEUE_STATUS_PENDING,
        batch_size,
    )


async def _set_status(
    pool: asyncpg.Pool, replay_id: str, user_id: str, status: str
) -> None:
    """Set a replay_queue row's status under the row's own user_id.

    The replay_queue UPDATE policy gates on ``user_id = app.user_id`` (the
    sentinel is NOT admitted for writes), so the GUC must carry the row's user.
    """
    _validate_user_id_for_write(user_id)  # rejects the sentinel + unsafe ids
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{user_id}'")
            await conn.execute(
                "UPDATE replay_queue SET status = $1 WHERE id = $2",
                status,
                replay_id,
            )


async def _set_terminal_status_with_retry(
    pool: asyncpg.Pool,
    replay_id: str,
    user_id: str,
    status: str,
) -> bool:
    """Persist a terminal status, retrying once before leaving the row pending.

    A pending row is safer than pretending a terminal write succeeded, because
    the stale-reaper will eventually drain it. The retry and durable counter make
    this rare lifecycle seam observable instead of silently pinning work.
    """
    for attempt in range(2):
        try:
            await _set_status(pool, replay_id, user_id, status)
            return True
        except Exception:
            if attempt == 0:
                logger.warning(
                    "replay_executor.terminal_status_retry: id=%s status=%s",
                    replay_id,
                    status,
                    exc_info=True,
                )
                continue
            logger.exception(
                "replay_executor.terminal_status_failed: id=%s status=%s (row stays pending)",
                replay_id,
                status,
            )
            await increment_counter(pool, COUNTER_REPLAY_TERMINAL_STATUS_FAILED)
            return False
    return False


async def reap_stale_pending_replays(
    pool: asyncpg.Pool,
    *,
    staleness_days: int = REPLAY_QUEUE_STALENESS_DAYS,
    batch_size: int = 500,
) -> int:
    """Drive orphaned stale 'pending' replay_queue rows to terminal 'failed'.

    Complements the L4 retention age-bound (weft-99cac4e5): retention STOPS
    HONORING a 'pending' row older than ``REPLAY_QUEUE_STALENESS_DAYS`` (its
    turns shed), but the row itself lingers 'pending' forever — inflating
    ``count_stale_pending_replays`` and never draining the queue. A pending row
    this old is presumed orphaned: real replays reach a terminal status in
    minutes/hours, so 14d+ means the executor never ran, the row's user_id fails
    the UPDATE policy, or it is a poison row. This sweep transitions them to
    'failed' (the same terminal the executor uses for unrecoverable rows), so the
    queue state matches what retention already assumes.

    Best-effort per row: reads under the system sentinel (to see every user's
    rows); writes per-row under the row's own user_id (the replay_queue UPDATE
    policy does not admit the sentinel — see :func:`_set_status`). A row whose
    write throws (unwritable user_id, poison row) is logged and SKIPPED — never
    aborting the sweep — and is retried next pass. Returns the count reaped and
    bumps :data:`COUNTER_REPLAY_STALE_REAPED` per reap. Idempotent: already-
    terminal rows are never selected.
    """
    # Phase 1 — read stale pending rows under the sentinel (across all users).
    token = current_user_id.set(SYSTEM_GLOBAL_USER_ID)
    try:
        async with acquire(pool):
            rows = await get_db(pool).fetch(
                """
                SELECT id, user_id
                FROM replay_queue
                WHERE status = $1
                  AND created_at <= now() - ($2 || ' days')::interval
                ORDER BY created_at ASC
                LIMIT $3
                """,
                REPLAY_QUEUE_STATUS_PENDING,
                str(staleness_days),
                batch_size,
            )
    finally:
        current_user_id.reset(token)

    # Phase 2 — per-row terminal write, isolated so one bad row can't abort it.
    reaped = 0
    for row in rows:
        replay_id = row["id"]
        user_id = row["user_id"]
        if await _set_terminal_status_with_retry(
            pool, replay_id, user_id, REPLAY_QUEUE_STATUS_FAILED
        ):
            await increment_counter(pool, COUNTER_REPLAY_STALE_REAPED)
            reaped += 1
    if reaped:
        logger.info("replay_executor.reaped_stale_pending: reaped=%d", reaped)
    return reaped


def _needs_escalation(claims: list[ClaimUpdate]) -> bool:
    """True when the cheap pass produced nothing usable above the review bar.

    Escalate when the detector abstained (no claims) or when every surviving
    claim scored below :data:`REVIEW_CONFIDENCE` (the 0.6–0.85 "needs review"
    band). A single claim at/above the bar is a confident hit — no escalation.
    """
    if not claims:
        return True
    return max(c.confidence for c in claims) < REVIEW_CONFIDENCE


async def _detect_with_escalation(
    turns: list,
) -> tuple[list[ClaimUpdate], str]:
    """Run the Haiku aggregate pass, escalating once to Sonnet 4.6 if weak.

    Returns ``(claims, detector_version)`` where the version marks which tier
    produced the adopted claims (so the persisted claim is attributable). The
    Sonnet result is adopted only when it yields claims; if Sonnet also comes up
    empty, the Haiku result stands (and the version stays Haiku). Opus is never
    invoked.
    """
    claims = await detect_aggregate_claims(turns)  # cheap Haiku pass
    if not _needs_escalation(claims):
        return claims, REPLAY_AGGREGATE_DETECTOR_VERSION

    logger.info(
        "replay_executor.escalating: turns=%d haiku_claims=%d -> %s",
        len(turns),
        len(claims),
        SONNET_ESCALATION_MODEL,
    )
    escalated = await detect_aggregate_claims(turns, model=SONNET_ESCALATION_MODEL)
    if escalated:
        return escalated, REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION
    # Sonnet abstained too — keep whatever Haiku had (possibly empty).
    return claims, REPLAY_AGGREGATE_DETECTOR_VERSION


async def run_replay_executor(
    pool: asyncpg.Pool, *, batch_size: int = 50
) -> ReplayExecutorResult:
    """Drain pending replay_queue rows once: detect aggregates, write, mark terminal.

    Single-pass (not a loop). A scheduler can call this on an interval; per-row
    failures are isolated and do not abort the pass. Returns a
    :class:`ReplayExecutorResult` summarizing the pass.
    """
    result = ReplayExecutorResult()

    # Phase 0 — reap orphaned stale-pending rows to terminal 'failed' so the
    # queue drains (and the executor doesn't re-attempt poison rows every pass).
    result.rows_reaped = await reap_stale_pending_replays(pool)

    # Phase 1 — read pending rows + hydrate their turns under the system
    # sentinel (one short-lived read transaction; no LLM/write work held here).
    token = current_user_id.set(SYSTEM_GLOBAL_USER_ID)
    try:
        async with acquire(pool):
            rows = await _fetch_pending_rows(pool, batch_size)
            work: list[tuple[str, str, list]] = []
            for row in rows:
                turns = await get_turns_by_ids(pool, list(row["turn_ids"]))
                work.append((row["id"], row["user_id"], turns))
    finally:
        current_user_id.reset(token)

    # Phase 2 — detect (+escalate) + write + terminal status, per row, isolated.
    for replay_id, user_id, turns in work:
        result.rows_processed += 1
        try:
            # Cheap Haiku pass; escalate once to Sonnet 4.6 on abstention or
            # all-below-review-threshold (E3.L9). The returned version marks the
            # tier that produced the adopted claims.
            claims, detector_version = await _detect_with_escalation(turns)
            written = await _write_replay_claims(
                pool, turns, claims, detector_version=detector_version
            )
            result.claims_written += written["written"]
            result.claims_superseded += written["superseded"]

            if await _set_terminal_status_with_retry(
                pool, replay_id, user_id, REPLAY_QUEUE_STATUS_DONE
            ):
                result.rows_done += 1
        except Exception as exc:  # noqa: BLE001 — isolate per-row; never abort the pass
            logger.error(
                "replay_executor.row_failed: id=%s user_id=%s error=%s",
                replay_id,
                user_id,
                exc,
            )
            await increment_counter(pool, COUNTER_REPLAY_EXECUTOR_FAILED)
            # Drive to a terminal status so the row cannot pin its turns forever.
            if await _set_terminal_status_with_retry(
                pool, replay_id, user_id, REPLAY_QUEUE_STATUS_FAILED
            ):
                result.rows_failed += 1

    if result.rows_processed:
        logger.info(
            "replay_executor.pass_complete: processed=%d done=%d failed=%d "
            "claims_written=%d",
            result.rows_processed,
            result.rows_done,
            result.rows_failed,
            result.claims_written,
        )
    return result


async def _write_replay_claims(
    pool: asyncpg.Pool,
    turns: list,
    claims: list,
    *,
    detector_version: str = REPLAY_AGGREGATE_DETECTOR_VERSION,
) -> dict:
    """Re-stamp aggregate claims with the replay- PROOF prefix and persist them.

    ``detector_version`` defaults to the Haiku-tier replay version; the inline
    executor passes the Sonnet-tier version when a turn-set was escalated, so the
    persisted claim records which model produced it. Both keep the 'replay-'
    prefix the replay_claims_30d PROOF metric matches.

    Anchors the write to the latest contributing turn (the aggregate "becomes
    true" as of the most recent evidence; the anchor supplies occurred_at +
    user_id). Returns the materializer's {"written", "superseded"} counts, or
    zeros when there is nothing to write.
    """
    replay_claims = [
        replace(c, detector_version=detector_version) for c in claims
    ]
    if not (replay_claims and turns):
        return {"written": 0, "superseded": 0}
    anchor = max(turns, key=lambda t: t.occurred_at)
    return await materialize_turn(pool, anchor, replay_claims)


async def _read_pending_work(
    pool: asyncpg.Pool, batch_size: int
) -> list[tuple[str, str, list]]:
    """Read pending rows + hydrate their turns under the system sentinel.

    Shared Phase-1 read for both the inline and Batch-API drains. Returns a list
    of (replay_id, user_id, turns).
    """
    token = current_user_id.set(SYSTEM_GLOBAL_USER_ID)
    try:
        async with acquire(pool):
            rows = await _fetch_pending_rows(pool, batch_size)
            work: list[tuple[str, str, list]] = []
            for row in rows:
                turns = await get_turns_by_ids(pool, list(row["turn_ids"]))
                work.append((row["id"], row["user_id"], turns))
    finally:
        current_user_id.reset(token)
    return work


def _build_batch_request(
    replay_id: str, request: AggregateRequest, model: str
) -> dict:
    """One ``messages.batches`` request envelope for a turn-set at a given model.

    The cheap Haiku pass and the Sonnet escalation pass share every param except
    ``model`` — structured outputs via output_config.format, the aggregate system
    prompt, and the per-row user message.
    """
    return {
        "custom_id": replay_id,
        "params": {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "system": _AGGREGATE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": request.user_message}],
            "output_config": {
                "format": {"type": "json_schema", "schema": AGGREGATE_OUTPUT_SCHEMA}
            },
        },
    }


async def _submit_and_poll_batch(
    client,
    batch_requests: list[dict],
    *,
    poll_interval_s: float,
    poll_timeout_s: float,
):
    """Submit one batch and poll it to completion.

    Returns the async results iterator on success, or ``None`` if submission
    failed or polling timed out — the caller leaves the in-flight rows pending
    (never silently failed) for a later pass to re-drain.
    """
    try:
        batch = await client.messages.batches.create(requests=batch_requests)
    except Exception:
        # Submission failed for the whole batch: no per-row attribution to fail on.
        logger.exception("replay_executor.batch_submit_error: rows=%d", len(batch_requests))
        return None

    status = batch.processing_status
    waited = 0.0
    while status != _BATCH_ENDED_STATUS:
        if waited >= poll_timeout_s:
            logger.warning(
                "replay_executor.batch_poll_timeout: id=%s status=%s rows=%d "
                "(left pending for next pass)",
                batch.id,
                status,
                len(batch_requests),
            )
            return None
        await asyncio.sleep(poll_interval_s)
        waited += poll_interval_s
        refreshed = await client.messages.batches.retrieve(batch.id)
        status = refreshed.processing_status

    return await client.messages.batches.results(batch.id)


def _parse_batch_item(item, request: AggregateRequest) -> list[ClaimUpdate]:
    """Parse one batch result item into aggregate claims.

    Raises when the request did not succeed — the caller treats that as a
    per-row failure (drives the row to terminal 'failed').
    """
    if item.result.type != "succeeded":
        raise RuntimeError(f"batch result not succeeded: {item.result.type}")
    message = item.result.message
    raw_json = next(
        (b.text for b in message.content if getattr(b, "type", None) == "text"),
        "",
    )
    return _parse_aggregate_claims(
        raw_json, request.valid_turn_ids, request.role_by_turn_id
    )


async def run_replay_executor_batch(
    pool: asyncpg.Pool,
    *,
    batch_size: int = 50,
    poll_interval_s: float = _BATCH_POLL_INTERVAL_S,
    poll_timeout_s: float = _BATCH_POLL_TIMEOUT_S,
) -> ReplayExecutorResult:
    """Drain pending replay_queue rows via the Anthropic Batch API.

    Same contract as :func:`run_replay_executor` (detect aggregates → write with
    the replay- PROOF prefix → drive each row terminal) but the Haiku detector
    calls go through ``messages.batches`` — one request per queued turn-set,
    submitted as a single batch (50% off, structured outputs via
    ``output_config.format``). This is the path consolidate() wires in as its
    4th sub-pass (E2.L8); the inline executor stays for direct/scheduler use.

    Escalation (E3.L9 in the batch path): after the Haiku batch lands, turn-sets
    that abstained or scored every claim below REVIEW_CONFIDENCE (per
    :func:`_needs_escalation`) are re-submitted as a SECOND batch at
    :data:`SONNET_ESCALATION_MODEL`. This is what makes the Sonnet tier live in
    the production consolidate() path — the inline :func:`run_replay_executor`
    (which carries the same escalation) has no production caller. Opus is never
    used. A Sonnet result is adopted only when it yields claims; if Sonnet also
    abstains, the Haiku result stands (Haiku tier), mirroring the inline path.

    The pass submits each batch and polls it to completion so a single call moves
    queued rows to a terminal status. On poll timeout the in-flight rows are left
    ``pending`` (never silently failed) — a later pass re-drains them.
    """
    result = ReplayExecutorResult()

    # Phase 0 — reap orphaned stale-pending rows to terminal 'failed' so the
    # queue drains (and the batch doesn't re-attempt poison rows every pass).
    result.rows_reaped = await reap_stale_pending_replays(pool)

    # Phase 1 — read pending rows + hydrate their turns (system sentinel).
    work = await _read_pending_work(pool, batch_size)
    if not work:
        return result

    # Phase 2 — gate + build one batch request per row. Rows that gate to nothing
    # (fewer than 2 belief-bearing turns) carry no aggregate and are consumed now.
    turns_by_id: dict[str, tuple[str, list]] = {}
    request_by_id: dict[str, AggregateRequest] = {}
    batch_requests: list[dict] = []
    for replay_id, user_id, turns in work:
        turns_by_id[replay_id] = (user_id, turns)
        request = build_aggregate_request(turns)
        if request is None:
            result.rows_processed += 1
            if await _set_terminal_status_with_retry(
                pool, replay_id, user_id, REPLAY_QUEUE_STATUS_DONE
            ):
                result.rows_done += 1
            continue
        request_by_id[replay_id] = request
        batch_requests.append(
            _build_batch_request(replay_id, request, belief_detector._MODEL)
        )

    if not batch_requests:
        return result

    client = belief_detector._get_client()

    # Per-row terminal helpers, closing over `result` so terminal accounting
    # (rows_processed / done / failed / claims) lives in one place. Reused by the
    # Haiku phase and the Sonnet escalation phase.
    async def _finish(replay_id, user_id, turns, claims, detector_version):
        result.rows_processed += 1
        written = await _write_replay_claims(
            pool, turns, claims, detector_version=detector_version
        )
        result.claims_written += written["written"]
        result.claims_superseded += written["superseded"]
        if await _set_terminal_status_with_retry(
            pool, replay_id, user_id, REPLAY_QUEUE_STATUS_DONE
        ):
            result.rows_done += 1

    async def _fail(replay_id, user_id, exc):
        result.rows_processed += 1
        logger.error(
            "replay_executor.batch_row_failed: id=%s user_id=%s error=%s",
            replay_id,
            user_id,
            exc,
        )
        await increment_counter(pool, COUNTER_REPLAY_EXECUTOR_FAILED)
        if await _set_terminal_status_with_retry(
            pool, replay_id, user_id, REPLAY_QUEUE_STATUS_FAILED
        ):
            result.rows_failed += 1

    # Phase 3 — submit + poll the cheap Haiku batch. None => left pending.
    results_iter = await _submit_and_poll_batch(
        client, batch_requests, poll_interval_s=poll_interval_s, poll_timeout_s=poll_timeout_s
    )
    if results_iter is None:
        return result

    # Phase 4 — parse Haiku results. Confident rows are written + closed now;
    # weak rows (abstain / all-below-review, per _needs_escalation) are deferred
    # to a Sonnet escalation batch (Phase 5), keeping their Haiku claims as the
    # fallback if Sonnet also abstains.
    escalate_ids: list[str] = []
    haiku_fallback: dict[str, list] = {}
    async for item in results_iter:
        replay_id = item.custom_id
        entry = turns_by_id.get(replay_id)
        request = request_by_id.get(replay_id)
        if entry is None or request is None:
            logger.warning("replay_executor.batch_unknown_custom_id: id=%s", replay_id)
            continue
        user_id, turns = entry
        try:
            claims = _parse_batch_item(item, request)
        except Exception as exc:  # noqa: BLE001 — isolate per-row; never abort the pass
            await _fail(replay_id, user_id, exc)
            continue
        if _needs_escalation(claims):
            escalate_ids.append(replay_id)
            haiku_fallback[replay_id] = claims
        else:
            await _finish(
                replay_id, user_id, turns, claims, REPLAY_AGGREGATE_DETECTOR_VERSION
            )

    # Phase 5 — Sonnet escalation batch for the weak turn-sets (makes the Sonnet
    # tier live in the production path). Opus is never used. On whole-batch
    # submit/timeout failure the escalated rows stay pending (re-drained next
    # pass) — the already-closed confident Haiku rows are unaffected.
    if escalate_ids:
        sonnet_requests = [
            _build_batch_request(rid, request_by_id[rid], SONNET_ESCALATION_MODEL)
            for rid in escalate_ids
        ]
        sonnet_iter = await _submit_and_poll_batch(
            client, sonnet_requests, poll_interval_s=poll_interval_s, poll_timeout_s=poll_timeout_s
        )
        if sonnet_iter is None:
            return result
        async for item in sonnet_iter:
            replay_id = item.custom_id
            entry = turns_by_id.get(replay_id)
            request = request_by_id.get(replay_id)
            if entry is None or request is None:
                logger.warning("replay_executor.batch_unknown_custom_id: id=%s", replay_id)
                continue
            user_id, turns = entry
            try:
                escalated = _parse_batch_item(item, request)
            except Exception as exc:  # noqa: BLE001 — isolate per-row; never abort the pass
                await _fail(replay_id, user_id, exc)
                continue
            if escalated:
                await _finish(
                    replay_id,
                    user_id,
                    turns,
                    escalated,
                    REPLAY_AGGREGATE_SONNET_DETECTOR_VERSION,
                )
            else:
                # Sonnet abstained too — keep the Haiku result (possibly empty),
                # stamped Haiku tier. Mirrors _detect_with_escalation's fallback.
                await _finish(
                    replay_id,
                    user_id,
                    turns,
                    haiku_fallback[replay_id],
                    REPLAY_AGGREGATE_DETECTOR_VERSION,
                )

    if result.rows_processed:
        logger.info(
            "replay_executor.batch_pass_complete: processed=%d done=%d failed=%d "
            "claims_written=%d",
            result.rows_processed,
            result.rows_done,
            result.rows_failed,
            result.claims_written,
        )
    return result
