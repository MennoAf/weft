#!/usr/bin/env python3
"""
materialize.py — Per-question belief-view materialization for the harness.

The production materializer (``weft.views.materializer.materialize_pending_turns``)
walks ``episode_turns`` with a single global, monotonic cursor stored in
``weft_metadata``. That model is correct for continuous production processing
but wrong for the benchmark: each question is a per-``project_id`` sandbox whose
turns carry the haystack's *original* dates (often years apart and out of order
across questions). A global cursor advanced past question A's 2024 turns would
silently skip question B's 2022 turns. The benchmark also cleans each question
up before the next runs, so there is no "resume" semantics to preserve.

So this module provides a sandbox-scoped entrypoint that ignores the cursor
and materializes exactly one question's turns. It reuses the production
per-turn supersession writer (``materialize_turn``) verbatim — only the
turn-selection and ordering policy differs.

Ordering: turns are processed ``occurred_at ASC`` so supersession resolves the
same way it would in a live session (oldest claim first, newest ends up
``status='active'``). ``materialize_turn``'s late-arrival branch handles any
residual out-of-order arrivals, but ASC keeps the common path on the simple
INSERT / supersede branches.

Author:  Jason Bauman
Python:  >= 3.12
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import asyncpg

from weft.models import EpisodeTurn, TurnRole
from weft.views.belief_detector import ClaimUpdate
from weft.views.materializer import _MIN_CONFIDENCE, materialize_turn

logger = logging.getLogger(__name__)

# A detector is any callable taking an EpisodeTurn and returning (sync or
# async) a list of ClaimUpdate. Injected in tests to avoid live Haiku calls.
Detector = Callable[[EpisodeTurn], "list[ClaimUpdate] | Awaitable[list[ClaimUpdate]]"]


class MaterializationAborted(RuntimeError):
    """Raised when consecutive detector/write failures exceed the abort cap.

    A burst of consecutive failures is the signature of a dead dependency
    (expired API key, rate-limit lockout, DB gone) rather than a bad turn —
    continuing would complete the run with silently under-materialized claims
    and produce a bogus gate score after real Reader spend. The harness treats
    this as fatal to the whole run, not just the current question.
    """


@dataclass(slots=True)
class MaterializeQuestionStats:
    """Summary of one ``materialize_question`` invocation."""

    turns_total: int = 0
    turns_processed: int = 0
    claims_written: int = 0
    claims_superseded: int = 0
    abstentions: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "turns_total": self.turns_total,
            "turns_processed": self.turns_processed,
            "claims_written": self.claims_written,
            "claims_superseded": self.claims_superseded,
            "abstentions": self.abstentions,
            "errors": self.errors,
        }


async def _fetch_question_turns(
    pool: asyncpg.Pool,
    project_id: str,
) -> list[EpisodeTurn]:
    """Load every turn for one question's sandbox, ordered for supersession.

    Joins ``episode_turns`` through ``episodes`` to scope by ``project_id`` —
    that join IS the sandbox boundary; no per-user filter is needed because
    every benchmark row is written under one sentinel identity anyway.

    Identity/RLS note: this SELECT runs on a bare pool connection. It passes
    RLS because (a) the adapter's pool ``setup`` callback (``_bench_setup``)
    issues a session-level ``SET app.user_id`` on every connection, and (b)
    the benchmark role typically owns the tables (RLS is ``ENABLE``, not
    ``FORCE``, so the owner bypasses policies regardless). Do not add a
    ``SET LOCAL`` here — outside an explicit transaction block it is a
    silent no-op and only *looks* like it scopes the query.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT et.id, et.episode_id, et.turn_index, et.role, et.content,
                   et.occurred_at, et.trace_id, et.importance_score,
                   et.token_count, et.user_id, et.created_at
            FROM episode_turns et
            JOIN episodes e ON et.episode_id = e.id
            WHERE e.project_id = $1
            ORDER BY et.occurred_at ASC, et.id ASC
            """,
            project_id,
        )
    return [
        EpisodeTurn(
            id=r["id"],
            episode_id=r["episode_id"],
            turn_index=r["turn_index"],
            role=TurnRole(r["role"]),
            content=r["content"],
            occurred_at=r["occurred_at"],
            trace_id=r["trace_id"],
            importance_score=r["importance_score"],
            token_count=r["token_count"],
            user_id=r["user_id"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


async def materialize_question(
    pool: asyncpg.Pool,
    project_id: str,
    *,
    detector: Detector | None = None,
    min_confidence: float = _MIN_CONFIDENCE,
    max_consecutive_errors: int = 5,
) -> MaterializeQuestionStats:
    """Materialize belief claims for one benchmark question's turns.

    Runs the detector over every turn in the ``project_id`` sandbox and writes
    actionable claims via the production supersession writer. Cursor-free and
    idempotent across re-runs (``materialize_turn`` skips below-threshold and
    null-value updates; the detector-version guard in the underlying writer
    de-dupes replays).

    Identity: claims inherit ``user_id`` from each turn row inside
    ``materialize_turn`` — there is no separate identity input here.

    Args:
        pool: asyncpg pool against the Weft Postgres instance.
        project_id: The question's sandbox (``lme_<qid>``).
        detector: Callable ``(EpisodeTurn) -> list[ClaimUpdate]`` (sync or
            async). Defaults to the real Haiku detector — inject a fake in
            tests to avoid live API calls.
        min_confidence: Floor below which updates are dropped (defense in
            depth — the detector already drops below 0.6).
        max_consecutive_errors: Abort cap. Isolated turn failures are logged
            and skipped, but this many failures *in a row* means the detector
            or DB is down — raise ``MaterializationAborted`` instead of
            completing a silently under-materialized run.

    Returns:
        MaterializeQuestionStats with per-question counts.

    Raises:
        MaterializationAborted: after ``max_consecutive_errors`` consecutive
            detector/write failures.
    """
    if detector is None:
        from weft.views.belief_detector import detect_belief_updates

        detector = detect_belief_updates

    stats = MaterializeQuestionStats()
    turns = await _fetch_question_turns(pool, project_id)
    stats.turns_total = len(turns)
    consecutive_errors = 0

    def _record_error(kind: str, turn_id: str, exc: Exception) -> None:
        nonlocal consecutive_errors
        logger.warning(
            "materialize_question.%s: turn_id=%s error=%s", kind, turn_id, exc,
        )
        stats.errors += 1
        consecutive_errors += 1
        if consecutive_errors >= max_consecutive_errors:
            raise MaterializationAborted(
                f"{consecutive_errors} consecutive materialization failures in "
                f"project_id={project_id} (last: {kind} on turn {turn_id}: "
                f"{exc}) — detector/DB looks down, aborting the run"
            ) from exc

    for turn in turns:
        try:
            maybe = detector(turn)
            claim_updates: list[ClaimUpdate] = (
                await maybe if inspect.isawaitable(maybe) else maybe
            )
        except MaterializationAborted:
            raise
        except Exception as exc:  # noqa: BLE001
            _record_error("detector_error", turn.id, exc)
            continue

        actionable = [
            u
            for u in claim_updates
            if u.attribute is not None and u.confidence >= min_confidence
        ]
        if not actionable:
            stats.abstentions += 1
            consecutive_errors = 0
            continue

        try:
            result = await materialize_turn(pool, turn, actionable)
        except Exception as exc:  # noqa: BLE001
            _record_error("write_error", turn.id, exc)
            continue

        stats.turns_processed += 1
        stats.claims_written += result["written"]
        stats.claims_superseded += result["superseded"]
        consecutive_errors = 0

    logger.info(
        "materialize_question: project_id=%s turns=%d/%d written=%d "
        "superseded=%d abstentions=%d errors=%d",
        project_id,
        stats.turns_processed,
        stats.turns_total,
        stats.claims_written,
        stats.claims_superseded,
        stats.abstentions,
        stats.errors,
    )
    return stats
