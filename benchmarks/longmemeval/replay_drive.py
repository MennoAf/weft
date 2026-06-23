#!/usr/bin/env python3
"""
replay_drive.py — drive the recall-gap replay loop inline for one question.

WHY THIS MODULE EXISTS (the inert-flag scar, do not repeat it):

    Production enqueues replay rows from the background scheduler's re-ask pass
    (``store.apply_reask_feedback`` → ``enqueue_replay_on_miss``) and drains them
    in ``consolidate()`` (the Batch-API executor, E2.L8). NEITHER runs during a
    benchmark: ``adapter.run_benchmark`` is a synchronous, question-at-a-time
    pipeline with no scheduler and no consolidation pass. So a naive
    before/after over the replay substrate measures *nothing* — the code exists,
    is correct, and is never executed on the benchmark control flow. Any score
    delta would be a confound, not a replay effect (cf. the WEFT_HIERARCHICAL
    inert-flag falsification).

    This module wires the two halves explicitly, per question:

        enqueue_replay_on_miss(question)  →  run_replay_executor[_batch]()

    so the multi-turn *aggregate* (enumeration) claims that the single-turn
    materializer can never produce actually land in ``belief_claims`` BEFORE the
    second recall reads them. The ``--tier replay`` path in adapter.py runs this
    after per-turn materialization, isolating the aggregate detector's
    contribution against the ``--tier belief-view`` baseline.

EXECUTOR CHOICE (inline vs batch):

    Both entrypoints share the detector (``detect_aggregate_claims`` +
    Sonnet escalation) and the write path (``_write_replay_claims`` →
    ``materialize_turn``), so they produce IDENTICAL claims. Only LLM *dispatch*
    differs:

      * ``inline`` (default) — synchronous direct Haiku/Sonnet calls. Deterministic
        per-question timing, no batch polling. Best for iteration.
      * ``batch`` — the Anthropic Batch API, exactly what production
        ``consolidate()`` drives (50% cheaper) but polls up to ~300s per question.
        Use for a production-exact validation pass.

    The claim content the second recall sees is the same either way; the flag
    exists for fidelity-vs-speed, not for a different result.

Identity: all writes land under the turn-set's ``user_id`` (the benchmark
sentinel). The executor reads pending rows under the system sentinel (so it sees
the benchmark's rows) and writes each claim / status under the row's own user_id
via ``materialize_turn`` / ``_set_status`` — composes cleanly with the adapter
pool's ``setup`` GUC. See the trace in the harness PR for the full chain.

Author:  Jason Bauman
Python:  >= 3.12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import asyncpg

from weft.replay import enqueue_replay_on_miss
from weft.replay_executor import (
    ReplayExecutorResult,
    run_replay_executor,
    run_replay_executor_batch,
)

logger = logging.getLogger(__name__)

# Which executor dispatch to use. 'inline' = synchronous direct LLM calls;
# 'batch' = Anthropic Batch API (production consolidate() path). Identical claims.
ReplayExecutorKind = Literal["inline", "batch"]


@dataclass(slots=True)
class ReplayDriveStats:
    """Summary of one ``drive_replay`` invocation (one benchmark question)."""

    rows_enqueued: int = 0
    rows_processed: int = 0
    rows_done: int = 0
    rows_failed: int = 0
    claims_written: int = 0
    claims_superseded: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "rows_enqueued": self.rows_enqueued,
            "rows_processed": self.rows_processed,
            "rows_done": self.rows_done,
            "rows_failed": self.rows_failed,
            "claims_written": self.claims_written,
            "claims_superseded": self.claims_superseded,
        }


async def drive_replay(
    pool: asyncpg.Pool,
    *,
    question: str,
    user_id: str,
    executor: ReplayExecutorKind = "inline",
    reason: str = "benchmark-replay",
) -> ReplayDriveStats:
    """Enqueue + drain the replay loop for one question's just-materialized turns.

    Treats the benchmark question as the "missed query" (the same input the
    production re-ask path feeds ``enqueue_replay_on_miss``): resolves the
    implicated episode turns, enqueues one pending ``replay_queue`` row per
    distinct episode, then drains the queue through the chosen executor so the
    aggregate detector writes its ``replay-``-stamped enumeration claims into
    ``belief_claims``.

    Must be called AFTER ``materialize_question`` (so the belief-claim resolution
    path has per-turn claims to match) and BEFORE the recall that should see the
    aggregate claims.

    Args:
        pool: asyncpg pool whose connections carry the benchmark ``app.user_id``
            GUC (the adapter's ``setup`` callback). The enqueue INSERT runs under
            that GUC; the executor manages its own per-row read/write identity.
        question: The question text, used as the missed-query input for turn
            resolution (forwarded to ``enqueue_replay_on_miss``).
        user_id: The benchmark identity owning the enqueued rows + written claims.
        executor: ``'inline'`` (default, synchronous) or ``'batch'`` (Batch API,
            production-exact). Identical claims; differ only in LLM dispatch.
        reason: Free-text label stored on the ``replay_queue`` row.

    Returns:
        ReplayDriveStats. When nothing is enqueued (no implicated turns, or an
        episode already had a pending row), the drain is skipped — an empty
        queue drain is a wasted LLM/batch round-trip — and the executor counts
        stay zero.
    """
    enqueued = await enqueue_replay_on_miss(pool, question, user_id, reason=reason)
    stats = ReplayDriveStats(rows_enqueued=enqueued)

    if enqueued == 0:
        logger.debug(
            "drive_replay: nothing enqueued for q=%r user=%s — skipping drain",
            question[:60], user_id,
        )
        return stats

    result: ReplayExecutorResult = (
        await run_replay_executor_batch(pool)
        if executor == "batch"
        else await run_replay_executor(pool)
    )
    stats.rows_processed = result.rows_processed
    stats.rows_done = result.rows_done
    stats.rows_failed = result.rows_failed
    stats.claims_written = result.claims_written
    stats.claims_superseded = result.claims_superseded

    logger.info(
        "drive_replay: q=%r executor=%s enqueued=%d processed=%d done=%d "
        "failed=%d claims_written=%d",
        question[:60], executor, enqueued, result.rows_processed,
        result.rows_done, result.rows_failed, result.claims_written,
    )
    return stats
