"""Recall canary: enrollment + daily fixed-materialization audit.

Phase 0.5 of the recall health monitoring system (PRD §V5, loom-c27ab1d2).

## Design (RI-4 guidance)

Probes are enrolled when memories are written (O(1) INSERT, no LLM call) and
audited daily against *fixed materialization* — deterministic vector search via
local FastEmbed with the tie-break ``ORDER BY (embedding <=> ...), id`` from task 0.3.
The same probe always produces the same ranking, so a miss is a real ranking miss,
not run-to-run noise.

### Two probe types (both stored in ``recall_canary``):

1. **``active``** (synthetic, SECONDARY): Enrolled by ``weft_remember`` at write time.
   The probe text is the memory's own content (truncated to ``PROBE_TEXT_MAX_CHARS``).
   Active probes are gated behind ``active_probing_enabled`` in the audit so their
   miss rate can be calibrated before it's trusted.  Default: disabled.

2. **``reask-bootstrap``** (high-confidence, PRIMARY): Auto-enrolled at audit time from
   ``weft_recall_queries`` rows where ``is_reask_miss = TRUE`` and a satisfying memory
   was recorded.  A real re-ask miss is a proven known-answer case: the original query
   text → satisfying memory.  These probes always run in the audit (not gated by the
   flag).

The bootstrap-first design follows RI-4: the ``is_reask_miss`` signal is PROVEN; active
synthetic probing is SECONDARY and unproven until calibrated.  Active probes accumulate
in the table during collection but are excluded from the audit (and thus the
``canary.miss`` counter) until ``active_probing_enabled`` is flipped on.

### Non-degeneracy contract (done_when gate, loom-c27ab1d2):
- ``probes_checked > 0``
- A probe with a probe_text that doesn't match its memory's content is recorded
  as a ``canary_miss``
- A probe whose probe_text naturally retrieves its memory is NOT flagged

### Scheduler note:
The daily audit is a callable entrypoint (``run_canary_audit``).  No live scheduler
is wired up — a cron job or a Loom-scheduled worker calls this function.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import asyncpg

from weft.counters import increment_counter
from weft.db.connection import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Global telemetry counter name — matches the weft_counters pattern from v54.
# Import this in ``weft.counters`` callers instead of hardcoding the string.
COUNTER_CANARY_MISS = "canary.miss"

# Active probes truncate long content at enrollment so probe_text stays small.
PROBE_TEXT_MAX_CHARS = 512

# Default top-K for the daily audit.  A probe must surface within this many
# results to be a hit; anything beyond is a miss.
DEFAULT_AUDIT_TOP_K = 10


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


async def enroll_canary(
    pool: asyncpg.Pool,
    memory_id: str,
    probe_text: str,
    *,
    probe_type: str = "active",
) -> str:
    """Enroll a memory as a recall canary probe.

    O(1) cheap INSERT — no LLM call, no embedding.  Called from the
    ``weft_remember`` write path immediately after ``store_memory``.

    The ``user_id`` column is populated by the GUC default
    ``nullif(current_setting('app.user_id', true), '')``, which matches the
    identity set by ``acquire()`` or the pool's ``setup`` callback.  No explicit
    user_id is required from the caller.

    Args:
        pool: asyncpg connection pool.
        memory_id: The memory to probe.  The audit checks that this memory
            surfaces when ``probe_text`` is searched.
        probe_text: The query text to use in the audit.  For active probes,
            use the memory's own content (truncated).  For reask-bootstrap
            probes, use the original missed query text.  Truncated to
            ``PROBE_TEXT_MAX_CHARS`` at enrollment.
        probe_type: ``'active'`` (synthetic, gated) or ``'reask-bootstrap'``
            (primary, always audited).

    Returns:
        The ``probe_id`` of the new canary probe (``cp-{shortid}`` format).
    """
    probe_id = f"cp-{uuid.uuid4().hex[:10]}"
    probe_text = probe_text[:PROBE_TEXT_MAX_CHARS]
    await get_db(pool).execute(
        """
        INSERT INTO recall_canary (probe_id, memory_id, probe_text, probe_type)
        VALUES ($1, $2, $3, $4)
        """,
        probe_id,
        memory_id,
        probe_text,
        probe_type,
    )
    logger.debug(
        "enroll_canary: probe_id=%s memory_id=%s probe_type=%s",
        probe_id,
        memory_id,
        probe_type,
    )
    return probe_id


# ---------------------------------------------------------------------------
# Bootstrap: is_reask_miss → reask-bootstrap probes
# ---------------------------------------------------------------------------


async def _sync_reask_bootstrap_probes(pool: asyncpg.Pool) -> int:
    """Auto-enroll new is_reask_miss events as high-confidence ``reask-bootstrap`` probes.

    Queries ``weft_recall_queries`` for rows where ``is_reask_miss = TRUE`` and a
    ``reask_satisfying_memory_id`` was recorded, then inserts a new
    ``recall_canary`` probe for each one that isn't already enrolled.

    Scoping: operates under the current GUC ``app.user_id`` — both the SELECT
    (via RLS on ``weft_recall_queries``) and the INSERT (via RLS + DEFAULT on
    ``recall_canary``) are automatically scoped to the current user.  The
    scheduler's per-user fan-out (``get_distinct_reask_user_ids``) ensures this
    is called once per user per audit cycle.

    ``ON CONFLICT DO NOTHING`` handles expected PK collisions (e.g. rare probe_id
    hash collision) silently.  Any other exception is an UNEXPECTED error and is
    logged at ERROR level so it surfaces rather than being silently swallowed.

    Returns:
        Number of new ``reask-bootstrap`` probes enrolled (skipped conflicts and
        errors are excluded).  Callers can detect partial failure by comparing
        this against the WARNING log line that names attempted vs enrolled.
    """
    # Find is_reask_miss rows not yet enrolled as reask-bootstrap probes.
    # The NOT EXISTS subquery avoids duplicate enrollment for the same
    # (memory_id, probe_text) pair that may arise from multiple reask events
    # pointing at the same satisfying memory with similar query text.
    rows = await get_db(pool).fetch(
        """
        SELECT q.query_text, q.reask_satisfying_memory_id
        FROM weft_recall_queries q
        WHERE q.is_reask_miss = TRUE
          AND q.reask_satisfying_memory_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM recall_canary c
              WHERE c.memory_id = q.reask_satisfying_memory_id
                AND c.probe_type = 'reask-bootstrap'
                AND c.probe_text = left(q.query_text, $1)
          )
        """,
        PROBE_TEXT_MAX_CHARS,
    )

    attempted = len(rows)
    enrolled = 0
    errors = 0

    for row in rows:
        probe_id = f"cp-{uuid.uuid4().hex[:10]}"
        try:
            result = await get_db(pool).execute(
                """
                INSERT INTO recall_canary (probe_id, memory_id, probe_text, probe_type)
                VALUES ($1, $2, $3, 'reask-bootstrap')
                ON CONFLICT DO NOTHING
                """,
                probe_id,
                row["reask_satisfying_memory_id"],
                row["query_text"][:PROBE_TEXT_MAX_CHARS],
            )
            # asyncpg returns "INSERT 0 1" when a row was inserted, "INSERT 0 0"
            # when ON CONFLICT skipped it.  Only count actual inserts.
            if result.split()[-1] == "1":
                enrolled += 1
                logger.debug(
                    "_sync_reask_bootstrap_probes: enrolled probe_id=%s memory_id=%s",
                    probe_id,
                    row["reask_satisfying_memory_id"],
                )
            # else: expected conflict/skip — fine, no logging needed
        except Exception as exc:
            # UNEXPECTED error (network, schema issue, RLS violation, …).
            # Log at ERROR so it surfaces; do not silently continue past it.
            errors += 1
            logger.error(
                "_sync_reask_bootstrap_probes: unexpected error for memory_id=%s: %s",
                row["reask_satisfying_memory_id"],
                exc,
            )

    if errors:
        logger.warning(
            "_sync_reask_bootstrap_probes: %d/%d probes enrolled, %d unexpected errors",
            enrolled,
            attempted,
            errors,
        )
    elif attempted:
        logger.debug(
            "_sync_reask_bootstrap_probes: %d/%d probes enrolled",
            enrolled,
            attempted,
        )

    return enrolled


# ---------------------------------------------------------------------------
# Daily audit
# ---------------------------------------------------------------------------


async def run_canary_audit(
    pool: asyncpg.Pool,
    embedder: Any,
    *,
    user_id: str | None = None,
    top_k: int = DEFAULT_AUDIT_TOP_K,
    active_probing_enabled: bool = False,
) -> dict[str, Any]:
    """Run the daily recall canary audit against fixed materialization.

    For each enabled probe, embeds the ``probe_text`` and runs
    ``search_by_vector`` (deterministic: local FastEmbed ONNX + tie-break from
    task 0.3).  If the probe's ``memory_id`` does NOT appear within the top-K
    results, the probe is a *canary miss*: the miss_count is incremented, the
    global ``canary.miss`` counter is bumped, and ``last_audit_at`` is updated.

    Audit scope:
    - ``active_probing_enabled=False`` (default): only ``'reask-bootstrap'``
      probes run.  Their miss rate is immediately trustworthy because they
      derive from the proven ``is_reask_miss`` signal.
    - ``active_probing_enabled=True``: both ``'active'`` and ``'reask-bootstrap'``
      probes run.  Enable only after calibrating the active miss baseline.

    **Fixed materialization guarantee**: FastEmbed is deterministic (local ONNX,
    no random seed), and ``search_by_vector`` breaks ties by ``id`` column
    (task 0.3). The same probe always produces the same top-K ranking on the
    same data, making the audit reproducible.

    Args:
        pool: asyncpg connection pool.
        embedder: An ``EmbeddingProvider`` (e.g. ``FastEmbedProvider``).
            Must implement ``async embed(text: str) -> list[float]``.
        user_id: If supplied, passed to ``search_by_vector`` to scope memory
            retrieval to this user's memories.  ``None`` lets RLS handle scoping
            (correct when the pool's ``setup`` callback has already set
            ``app.user_id``).
        top_k: Number of top results to retrieve per probe.  A probe's
            ``memory_id`` must appear within these results to be a hit.
        active_probing_enabled: RI-4 gate.  ``False`` (default) runs only
            ``reask-bootstrap`` probes.  ``True`` also runs ``active`` probes.

    Returns:
        ``dict`` with keys:

        * ``probes_checked`` (int): number of probes actually evaluated.
        * ``misses`` (int): number of probes that failed to surface their memory.
        * ``miss_rate`` (float): ``misses / probes_checked``, or 0.0 if none checked.
        * ``bootstrap_synced`` (int): new ``reask-bootstrap`` probes enrolled this run.
        * ``audit_valid`` (bool): ``True`` when the audit ran with ≥1 probe and valid
          user scoping.  ``False`` when the audit was a no-op (skipped).  A caller
          MUST check this before treating ``miss_rate=0.0`` as a healthy signal —
          a skipped audit and a true all-pass both produce ``miss_rate=0.0``.
        * ``status`` (str): ``'ok'`` for a valid run, ``'skipped'`` for a no-op.
    """
    # Late import to avoid circular dependency: canary ← store ← (many things).
    from weft.store import search_by_vector

    # --- Guard: user scoping must be established ---
    # Without a user_id argument AND without the app.user_id GUC the audit
    # would silently run against an empty (or system-only) probe set, producing
    # a miss_rate=0.0 that looks healthy but measured nothing meaningful.
    if user_id is None:
        guc_uid = await get_db(pool).fetchval(
            "SELECT nullif(current_setting('app.user_id', true), '')"
        )
        if guc_uid is None:
            logger.warning(
                "canary audit: no user scoping in effect — user_id not passed and "
                "app.user_id GUC is empty. Returning audit_valid=False (skipped)."
            )
            return {
                "probes_checked": 0,
                "misses": 0,
                "miss_rate": 0.0,
                "bootstrap_synced": 0,
                "audit_valid": False,
                "status": "skipped",
            }

    # --- Phase 1: bootstrap from is_reask_miss events ---
    bootstrap_synced = await _sync_reask_bootstrap_probes(pool)

    # --- Phase 2: select enabled probes (parameterized — no f-string interpolation) ---
    probe_types = ["active", "reask-bootstrap"] if active_probing_enabled else ["reask-bootstrap"]

    probes = await get_db(pool).fetch(
        """
        SELECT probe_id, memory_id, probe_text, probe_type
        FROM recall_canary
        WHERE enabled = TRUE AND probe_type = ANY($1::text[])
        ORDER BY probe_id
        """,
        probe_types,
    )

    # --- Guard: 0 enabled probes → audit_valid=False ---
    # A 0-probe result has miss_rate=0.0, identical to a healthy all-pass.
    # Signal explicitly so schedulers/callers can distinguish the two.
    if not probes:
        logger.warning(
            "canary audit: 0 enabled probes found — audit_valid=False "
            "(bootstrap_synced=%d). This is NOT a genuine all-pass result.",
            bootstrap_synced,
        )
        return {
            "probes_checked": 0,
            "misses": 0,
            "miss_rate": 0.0,
            "bootstrap_synced": bootstrap_synced,
            "audit_valid": False,
            "status": "skipped",
        }

    probes_checked = 0
    misses = 0

    for probe in probes:
        probe_id: str = probe["probe_id"]
        memory_id: str = probe["memory_id"]
        probe_text: str = probe["probe_text"]

        # Embed the probe text.  FastEmbed ONNX is deterministic — same text
        # always produces the same vector.
        try:
            embedding = await embedder.embed(probe_text)
        except Exception as exc:
            logger.warning(
                "canary audit: embed failed for probe_id=%s: %s", probe_id, exc,
            )
            continue

        # Vector search against fixed materialization.
        # threshold=0.0 so all memories with embeddings are candidates.
        # tie-break ORDER BY ..., id from task 0.3 makes the ranking deterministic.
        try:
            results = await search_by_vector(
                pool,
                embedding,
                limit=top_k,
                threshold=0.0,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(
                "canary audit: search_by_vector failed for probe_id=%s: %s",
                probe_id,
                exc,
            )
            continue

        probes_checked += 1
        result_ids = {r.memory.id for r in results}
        is_miss = memory_id not in result_ids

        if is_miss:
            misses += 1
            await get_db(pool).execute(
                """
                UPDATE recall_canary
                SET miss_count   = miss_count + 1,
                    audit_count  = audit_count + 1,
                    last_audit_at = now()
                WHERE probe_id = $1
                """,
                probe_id,
            )
            # Best-effort counter: never raises (counters.py contract).
            await increment_counter(pool, COUNTER_CANARY_MISS)
            logger.debug(
                "canary miss: probe_id=%s memory_id=%s probe_type=%s",
                probe_id,
                memory_id,
                probe["probe_type"],
            )
        else:
            await get_db(pool).execute(
                """
                UPDATE recall_canary
                SET audit_count  = audit_count + 1,
                    last_audit_at = now()
                WHERE probe_id = $1
                """,
                probe_id,
            )
            logger.debug(
                "canary hit: probe_id=%s memory_id=%s probe_type=%s",
                probe_id,
                memory_id,
                probe["probe_type"],
            )

    miss_rate = misses / probes_checked if probes_checked > 0 else 0.0
    logger.info(
        "canary audit complete: probes_checked=%d misses=%d miss_rate=%.3f "
        "bootstrap_synced=%d active_probing_enabled=%s",
        probes_checked,
        misses,
        miss_rate,
        bootstrap_synced,
        active_probing_enabled,
    )
    return {
        "probes_checked": probes_checked,
        "misses": misses,
        "miss_rate": miss_rate,
        "bootstrap_synced": bootstrap_synced,
        "audit_valid": True,
        "status": "ok",
    }
