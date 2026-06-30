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

2. **``reaREDACTED``** (high-confidence, PRIMARY): Auto-enrolled at audit time from
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
# Eval-case minting (CL1 compounding loop, loom-add4d5c8)
# ---------------------------------------------------------------------------

# Sentinel: "use the default path from benchmarks.enumeration_eval.mint if
# the package is available; otherwise no-op."  Distinct from None so callers
# can explicitly pass None to DISABLE minting even when benchmarks is present.
_EVAL_CASE_SENTINEL = object()


def _default_eval_case_store_path():
    """Return DEFAULT_MINTED_CASES_PATH from benchmarks, or None if unavailable."""
    try:
        from benchmarks.enumeration_eval.mint import DEFAULT_MINTED_CASES_PATH  # noqa: PLC0415
        return DEFAULT_MINTED_CASES_PATH
    except ImportError:
        return None


def _try_mint_eval_case(
    query: str,
    satisfying_memory_id: str,
    source: str,
    *,
    probe_id: str | None = None,
    path: "Any",
) -> None:
    """Best-effort eval case minting.  Logs and swallows all errors.

    Args:
        query: The query / probe_text to record.
        satisfying_memory_id: The memory that should surface for this query.
        source: ``'canary'`` (direct miss) or ``'reask'`` (is_reask_miss event).
        probe_id: The recall_canary probe_id for traceability.
        path: Destination JSONL path (resolved before this call).  None means skip.
    """
    if path is None:
        return
    try:
        from benchmarks.enumeration_eval.mint import mint_eval_case  # noqa: PLC0415
        minted = mint_eval_case(
            query,
            satisfying_memory_id,
            source,
            probe_id=probe_id,
            path=path,
        )
        if minted:
            logger.debug(
                "_try_mint_eval_case: minted eval case memory_id=%s source=%s",
                satisfying_memory_id,
                source,
            )
        # else: duplicate — fine, no log needed
    except ImportError:
        logger.debug("_try_mint_eval_case: benchmarks package not available, skipping mint")
    except Exception as exc:
        # Never crash the audit over a minting failure — best-effort only.
        logger.warning("_try_mint_eval_case: unexpected error: %s", exc)


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
            use the memory's own content (truncated).  For reaREDACTED
            probes, use the original missed query text.  Truncated to
            ``PROBE_TEXT_MAX_CHARS`` at enrollment.
        probe_type: ``'active'`` (synthetic, gated) or ``'reaREDACTED'``
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
# Bootstrap: is_reask_miss → reaREDACTED probes
# ---------------------------------------------------------------------------


async def _sync_reask_bootstrap_probes(
    pool: asyncpg.Pool,
    *,
    eval_case_store_path: "Any" = None,
) -> int:
    """Auto-enroll new is_reask_miss events as high-confidence ``reaREDACTED`` probes.

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
        Number of new ``reaREDACTED`` probes enrolled (skipped conflicts and
        errors are excluded).  Callers can detect partial failure by comparing
        this against the WARNING log line that names attempted vs enrolled.
    """
    # Find is_reask_miss rows not yet enrolled as reaREDACTED probes.
    # The NOT EXISTS subquery avoids duplicate enrollment for the same
    # (memory_id, probe_text) pair that may arise from multiple reask events
    # pointing at the same satisfying memory with similar query text.
    rows = await get_db(pool).fetch(
        """
        SELECT q.query_text, q.reask_satisfying_memory_id, q.user_id
        FROM weft_recall_queries q
        WHERE q.is_reask_miss = TRUE
          AND q.reask_satisfying_memory_id IS NOT NULL
          AND q.user_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM recall_canary c
              WHERE c.memory_id = q.reask_satisfying_memory_id
                AND c.probe_type = 'reaREDACTED'
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
            # user_id is set EXPLICITLY from the query row, not left to the
            # recall_canary.user_id GUC default. This path runs on the raw pool
            # (scheduler / audit context) where ``app.user_id`` is unset, so the
            # ``nullif(current_setting('app.user_id', true), '')`` column default
            # resolves to NULL and trips the NOT NULL constraint — every
            # reaREDACTED insert failed silently before this. (The active
            # enroll path works only because it rides weft_remember's acquire()
            # GUC.)
            result = await get_db(pool).execute(
                """
                INSERT INTO recall_canary (probe_id, memory_id, user_id, probe_text, probe_type)
                VALUES ($1, $2, $3, $4, 'reaREDACTED')
                ON CONFLICT DO NOTHING
                """,
                probe_id,
                row["reask_satisfying_memory_id"],
                row["user_id"],
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
                # CL1: mint a known-answer eval case for each new reask probe.
                # The original query → satisfying memory is a proven known-answer
                # pair from the is_reask_miss signal.
                _try_mint_eval_case(
                    row["query_text"],
                    row["reask_satisfying_memory_id"],
                    "reask",
                    probe_id=probe_id,
                    path=eval_case_store_path,
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
    eval_case_store_path: Any = _EVAL_CASE_SENTINEL,
) -> dict[str, Any]:
    """Run the daily recall canary audit against fixed materialization.

    For each enabled probe, embeds the ``probe_text`` and runs
    ``search_by_vector`` (deterministic: local FastEmbed ONNX + tie-break from
    task 0.3).  If the probe's ``memory_id`` does NOT appear within the top-K
    results, the probe is a *canary miss*: the miss_count is incremented, the
    global ``canary.miss`` counter is bumped, and ``last_audit_at`` is updated.

    Audit scope:
    - ``active_probing_enabled=False`` (default): only ``'reaREDACTED'``
      probes run.  Their miss rate is immediately trustworthy because they
      derive from the proven ``is_reask_miss`` signal.
    - ``active_probing_enabled=True``: both ``'active'`` and ``'reaREDACTED'``
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
            ``reaREDACTED`` probes.  ``True`` also runs ``active`` probes.
        eval_case_store_path: Path to the JSONL eval-case store for the CL1
            compounding loop.  On a miss, a known-answer case is appended so
            the ``benchmarks.enumeration_eval`` harness can exercise it on the
            next run.  Default (``_EVAL_CASE_SENTINEL``): use
            ``benchmarks.enumeration_eval.mint.DEFAULT_MINTED_CASES_PATH`` when
            the ``benchmarks`` package is available; silently skip when it is
            not.  Pass ``None`` to disable minting explicitly.  Pass an explicit
            ``Path`` to redirect minting (e.g. to a ``tmp_path`` in tests).

    Returns:
        ``dict`` with keys:

        * ``probes_checked`` (int): number of probes actually evaluated.
        * ``misses`` (int): number of probes that failed to surface their memory.
        * ``miss_rate`` (float): ``misses / probes_checked``, or 0.0 if none checked.
        * ``bootstrap_synced`` (int): new ``reaREDACTED`` probes enrolled this run.
        * ``audit_valid`` (bool): ``True`` when the audit ran with ≥1 probe and valid
          user scoping.  ``False`` when the audit was a no-op (skipped).  A caller
          MUST check this before treating ``miss_rate=0.0`` as a healthy signal —
          a skipped audit and a true all-pass both produce ``miss_rate=0.0``.
        * ``status`` (str): ``'ok'`` for a valid run, ``'skipped'`` for a no-op.
    """
    # Late import to avoid circular dependency: canary ← store ← (many things).
    from weft.store import search_by_vector

    # --- Resolve eval-case store path (CL1) ---
    # Sentinel means "auto-detect": use the benchmarks default if available.
    if eval_case_store_path is _EVAL_CASE_SENTINEL:
        eval_case_store_path = _default_eval_case_store_path()

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
    # eval_case_store_path is threaded in so reask misses also mint eval cases.
    bootstrap_synced = await _sync_reask_bootstrap_probes(
        pool, eval_case_store_path=eval_case_store_path
    )

    # --- Phase 2: select enabled probes (parameterized — no f-string interpolation) ---
    probe_types = ["active", "reaREDACTED"] if active_probing_enabled else ["reaREDACTED"]

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
            # CL1: mint a known-answer eval case for every new miss so the
            # enumeration harness can track and exercise it going forward.
            _try_mint_eval_case(
                probe_text,
                memory_id,
                "canary",
                probe_id=probe_id,
                path=eval_case_store_path,
            )
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


# A meter that hasn't audited within this window is treated as DARK. The audit
# runs daily (~23h min-age), so 48h means it has missed ~2 cycles — long enough
# to be a real failure, short enough to catch it fast on the next prime.
_CANARY_STALE_HOURS = 48.0


async def canary_health(
    pool: asyncpg.Pool, user_id: str | None = None
) -> dict | None:
    """Reconciliation-meter health summary for the primer and daily brief.

    Read-only aggregate over ``recall_canary`` grouped by probe arm
    (``active`` = uncalibrated completeness proxy; ``reaREDACTED`` =
    trustworthy, derived from the proven ``is_reask_miss`` signal). Per arm it
    reports enrolled/audited probe counts and a lifetime ``miss_rate``
    (``sum(miss_count) / sum(audit_count)`` — both bump on every hit and miss,
    so this equals the latest run's rate when stable).

    The load-bearing part is ``dark``: True when the meter has NEVER audited or
    its last audit is older than ``_CANARY_STALE_HOURS``. When dark, ``alert``
    carries a loud, human-readable line so a dead meter SCREAMS on the next
    prime instead of failing silent — the exact failure mode that let the meter
    sit unaudited for weeks. The watchman is watched by the one surface a human
    reads every session.

    ``user_id`` scopes the read EXPLICITLY (``WHERE user_id = $1``) rather than
    relying on the ``app.user_id`` GUC — the primer and scheduler contexts do
    not reliably set it (the same NULL-GUC gap that broke reaREDACTED
    enrollment). Returns None on any error — best-effort, never breaks prime.
    """
    from datetime import datetime, timezone

    try:
        if user_id is not None:
            rows = await get_db(pool).fetch(
                """
                SELECT probe_type,
                       count(*)                               AS probes,
                       count(*) FILTER (WHERE audit_count > 0) AS audited,
                       coalesce(sum(miss_count), 0)           AS misses,
                       coalesce(sum(audit_count), 0)          AS checks,
                       max(last_audit_at)                     AS last_audit_at
                FROM recall_canary
                WHERE enabled = TRUE AND user_id = $1
                GROUP BY probe_type
                """,
                user_id,
            )
        else:
            # No explicit scope: lean on RLS (app.user_id GUC) for tenant
            # boundary. Used by callers that already run inside acquire().
            rows = await get_db(pool).fetch(
                """
                SELECT probe_type,
                       count(*)                               AS probes,
                       count(*) FILTER (WHERE audit_count > 0) AS audited,
                       coalesce(sum(miss_count), 0)           AS misses,
                       coalesce(sum(audit_count), 0)          AS checks,
                       max(last_audit_at)                     AS last_audit_at
                FROM recall_canary
                WHERE enabled = TRUE
                GROUP BY probe_type
                """
            )
    except Exception:
        logger.debug("canary_health: aggregate query failed", exc_info=True)
        return None

    if not rows:
        return None

    arms: dict[str, dict] = {}
    overall_last = None
    for r in rows:
        checks = int(r["checks"] or 0)
        misses = int(r["misses"] or 0)
        last = r["last_audit_at"]
        arm = {
            "probes": int(r["probes"]),
            "audited": int(r["audited"]),
            "misses": misses,
            "checks": checks,
            "miss_rate": round(misses / checks, 4) if checks else None,
            "last_audit_at": last.isoformat() if last else None,
            "trustworthy": r["probe_type"] == "reaREDACTED",
        }
        if r["probe_type"] == "active":
            # The active arm is a real signal but its baseline was never
            # calibrated (RI-4) — surfaced, but labelled so it is not mistaken
            # for a trustworthy miss rate.
            arm["label"] = "uncalibrated"
        arms[r["probe_type"]] = arm
        if last is not None and (overall_last is None or last > overall_last):
            overall_last = last

    now = datetime.now(timezone.utc)
    age_hours = (
        (now - overall_last).total_seconds() / 3600 if overall_last else None
    )
    dark = age_hours is None or age_hours >= _CANARY_STALE_HOURS
    if age_hours is None:
        dark_reason = "never audited"
    elif dark:
        dark_reason = f"stale — last audit {age_hours:.0f}h ago"
    else:
        dark_reason = None

    health: dict = {
        "arms": arms,
        "last_audit_at": overall_last.isoformat() if overall_last else None,
        "audit_age_hours": round(age_hours, 1) if age_hours is not None else None,
        "dark": dark,
        "dark_reason": dark_reason,
    }
    if dark:
        health["alert"] = (
            f"⚠️ recall canary DARK ({dark_reason}) — the reconciliation meter "
            "is not measuring; silent recall misses are going undetected. "
            "Check WEFT_DEFAULT_USER_ID and the canary_audit loop."
        )
    return health
