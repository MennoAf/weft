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
The daily audit is wired into the MCP server lifespan through
``weft.scheduler.canary_audit_loop``. The loop enumerates owners with canary
work and invokes this module once per owner with explicit predicates; it does
not depend on request middleware or ``WEFT_DEFAULT_USER_ID`` for isolation.
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


def _is_degenerate_reask_probe(text: str | None) -> bool:
    """Return True when ``text`` is a machine reference, not a natural-language query.

    The reask-bootstrap arm is the canary's *trustworthy* signal: its probes come
    from real ``is_reask_miss`` events (a query that missed, then a satisfying
    memory recorded), so a miss is supposed to mean a genuine recall regression.
    But a query like ``task:loom-7bedb110`` (an internal task reference) embeds
    into a region of vector space unrelated to the memory it "satisfied," so it can
    NEVER retrieve that memory — a permanent, meaningless miss that poisons the arm
    the primer surfaces as trustworthy. We refuse to enroll such probes (and disable
    any already enrolled).

    Conservative by design — only a single token (no internal whitespace) that looks
    like an identifier is rejected; a genuine short natural-language query keeps its
    whitespace or lacks id structure and passes through untouched.
    """
    t = (text or "").strip()
    if not t:
        return True
    if any(c.isspace() for c in t):
        return False  # multi-word => natural-language query
    # Single token: reject reference/id shapes ("prefix:id", "loom-7bedb110").
    if ":" in t:
        return True
    if any(c.isdigit() for c in t) and ("-" in t or "_" in t):
        return True
    return False

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


async def _sync_reask_bootstrap_probes(
    pool: asyncpg.Pool,
    *,
    user_id: str,
    eval_case_store_path: "Any" = None,
) -> int:
    """Auto-enroll new is_reask_miss events as high-confidence ``reask-bootstrap`` probes.

    Queries ``weft_recall_queries`` for rows where ``is_reask_miss = TRUE`` and a
    ``reask_satisfying_memory_id`` was recorded, then inserts a new
    ``recall_canary`` probe for each one that isn't already enrolled.

    Scoping is explicit on every read and write. Background jobs run through a
    service role that may bypass RLS, so the GUC is defense-in-depth rather than
    the tenant boundary. The scheduler calls this once per user per audit cycle.

    ``ON CONFLICT DO NOTHING`` handles expected PK collisions (e.g. rare probe_id
    hash collision) silently.  Any other exception is an UNEXPECTED error and is
    logged at ERROR level so it surfaces rather than being silently swallowed.

    Returns:
        Number of new ``reask-bootstrap`` probes enrolled (skipped conflicts and
        errors are excluded).  Callers can detect partial failure by comparing
        this against the WARNING log line that names attempted vs enrolled.
    """
    # Hygiene: disable any already-enrolled reask-bootstrap probe whose probe_text
    # is a machine reference (enrolled before this guard existed). One-way disable
    # is correct here — probe_text is immutable, so a degenerate probe can never
    # become valid. Uses the same Python predicate as the enrollment filter so the
    # two never drift.
    existing = await get_db(pool).fetch(
        """
        SELECT probe_id, probe_text FROM recall_canary
        WHERE probe_type = 'reask-bootstrap' AND enabled = TRUE
          AND user_id = $1
        """,
        user_id,
    )
    poisoned = [r["probe_id"] for r in existing if _is_degenerate_reask_probe(r["probe_text"])]
    if poisoned:
        await get_db(pool).execute(
            "UPDATE recall_canary SET enabled = FALSE "
            "WHERE probe_id = ANY($1::text[]) AND user_id = $2",
            poisoned,
            user_id,
        )
        logger.info(
            "_sync_reask_bootstrap_probes: disabled %d degenerate reask-bootstrap "
            "probe(s) already enrolled", len(poisoned),
        )

    # Find is_reask_miss rows not yet enrolled as reask-bootstrap probes.
    # The NOT EXISTS subquery avoids duplicate enrollment for the same
    # (memory_id, probe_text) pair that may arise from multiple reask events
    # pointing at the same satisfying memory with similar query text.
    rows = await get_db(pool).fetch(
        """
        SELECT q.query_text, q.reask_satisfying_memory_id, q.user_id
        FROM weft_recall_queries q
        WHERE q.is_reask_miss = TRUE
          AND q.reask_satisfying_memory_id IS NOT NULL
          AND q.user_id = $2
          AND NOT EXISTS (
              SELECT 1 FROM recall_canary c
              WHERE c.memory_id = q.reask_satisfying_memory_id
                AND c.probe_type = 'reask-bootstrap'
                AND c.probe_text = left(q.query_text, $1)
                AND c.user_id = q.user_id
          )
        """,
        PROBE_TEXT_MAX_CHARS,
        user_id,
    )

    # Refuse machine-reference query texts up front: they can never embed-retrieve
    # their satisfying memory, so enrolling them manufactures a permanent miss on
    # the trustworthy arm (see _is_degenerate_reask_probe).
    degenerate = [r for r in rows if _is_degenerate_reask_probe(r["query_text"])]
    if degenerate:
        logger.info(
            "_sync_reask_bootstrap_probes: skipped %d degenerate probe(s) "
            "(non-natural-language query text, e.g. %r)",
            len(degenerate),
            (degenerate[0]["query_text"] or "")[:60],
        )
    rows = [r for r in rows if not _is_degenerate_reask_probe(r["query_text"])]

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
            # reask-bootstrap insert failed silently before this. (The active
            # enroll path works only because it rides weft_remember's acquire()
            # GUC.)
            result = await get_db(pool).execute(
                """
                INSERT INTO recall_canary (probe_id, memory_id, user_id, probe_text, probe_type)
                VALUES ($1, $2, $3, $4, 'reask-bootstrap')
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


async def list_canary_user_ids(pool: asyncpg.Pool) -> list[str]:
    """List owners with enabled probes or eligible re-ask bootstrap rows.

    This is the scheduler's only cross-user canary query. Every returned owner
    is subsequently processed by :func:`run_canary_audit` with explicit tenant
    predicates, so an RLS-bypassing service role cannot mix probe universes.
    """
    rows = await get_db(pool).fetch(
        """
        SELECT user_id
        FROM (
            SELECT user_id
            FROM recall_canary
            WHERE enabled = TRUE AND user_id IS NOT NULL
            UNION
            SELECT user_id
            FROM weft_recall_queries
            WHERE is_reask_miss = TRUE
              AND reask_satisfying_memory_id IS NOT NULL
              AND user_id IS NOT NULL
        ) owners
        ORDER BY user_id
        """
    )
    return [str(row["user_id"]) for row in rows]


async def run_canary_audit(
    pool: asyncpg.Pool,
    embedder: Any,
    *,
    user_id: str,
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
        user_id: Required owner scope for probe selection, maintenance, writes,
            and vector retrieval. Never inferred from RLS because production
            background roles may bypass it.
        top_k: Number of top results to retrieve per probe.  A probe's
            ``memory_id`` must appear within these results to be a hit.
        active_probing_enabled: RI-4 gate.  ``False`` (default) runs only
            ``reask-bootstrap`` probes.  ``True`` also runs ``active`` probes.
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
        * ``bootstrap_synced`` (int): new ``reask-bootstrap`` probes enrolled this run.
        * ``probes_disabled`` (int): orphan probes soft-disabled this run because
          their referenced memory is no longer ``active`` (probe hygiene).
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

    # --- Phase 1: bootstrap from is_reask_miss events ---
    # eval_case_store_path is threaded in so reask misses also mint eval cases.
    bootstrap_synced = await _sync_reask_bootstrap_probes(
        pool, user_id=user_id, eval_case_store_path=eval_case_store_path
    )

    # --- Phase 1.5: probe hygiene — disable probes whose memory is no longer active ---
    # Archival paths (revise/supersede, quarantine merge, consolidation decay and
    # duplicate-merge) flip memories to status!='active' via direct UPDATEs that
    # bypass delete_memory()'s probe-disable. Left enabled, those orphan probes can
    # NEVER surface their (now non-active) memory — search_by_vector only returns
    # active memories — so every audit would record them as misses (phantom miss
    # events) and inflate the reconciliation miss-rate. Disable them here
    # (idempotent, RLS-scoped, covers hard-deletes via NOT EXISTS) so the audit
    # loop stops probing them within one cycle, regardless of which path archived
    # the memory. (canary_health also JOINs the current active universe, so a
    # freshly-archived probe's recent events drop out of the windowed rate too.)
    probes_disabled = int(
        await get_db(pool).fetchval(
            """
            WITH stale AS (
                UPDATE recall_canary c
                SET enabled = FALSE
                WHERE c.enabled = TRUE
                  AND c.user_id = $1
                  AND NOT EXISTS (
                      SELECT 1 FROM memories m
                      WHERE m.id = c.memory_id AND m.status = 'active'
                  )
                RETURNING 1
            )
            SELECT count(*) FROM stale
            """,
            user_id,
        )
        or 0
    )
    if probes_disabled:
        logger.info(
            "canary audit: disabled %d orphan probe(s) whose memory is no longer active",
            probes_disabled,
        )

    # --- Phase 2: select enabled probes (parameterized — no f-string interpolation) ---
    # The JOIN mirrors search_by_vector's DEFAULT candidate filter EXACTLY:
    # status='active' AND review_status='active' AND write_provenance != 'agent'.
    # The audit resolves a miss by searching via that same default path, so a probe
    # whose memory the search would never return (quarantined pending_review, or
    # agent-provenance) is not a recall failure — it's the review/provenance gate
    # working as designed. Selecting it would count a GUARANTEED miss and inflate
    # the reconciliation rate (measured: 10 of 11 active-arm misses were
    # pending_review memories with self-similarity ~0.9 but permanently unrankable).
    # Unlike Phase 1.5's one-way disable (for permanently-archived memories),
    # review_status is TRANSIENT: a pending_review memory is skipped here without
    # disabling its probe, so it re-enters the audit automatically once approved.
    probe_types = ["active", "reask-bootstrap"] if active_probing_enabled else ["reask-bootstrap"]

    probes = await get_db(pool).fetch(
        """
        SELECT c.probe_id, c.memory_id, c.probe_text, c.probe_type, c.user_id
        FROM recall_canary c
        JOIN memories m ON m.id = c.memory_id
            AND m.status = 'active'
            AND m.review_status = 'active'
            AND m.write_provenance != 'agent'
        WHERE c.enabled = TRUE
          AND c.probe_type = ANY($1::text[])
          AND c.user_id = $2
        ORDER BY c.probe_id
        """,
        probe_types,
        user_id,
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
            "probes_disabled": probes_disabled,
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
                include_agent_provenance=False,
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
                WHERE probe_id = $1 AND user_id = $2
                """,
                probe_id,
                user_id,
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
                WHERE probe_id = $1 AND user_id = $2
                """,
                probe_id,
                user_id,
            )
            logger.debug(
                "canary hit: probe_id=%s memory_id=%s probe_type=%s",
                probe_id,
                memory_id,
                probe["probe_type"],
            )

        # Append this run's outcome to the windowed event log. canary_health
        # computes its miss rate over a trailing time window of these rows, so
        # stale outcomes age out instead of pinning the lifetime counters
        # forever (see v66 migration). user_id is set EXPLICITLY from the probe
        # row: the raw scheduler pool leaves app.user_id unset, so the column
        # default would resolve to NULL and trip NOT NULL (same trap the
        # reask-bootstrap enroll hit).
        await get_db(pool).execute(
            """
            INSERT INTO recall_canary_audit (probe_id, user_id, hit)
            VALUES ($1, $2, $3)
            """,
            probe_id,
            probe["user_id"],
            not is_miss,
        )

    # --- Retention: bound the event log so the windowed query stays cheap ---
    # Keep a buffer beyond the health window so the window is always fully
    # covered; older rows can never affect a windowed rate, so prune them.
    await get_db(pool).execute(
        "DELETE FROM recall_canary_audit "
        "WHERE audited_at < now() - make_interval(days => $1) "
        "AND user_id = $2",
        _CANARY_AUDIT_RETENTION_DAYS,
        user_id,
    )

    miss_rate = misses / probes_checked if probes_checked > 0 else 0.0
    logger.info(
        "canary audit complete: probes_checked=%d misses=%d miss_rate=%.3f "
        "bootstrap_synced=%d probes_disabled=%d active_probing_enabled=%s",
        probes_checked,
        misses,
        miss_rate,
        bootstrap_synced,
        probes_disabled,
        active_probing_enabled,
    )
    return {
        "probes_checked": probes_checked,
        "misses": misses,
        "miss_rate": miss_rate,
        "bootstrap_synced": bootstrap_synced,
        "probes_disabled": probes_disabled,
        "audit_valid": True,
        "status": "ok",
    }


# A meter that hasn't audited within this window is treated as DARK. The audit
# runs daily (~23h min-age), so 48h means it has missed ~2 cycles — long enough
# to be a real failure, short enough to catch it fast on the next prime.
_CANARY_STALE_HOURS = 48.0

# Minimum audited checks before an arm's miss_rate is trusted (and before the
# drift tripwire can fire). Below this the sample is too small: a single early
# miss would read as a regression. Mirrors the audit_valid ">=1 probe" discipline
# but at a threshold where a rate is statistically meaningful.
_CANARY_MIN_TRIP_SAMPLE = 30

# Absolute miss-rate ceiling for the drift tripwire. The calibrated active-arm
# baseline is ~2.6% (PR #31, after the review_status/provenance artifact was
# removed). An absolute ~10% ceiling (~4x baseline) trips on real regressions —
# it would have caught the original 14.5% artifact — without false-alarming at
# baseline. Deliberately an ABSOLUTE ceiling, not a multiple of a stored baseline:
# the baseline sample is small (3 misses / 115 probes), so an absolute bound is
# more robust than overfitting a constant to it.
_CANARY_MISS_RATE_ALERT_THRESHOLD = 0.10

# Trailing window (days) over which canary_health computes the miss rate from the
# recall_canary_audit event log. A WINDOWED rate — not the lifetime counter sum —
# so a burst of historical misses ages out once recall recovers, instead of
# pinning the rate (and the drift tripwire) forever. 14 days ≈ 14 daily audit
# cohorts, enough samples for the >=30-check trustworthy gate while staying recent.
_CANARY_HEALTH_WINDOW_DAYS = 14

# Recent sub-window (days) that decides whether the drift tripwire is CURRENTLY
# firing. The 14-day windowed rate is the reported metric, but tripping on it
# lets one catastrophic-but-resolved incident keep the tripwire screaming for
# two clean weeks: the Oct 2026 FastEmbed-fallback burst (~990 miss events over
# 2 runs) held the windowed rate above the ceiling until its events aged out,
# even after four consecutive 100%-hit audits. Tripping on the RECENT rate fires
# immediately on a real regression (an acute burst and a chronic elevated rate
# both cross the ceiling within the recent window) and clears within
# _CANARY_TRIP_WINDOW_DAYS of recovery. Low-frequency deployments whose recent
# window carries fewer than _CANARY_MIN_TRIP_SAMPLE checks fall back to the
# windowed rate, preserving the old behavior where the recent sample is too thin.
_CANARY_TRIP_WINDOW_DAYS = 3

# Retention horizon (days) for recall_canary_audit rows. Kept > the health window
# so the trailing window is always fully covered, with a buffer of history for
# ad-hoc diagnostics; rows older than this can never affect a windowed rate and
# are pruned each audit run to keep the log bounded.
_CANARY_AUDIT_RETENTION_DAYS = 30


async def canary_health(
    pool: asyncpg.Pool, user_id: str
) -> dict | None:
    """Reconciliation-meter health summary for the primer and daily brief.

    Per probe arm it reports enrolled/audited probe counts and a **windowed**
    ``miss_rate`` — misses / checks over the ``recall_canary_audit`` event log
    within the trailing ``_CANARY_HEALTH_WINDOW_DAYS`` (v66). This replaced the
    old *lifetime* aggregate (``sum(miss_count)/sum(audit_count)`` over monotonic
    counters), which could never fall: a burst of historical misses pinned the
    rate — and the drift tripwire — forever. Concretely, PR #31 skips (does not
    disable) pending_review/agent-provenance probes, so their banked artifact
    misses lingered in the lifetime sum and, once PR #32 made the arm trustworthy,
    fired the tripwire on stale data. Windowing ages those outcomes out.

    Two universe filters keep the rate honest:
    * TIME — only events within the trailing window count, so old outcomes decay.
    * CURRENT UNIVERSE — a JOIN to ``memories`` on the SAME default filter
      ``search_by_vector`` uses (``status='active' AND review_status='active' AND
      write_provenance != 'agent'``) drops probes whose memory is no longer
      searchable, even if they have recent events. This mirrors the audit's own
      Phase-2 selection (PR #31), one level up at the health surface.

    Liveness (``dark`` / ``last_audit_at``) still comes from
    ``recall_canary.last_audit_at`` — "did the audit RUN", independent of the
    outcome window — so a meter with probes but no recent audit still SCREAMS.

    Trustworthiness is sample-based, not probe-type-based: an arm is
    ``trustworthy`` once its WINDOWED ``checks >= _CANARY_MIN_TRIP_SAMPLE``
    (below that it carries ``label='uncalibrated'``). The drift ``tripwire``
    fires for a trustworthy arm whose RECENT rate — over the trailing
    ``_CANARY_TRIP_WINDOW_DAYS``, when that window carries at least
    ``_CANARY_MIN_TRIP_SAMPLE`` checks, else the 14-day rate — crosses
    ``_CANARY_MISS_RATE_ALERT_THRESHOLD``. Tripping on the recent rate keeps a
    resolved incident burst from holding the tripwire open while its events age
    out of the reported 14-day rate, without losing sensitivity to chronic or
    acute regressions.

    ``user_id`` scopes the read EXPLICITLY rather than relying on the
    ``app.user_id`` GUC — the primer and scheduler contexts do not reliably set
    it (the same NULL-GUC gap that broke reask-bootstrap enrollment). Returns
    an explicit dark ``no_probes`` status when no enabled probe is in the
    current universe, and None only on query error — best-effort, never breaks
    prime.
    """
    from datetime import datetime, timezone

    _UNIVERSE = (
        "m.status = 'active' AND m.review_status = 'active' "
        "AND m.write_provenance != 'agent'"
    )
    try:
        # Query 1 — enrolled-probe skeleton + liveness. Drives arm presence, the
        # ``probes`` count, and dark/stale (last_audit_at = "when did we audit",
        # universe-independent of the outcome window).
        skeleton = await get_db(pool).fetch(
            f"""
            SELECT c.probe_type,
                   count(*)             AS probes,
                   max(c.last_audit_at) AS last_audit_at
            FROM recall_canary c
            JOIN memories m ON m.id = c.memory_id AND {_UNIVERSE}
            WHERE c.enabled = TRUE
              AND c.user_id = $1
            GROUP BY c.probe_type
            """,
            user_id,
        )
        # Query 2 — windowed outcomes from the event log, same universe filter.
        window = await get_db(pool).fetch(
            f"""
            SELECT c.probe_type,
                   count(a.id)                             AS checks,
                   count(a.id) FILTER (WHERE NOT a.hit)    AS misses,
                   count(DISTINCT a.probe_id)              AS audited
            FROM recall_canary_audit a
            JOIN recall_canary c ON c.probe_id = a.probe_id AND c.enabled = TRUE
            JOIN memories m ON m.id = c.memory_id AND {_UNIVERSE}
            WHERE a.audited_at > now() - make_interval(days => $1)
              AND a.user_id = $2
            GROUP BY c.probe_type
            """,
            _CANARY_HEALTH_WINDOW_DAYS,
            user_id,
        )
        # Query 2b — RECENT outcomes over the trip sub-window. The drift tripwire
        # fires on this rate (see _CANARY_TRIP_WINDOW_DAYS); the 14-day rate above
        # stays the reported metric so a resolved incident decays visibly instead
        # of vanishing.
        recent = await get_db(pool).fetch(
            f"""
            SELECT c.probe_type,
                   count(a.id)                             AS checks,
                   count(a.id) FILTER (WHERE NOT a.hit)    AS misses
            FROM recall_canary_audit a
            JOIN recall_canary c ON c.probe_id = a.probe_id AND c.enabled = TRUE
            JOIN memories m ON m.id = c.memory_id AND {_UNIVERSE}
            WHERE a.audited_at > now() - make_interval(days => $1)
              AND a.user_id = $2
            GROUP BY c.probe_type
            """,
            _CANARY_TRIP_WINDOW_DAYS,
            user_id,
        )
    except Exception:
        logger.debug("canary_health: aggregate query failed", exc_info=True)
        return None

    if not skeleton:
        return {
            "status": "no_probes",
            "arms": {},
            "last_audit_at": None,
            "audit_age_hours": None,
            "dark": True,
            "dark_reason": "no active probes",
            "alert": (
                "⚠️ recall canary DARK (no active probes) — the reconciliation "
                "meter has nothing enrolled to measure."
            ),
        }

    windowed = {r["probe_type"]: r for r in window}
    recent = {r["probe_type"]: r for r in recent}

    arms: dict[str, dict] = {}
    overall_last = None
    tripped_arms: list[tuple[str, float]] = []
    for r in skeleton:
        w = windowed.get(r["probe_type"])
        checks = int(w["checks"]) if w else 0
        misses = int(w["misses"]) if w else 0
        audited = int(w["audited"]) if w else 0
        last = r["last_audit_at"]
        miss_rate = round(misses / checks, 4) if checks else None
        # Trustworthy once the WINDOWED sample is large enough for the rate to
        # mean something — arm-type-agnostic. Below the threshold the arm is
        # flagged 'uncalibrated' so a thin sample is never mistaken for a rate.
        trustworthy = checks >= _CANARY_MIN_TRIP_SAMPLE
        # Drift tripwire: decided on the RECENT sub-window rate when it carries
        # enough sample; fall back to the windowed rate when the deployment
        # audits too rarely for a meaningful recent sample. Either way the
        # min-sample gate is baked in, so a thin recent window can't trip.
        rec = recent.get(r["probe_type"])
        recent_checks = int(rec["checks"]) if rec else 0
        recent_misses = int(rec["misses"]) if rec else 0
        recent_miss_rate = (
            round(recent_misses / recent_checks, 4) if recent_checks else None
        )
        if recent_checks >= _CANARY_MIN_TRIP_SAMPLE:
            trip_rate = recent_miss_rate
        else:
            trip_rate = miss_rate if trustworthy else None
        tripped = (
            trip_rate is not None
            and trip_rate > _CANARY_MISS_RATE_ALERT_THRESHOLD
        )
        arm = {
            "probes": int(r["probes"]),
            "audited": audited,
            "misses": misses,
            "checks": checks,
            "miss_rate": miss_rate,
            "recent_checks": recent_checks,
            "recent_miss_rate": recent_miss_rate,
            "recent_window_days": _CANARY_TRIP_WINDOW_DAYS,
            "last_audit_at": last.isoformat() if last else None,
            "trustworthy": trustworthy,
            "tripped": tripped,
        }
        if not trustworthy:
            arm["label"] = "uncalibrated"
        if tripped:
            tripped_arms.append((r["probe_type"], trip_rate))
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
            "Check the per-user canary_audit loop and scheduler state."
        )
    if tripped_arms:
        worst_arm, worst_rate = max(tripped_arms, key=lambda t: t[1])
        health["tripwire"] = (
            f"⚠️ recall canary miss-rate {worst_rate:.1%} on the '{worst_arm}' arm "
            f"over the last {_CANARY_TRIP_WINDOW_DAYS} days exceeds the "
            f"{_CANARY_MISS_RATE_ALERT_THRESHOLD:.0%} ceiling — recall may be "
            "regressing (embedding drift, index corruption, or a filter "
            "mismatch). Investigate before trusting recall."
        )
    return health
