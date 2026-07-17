"""Tests for the recall canary system (Phase 0.5, loom-c27ab1d2).

Done-when gate (non-degeneracy):
  1. One audit cycle runs with probes_checked > 0.
  2. A deliberately-planted below-cutoff probe is recorded as a canary_miss.
  3. A known-surfacing probe is NOT flagged (miss_count stays 0).

A no-op meter that just returns rate=0 with probes_checked=0 MUST fail the test.

RI-4 design notes:
  - ReaREDACTED probes (derived from is_reask_miss signal) are the primary,
    always-audited probe type.
  - Active synthetic probes are gated behind active_probing_enabled and
    collected at weft_remember write time but NOT audited by default.
  - Fixed materialization: FastEmbed ONNX is deterministic; store.py tie-break
    (ORDER BY ..., id) removes the last nondeterminism.
"""

from __future__ import annotations

import pytest

from weft.canary import (
    COUNTER_CANARY_MISS,
    PROBE_TEXT_MAX_CHARS,
    _is_degenerate_reask_probe,
    canary_health,
    enroll_canary,
    run_canary_audit,
)
from tests.conftest import DEFAULT_TEST_USER_ID
from weft.counters import get_counter
from weft.db.connection import get_db
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.store import delete_memory, store_memory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder():
    """Local FastEmbed provider — deterministic ONNX, no API key."""
    return get_provider("fastembed")


@pytest.fixture(autouse=True)
async def clean_canary(pool):
    """Truncate recall_canary (+ its audit event log) before each test.

    conftest.py's TRUNCATE list predates these tables (v63/v66 are newer).  This
    autouse fixture fills the gap until conftest is updated.  Follow-up:
    add 'recall_canary' + 'recall_canary_audit' to the TRUNCATE list in
    tests/conftest.py. CASCADE covers the recall_canary_audit FK.
    """
    await pool.execute("TRUNCATE recall_canary, recall_canary_audit CASCADE")
    yield


# ---------------------------------------------------------------------------
# Enrollment tests
# ---------------------------------------------------------------------------


async def test_enroll_canary_inserts_probe(pool):
    """enroll_canary writes a row with correct probe_id / memory_id / probe_type."""
    probe_id = await enroll_canary(pool, "mem-abc123", "cats love sunny spots")
    row = await get_db(pool).fetchrow(
        "SELECT memory_id, probe_text, probe_type, enabled, miss_count "
        "FROM recall_canary WHERE probe_id = $1",
        probe_id,
    )
    assert row is not None
    assert row["memory_id"] == "mem-abc123"
    assert row["probe_text"] == "cats love sunny spots"
    assert row["probe_type"] == "active"
    assert row["enabled"] is True
    assert row["miss_count"] == 0


async def test_enroll_canary_truncates_long_probe_text(pool):
    """probe_text longer than PROBE_TEXT_MAX_CHARS is truncated at enrollment."""
    long_text = "x" * (PROBE_TEXT_MAX_CHARS + 100)
    probe_id = await enroll_canary(pool, "mem-trunc1", long_text)
    row = await get_db(pool).fetchrow(
        "SELECT probe_text FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row is not None
    assert len(row["probe_text"]) == PROBE_TEXT_MAX_CHARS


async def test_enroll_canary_reask_bootstrap_type(pool):
    """reaREDACTED probe_type is stored correctly."""
    probe_id = await enroll_canary(
        pool, "mem-rq001", "original missed query", probe_type="reaREDACTED",
    )
    row = await get_db(pool).fetchrow(
        "SELECT probe_type FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row is not None
    assert row["probe_type"] == "reaREDACTED"


# ---------------------------------------------------------------------------
# Non-degeneracy gate (the critical done_when test)
# ---------------------------------------------------------------------------


async def test_canary_audit_nondegeneracy(pool, embedder):
    """Non-degeneracy gate: probes_checked > 0, below-cutoff miss is caught,
    known-surfacing probe is NOT flagged.

    Design:
    - Memory A: "The orange cat sleeps in warm sunbeams" (semantic: cats)
    - Memory B: "Quantum entanglement and particle physics" (semantic: physics)

    Probe for A: probe_text = A's own content → vector search returns A → HIT.
    Probe for B: probe_text = A's content (cats) → vector search returns A,
                 NOT B, when top_k=1 → MISS.  This is the "below-cutoff" probe.

    Using top_k=1 guarantees:
    - A probe searching for cat content retrieves only the single most-similar
      result.  Memory A (the cat memory) is the nearest neighbour to its own
      embedding → A's probe hits.  Memory B (physics) is NOT the nearest
      neighbour to cat content → B's probe misses.
    """
    cat_content = "The orange cat sleeps in warm sunbeams by the window"
    physics_content = "Quantum entanglement and particle physics experiments"

    cat_emb = await embedder.embed(cat_content)
    physics_emb = await embedder.embed(physics_content)

    cat_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=cat_content, topic=["cats"]),
        embedding=cat_emb,
    )
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=physics_content, topic=["physics"]),
        embedding=physics_emb,
    )

    # Probe A: searching for cat content should surface the cat memory → HIT.
    await enroll_canary(
        pool, cat_mem.id, cat_content, probe_type="reaREDACTED",
    )

    # Probe B (deliberately-planted miss): also searches for cat content but
    # expects to find the physics memory — which it won't.  This is the
    # "below-cutoff probe" the done_when gate requires.
    await enroll_canary(
        pool, physics_mem.id, cat_content, probe_type="reaREDACTED",
    )

    # Run audit with top_k=1 (only the single most-similar result is returned).
    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=1
    )

    # --- Done-when assertions ---

    # 1. probes_checked > 0: the audit actually evaluated probes.
    assert result["probes_checked"] > 0, (
        "Audit must check at least one probe — a no-op meter logging rate=0 is rejected."
    )

    # 2. Below-cutoff probe was caught as a canary_miss.
    assert result["misses"] >= 1, (
        "The deliberately-mismatched probe (physics memory, cat probe_text) "
        "must be recorded as a canary_miss."
    )

    # 3. Known-surfacing probe is NOT flagged.
    assert result["misses"] < result["probes_checked"], (
        "The surfacing probe (cat memory, cat probe_text) must NOT be a miss."
    )

    # 4. Global canary.miss counter was incremented.
    total_misses = await get_counter(pool, COUNTER_CANARY_MISS)
    assert total_misses >= 1, "canary.miss counter must be incremented for each miss."

    # 5. Per-probe miss_count reflects the audit result.
    cat_row = await get_db(pool).fetchrow(
        "SELECT miss_count, audit_count FROM recall_canary WHERE memory_id = $1",
        cat_mem.id,
    )
    physics_row = await get_db(pool).fetchrow(
        "SELECT miss_count, audit_count FROM recall_canary WHERE memory_id = $1",
        physics_mem.id,
    )

    assert cat_row is not None
    assert physics_row is not None
    assert cat_row["miss_count"] == 0, "Surfacing probe must have miss_count=0."
    assert physics_row["miss_count"] == 1, "Below-cutoff probe must have miss_count=1."
    assert cat_row["audit_count"] == 1
    assert physics_row["audit_count"] == 1


# ---------------------------------------------------------------------------
# Active probing gate (RI-4: active probes are off by default)
# ---------------------------------------------------------------------------


async def test_active_probes_excluded_by_default(pool, embedder):
    """Active probes are NOT audited when active_probing_enabled=False (default).

    This verifies the RI-4 gate: active probes accumulate in the table but
    don't contribute to the canary_miss counter until explicitly enabled.
    """
    content = "Testing active probe gating"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["test"]),
        embedding=emb,
    )
    # Enroll an active probe with MISMATCHED probe_text — would be a miss if audited.
    await enroll_canary(pool, mem.id, "completely unrelated text", probe_type="active")

    # Default audit: active_probing_enabled=False.
    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )

    # No reaREDACTED probes exist, so no probes are checked.
    assert result["probes_checked"] == 0
    assert result["misses"] == 0


async def test_active_probes_included_when_flag_set(pool, embedder):
    """Active probes ARE audited when active_probing_enabled=True.

    Verifies the flag correctly enables the active probing path.
    """
    cat_content = "Cats enjoy sleeping in warm patches of sunlight"
    cat_emb = await embedder.embed(cat_content)
    cat_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=cat_content, topic=["cats"]),
        embedding=cat_emb,
    )

    physics_content = "String theory and extra dimensional compactification"
    physics_emb = await embedder.embed(physics_content)
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=physics_content, topic=["physics"]),
        embedding=physics_emb,
    )

    # Active probes: cat memory with matching probe (should hit), physics with cat probe (miss).
    await enroll_canary(pool, cat_mem.id, cat_content, probe_type="active")
    await enroll_canary(pool, physics_mem.id, cat_content, probe_type="active")

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        active_probing_enabled=True,
    )

    assert result["probes_checked"] == 2
    assert result["misses"] == 1
    assert result["miss_rate"] == 0.5


# ---------------------------------------------------------------------------
# ReaREDACTED sync
# ---------------------------------------------------------------------------


async def test_reask_bootstrap_sync_enrolls_from_is_reask_miss(pool, embedder):
    """The audit auto-enrolls is_reask_miss rows as reaREDACTED probes.

    Inserts a weft_recall_queries row with is_reask_miss=TRUE + a satisfying
    memory_id, then runs the audit.  The sync should enroll the probe and
    (if the memory exists and is retrievable) evaluate it.
    """
    content = "The satisfying memory that answered a re-ask"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["reask-test"]),
        embedding=emb,
    )

    # Simulate a recorded re-ask miss: original query → satisfying memory.
    await pool.execute(
        """
        INSERT INTO weft_recall_queries
            (query_id, query_text, tool_name, created_at,
             is_reask_miss, reask_satisfying_memory_id)
        VALUES ($1, $2, 'recall', now() - interval '5 minutes',
                TRUE, $3)
        """,
        "qid-test-reask-001",
        content[:200],  # original query approximates the memory content
        mem.id,
    )

    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )

    # The sync should have enrolled one new reaREDACTED probe.
    assert result["bootstrap_synced"] == 1

    # The audit should have checked the newly-enrolled probe.
    assert result["probes_checked"] == 1

    # A re-ask probe searching for its own content should find the memory → hit.
    assert result["misses"] == 0

    # Verify the probe was written to recall_canary.
    canary_row = await get_db(pool).fetchrow(
        "SELECT probe_type, memory_id FROM recall_canary WHERE memory_id = $1",
        mem.id,
    )
    assert canary_row is not None
    assert canary_row["probe_type"] == "reaREDACTED"


async def test_reask_bootstrap_sync_idempotent(pool, embedder):
    """Running the audit twice does not double-enroll reaREDACTED probes."""
    content = "Idempotency test memory"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["idempotency"]),
        embedding=emb,
    )

    await pool.execute(
        """
        INSERT INTO weft_recall_queries
            (query_id, query_text, tool_name, created_at,
             is_reask_miss, reask_satisfying_memory_id)
        VALUES ($1, $2, 'recall', now() - interval '2 minutes', TRUE, $3)
        """,
        "qid-idem-001",
        content[:200],
        mem.id,
    )

    result1 = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )
    result2 = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )

    # Second audit should sync 0 new probes (already enrolled).
    assert result1["bootstrap_synced"] == 1
    assert result2["bootstrap_synced"] == 0

    # Only one probe row for this memory.
    count = await get_db(pool).fetchval(
        "SELECT count(*) FROM recall_canary WHERE memory_id = $1 "
        "AND probe_type = 'reaREDACTED'",
        mem.id,
    )
    assert count == 1


# ---------------------------------------------------------------------------
# Fix #1: non-degenerate audit guard
# ---------------------------------------------------------------------------


async def test_zero_probe_audit_signals_invalid(pool, embedder):
    """A 0-probe audit must be distinguishable from a real all-pass audit.

    Done-when gate: when run_canary_audit checks 0 probes it must return
    audit_valid=False and status='skipped', NOT a healthy-looking miss_rate=0.0
    that a scheduler would silently treat as all-green.
    """
    # No probes enrolled — the audit has nothing to check.
    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )

    assert result["probes_checked"] == 0
    assert result["misses"] == 0
    assert result["audit_valid"] is False, (
        "A 0-probe audit must set audit_valid=False; "
        "miss_rate=0.0 with probes_checked=0 is NOT a valid all-pass result."
    )
    assert result["status"] == "skipped"


async def test_real_all_pass_audit_is_valid(pool, embedder):
    """A real all-pass (probes_checked > 0, misses=0) has audit_valid=True.

    Contrast with test_zero_probe_audit_signals_invalid: when probes actually
    ran and all hit, the result is genuinely healthy — audit_valid=True.
    """
    content = "The golden retriever played fetch on the sunny beach"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["dogs"]),
        embedding=emb,
    )
    await enroll_canary(pool, mem.id, content, probe_type="reaREDACTED")

    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )

    assert result["probes_checked"] == 1
    assert result["misses"] == 0
    assert result["audit_valid"] is True, (
        "A real all-pass audit (probes checked, none missed) must have audit_valid=True."
    )
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Fix #2: orphan-probe cleanup on delete/archive
# ---------------------------------------------------------------------------


async def test_orphan_probe_disabled_on_soft_delete(pool, embedder):
    """Soft-deleting (archiving) a memory must disable its canary probes.

    Rationale: an orphan probe for an archived memory always misses (the memory
    is no longer surfaced in active recall) → pollutes canary.miss and grows
    unbounded if not cleaned up.
    """
    content = "Memory that will be soft-deleted for orphan-probe test"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["orphan-test"]),
        embedding=emb,
    )
    probe_id = await enroll_canary(pool, mem.id, content, probe_type="reaREDACTED")

    # Confirm probe is enabled before deletion.
    row_before = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row_before["enabled"] is True

    # Soft-delete the memory.
    deleted = await delete_memory(pool, mem.id)
    assert deleted is True

    # Probe must now be disabled.
    row_after = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row_after["enabled"] is False, (
        "Soft-deleting a memory must set enabled=FALSE on its canary probes."
    )

    # Disabled probe must be excluded from the next audit.
    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )
    assert result["probes_checked"] == 0, (
        "The orphan probe for an archived memory must not appear in run_canary_audit."
    )
    assert result["audit_valid"] is False  # 0 probes → skipped


async def test_orphan_probe_disabled_on_hard_delete(pool, embedder):
    """Hard-deleting a memory must disable its canary probes (probe row is kept).

    The probe row is retained (for history) but flipped to enabled=FALSE so
    it cannot pollute future audit runs.
    """
    content = "Memory that will be hard-deleted for orphan-probe test"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["orphan-test-hard"]),
        embedding=emb,
    )
    probe_id = await enroll_canary(pool, mem.id, content, probe_type="reaREDACTED")

    # Hard-delete the memory.
    deleted = await delete_memory(pool, mem.id, hard=True)
    assert deleted is True

    # Probe row must still exist but be disabled.
    row = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row is not None, "Probe row must be retained after hard-delete of its memory."
    assert row["enabled"] is False, (
        "Hard-deleting a memory must set enabled=FALSE on its canary probes."
    )

    # Audit must exclude the disabled probe.
    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )
    assert result["probes_checked"] == 0
    assert result["audit_valid"] is False  # 0 probes → skipped


async def test_audit_self_heals_probe_archived_outside_delete_memory(pool, embedder):
    """REGRESSION (probe hygiene): the audit disables probes whose memory was
    archived via a path that bypasses delete_memory().

    revise/supersede, quarantine merge, and consolidation (decay + duplicate
    merge) all set memories.status directly, NEVER routing through
    delete_memory()'s probe-disable. Those orphan probes can never surface their
    now-inactive memory, so each audit counted them as misses and inflated the
    reconciliation miss-rate. run_canary_audit must self-heal them regardless of
    which path archived the memory.
    """
    content = "Memory archived by a direct status UPDATE (revise/consolidation path)"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["hygiene-test"]),
        embedding=emb,
    )
    probe_id = await enroll_canary(pool, mem.id, content, probe_type="reaREDACTED")

    # Archive the memory WITHOUT going through delete_memory() — this is exactly
    # what revise.py / quarantine.py / consolidation.py do, so the probe is left
    # enabled=TRUE (the bug).
    await pool.execute(
        "UPDATE memories SET status = 'archived', updated_at = now() WHERE id = $1",
        mem.id,
    )
    row_before = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row_before["enabled"] is True, (
        "Direct status UPDATE must leave the probe enabled — that is the bug the "
        "audit self-heal exists to correct."
    )

    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID
    )

    # The audit disabled the orphan probe and did NOT count it as a miss.
    assert result["probes_disabled"] == 1
    assert result["probes_checked"] == 0, (
        "An orphan probe for an archived memory must not be audited (no phantom miss)."
    )
    assert result["misses"] == 0
    row_after = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", probe_id,
    )
    assert row_after["enabled"] is False, (
        "run_canary_audit must soft-disable probes whose memory is no longer active."
    )

    # canary_health must now exclude the disabled probe from its aggregate.
    health = await canary_health(pool, user_id=DEFAULT_TEST_USER_ID)
    assert health is None or "reaREDACTED" not in health.get("arms", {}), (
        "The disabled orphan probe must drop out of the canary_health aggregate."
    )


async def test_audit_keeps_active_probe_and_disables_only_orphans(pool, embedder):
    """The self-heal must be surgical: an active-memory probe survives while a
    sibling orphan probe (archived memory) is disabled in the same run.
    """
    live_content = "Live memory whose probe must keep being audited"
    dead_content = "Archived memory whose probe must be disabled"
    live_emb = await embedder.embed(live_content)
    dead_emb = await embedder.embed(dead_content)
    live = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=live_content, topic=["mixed"]),
        embedding=live_emb,
    )
    dead = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=dead_content, topic=["mixed"]),
        embedding=dead_emb,
    )
    live_probe = await enroll_canary(pool, live.id, live_content, probe_type="reaREDACTED")
    await enroll_canary(pool, dead.id, dead_content, probe_type="reaREDACTED")
    await pool.execute(
        "UPDATE memories SET status = 'archived', updated_at = now() WHERE id = $1",
        dead.id,
    )

    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )

    assert result["probes_disabled"] == 1
    assert result["probes_checked"] == 1, "Only the live-memory probe should be audited."
    # The live probe survives enabled; the dead one is disabled.
    live_enabled = await get_db(pool).fetchval(
        "SELECT enabled FROM recall_canary WHERE probe_id = $1", live_probe,
    )
    assert live_enabled is True


# ---------------------------------------------------------------------------
# Bug 1 regression: reaREDACTED enrollment must set user_id explicitly
# ---------------------------------------------------------------------------


async def test_reask_bootstrap_enroll_populates_user_id_from_query(pool, embedder):
    """REGRESSION: reaREDACTED enrollment populates recall_canary.user_id
    EXPLICITLY from the source weft_recall_queries row, not from the
    app.user_id GUC default.

    In prod the audit runs via get_db(pool) with no SET LOCAL, so the
    ``nullif(current_setting('app.user_id', true), '')`` column default
    resolved to NULL and every reaREDACTED insert hit the NOT NULL
    constraint — the trustworthy probe arm stayed permanently empty.

    The source query is owned by a user DIFFERENT from the session GUC, so a
    probe that merely inherited the GUC default would carry the wrong owner;
    asserting the probe carries the QUERY's owner proves the value is sourced
    explicitly.
    """
    owner = "reaREDACTED"  # deliberately != DEFAULT_TEST_USER_ID
    content = "Canary reask owner-scoping regression memory"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["reask-owner"]),
        embedding=emb,
    )

    # Insert the source query row owned by `owner` (transaction-local GUC so
    # NOT NULL default + RLS WITH CHECK are satisfied for that owner).
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL app.user_id = '{owner}'")
            await conn.execute(
                """
                INSERT INTO weft_recall_queries
                    (query_id, query_text, tool_name, created_at,
                     is_reask_miss, reask_satisfying_memory_id)
                VALUES ($1, $2, 'recall', now() - interval '5 minutes', TRUE, $3)
                """,
                "qid-owner-001",
                content[:200],
                mem.id,
            )

    result = await run_canary_audit(
        pool, embedder, user_id=owner, top_k=5
    )
    assert result["bootstrap_synced"] == 1, "reaREDACTED probe failed to enroll"

    probe = await get_db(pool).fetchrow(
        "SELECT user_id, probe_type FROM recall_canary WHERE memory_id = $1 "
        "AND probe_type = 'reaREDACTED'",
        mem.id,
    )
    assert probe is not None
    assert probe["user_id"] is not None, "user_id must not be NULL (the bug)"
    assert probe["user_id"] == owner, (
        "probe user_id must come from the source query row, not the GUC default"
    )


# ---------------------------------------------------------------------------
# canary_health — the load-bearing primer/brief surface
# ---------------------------------------------------------------------------


async def _store_active_memory(pool, embedder, content, topic="health"):
    """Store a real, searchable memory (current active universe).

    canary_health JOINs the same universe search_by_vector uses (status +
    review_status='active' + write_provenance!='agent'), so its probes must
    reference real active memories — a bare fake memory_id no longer surfaces
    in the aggregate.
    """
    emb = await embedder.embed(content)
    return await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=[topic]),
        embedding=emb,
    )


async def _record_audit_event(pool, probe_id, *, hit, age_days=0.0):
    """Insert one recall_canary_audit event with a controllable age.

    age_days lets a test place an outcome inside or outside the trailing
    health window to exercise windowing.
    """
    await pool.execute(
        "INSERT INTO recall_canary_audit (probe_id, user_id, hit, audited_at) "
        "VALUES ($1, $2, $3, now() - make_interval(secs => $4))",
        probe_id,
        DEFAULT_TEST_USER_ID,
        hit,
        float(age_days) * 86400.0,
    )


async def test_canary_health_dark_when_never_audited(pool, embedder):
    """A meter with probes but no audit is DARK with a loud alert — the
    failure mode that let the meter sit unaudited for weeks."""
    mem = await _store_active_memory(pool, embedder, "health probe one")
    await enroll_canary(pool, mem.id, "health probe one", probe_type="active")
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    assert health is not None
    assert health["dark"] is True
    assert health["dark_reason"] == "never audited"
    assert "alert" in health and "DARK" in health["alert"]
    assert "active" in health["arms"]
    # No windowed events yet → thin sample → uncalibrated.
    assert health["arms"]["active"]["label"] == "uncalibrated"
    assert health["arms"]["active"]["checks"] == 0


async def test_canary_health_is_explicit_when_no_probes_are_enrolled(pool):
    """An empty meter must be visible as dark, not disappear as None."""
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)

    assert health["status"] == "no_probes"
    assert health["dark"] is True
    assert health["dark_reason"] == "no active probes"
    assert "DARK" in health["alert"]


async def test_canary_health_fresh_after_audit(pool, embedder):
    """A recently-audited meter is not dark and reports the windowed miss_rate."""
    mem = await _store_active_memory(pool, embedder, "health probe two")
    pid = await enroll_canary(pool, mem.id, "health probe two", probe_type="active")
    # Liveness comes from recall_canary.last_audit_at; the rate from the log.
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    await _record_audit_event(pool, pid, hit=True)
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    assert health is not None
    assert health["dark"] is False
    assert health["dark_reason"] is None
    assert "alert" not in health
    assert health["arms"]["active"]["miss_rate"] == 0.0
    assert health["arms"]["active"]["audited"] == 1
    assert health["arms"]["active"]["checks"] == 1


async def test_canary_health_dark_when_stale(pool, embedder):
    """An audit older than the stale window flips the meter back to DARK."""
    mem = await _store_active_memory(pool, embedder, "health probe three")
    pid = await enroll_canary(pool, mem.id, "health probe three", probe_type="active")
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() - interval '60 hours' "
        "WHERE probe_id = $1",
        pid,
    )
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    assert health is not None
    assert health["dark"] is True
    assert health["dark_reason"].startswith("stale")
    assert "alert" in health


async def test_canary_health_miss_rate_aggregates(pool, embedder):
    """miss_rate = windowed misses / checks across the arm's event log."""
    m1 = await _store_active_memory(pool, embedder, "probe miss content")
    m2 = await _store_active_memory(pool, embedder, "probe hit content")
    p1 = await enroll_canary(pool, m1.id, "probe miss content", probe_type="active")
    p2 = await enroll_canary(pool, m2.id, "probe hit content", probe_type="active")
    await pool.execute("UPDATE recall_canary SET last_audit_at = now()")
    await _record_audit_event(pool, p1, hit=False)
    await _record_audit_event(pool, p2, hit=True)
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    assert health is not None
    assert health["arms"]["active"]["miss_rate"] == 0.5
    assert health["arms"]["active"]["misses"] == 1
    assert health["arms"]["active"]["checks"] == 2
    assert health["arms"]["active"]["audited"] == 2


# ---------------------------------------------------------------------------
# Windowed rate — the fix: stale outcomes age out; a freshly-departed universe
# member's recent events drop via the JOIN. Together these clear the false
# tripwire that a monotonic lifetime aggregate would keep firing forever.
# ---------------------------------------------------------------------------


async def test_canary_health_windowed_rate_ages_out_old_misses(pool, embedder):
    """Old misses outside the window do NOT inflate the rate — the whole point.

    A lifetime aggregate would report ~50% here (40 misses / 80 checks) and trip
    the 10% ceiling forever. The windowed rate sees only the recent cohort.
    """
    mem = await _store_active_memory(pool, embedder, "aging probe content")
    pid = await enroll_canary(pool, mem.id, "aging probe content", probe_type="active")
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # 40 misses long ago (outside the 14d window) + 40 recent hits (inside it).
    for _ in range(40):
        await _record_audit_event(pool, pid, hit=False, age_days=20)
        await _record_audit_event(pool, pid, hit=True, age_days=1)

    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    arm = health["arms"]["active"]
    assert arm["checks"] == 40, "only in-window events count"
    assert arm["misses"] == 0
    assert arm["miss_rate"] == 0.0
    assert arm["trustworthy"] is True  # 40 >= 30-check sample gate
    assert arm["tripped"] is False, "stale misses aged out — tripwire must clear"
    assert "tripwire" not in health


async def test_canary_health_join_excludes_departed_universe_probe(pool, embedder):
    """A probe whose memory just left the searchable universe drops out — even
    with recent MISS events — instead of firing the tripwire on a guaranteed miss.

    This is the artifact that started it all (PR #31), enforced one level up at
    the health surface: pending_review memories can never surface, so their
    recent 'misses' are not recall regressions.
    """
    mem = await _store_active_memory(pool, embedder, "departing probe content")
    pid = await enroll_canary(
        pool, mem.id, "departing probe content", probe_type="active"
    )
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # Enough recent misses to trip a trustworthy arm...
    for _ in range(35):
        await _record_audit_event(pool, pid, hit=False, age_days=1)
    # ...but the memory is now pending_review (outside the search universe).
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1", mem.id
    )

    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    # Only arm was 'active' and it left the universe → nothing to report, and
    # crucially NO tripwire fired on the guaranteed-miss events.
    assert health is None or "active" not in health.get("arms", {})
    if health is not None:
        assert not health.get("tripwire")


async def test_canary_health_windowed_tripwire_fires_on_real_regression(pool, embedder):
    """The windowed rate still SCREAMS for a genuine, recent recall regression."""
    mem = await _store_active_memory(pool, embedder, "regressing probe content")
    pid = await enroll_canary(
        pool, mem.id, "regressing probe content", probe_type="active"
    )
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # 30 recent checks, 12 recent misses → 40% in-window > 10% ceiling.
    for i in range(30):
        await _record_audit_event(pool, pid, hit=(i >= 12), age_days=1)

    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    arm = health["arms"]["active"]
    assert arm["trustworthy"] is True
    assert arm["miss_rate"] == 0.4
    assert arm["tripped"] is True
    assert "tripwire" in health and "10%" in health["tripwire"]


async def test_audit_prunes_events_past_retention(pool, embedder):
    """run_canary_audit prunes recall_canary_audit rows older than the retention
    horizon, keeping the windowed-rate query bounded, while writing this run's."""
    mem = await _store_active_memory(pool, embedder, "retention probe content")
    pid = await enroll_canary(
        pool, mem.id, "retention probe content", probe_type="reaREDACTED"
    )
    # An event well past the 30d retention horizon.
    await _record_audit_event(pool, pid, hit=True, age_days=45)
    old_before = await get_db(pool).fetchval(
        "SELECT count(*) FROM recall_canary_audit WHERE audited_at < now() - "
        "interval '40 days'"
    )
    assert old_before == 1

    await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )

    old_after = await get_db(pool).fetchval(
        "SELECT count(*) FROM recall_canary_audit WHERE audited_at < now() - "
        "interval '40 days'"
    )
    assert old_after == 0, "retention prune must delete rows past the horizon"
    fresh = await get_db(pool).fetchval(
        "SELECT count(*) FROM recall_canary_audit WHERE probe_id = $1 "
        "AND audited_at > now() - interval '1 hour'",
        pid,
    )
    assert fresh == 1, "the audit must log this run's outcome as a fresh event"


async def test_audit_does_not_mutate_another_owner(pool, embedder):
    """Every maintenance and audit phase stays inside the explicit owner scope."""
    owner_b = "canary-owner-b"
    mem = await _store_active_memory(pool, embedder, "owner A live probe")
    await enroll_canary(
        pool, mem.id, "owner A live probe", probe_type="reaREDACTED"
    )
    await pool.execute(
        """
        INSERT INTO recall_canary
            (probe_id, memory_id, user_id, probe_text, probe_type,
             audit_count, miss_count, last_audit_at)
        VALUES
            ('owner-b-orphan', 'missing-owner-b-memory', $1,
             'task:owner-b-machine-id', 'reaREDACTED', 1, 1,
             now() - interval '45 days')
        """,
        owner_b,
    )
    await pool.execute(
        """
        INSERT INTO recall_canary_audit
            (probe_id, user_id, audited_at, hit)
        VALUES ('owner-b-orphan', $1, now() - interval '45 days', FALSE)
        """,
        owner_b,
    )
    await pool.execute(
        """
        INSERT INTO weft_recall_queries
            (query_id, query_text, tool_name, created_at, user_id,
             is_reask_miss, reask_satisfying_memory_id)
        VALUES
            ('owner-b-reask', 'Owner B natural-language bootstrap query',
             'recall', now() - interval '5 minutes', $1, TRUE,
             'missing-owner-b-memory')
        """,
        owner_b,
    )

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
    )

    assert result["probes_checked"] == 1
    owner_b_rows = await pool.fetch(
        "SELECT probe_id, enabled, audit_count, miss_count FROM recall_canary "
        "WHERE user_id = $1 ORDER BY probe_id",
        owner_b,
    )
    assert [dict(row) for row in owner_b_rows] == [
        {
            "probe_id": "owner-b-orphan",
            "enabled": True,
            "audit_count": 1,
            "miss_count": 1,
        }
    ]
    assert await pool.fetchval(
        "SELECT count(*) FROM recall_canary_audit WHERE user_id = $1",
        owner_b,
    ) == 1


# ---------------------------------------------------------------------------
# Calibration: audit probe universe == search_by_vector default universe
#
# search_by_vector defaults exclude review_status!='active' and
# write_provenance='agent'. enroll_canary probes every memory, so a probe whose
# memory the search can never return is a GUARANTEED miss, not a recall failure.
# The audit must skip those memories instead of counting them.
# ---------------------------------------------------------------------------


async def test_pending_review_probe_skipped_not_counted(pool, embedder):
    """A pending_review memory's active probe is skipped (transient), not missed.

    review_status='pending_review' is excluded from search_by_vector's default
    candidate pool, so the memory could NEVER surface — counting it as a miss
    inflated the reconciliation rate (10/11 real active-arm misses were this).
    The probe must be skipped AND left enabled so it re-enters once approved.
    """
    content = "Quarterly OKR review cadence moved from monthly to biweekly"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["okr"]),
        embedding=emb,
    )
    # Probe text == content, so it WOULD hit if the memory were searchable.
    pid = await enroll_canary(pool, mem.id, content, probe_type="active")
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1", mem.id
    )

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
        active_probing_enabled=True,
    )

    assert result["probes_checked"] == 0, "pending_review memory must not be probed"
    assert result["misses"] == 0
    row = await get_db(pool).fetchrow(
        "SELECT enabled, miss_count FROM recall_canary WHERE probe_id = $1", pid
    )
    assert row["enabled"] is True, "review_status is transient — do not disable the probe"
    assert row["miss_count"] == 0


async def test_pending_review_probe_reenters_after_approval(pool, embedder):
    """Once a pending_review memory becomes active, its probe is audited again."""
    content = "The migration runbook lives in docs/ops/migrations.md"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["ops"]),
        embedding=emb,
    )
    await enroll_canary(pool, mem.id, content, probe_type="active")
    await pool.execute(
        "UPDATE memories SET review_status = 'pending_review' WHERE id = $1", mem.id
    )
    skipped = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
        active_probing_enabled=True,
    )
    assert skipped["probes_checked"] == 0

    # Approve the memory.
    await pool.execute(
        "UPDATE memories SET review_status = 'active' WHERE id = $1", mem.id
    )
    audited = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
        active_probing_enabled=True,
    )
    assert audited["probes_checked"] == 1
    assert audited["misses"] == 0  # probe_text == content, surfaces at rank 1


async def test_agent_provenance_probe_skipped(pool, embedder):
    """An agent-provenance memory's probe is skipped — search excludes it by default."""
    content = "Agent-authored note about the nightly batch window"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["batch"]),
        embedding=emb,
    )
    await enroll_canary(pool, mem.id, content, probe_type="active")
    await pool.execute(
        "UPDATE memories SET write_provenance = 'agent' WHERE id = $1", mem.id
    )

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
        active_probing_enabled=True,
    )
    assert result["probes_checked"] == 0
    assert result["misses"] == 0


# ---------------------------------------------------------------------------
# Degenerate reaREDACTED probe guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,degenerate",
    [
        ("task:loom-7bedb110", True),        # colon-prefixed machine reference
        ("id:abc", True),
        ("loom-7bedb110", True),             # bare slug/hash id (digit + hyphen)
        ("ticket_42", True),                 # digit + underscore
        ("", True),                          # empty
        ("   ", True),                       # whitespace-only
        ("what did we decide about auth?", False),   # natural-language phrase
        ("authentication", False),           # single real word, no id structure
        ("the loom-7bedb110 outcome", False),        # id embedded in a phrase
        ("PostgreSQL", False),
    ],
)
def test_is_degenerate_reask_probe(text, degenerate):
    assert _is_degenerate_reask_probe(text) is degenerate


async def test_degenerate_reask_query_not_enrolled(pool, embedder):
    """A machine-reference reask query is not enrolled as a bootstrap probe."""
    content = "Per-finding tickets now lead with YAML frontmatter"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["tickets"]),
        embedding=emb,
    )
    await pool.execute(
        """
        INSERT INTO weft_recall_queries
            (query_id, query_text, tool_name, created_at,
             is_reask_miss, reask_satisfying_memory_id)
        VALUES ($1, $2, 'recall', now() - interval '5 minutes', TRUE, $3)
        """,
        "qid-degenerate-001",
        "task:loom-7bedb110",  # the exact poison observed in production
        mem.id,
    )

    result = await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )
    assert result["bootstrap_synced"] == 0
    count = await get_db(pool).fetchval(
        "SELECT count(*) FROM recall_canary WHERE probe_type = 'reaREDACTED'"
    )
    assert count == 0


async def test_existing_degenerate_reask_probe_disabled(pool, embedder):
    """An already-enrolled degenerate reaREDACTED probe self-heals to disabled."""
    content = "The satisfying memory for a poisoned probe"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["poison"]),
        embedding=emb,
    )
    # Enrolled before the guard existed: degenerate probe_text, enabled.
    await pool.execute(
        """
        INSERT INTO recall_canary (probe_id, memory_id, user_id, probe_text, probe_type)
        VALUES ($1, $2, $3, 'task:loom-7bedb110', 'reaREDACTED')
        """,
        "cp-poisoned01",
        mem.id,
        DEFAULT_TEST_USER_ID,
    )

    await run_canary_audit(
        pool, embedder, user_id=DEFAULT_TEST_USER_ID, top_k=5
    )

    row = await get_db(pool).fetchrow(
        "SELECT enabled FROM recall_canary WHERE probe_id = 'cp-poisoned01'"
    )
    assert row["enabled"] is False


# ---------------------------------------------------------------------------
# Sample-based trustworthiness + drift tripwire
# ---------------------------------------------------------------------------


async def test_canary_health_trustworthy_when_sample_sufficient(pool, embedder):
    """At/above the min WINDOWED sample with a healthy rate: trustworthy, no
    'uncalibrated', no tripwire."""
    mem = await _store_active_memory(pool, embedder, "trust probe content")
    pid = await enroll_canary(pool, mem.id, "trust probe content", probe_type="active")
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # 40 in-window checks, 1 miss => 2.5% (below the 10% ceiling), sample >= 30.
    for i in range(40):
        await _record_audit_event(pool, pid, hit=(i != 0), age_days=1)
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    active = health["arms"]["active"]
    assert active["trustworthy"] is True
    assert "label" not in active, "sufficient sample must drop the 'uncalibrated' label"
    assert active["tripped"] is False
    assert "tripwire" not in health


async def test_canary_health_tripwire_fires_on_high_miss_rate(pool, embedder):
    """A trustworthy arm whose windowed miss_rate crosses the ceiling raises a
    loud tripwire."""
    mem = await _store_active_memory(pool, embedder, "trip probe content")
    pid = await enroll_canary(pool, mem.id, "trip probe content", probe_type="active")
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # 40 in-window checks, 8 misses => 20% (above the 10% ceiling), sample >= 30.
    for i in range(40):
        await _record_audit_event(pool, pid, hit=(i >= 8), age_days=1)
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    active = health["arms"]["active"]
    assert active["trustworthy"] is True
    assert active["tripped"] is True
    assert "tripwire" in health
    assert "20.0%" in health["tripwire"]
    assert "active" in health["tripwire"]


async def test_canary_health_no_trip_below_min_sample(pool, embedder):
    """A high miss_rate on a thin WINDOWED sample must NOT trip — uncalibrated."""
    mem = await _store_active_memory(pool, embedder, "thin probe content")
    pid = await enroll_canary(pool, mem.id, "thin probe content", probe_type="active")
    await pool.execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", pid
    )
    # 5 in-window checks, 5 misses => 100% rate but sample < 30: guarded.
    for _ in range(5):
        await _record_audit_event(pool, pid, hit=False, age_days=1)
    health = await canary_health(pool, DEFAULT_TEST_USER_ID)
    active = health["arms"]["active"]
    assert active["trustworthy"] is False
    assert active["tripped"] is False
    assert active.get("label") == "uncalibrated"
    assert "tripwire" not in health
