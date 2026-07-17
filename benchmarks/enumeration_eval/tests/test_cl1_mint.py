"""CL1 compounding loop — canary/reask miss auto-mints an eval case.

Done-when gates (loom-add4d5c8):
  1. Forcing ONE canary miss appends EXACTLY ONE eval case to the JSONL store,
     referencing the missed memory_id and source='canary'.
  2. Re-running the harness via run_minted_case_eval EXERCISES that minted case
     (returns a result row for it).
  3. DEAD TELL: if miss count rises but eval-case count stays flat, the test
     FAILS — proving the mint is genuinely wired, not a no-op.
  4. Idempotency: running two audits of the same miss mints exactly ONE case.
  5. is_reask_miss events mint eval cases via bootstrap sync (source='reask').

House rules: synthetic persona "Jim Boblaw" is not used here because these tests
do not exercise the seeded enumeration collections; they exercise the CL1 signal
path directly with test-user-default memories.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from benchmarks.enumeration_eval.mint import load_minted_cases, mint_eval_case
from benchmarks.enumeration_eval.harness import run_minted_case_eval
from weft.canary import enroll_canary, run_canary_audit
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.store import store_memory


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder():
    """Local FastEmbed provider — deterministic ONNX, no API key."""
    return get_provider("fastembed")


@pytest.fixture
def store_path(tmp_path) -> Path:
    """Isolated JSONL store path per test — avoids polluting minted_cases.jsonl."""
    return tmp_path / "minted_cases.jsonl"


# Semantically distant memories used across multiple tests.
CAT_CONTENT = "The orange cat sleeps in warm sunbeams by the window"
PHYSICS_CONTENT = "Quantum entanglement and particle physics experiments"
DEFAULT_TEST_USER_ID = "test-user-default"


# ---------------------------------------------------------------------------
# Test 1 (core): one miss → one eval case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_miss_mints_one_eval_case(pool, embedder, store_path):
    """Forcing ONE canary miss appends EXACTLY ONE eval case to the JSONL store.

    Probe setup:
      - cat_mem  + cat probe_text  → surfaces at top_k=1 → HIT  (no case minted)
      - physics_mem + cat probe_text → does NOT surface → MISS (one case minted)

    Result: exactly 1 new eval case with satisfying_memory_id == physics_mem.id
    and source == 'canary'.
    """
    cat_emb = await embedder.embed(CAT_CONTENT)
    physics_emb = await embedder.embed(PHYSICS_CONTENT)

    cat_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=CAT_CONTENT, topic=["cats"]),
        embedding=cat_emb,
    )
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=PHYSICS_CONTENT, topic=["physics"]),
        embedding=physics_emb,
    )

    # Hit probe: searching cat content for the cat memory → should surface → HIT.
    await enroll_canary(pool, cat_mem.id, CAT_CONTENT, probe_type="reaREDACTED")

    # Miss probe: searching cat content but expecting physics memory → MISS.
    await enroll_canary(pool, physics_mem.id, CAT_CONTENT, probe_type="reaREDACTED")

    assert len(load_minted_cases(store_path)) == 0, "Pre-condition: store is empty"

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        eval_case_store_path=store_path,
    )

    assert result["misses"] == 1, (
        f"Expected exactly 1 miss (physics probe), got {result['misses']}"
    )

    cases = load_minted_cases(store_path)
    assert len(cases) == 1, (
        f"Expected 1 minted eval case (one per miss), got {len(cases)}"
    )
    assert cases[0]["satisfying_memory_id"] == physics_mem.id, (
        "Minted case must reference the MISSED memory (physics_mem)"
    )
    assert cases[0]["source"] == "canary"
    assert cases[0]["query"] == CAT_CONTENT[:512]
    assert "minted_at" in cases[0]


# ---------------------------------------------------------------------------
# Test 2: harness exercises the minted case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_exercises_minted_case(pool, embedder, store_path):
    """Re-running run_minted_case_eval exercises (returns a result for) the minted case.

    The harness must read the JSONL and return a result row for each minted case.
    The row tells whether the recall search now hits or still misses — that's the
    'Feedback' leg of the CL1 loop.  Here we only assert the case is EXERCISED
    (a result exists), not that it passes (it's still a miss in the same DB).
    """
    cat_emb = await embedder.embed(CAT_CONTENT)
    physics_emb = await embedder.embed(PHYSICS_CONTENT)

    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=CAT_CONTENT, topic=["cats"]),
        embedding=cat_emb,
    )
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=PHYSICS_CONTENT, topic=["physics"]),
        embedding=physics_emb,
    )

    await enroll_canary(pool, physics_mem.id, CAT_CONTENT, probe_type="reaREDACTED")

    # Force the miss → mint the eval case.
    await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        eval_case_store_path=store_path,
    )
    assert len(load_minted_cases(store_path)) == 1, "Pre-condition: exactly 1 minted case"

    # Exercise the minted cases via the harness.
    # Use the pool's default user so the search finds the memories we stored.
    exercise_results = await run_minted_case_eval(
        pool,
        user_id="test-user-default",
        path=store_path,
        embedder=embedder,
        top_k=10,
    )

    assert len(exercise_results) == 1, (
        f"Harness must return one result per minted case; expected 1, got {len(exercise_results)}"
    )
    assert exercise_results[0]["satisfying_memory_id"] == physics_mem.id
    assert "hit" in exercise_results[0], "Result must carry 'hit' boolean"


# ---------------------------------------------------------------------------
# Test 3 (dead-tell): miss rises + eval count stays flat → FAIL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dead_tell_miss_must_grow_eval_case_count(pool, embedder, store_path):
    """DEAD TELL: if misses > 0 but eval-case count did not grow, this test fails.

    This is the compulsory wiring proof: if run_canary_audit stops calling
    _try_mint_eval_case, `misses` will be non-zero but `final_count` stays at
    `initial_count`.  The assertion below catches exactly that regression.

    Steps:
      1. Record the initial eval-case count.
      2. Force a guaranteed miss (physics probe with cat query, top_k=1).
      3. Assert that EVERY miss produced a new case (final > initial).

    If the assertion fires, the failure message tells you exactly what broke:
    "DEAD TELL: N miss(es) detected but eval case count stayed at M."
    """
    cat_emb = await embedder.embed(CAT_CONTENT)
    physics_emb = await embedder.embed(PHYSICS_CONTENT)

    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=CAT_CONTENT, topic=["cats"]),
        embedding=cat_emb,
    )
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=PHYSICS_CONTENT, topic=["physics"]),
        embedding=physics_emb,
    )

    # Guaranteed miss: probe expects physics_mem but query is about cats.
    await enroll_canary(pool, physics_mem.id, CAT_CONTENT, probe_type="reaREDACTED")

    initial_count = len(load_minted_cases(store_path))

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        eval_case_store_path=store_path,
    )

    misses = result["misses"]
    final_count = len(load_minted_cases(store_path))

    # DEAD TELL ASSERTION — if this fires, minting is broken / a no-op.
    if misses > 0:
        assert final_count > initial_count, (
            f"DEAD TELL: {misses} miss(es) detected but eval case count stayed "
            f"at {initial_count}. "
            "Minting is not wired — every canary miss must append a new eval case "
            "to the JSONL store. Check that run_canary_audit calls "
            "_try_mint_eval_case on the miss branch."
        )

    # Additionally assert the exact numbers for this test's single-miss setup.
    assert misses == 1
    assert final_count == initial_count + 1


# ---------------------------------------------------------------------------
# Test 4: idempotency — same miss, two audit runs → one case only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mint_is_idempotent(pool, embedder, store_path):
    """Running two audits with the same miss mints exactly ONE eval case.

    Repeated audits of the same (query, satisfying_memory_id) pair must not
    append duplicate rows to the JSONL store.
    """
    cat_emb = await embedder.embed(CAT_CONTENT)
    physics_emb = await embedder.embed(PHYSICS_CONTENT)

    await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=CAT_CONTENT, topic=["cats"]),
        embedding=cat_emb,
    )
    physics_mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=PHYSICS_CONTENT, topic=["physics"]),
        embedding=physics_emb,
    )

    await enroll_canary(pool, physics_mem.id, CAT_CONTENT, probe_type="reaREDACTED")

    # First audit: 1 miss → 1 case minted.
    result1 = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        eval_case_store_path=store_path,
    )
    assert result1["misses"] == 1
    assert len(load_minted_cases(store_path)) == 1

    # Second audit: same probe misses again → still exactly 1 case (no duplicate).
    result2 = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=1,
        eval_case_store_path=store_path,
    )
    assert result2["misses"] == 1
    cases = load_minted_cases(store_path)
    assert len(cases) == 1, (
        "Re-running the same miss must NOT append a duplicate eval case. "
        f"Expected 1, got {len(cases)}"
    )


# ---------------------------------------------------------------------------
# Test 5: is_reask_miss → eval case via bootstrap sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reask_miss_mints_eval_case(pool, embedder, store_path):
    """An is_reask_miss event triggers eval case minting during bootstrap sync.

    _sync_reask_bootstrap_probes, called at the start of run_canary_audit,
    enrolls new reask probes from weft_recall_queries.  For each new probe
    enrolled it must also mint an eval case with source='reask'.
    """
    content = "The satisfying memory that answered a re-ask query"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["reaREDACTED"]),
        embedding=emb,
    )

    # Simulate a re-ask miss event in the query log.
    await pool.execute(
        """
        INSERT INTO weft_recall_queries
            (query_id, query_text, tool_name, created_at,
             is_reask_miss, reask_satisfying_memory_id)
        VALUES ($1, $2, 'recall', now() - interval '5 minutes', TRUE, $3)
        """,
        "qid-cl1-reaREDACTED",
        "original query text that was re-asked and answered",
        mem.id,
    )

    assert len(load_minted_cases(store_path)) == 0

    result = await run_canary_audit(
        pool,
        embedder,
        user_id=DEFAULT_TEST_USER_ID,
        top_k=5,
        eval_case_store_path=store_path,
    )

    assert result["bootstrap_synced"] == 1, (
        "Audit must have enrolled 1 new reaREDACTED probe"
    )

    cases = load_minted_cases(store_path)
    assert len(cases) == 1, (
        f"Expected 1 reask eval case minted during bootstrap sync, got {len(cases)}"
    )
    assert cases[0]["satisfying_memory_id"] == mem.id
    assert cases[0]["source"] == "reask"


# ---------------------------------------------------------------------------
# Unit test: mint_eval_case in isolation
# ---------------------------------------------------------------------------


def test_mint_eval_case_dedup(tmp_path):
    """mint_eval_case is idempotent on (query, satisfying_memory_id)."""
    p = tmp_path / "cases.jsonl"

    added1 = mint_eval_case("what does Jim eat?", "mem-001", "canary", path=p)
    assert added1 is True
    assert len(load_minted_cases(p)) == 1

    # Same pair → no-op.
    added2 = mint_eval_case("what does Jim eat?", "mem-001", "canary", path=p)
    assert added2 is False
    assert len(load_minted_cases(p)) == 1

    # Different memory_id → new case.
    added3 = mint_eval_case("what does Jim eat?", "mem-002", "reask", path=p)
    assert added3 is True
    assert len(load_minted_cases(p)) == 2


def test_mint_eval_case_query_truncation(tmp_path):
    """Query exceeding 512 chars is stored truncated — dedup uses the truncated form."""
    p = tmp_path / "cases.jsonl"
    long_query = "x" * 600

    mint_eval_case(long_query, "mem-trunc", "canary", path=p)
    cases = load_minted_cases(p)
    assert len(cases) == 1
    assert len(cases[0]["query"]) == 512

    # Re-minting with the same long query (dedup on truncated form) → no-op.
    added = mint_eval_case(long_query, "mem-trunc", "canary", path=p)
    assert added is False
    assert len(load_minted_cases(p)) == 1


def test_load_minted_cases_missing_file(tmp_path):
    """load_minted_cases returns empty list when the file does not exist."""
    p = tmp_path / "nonexistent.jsonl"
    assert load_minted_cases(p) == []
