"""Tests for the recall-canary daily scheduler loop (weft.scheduler).

Covers the wiring between the always-on background loop and
``weft.canary.run_canary_audit``:

  * ``_canary_audit_due`` — the restart-safe "is a daily audit due?" gate.
  * ``canary_audit_loop`` — discovers owners and runs explicitly scoped audits
    without relying on ``WEFT_DEFAULT_USER_ID``.

The loop must pass an explicit ``user_id`` because its service connection may
bypass RLS and must never mix owners' probe and retrieval universes.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import DEFAULT_TEST_USER_ID
from weft.auth import current_user_id
from weft.canary import enroll_canary
from weft.db.connection import acquire, get_db
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.scheduler import (
    CanaryAuditRuntimeState,
    _canary_audit_due,
    _run_canary_audit_pass,
    canary_audit_loop,
)
from weft.store import store_memory


@pytest.fixture
def embedder():
    """Local FastEmbed provider — deterministic ONNX, no API key."""
    return get_provider("fastembed")


@pytest.fixture(autouse=True)
async def clean_canary(pool):
    """Truncate recall_canary (+ its v66 audit event log) before each test."""
    await pool.execute("TRUNCATE recall_canary, recall_canary_audit CASCADE")
    yield


# ---------------------------------------------------------------------------
# _canary_audit_due — the restart-safe daily gate
# ---------------------------------------------------------------------------


async def test_due_true_when_no_probes(pool):
    """No probes at all → max(last_audit_at) is NULL → due (fail-open first run)."""
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is True


async def test_due_true_when_never_audited(pool):
    """An enrolled-but-never-audited probe has last_audit_at NULL → due."""
    await enroll_canary(pool, "mem-never", "some probe text", probe_type="reaREDACTED")
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is True


async def test_not_due_right_after_audit(pool):
    """A probe audited 'now' is inside the min-age window → NOT due."""
    probe_id = await enroll_canary(pool, "mem-fresh", "probe", probe_type="reaREDACTED")
    await get_db(pool).execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", probe_id
    )
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is False


async def test_due_again_after_min_age(pool):
    """A probe last audited > min_age_hours ago is due again (restart-safe cadence)."""
    probe_id = await enroll_canary(pool, "mem-old", "probe", probe_type="reaREDACTED")
    await get_db(pool).execute(
        "UPDATE recall_canary SET last_audit_at = now() - interval '25 hours' "
        "WHERE probe_id = $1",
        probe_id,
    )
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is True
    # And a tighter window would consider it not-yet-due.
    assert (
        await _canary_audit_due(pool, DEFAULT_TEST_USER_ID, min_age_hours=48) is False
    )


# ---------------------------------------------------------------------------
# canary_audit_loop — wiring
# ---------------------------------------------------------------------------


async def test_loop_runs_without_default_user(pool, embedder, monkeypatch):
    """Per-user fan-out no longer depends on WEFT_DEFAULT_USER_ID."""
    monkeypatch.delenv("WEFT_DEFAULT_USER_ID", raising=False)
    task = asyncio.create_task(canary_audit_loop(pool, embedder, interval=3600))
    try:
        await asyncio.sleep(0.1)
        assert not task.done(), "scheduler unexpectedly stopped without owner env"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_loop_records_failure_and_clears_it_after_recovery(
    pool, embedder, monkeypatch
):
    """A live loop exposes its last exception and clears it after recovery."""
    attempts = 0

    async def fail_then_recover(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("production-like RLS failure")
        return {
            "owners_considered": 1,
            "owners_audited": 1,
            "probes_checked": 1,
            "misses": 0,
        }

    monkeypatch.setattr(
        "weft.scheduler._run_canary_audit_pass", fail_then_recover
    )
    state = CanaryAuditRuntimeState()
    task = asyncio.create_task(
        canary_audit_loop(pool, embedder, interval=0.05, runtime_state=state)
    )
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            if state.consecutive_failures == 1:
                break
        assert state.last_attempt_at is not None
        assert state.last_success_at is None
        assert state.consecutive_failures == 1
        assert state.last_exception == "RuntimeError: production-like RLS failure"
        assert not task.done(), "a failed pass must not kill the retry loop"

        for _ in range(20):
            await asyncio.sleep(0.01)
            if state.last_success_at is not None:
                break
        assert state.last_success_at is not None
        assert state.consecutive_failures == 0
        assert state.last_exception is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def _store_probe_for_user(pool, embedder, user_id: str, content: str) -> str:
    """Create one self-retrieving active probe under an explicit owner."""
    token = current_user_id.set(user_id)
    try:
        async with acquire(pool):
            embedding = await embedder.embed(content)
            memory = await store_memory(
                pool,
                MemoryCreate(type=MemoryType.fact, content=content, topic=[user_id]),
                embedding=embedding,
            )
            return await enroll_canary(
                pool, memory.id, content, probe_type="active"
            )
    finally:
        current_user_id.reset(token)


async def test_configured_owner_bypasses_empty_rls_discovery(
    pool, embedder, monkeypatch
):
    """Hosted scheduler audits its configured owner even if discovery sees none."""
    owner = "configured-canary-owner"
    probe_id = await _store_probe_for_user(
        pool, embedder, owner, "The configured owner remembers the silver orchard"
    )
    monkeypatch.setenv("WEFT_DEFAULT_USER_ID", owner)

    async def empty_discovery(_pool):
        return []

    monkeypatch.setattr("weft.canary.list_canary_user_ids", empty_discovery)
    summary = await _run_canary_audit_pass(pool, embedder, min_age_hours=0)

    assert summary == {
        "owners_considered": 1,
        "owners_audited": 1,
        "probes_checked": 1,
        "misses": 0,
    }
    row = await pool.fetchrow(
        "SELECT audit_count, last_audit_at FROM recall_canary WHERE probe_id = $1",
        probe_id,
    )
    assert row["audit_count"] == 1
    assert row["last_audit_at"] is not None


async def test_pass_fans_out_without_cross_owner_misses(pool, embedder):
    """REGRESSION: a BYPASSRLS scheduler must search each probe in its owner corpus."""
    owner_a = "canary-owner-a"
    owner_b = "canary-owner-b"
    probe_a = await _store_probe_for_user(
        pool, embedder, owner_a, "Owner A remembers the amber lighthouse"
    )
    probe_b = await _store_probe_for_user(
        pool, embedder, owner_b, "Owner B remembers the cobalt observatory"
    )

    summary = await _run_canary_audit_pass(pool, embedder, min_age_hours=0)

    assert summary == {
        "owners_considered": 2,
        "owners_audited": 2,
        "probes_checked": 2,
        "misses": 0,
    }
    rows = await pool.fetch(
        "SELECT probe_id, audit_count, miss_count FROM recall_canary "
        "WHERE probe_id = ANY($1::text[]) ORDER BY probe_id",
        [probe_a, probe_b],
    )
    assert [(row["audit_count"], row["miss_count"]) for row in rows] == [
        (1, 0),
        (1, 0),
    ]


async def test_loop_runs_audit_when_due(pool, embedder, monkeypatch):
    """End-to-end: the loop runs an audit on first tick and stamps audit_count."""
    monkeypatch.setenv("WEFT_DEFAULT_USER_ID", DEFAULT_TEST_USER_ID)

    content = "The orange cat sleeps in warm sunbeams by the window"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["cats"]),
        embedding=emb,
    )
    # A probe that surfaces its own memory → a hit, but audit_count still ticks.
    probe_id = await enroll_canary(
        pool, mem.id, content, probe_type="reaREDACTED"
    )

    # Short poll interval so the post-audit sleep doesn't stall teardown.
    task = asyncio.create_task(canary_audit_loop(pool, embedder, interval=3600))
    try:
        # First iteration runs immediately (probe never audited → due).
        for _ in range(50):
            await asyncio.sleep(0.05)
            row = await get_db(pool).fetchrow(
                "SELECT audit_count, miss_count FROM recall_canary WHERE probe_id = $1",
                probe_id,
            )
            if row["audit_count"] > 0:
                break
        assert row["audit_count"] == 1, "loop did not run the canary audit"
        assert row["miss_count"] == 0, "self-probe should hit, not miss"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_loop_skips_when_not_due(pool, embedder, monkeypatch):
    """If a recent audit already ran, the loop's first tick does NOT re-audit.

    Guards the restart-safe property: a server restart must not re-run the
    daily audit and double-count the per-probe counters.
    """
    monkeypatch.setenv("WEFT_DEFAULT_USER_ID", DEFAULT_TEST_USER_ID)

    content = "Quantum entanglement and particle physics experiments"
    emb = await embedder.embed(content)
    mem = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=content, topic=["physics"]),
        embedding=emb,
    )
    probe_id = await enroll_canary(
        pool, mem.id, content, probe_type="reaREDACTED"
    )
    # Mark as already audited just now → not due.
    await get_db(pool).execute(
        "UPDATE recall_canary SET last_audit_at = now(), audit_count = 1 "
        "WHERE probe_id = $1",
        probe_id,
    )

    task = asyncio.create_task(canary_audit_loop(pool, embedder, interval=3600))
    try:
        # Give the loop a few ticks; audit_count must stay at 1 (no re-run).
        for _ in range(10):
            await asyncio.sleep(0.05)
        row = await get_db(pool).fetchrow(
            "SELECT audit_count FROM recall_canary WHERE probe_id = $1", probe_id
        )
        assert row["audit_count"] == 1, "loop re-audited despite a recent run"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
