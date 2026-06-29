"""Tests for the recall-canary daily scheduler loop (weft.scheduler).

Covers the wiring between the always-on background loop and
``weft.canary.run_canary_audit``:

  * ``_canary_audit_due`` — the restart-safe "is a daily audit due?" gate.
  * ``canary_audit_loop`` — disabled without ``WEFT_DEFAULT_USER_ID``; runs
    an audit (incrementing per-probe ``audit_count``) when enabled.

The loop must pass an explicit ``user_id`` because background tasks carry no
HTTP middleware, so ``run_canary_audit``'s guard would otherwise skip.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import DEFAULT_TEST_USER_ID
from weft.canary import enroll_canary
from weft.db.connection import get_db
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.scheduler import _canary_audit_due, canary_audit_loop
from weft.store import store_memory


@pytest.fixture
def embedder():
    """Local FastEmbed provider — deterministic ONNX, no API key."""
    return get_provider("fastembed")


@pytest.fixture(autouse=True)
async def clean_canary(pool):
    """Truncate recall_canary before each test (v63 table)."""
    await pool.execute("TRUNCATE recall_canary")
    yield


# ---------------------------------------------------------------------------
# _canary_audit_due — the restart-safe daily gate
# ---------------------------------------------------------------------------


async def test_due_true_when_no_probes(pool):
    """No probes at all → max(last_audit_at) is NULL → due (fail-open first run)."""
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is True


async def test_due_true_when_never_audited(pool):
    """An enrolled-but-never-audited probe has last_audit_at NULL → due."""
    await enroll_canary(pool, "mem-never", "some probe text", probe_type="reask-bootstrap")
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is True


async def test_not_due_right_after_audit(pool):
    """A probe audited 'now' is inside the min-age window → NOT due."""
    probe_id = await enroll_canary(pool, "mem-fresh", "probe", probe_type="reask-bootstrap")
    await get_db(pool).execute(
        "UPDATE recall_canary SET last_audit_at = now() WHERE probe_id = $1", probe_id
    )
    assert await _canary_audit_due(pool, DEFAULT_TEST_USER_ID) is False


async def test_due_again_after_min_age(pool):
    """A probe last audited > min_age_hours ago is due again (restart-safe cadence)."""
    probe_id = await enroll_canary(pool, "mem-old", "probe", probe_type="reask-bootstrap")
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


async def test_loop_disabled_without_default_user(pool, embedder, monkeypatch):
    """Without WEFT_DEFAULT_USER_ID the loop returns immediately (no infinite loop).

    If this regressed to an infinite loop, the test would hang — so the await
    completing IS the assertion. asyncio.wait_for guards against a hang.
    """
    monkeypatch.delenv("WEFT_DEFAULT_USER_ID", raising=False)
    await asyncio.wait_for(canary_audit_loop(pool, embedder), timeout=5)


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
        pool, mem.id, content, probe_type="reask-bootstrap"
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
        pool, mem.id, content, probe_type="reask-bootstrap"
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
