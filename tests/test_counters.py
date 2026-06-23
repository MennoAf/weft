"""Tests for weft/counters.py — global named telemetry counters.

Requires a real PostgreSQL instance (testcontainers via conftest.py).

Spec: Loom task loom-3c4a0be3.
"""

from __future__ import annotations

import pytest

from weft.counters import (
    COUNTER_REPLAY_ENQUEUE_FAILED,
    FAILURE_COUNTERS,
    get_counter,
    get_counters,
    increment_counter,
)

pytestmark = pytest.mark.asyncio


async def test_increment_creates_then_accumulates(pool):
    """First increment creates the row at `by`; subsequent ones accumulate."""
    name = "test.counter.alpha"

    assert await get_counter(pool, name) == 0, "unseen counter must read 0"

    await increment_counter(pool, name)
    assert await get_counter(pool, name) == 1

    await increment_counter(pool, name)
    await increment_counter(pool, name, by=3)
    assert await get_counter(pool, name) == 5


async def test_get_counters_defaults_missing_to_zero(pool):
    """get_counters returns every requested name, 0 for those never fired."""
    await increment_counter(pool, COUNTER_REPLAY_ENQUEUE_FAILED, by=2)

    counters = await get_counters(pool, FAILURE_COUNTERS)

    # Every reserved failure counter is present in the result...
    assert set(counters) == set(FAILURE_COUNTERS)
    # ...with the one we bumped reflecting its value and the rest defaulting to 0.
    assert counters[COUNTER_REPLAY_ENQUEUE_FAILED] == 2
    assert all(
        counters[name] == 0
        for name in FAILURE_COUNTERS
        if name != COUNTER_REPLAY_ENQUEUE_FAILED
    )


async def test_increment_is_best_effort_never_raises(pool, monkeypatch):
    """A DB error during increment is swallowed — the caller is never aborted.

    Counters instrument failure-handling paths; an increment that could throw
    would defeat the "must not abort the path" contract it exists to support.
    """
    import weft.counters as counters_mod

    class _Boom:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("simulated counter-write failure")

    monkeypatch.setattr(counters_mod, "get_db", lambda _pool: _Boom())

    # Must not raise despite the underlying execute blowing up.
    await increment_counter(pool, "test.counter.besteffort")
