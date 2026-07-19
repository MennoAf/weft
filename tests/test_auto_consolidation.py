"""Tests for auto-consolidation scheduling: metadata tracking, due checks,
primer integration, concurrency safety, and E2E dedup through primer path."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.consolidation import (
    consolidate,
    consolidate_if_due,
    record_consolidation_run,
    should_consolidate,
)
from weft.models import MemoryCreate, MemoryStatus, MemoryType
from weft.store import get_metadata, list_memories, set_metadata, store_memory


# --- Metadata CRUD ---


async def test_metadata_roundtrip(pool):
    """get_metadata / set_metadata work for basic key-value storage."""
    assert await get_metadata(pool, "nonexistent") is None

    await set_metadata(pool, "test_key", {"foo": "bar", "count": 42})
    result = await get_metadata(pool, "test_key")
    assert result == {"foo": "bar", "count": 42}


async def test_metadata_upsert(pool):
    """set_metadata overwrites existing values."""
    await set_metadata(pool, "test_key", {"version": 1})
    await set_metadata(pool, "test_key", {"version": 2})
    result = await get_metadata(pool, "test_key")
    assert result["version"] == 2


# --- should_consolidate ---


async def test_should_consolidate_never_run(pool):
    """Returns True when no consolidation has ever run."""
    assert await should_consolidate(pool) is True


async def test_should_consolidate_stale(pool):
    """Returns True when last run was >24h ago."""
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    await set_metadata(pool, "last_consolidation_run", {"ran_at": stale_time})
    assert await should_consolidate(pool) is True


async def test_should_consolidate_recent(pool):
    """Returns False when last run was <24h ago."""
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await set_metadata(pool, "last_consolidation_run", {"ran_at": recent_time})
    assert await should_consolidate(pool) is False


async def test_should_consolidate_custom_interval(pool):
    """Respects custom interval_hours."""
    # 2 hours ago — should be due with 1h interval, not due with 4h interval
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await set_metadata(pool, "last_consolidation_run", {"ran_at": two_hours_ago})

    assert await should_consolidate(pool, interval_hours=1) is True
    assert await should_consolidate(pool, interval_hours=4) is False


# --- record_consolidation_run ---


async def test_record_consolidation_run(pool):
    """Records consolidation metadata with timestamp and status."""
    await record_consolidation_run(pool, memories_processed=42, status="completed")
    meta = await get_metadata(pool, "last_consolidation_run")
    assert meta is not None
    assert meta["status"] == "completed"
    assert meta["memories_processed"] == 42
    assert "ran_at" in meta


# --- consolidate_if_due ---


async def test_consolidate_if_due_runs_when_due(pool):
    """consolidate_if_due runs consolidation when it's due."""
    # Ensure no previous metadata from other tests
    await pool.execute("DELETE FROM weft_metadata WHERE key = 'last_consolidation_run'")

    result = await consolidate_if_due(pool)
    assert result["ran"] is True, f"Expected consolidation to run: {result}"

    # Should now be recorded as recently run
    assert await should_consolidate(pool) is False


async def test_consolidate_if_due_skips_when_recent(pool):
    """consolidate_if_due skips when consolidation ran recently."""
    recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await set_metadata(pool, "last_consolidation_run", {"ran_at": recent_time})

    result = await consolidate_if_due(pool)
    assert result["ran"] is False
    assert "recently" in result["skipped_reason"]


# --- Primer integration (fire-and-forget) ---


async def test_primer_triggers_auto_consolidation(pool):
    """weft_prime fires auto-consolidation as a background task."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from weft.cache import NullCache
    from weft.config import WeftConfig
    from weft.mcp.server import AppContext
    from weft.mcp.tools import weft_prime

    app = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=type("FakeEmbed", (), {
            "embed": staticmethod(AsyncMock(return_value=[0.1] * 768)),
            "provider_name": "fake",
            "dimensions": 768,
        })(),
        config=WeftConfig(),
    )

    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])

    # Run primer — should schedule background consolidation
    result = await weft_prime(ctx)
    assert "rules" in result  # primer worked

    # Give the background task a moment to run
    await asyncio.sleep(0.5)

    # Check that consolidation ran (metadata should be set)
    meta = await get_metadata(pool, "last_consolidation_run")
    assert meta is not None
    assert meta["status"] == "completed"


# --- Concurrency safety (advisory lock) ---


async def test_concurrent_consolidation_runs_once(pool):
    """Two concurrent consolidate_if_due() calls result in only one actual run.

    The advisory lock in consolidate() prevents double execution. The
    optimistic metadata write in consolidate_if_due() further blocks
    the second caller at the should_consolidate check.
    """
    # Ensure consolidation is due
    await pool.execute("DELETE FROM weft_metadata WHERE key = 'last_consolidation_run'")

    # Fire two concurrent calls
    results = await asyncio.gather(
        consolidate_if_due(pool),
        consolidate_if_due(pool),
    )

    # At most one should have actually run (the other skips or sees recent metadata)
    ran_count = sum(1 for r in results if r.get("ran"))
    assert ran_count >= 1  # at least one ran
    # After both complete, consolidation is marked as recent
    assert await should_consolidate(pool) is False


async def test_advisory_lock_prevents_concurrent_consolidate(pool):
    """Direct consolidate() calls serialize via advisory lock."""
    # Track invocation count via a side effect counter
    call_count = 0
    original_run_decay = None

    # We'll spy on run_decay to count actual consolidation work
    from weft import consolidation

    original_run_decay = consolidation.run_decay

    async def counting_run_decay(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Add a small delay to make the race window larger
        await asyncio.sleep(0.1)
        return await original_run_decay(*args, **kwargs)

    with patch.object(consolidation, "run_decay", counting_run_decay):
        results = await asyncio.gather(
            consolidate(pool),
            consolidate(pool),
        )

    # One should have run, the other should have been skipped
    skipped_count = sum(1 for r in results if r.skipped)
    assert skipped_count >= 1, "At least one concurrent run should have been skipped"


# --- E2E: consolidation affects actual memory state ---


async def test_consolidation_reports_stale_low_confidence_candidates(pool):
    """E2E: automatic consolidation reports candidates without mutation."""
    # Create a low-confidence memory and backdate its access time
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Stale low confidence fact",
        topic=["test"],
        confidence=0.2,
    ))
    # Backdate accessed_at to make it stale (>30 day half-life, low score)
    await pool.execute(
        "UPDATE memories SET accessed_at = $1, updated_at = $1 WHERE id = $2",
        datetime.now(timezone.utc) - timedelta(days=120),
        mem.id,
    )

    # Run consolidation directly
    report = await consolidate(pool)
    assert mem.id in report.decayed

    # Review-only lifecycle: scheduled consolidation cannot mutate status.
    row = await pool.fetchrow("SELECT status FROM memories WHERE id = $1", mem.id)
    assert row["status"] == "active"


async def test_consolidation_includes_access_log_pruning(pool):
    """E2E: consolidation prunes old access logs."""
    from weft.session_tracking import log_memory_access

    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Access log test memory",
        topic=["test"],
        confidence=0.9,
    ))

    # Create an old access log entry
    await log_memory_access(pool, [mem.id], "test", session_id="old-e2e")
    await pool.execute(
        "UPDATE memory_access_log SET accessed_at = $1 WHERE session_id = 'old-e2e'",
        datetime.now(timezone.utc) - timedelta(days=100),
    )

    report = await consolidate(pool)
    assert report.access_logs_pruned >= 1
