"""Tests for auto-consolidation scheduling: metadata tracking, due checks, and primer integration."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from weft.consolidation import (
    consolidate_if_due,
    record_consolidation_run,
    should_consolidate,
)
from weft.store import get_metadata, set_metadata


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
