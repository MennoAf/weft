"""Tests for weft.primer_sections.rules — rules section builder."""

from __future__ import annotations

import pytest

from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.primer import build_primer
from weft.primer_sections.context import PrimerContext
from weft.primer_sections.rules import build_rules_section
from weft.store import store_memory


def _make_ctx(pool, *, project_id=None, budget_tokens=2400):
    return PrimerContext(
        user_id="test-user", project_id=project_id, agent_id=None,
        pool=pool, budget_tokens=budget_tokens, query=None, query_vec=None,
        disclosure="full", mode=None,
    )


async def test_empty_when_no_pinned(pool):
    """No pinned memories → empty section."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers dark mode",
        confidence=1.0,
        pinned=False,
    ))

    ctx = _make_ctx(pool)
    result = await build_rules_section(ctx)

    assert result.items == []
    assert result.tokens_used == 0


async def test_returns_pinned_memories(pool):
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Always use pytest for testing.",
        confidence=1.0,
        pinned=True,
    ))

    ctx = _make_ctx(pool)
    result = await build_rules_section(ctx)

    assert len(result.items) == 1
    assert "pytest" in result.items[0]["content"]
    assert result.items[0]["pinned"] is True
    assert result.tokens_used > 0


async def test_updates_seen_ids(pool):
    """Rule memories are added to seen_ids to prevent duplicates."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Never skip tests.",
        confidence=1.0,
        pinned=True,
    ))

    ctx = _make_ctx(pool)
    await build_rules_section(ctx)

    assert mem.id in ctx.seen_ids


async def test_review_after_annotation(pool):
    """review_after field is annotated with review_due boolean."""
    from datetime import datetime, timedelta, timezone

    past = datetime.now(timezone.utc) - timedelta(days=30)
    await store_memory(pool, MemoryCreate(
        type=MemoryType.decision,
        content="Use asyncpg for all DB access.",
        confidence=0.9,
        pinned=True,
        review_after=past,
    ))

    ctx = _make_ctx(pool)
    result = await build_rules_section(ctx)

    assert len(result.items) == 1
    assert result.items[0]["review_due"] is True


async def test_sorted_by_confidence_desc(pool):
    """Higher confidence rules appear first."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Low confidence rule.",
        confidence=0.5,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="High confidence rule.",
        confidence=1.0,
        pinned=True,
    ))

    ctx = _make_ctx(pool)
    result = await build_rules_section(ctx)

    assert len(result.items) == 2
    assert result.items[0]["confidence"] >= result.items[1]["confidence"]


async def test_matches_monolithic_primer(pool):
    """Rules from section builder match the monolithic build_primer."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Weft uses pgvector for search.",
        confidence=1.0,
        pinned=True,
    ))

    mono = await build_primer(pool, budget_tokens=2400, disclosure="full")
    ctx = _make_ctx(pool, budget_tokens=2400)
    result = await build_rules_section(ctx)

    assert len(result.items) == len(mono["rules"])
    for mono_item, sect_item in zip(mono["rules"], result.items):
        assert mono_item["id"] == sect_item["id"]
        assert mono_item["content"] == sect_item["content"]
    assert result.tokens_used == mono["section_tokens"]["rules"]
