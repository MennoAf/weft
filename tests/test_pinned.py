"""Tests for pinned memory feature."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from weft.consolidation import compute_decay_score, find_duplicates
from weft.models import Memory, MemoryCreate, MemorySource, MemoryStatus, MemoryType
from weft.primer import build_primer
from weft.store import list_memories, store_memory, update_memory


# --- Store layer ---


async def test_store_pinned_memory(pool):
    """Pinned flag is persisted and returned."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Always use uv run python -m loom",
        topic=["conventions"],
        confidence=1.0,
        pinned=True,
    ))
    assert mem.pinned is True


async def test_store_unpinned_by_default(pool):
    """Memories are not pinned by default."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Some regular memory",
        confidence=0.7,
    ))
    assert mem.pinned is False


async def test_list_memories_pinned_filter(pool):
    """list_memories(pinned=True) returns only pinned memories."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="Pinned one", pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="Regular one",
    ))

    pinned = await list_memories(pool, pinned=True)
    assert len(pinned) == 1
    assert pinned[0].content == "Pinned one"

    all_mems = await list_memories(pool)
    assert len(all_mems) == 2


async def test_update_memory_pin_toggle(pool):
    """update_memory can pin and unpin a memory."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact, content="Toggle me",
    ))
    assert mem.pinned is False

    updated = await update_memory(pool, mem.id, pinned=True)
    assert updated.pinned is True

    updated2 = await update_memory(pool, mem.id, pinned=False)
    assert updated2.pinned is False


# --- Primer ---


async def test_primer_includes_pinned_section(pool):
    """Pinned memories appear in the 'pinned' section of primer output."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Critical convention: always use snake_case",
        topic=["conventions"],
        confidence=1.0,
        pinned=True,
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Some regular fact",
        confidence=0.5,
    ))

    result = await build_primer(pool, budget_tokens=4000)
    assert len(result["pinned"]) == 1
    assert "snake_case" in result["pinned"][0]["content"]


async def test_pinned_not_duplicated_in_other_sections(pool):
    """A pinned preference doesn't appear in both pinned and preferences sections."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers dark mode",
        confidence=1.0,
        pinned=True,
    ))

    result = await build_primer(pool, budget_tokens=4000)
    all_ids = (
        [m["id"] for m in result["pinned"]]
        + [m["id"] for m in result["preferences"]]
        + [m["id"] for m in result["recent_work"]]
        + [m["id"] for m in result["relevant"]]
    )
    assert len(all_ids) == len(set(all_ids)), "Pinned memory duplicated across sections"


async def test_pinned_takes_priority_in_budget(pool):
    """Pinned memories consume budget before other sections."""
    # Create a pinned memory
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Critical convention pinned",
        pinned=True,
    ))
    # Create a preference
    await store_memory(pool, MemoryCreate(
        type=MemoryType.preference,
        content="User prefers short responses",
        confidence=1.0,
    ))

    result = await build_primer(pool, budget_tokens=4000)
    # Pinned should appear first, preferences second
    assert len(result["pinned"]) == 1
    assert "Critical convention pinned" in result["pinned"][0]["content"]
    # Budget math should add up
    assert result["total_tokens"] + result["budget_remaining"] == result["budget_tokens"]


# --- Decay protection ---


def test_pinned_memory_exempt_from_decay():
    """Pinned memories always get a decay score of 1.0."""
    old_time = datetime.now(timezone.utc) - timedelta(days=365)
    mem = Memory(
        type=MemoryType.fact,
        content="Old but pinned",
        confidence=0.1,
        accessed_at=old_time,
        access_count=0,
        pinned=True,
    )
    score = compute_decay_score(mem)
    assert score == 1.0


def test_unpinned_memory_decays_normally():
    """Unpinned memories still decay as expected."""
    old_time = datetime.now(timezone.utc) - timedelta(days=365)
    mem = Memory(
        type=MemoryType.fact,
        content="Old and unpinned",
        confidence=0.1,
        accessed_at=old_time,
        access_count=0,
        pinned=False,
    )
    score = compute_decay_score(mem)
    assert score < 1.0


# --- Dedup protection ---


async def test_dedup_keeps_pinned_memory(pool):
    """When a pinned and unpinned memory are duplicates, the pinned one is kept."""
    embedding = [0.1] * 384  # same embedding = near-duplicate

    pinned = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Important convention",
        confidence=0.5,
        pinned=True,
    ), embedding=embedding)

    unpinned = await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Important convention",
        confidence=0.9,  # higher confidence but not pinned
    ), embedding=embedding)

    merged = await find_duplicates(pool, threshold=0.95)

    if merged:
        kept_ids = {k for k, _ in merged}
        archived_ids = {a for _, a in merged}
        # Pinned should be kept even though it has lower confidence
        assert pinned.id in kept_ids or pinned.id not in archived_ids


# --- MCP tool registration ---


async def test_weft_pin_tool_registered():
    """weft_pin tool is registered."""
    from weft.mcp import tools
    assert hasattr(tools, "weft_pin")
    assert callable(getattr(tools, "weft_pin"))
