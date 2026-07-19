"""Phase 3 end-to-end tests — relationships, consolidation, proactive contradiction.

Realistic scenarios that exercise the full Phase 3 feature set working together.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.consolidation import (
    ConsolidationConfig,
    DecayConfig,
    check_contradictions_on_store,
    consolidate,
)
from weft.embeddings import get_provider
from weft.models import (
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    RelationType,
)
from weft.store import (
    add_relationship,
    get_memory,
    get_relationships,
    list_memories,
    remove_relationship,
    store_memory,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


# ---------------------------------------------------------------------------
# 1. Contradiction auto-detection on store
# ---------------------------------------------------------------------------


async def test_contradicting_memories_auto_detected(pool, provider):
    """When storing a memory that contradicts an existing one,
    check_contradictions_on_store should return warnings and create
    a contradicts relationship."""
    # Store a fact
    content_a = "Weft uses Redis for caching frequently accessed data in the application layer"
    emb_a = await provider.embed(content_a)
    mem_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            topic=["infrastructure"],
            confidence=0.9,
        ),
        embedding=emb_a,
    )

    # Store a contradicting fact
    content_b = "Weft does not use Redis for caching frequently accessed data in the application layer"
    emb_b = await provider.embed(content_b)
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            topic=["infrastructure"],
            confidence=0.6,
        ),
        embedding=emb_b,
    )

    # Run proactive contradiction check
    warnings = await check_contradictions_on_store(pool, mem_b.id, emb_b)
    assert len(warnings) >= 1
    assert any(w["memory_id"] == mem_a.id for w in warnings)

    # Verify the contradicts relationship was auto-created
    rels = await get_relationships(pool, mem_b.id, relation=RelationType.contradicts)
    assert len(rels) >= 1
    assert any(r.target_id == mem_a.id for r in rels)


# ---------------------------------------------------------------------------
# 2. Decay archives stale memories while preserving immortals
# ---------------------------------------------------------------------------


async def test_decay_archives_stale_preserves_immortals(pool, provider):
    """Full consolidation should decay old low-confidence facts but
    preserve preferences and user_model memories regardless of age."""
    # A stale fact — should be decayed
    stale_fact = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Some old fact that is no longer relevant to anyone",
            confidence=0.15,
            topic=["stale"],
        ),
        embedding=await provider.embed("Some old fact that is no longer relevant to anyone"),
    )
    old_dt = datetime.now(timezone.utc) - timedelta(days=180)
    await pool.execute(
        "UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, stale_fact.id
    )

    # An old preference — should survive
    old_pref = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.preference,
            content="Always use dark mode in the IDE",
            confidence=0.5,
            topic=["workflow"],
        ),
        embedding=await provider.embed("Always use dark mode in the IDE"),
    )
    await pool.execute(
        "UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, old_pref.id
    )

    # An old user_model — should survive
    old_um = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.user_model,
            content="User prefers concise responses over detailed explanations",
            confidence=0.4,
            topic=["communication"],
        ),
        embedding=await provider.embed("User prefers concise responses over detailed explanations"),
    )
    await pool.execute(
        "UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, old_um.id
    )

    # Run full consolidation
    report = await consolidate(pool)

    # Stale fact should be reported as a review candidate.
    assert stale_fact.id in report.decayed

    # Preference and user_model should NOT be decayed
    assert old_pref.id not in report.decayed
    assert old_um.id not in report.decayed

    # Automatic consolidation is review-only and cannot change status.
    assert (await get_memory(pool, stale_fact.id)).status == MemoryStatus.active
    assert (await get_memory(pool, old_pref.id)).status == MemoryStatus.active
    assert (await get_memory(pool, old_um.id)).status == MemoryStatus.active


# ---------------------------------------------------------------------------
# 3. Near-duplicate merging
# ---------------------------------------------------------------------------


async def test_near_duplicates_merged(pool, provider):
    """Two memories with nearly identical content should be merged:
    the lower-confidence one archived, supersedes relationship created."""
    content_a = "FastEmbed generates embeddings locally using ONNX runtime"
    content_b = "FastEmbed generates local embeddings using the ONNX runtime"

    mem_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_a,
            confidence=0.9,
            topic=["embeddings"],
        ),
        embedding=await provider.embed(content_a),
    )
    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content=content_b,
            confidence=0.5,
            topic=["embeddings"],
        ),
        embedding=await provider.embed(content_b),
    )

    config = ConsolidationConfig(duplicate_threshold=0.85)
    report = await consolidate(pool, config=config)

    # At least one pair should be merged
    assert len(report.duplicates_merged) >= 1

    # The lower confidence memory should be archived
    archived_ids = [pair[1] for pair in report.duplicates_merged]
    assert mem_b.id in archived_ids

    # Verify supersedes relationship
    kept_id = report.duplicates_merged[0][0]
    rels = await get_relationships(pool, kept_id)
    assert any(r.relation == RelationType.supersedes for r in rels)


# ---------------------------------------------------------------------------
# 4. Consolidation report accuracy
# ---------------------------------------------------------------------------


async def test_consolidation_report_complete(pool, provider):
    """The consolidation report should accurately reflect all actions taken."""
    # Set up a scenario with multiple consolidation actions
    # 1. Stale memory for decay
    stale = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="Stale e2e test memory", confidence=0.1),
        embedding=await provider.embed("Stale e2e test memory"),
    )
    old_dt = datetime.now(timezone.utc) - timedelta(days=200)
    await pool.execute("UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, stale.id)

    # 2. Contradiction pair
    c_a = "Python 3.12 is the latest version"
    c_b = "Python 3.13 is the latest version"
    mem_ca = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=c_a, confidence=0.8, topic=["python"]),
        embedding=await provider.embed(c_a),
    )
    mem_cb = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content=c_b, confidence=0.7, topic=["python"]),
        embedding=await provider.embed(c_b),
    )

    config = ConsolidationConfig(
        decay=DecayConfig(min_confidence=0.3),
        contradiction_similarity_min=0.7,
    )
    report = await consolidate(pool, config=config)

    # Report should have the right structure
    d = report.to_dict()
    assert "decayed_count" in d
    assert "duplicates_merged_count" in d
    assert "contradictions_flagged_count" in d
    assert "total_actions" in d
    assert d["total_actions"] == (
        d["decayed_count"] + d["duplicates_merged_count"] + d["contradictions_flagged_count"]
    )

    # Stale memory should be decayed
    assert stale.id in d["decayed"]


# ---------------------------------------------------------------------------
# 5. weft_relate workflow
# ---------------------------------------------------------------------------


async def test_relate_full_workflow(pool):
    """Test the full relationship lifecycle: add, get, filter, remove."""
    m1 = await store_memory(
        pool, MemoryCreate(type=MemoryType.fact, content="memory about architecture")
    )
    m2 = await store_memory(
        pool, MemoryCreate(type=MemoryType.fact, content="memory about testing")
    )
    m3 = await store_memory(
        pool, MemoryCreate(type=MemoryType.fact, content="memory about deployment")
    )

    # Add multiple relationships
    await add_relationship(pool, m1.id, m2.id, RelationType.related_to)
    await add_relationship(pool, m1.id, m3.id, RelationType.derived_from)
    await add_relationship(pool, m2.id, m3.id, RelationType.contradicts)

    # Get all relationships for m1
    rels = await get_relationships(pool, m1.id)
    assert len(rels) == 2

    # Filter by type
    related = await get_relationships(pool, m1.id, relation=RelationType.related_to)
    assert len(related) == 1
    assert related[0].target_id == m2.id

    derived = await get_relationships(pool, m1.id, relation=RelationType.derived_from)
    assert len(derived) == 1
    assert derived[0].target_id == m3.id

    # Get relationships where m3 is target
    m3_rels = await get_relationships(pool, m3.id)
    assert len(m3_rels) == 2  # derived_from + contradicts

    # Remove a relationship
    removed = await remove_relationship(pool, m1.id, m2.id, RelationType.related_to)
    assert removed is True

    # Verify removal
    rels_after = await get_relationships(pool, m1.id)
    assert len(rels_after) == 1
    assert rels_after[0].relation == RelationType.derived_from


# ---------------------------------------------------------------------------
# 6. Consolidation respects already-archived memories
# ---------------------------------------------------------------------------


async def test_consolidation_skips_archived(pool, provider):
    """Consolidation should only process active memories, not re-process
    already archived or decayed ones."""
    # Create and immediately archive a memory
    mem = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact, content="Already archived memory", confidence=0.1
        ),
        embedding=await provider.embed("Already archived memory"),
    )
    old_dt = datetime.now(timezone.utc) - timedelta(days=200)
    await pool.execute("UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, mem.id)
    await pool.execute(
        "UPDATE memories SET status = $1 WHERE id = $2", MemoryStatus.archived.value, mem.id
    )

    report = await consolidate(pool)

    # The already-archived memory should NOT appear in the decayed list
    assert mem.id not in report.decayed


# ---------------------------------------------------------------------------
# 7. Dry run doesn't modify state
# ---------------------------------------------------------------------------


async def test_consolidation_dry_run_no_side_effects(pool, provider):
    """A dry-run consolidation should report what would happen but
    leave all memories unchanged."""
    # Create a stale memory
    stale = await store_memory(
        pool,
        MemoryCreate(type=MemoryType.fact, content="Dry run e2e stale", confidence=0.1),
        embedding=await provider.embed("Dry run e2e stale"),
    )
    old_dt = datetime.now(timezone.utc) - timedelta(days=200)
    await pool.execute("UPDATE memories SET accessed_at = $1 WHERE id = $2", old_dt, stale.id)

    # Dry run
    report = await consolidate(pool, dry_run=True)
    assert stale.id in report.decayed

    # Memory should still be active
    fetched = await get_memory(pool, stale.id)
    assert fetched.status == MemoryStatus.active

    # Now run for real
    report2 = await consolidate(pool)
    assert stale.id in report2.decayed

    # Non-dry-run consolidation is also review-only; an operator-controlled
    # apply path is required for destructive status transitions.
    fetched2 = await get_memory(pool, stale.id)
    assert fetched2.status == MemoryStatus.active
