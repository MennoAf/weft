"""Tests for backup and restore roundtrip."""

from __future__ import annotations

import json

import pytest

from weft.backup import BACKUP_VERSION, backup_all, restore_all, verify_backup
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import add_relationship, get_memory, get_relationships, store_memory
from weft.models import RelationType


# --- Helpers ---


async def _seed_test_data(pool, embedding_dim=384):
    """Insert a few memories with embeddings and relationships for testing."""
    embed_a = [0.1] * embedding_dim
    embed_b = [0.2] * embedding_dim
    embed_c = [0.3] * embedding_dim

    mem_a = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.fact,
            content="Python uses indentation for blocks",
            topic=["python", "syntax"],
            source=MemorySource.conversation,
            confidence=0.9,
            project_id="test-project",
            pinned=True,
        ),
        embedding=embed_a,
    )

    mem_b = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.decision,
            content="Use PostgreSQL with pgvector for storage",
            topic=["architecture", "database"],
            source=MemorySource.conversation,
            confidence=0.85,
            project_id="test-project",
        ),
        embedding=embed_b,
    )

    mem_c = await store_memory(
        pool,
        MemoryCreate(
            type=MemoryType.preference,
            content="Always run tests before committing",
            topic=["workflow"],
            source=MemorySource.conversation,
            confidence=1.0,
        ),
        embedding=embed_c,
    )

    # Add relationships
    await add_relationship(pool, mem_a.id, mem_b.id, RelationType.related_to)
    await add_relationship(pool, mem_b.id, mem_a.id, RelationType.derived_from)

    return mem_a, mem_b, mem_c


# --- backup_all tests ---


@pytest.mark.asyncio
async def test_backup_produces_valid_structure(pool):
    await _seed_test_data(pool)
    data = await backup_all(pool)

    assert data["version"] == BACKUP_VERSION
    assert data["memory_count"] == 3
    assert data["relationship_count"] == 2
    assert data["checksum"]
    assert data["schema_version"] >= 1
    assert data["exported_at"]
    assert len(data["memories"]) == 3
    assert len(data["relationships"]) == 2


@pytest.mark.asyncio
async def test_backup_includes_embeddings(pool):
    await _seed_test_data(pool)
    data = await backup_all(pool)

    for m in data["memories"]:
        assert m["embedding"] is not None
        assert len(m["embedding"]) == 384
        assert all(isinstance(v, float) for v in m["embedding"])


@pytest.mark.asyncio
async def test_backup_includes_all_fields(pool):
    await _seed_test_data(pool)
    data = await backup_all(pool)

    m = data["memories"][0]
    required = {
        "id", "type", "topic", "content", "source", "confidence",
        "token_count", "created_at", "updated_at", "accessed_at",
        "access_count", "project_id", "agent_id", "status", "pinned",
        "usefulness_score", "usefulness_count", "review_after", "embedding",
    }
    assert required.issubset(set(m.keys()))


@pytest.mark.asyncio
async def test_backup_preserves_relationships(pool):
    mem_a, mem_b, _ = await _seed_test_data(pool)
    data = await backup_all(pool)

    rels = data["relationships"]
    assert len(rels) == 2
    source_ids = {r["source_id"] for r in rels}
    assert mem_a.id in source_ids
    assert mem_b.id in source_ids


@pytest.mark.asyncio
async def test_backup_empty_db(pool):
    data = await backup_all(pool)
    assert data["memory_count"] == 0
    assert data["relationship_count"] == 0
    assert data["memories"] == []
    assert data["relationships"] == []


# --- verify_backup tests ---


def test_verify_valid_backup():
    data = {
        "version": BACKUP_VERSION,
        "schema_version": 9,
        "exported_at": "2026-01-01T00:00:00+00:00",
        "checksum": "abc",
        "memory_count": 1,
        "relationship_count": 0,
        "memories": [{"id": "weft-test1", "type": "fact", "content": "hello"}],
        "relationships": [],
    }
    # Recompute checksum to match
    import hashlib
    data["checksum"] = hashlib.sha256(
        json.dumps(["weft-test1" + "hello"], sort_keys=True).encode()
    ).hexdigest()

    report = verify_backup(data)
    assert report["valid"] is True
    assert report["memory_count"] == 1


def test_verify_detects_missing_version():
    report = verify_backup({"memories": [], "relationships": []})
    assert report["valid"] is False
    assert any("version" in i for i in report["issues"])


def test_verify_detects_count_mismatch():
    report = verify_backup({
        "version": BACKUP_VERSION,
        "memory_count": 5,
        "memories": [{"id": "a", "type": "fact", "content": "x"}],
        "relationships": [],
    })
    assert report["valid"] is False
    assert any("count mismatch" in i for i in report["issues"])


def test_verify_detects_checksum_mismatch():
    report = verify_backup({
        "version": BACKUP_VERSION,
        "checksum": "bad_checksum",
        "memories": [{"id": "a", "type": "fact", "content": "x"}],
        "relationships": [],
    })
    assert report["valid"] is False
    assert any("Checksum" in i for i in report["issues"])


def test_verify_detects_dangling_relationship():
    report = verify_backup({
        "version": BACKUP_VERSION,
        "memories": [{"id": "a", "type": "fact", "content": "x"}],
        "relationships": [{"source_id": "a", "target_id": "missing", "relation": "related_to"}],
    })
    assert report["valid"] is False
    assert any("unknown target_id" in i for i in report["issues"])


# --- restore_all tests ---


@pytest.mark.asyncio
async def test_restore_roundtrip(pool):
    """The core test: backup → fresh DB → restore → verify data matches."""
    mem_a, mem_b, mem_c = await _seed_test_data(pool)

    # Backup
    data = await backup_all(pool)
    assert data["memory_count"] == 3

    # Wipe the database
    await pool.execute("TRUNCATE memory_relationships, memories")

    # Verify it's empty
    count = await pool.fetchval("SELECT COUNT(*) FROM memories")
    assert count == 0

    # Restore
    report = await restore_all(pool, data)
    assert report["memories_restored"] == 3
    assert report["memories_skipped"] == 0
    assert report["relationships_restored"] == 2
    assert report["relationships_skipped"] == 0
    assert report["errors"] == []

    # Verify memories exist with correct content
    restored_a = await get_memory(pool, mem_a.id)
    assert restored_a is not None
    assert restored_a.content == mem_a.content
    assert restored_a.type == mem_a.type
    assert restored_a.topic == mem_a.topic
    assert abs(restored_a.confidence - mem_a.confidence) < 1e-6
    assert restored_a.project_id == mem_a.project_id
    assert restored_a.pinned == mem_a.pinned

    restored_c = await get_memory(pool, mem_c.id)
    assert restored_c is not None
    assert restored_c.content == mem_c.content
    assert restored_c.project_id is None  # global memory

    # Verify relationships restored
    rels = await get_relationships(pool, mem_a.id)
    assert len(rels) == 2


@pytest.mark.asyncio
async def test_restore_preserves_embeddings(pool):
    """Verify embeddings survive the roundtrip."""
    await _seed_test_data(pool)
    data = await backup_all(pool)
    await pool.execute("TRUNCATE memory_relationships, memories")

    await restore_all(pool, data)

    # Check embeddings are present
    row = await pool.fetchrow(
        "SELECT embedding::text AS emb FROM memories WHERE embedding IS NOT NULL LIMIT 1"
    )
    assert row is not None
    assert row["emb"] is not None
    # Should be a pgvector string
    assert row["emb"].startswith("[")


@pytest.mark.asyncio
async def test_restore_skips_duplicates(pool):
    """Restoring into a DB that already has the data should skip, not fail."""
    await _seed_test_data(pool)
    data = await backup_all(pool)

    # Restore without wiping — all 3 should be skipped
    report = await restore_all(pool, data, skip_duplicates=True)
    assert report["memories_restored"] == 0
    assert report["memories_skipped"] == 3

    # Total count unchanged
    count = await pool.fetchval("SELECT COUNT(*) FROM memories")
    assert count == 3


@pytest.mark.asyncio
async def test_restore_dry_run(pool):
    """Dry run should report what would happen without modifying the DB."""
    await _seed_test_data(pool)
    data = await backup_all(pool)
    await pool.execute("TRUNCATE memory_relationships, memories")

    report = await restore_all(pool, data, dry_run=True)
    assert report["memories_restored"] == 3
    assert report["relationships_restored"] == 2

    # DB should still be empty
    count = await pool.fetchval("SELECT COUNT(*) FROM memories")
    assert count == 0


@pytest.mark.asyncio
async def test_restore_partial_overlap(pool):
    """Restore with some existing and some new memories."""
    mem_a, mem_b, mem_c = await _seed_test_data(pool)
    data = await backup_all(pool)

    # Delete only mem_c, keep a and b
    await pool.execute("DELETE FROM memories WHERE id = $1", mem_c.id)

    report = await restore_all(pool, data, skip_duplicates=True)
    assert report["memories_restored"] == 1  # only mem_c
    assert report["memories_skipped"] == 2  # mem_a and mem_b

    count = await pool.fetchval("SELECT COUNT(*) FROM memories")
    assert count == 3


@pytest.mark.asyncio
async def test_backup_restore_json_serialization(pool):
    """Verify the backup survives JSON serialization/deserialization."""
    await _seed_test_data(pool)
    data = await backup_all(pool)

    # Serialize and deserialize (simulates writing to file and reading back)
    json_str = json.dumps(data, indent=2)
    restored_data = json.loads(json_str)

    # Verify it's still valid
    report = verify_backup(restored_data)
    assert report["valid"] is True

    # Restore from deserialized data
    await pool.execute("TRUNCATE memory_relationships, memories")
    result = await restore_all(pool, restored_data)
    assert result["memories_restored"] == 3
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_restore_without_embeddings(pool):
    """Memories without embeddings should restore without error."""
    await _seed_test_data(pool)
    data = await backup_all(pool)

    # Strip embeddings from backup
    for m in data["memories"]:
        m["embedding"] = None

    await pool.execute("TRUNCATE memory_relationships, memories")
    report = await restore_all(pool, data)
    assert report["memories_restored"] == 3
    assert report["errors"] == []

    # Embeddings should be NULL in DB
    row = await pool.fetchrow(
        "SELECT COUNT(*) AS c FROM memories WHERE embedding IS NOT NULL"
    )
    assert row["c"] == 0
