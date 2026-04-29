"""Tests for backup and restore roundtrip."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from weft.auth import current_user_id
from weft.backup import BACKUP_VERSION, backup_all, restore_all, verify_backup
from weft.behaviors import store_behavior
from weft.db.connection import acquire
from weft.entities import link_mention, store_entity
from weft.episodes import add_memory_to_episode, create_episode
from weft.models import (
    BehaviorCreate,
    BehaviorScope,
    EntityCreate,
    EntityType,
    EpisodeCreate,
    MemoryCreate,
    MemorySource,
    MemoryType,
    ModeCreate,
    ModeWeights,
    NudgeMode,
    RelationType,
    TrackerCreate,
    TrackerKind,
)
from weft.modes import upsert_mode
from weft.store import add_relationship, get_memory, get_relationships, store_memory
from weft.trackers import create_tracker
from weft.workspaces import add_member, create_workspace


# --- Helpers ---


async def _seed_test_data(pool, embedding_dim=768):
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
        assert len(m["embedding"]) == 768
        assert all(isinstance(v, float) for v in m["embedding"])


@pytest.mark.asyncio
async def test_backup_includes_all_fields(pool):
    await _seed_test_data(pool)
    data = await backup_all(pool)

    m = data["memories"][0]
    required = {
        "id", "user_id", "type", "topic", "content", "source", "confidence",
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
async def test_backup_includes_user_id(pool):
    """Backup includes user_id for each memory. After migration 36 every row
    has a non-NULL user_id; the test fixture's session default supplies one
    when an INSERT doesn't specify one."""
    await _seed_test_data(pool)

    # Insert a user-scoped memory directly
    await pool.execute(
        """
        INSERT INTO memories (id, user_id, type, content, source, confidence, status)
        VALUES ('weft-user1', 'user-abc', 'fact', 'user scoped', 'conversation', 0.9, 'active')
        """
    )

    data = await backup_all(pool)
    user_ids = {m["id"]: m["user_id"] for m in data["memories"]}
    assert user_ids["weft-user1"] == "user-abc"
    # Other memories pick up the test fixture's default user_id from the
    # column DEFAULT (no longer NULL).
    for m in data["memories"]:
        if m["id"] != "weft-user1":
            assert m["user_id"] is not None


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
    await pool.execute("TRUNCATE entity_mentions, episode_memories, memory_relationships, entities, episodes, memories CASCADE")

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
    await pool.execute("TRUNCATE entity_mentions, episode_memories, memory_relationships, entities, episodes, memories CASCADE")

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
    await pool.execute("TRUNCATE entity_mentions, episode_memories, memory_relationships, entities, episodes, memories CASCADE")

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
    await pool.execute("TRUNCATE entity_mentions, episode_memories, memory_relationships, entities, episodes, memories CASCADE")
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

    await pool.execute("TRUNCATE entity_mentions, episode_memories, memory_relationships, entities, episodes, memories CASCADE")
    report = await restore_all(pool, data)
    assert report["memories_restored"] == 3
    assert report["errors"] == []

    # Embeddings should be NULL in DB
    row = await pool.fetchrow(
        "SELECT COUNT(*) AS c FROM memories WHERE embedding IS NOT NULL"
    )
    assert row["c"] == 0


# --- v1.2 full-coverage roundtrip ---


_BACKUP_USER = "rls-backup-user-1"

_TRUNCATE_ALL = (
    "TRUNCATE "
    "trackers, modes, episode_memories, episodes, "
    "entity_mentions, entities, behaviors, "
    "memory_relationships, memories, "
    "workspace_members, workspaces "
    "CASCADE"
)


async def _seed_full_inventory(pool):
    """Seed at least one row per backed-up table for a single user.

    Returns a dict of seeded IDs keyed by table for downstream verification.
    """
    tok = current_user_id.set(_BACKUP_USER)
    try:
        async with acquire(pool):
            ws = await create_workspace(
                pool,
                name="Backup Test WS",
                description="for v1.2 roundtrip",
                created_by=_BACKUP_USER,
            )
            await add_member(
                pool,
                workspace_id=ws.id,
                user_id="second-member",
                role="reader",
                added_by=_BACKUP_USER,
            )

            mem_a = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content="round-trip memory A",
                    topic=["roundtrip", "alpha"],
                ),
                embedding=[0.11] * 768,
            )
            mem_b = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.decision,
                    content="round-trip memory B",
                    confidence=0.85,
                ),
                embedding=[0.22] * 768,
            )
            await add_relationship(pool, mem_a.id, mem_b.id, RelationType.related_to)

            beh = await store_behavior(
                pool,
                BehaviorCreate(
                    trigger_pattern="when running roundtrip",
                    action="restore everything",
                    scope=BehaviorScope.global_,
                ),
            )

            ent = await store_entity(
                pool,
                EntityCreate(
                    name="Backup Test Entity",
                    entity_type=EntityType.concept,
                    aliases=["bte"],
                    description="seeded for roundtrip",
                ),
            )
            await link_mention(pool, ent.id, mem_a.id)

            ep = await create_episode(pool, EpisodeCreate(title="roundtrip episode"))
            await add_memory_to_episode(pool, ep.id, mem_a.id)

            mode = await upsert_mode(
                pool,
                ModeCreate(
                    name="roundtrip-mode",
                    description="for roundtrip test",
                    weights=ModeWeights(),
                ),
            )

            trk = await create_tracker(
                pool,
                TrackerCreate(
                    kind=TrackerKind.task,
                    title="roundtrip tracker",
                    context={"shopping": ["milk", "eggs"]},
                    nudge_mode=NudgeMode.recur,
                    nudge_interval=timedelta(hours=24),
                ),
            )
    finally:
        current_user_id.reset(tok)

    return {
        "workspace": ws,
        "mem_a": mem_a,
        "mem_b": mem_b,
        "behavior": beh,
        "entity": ent,
        "episode": ep,
        "mode": mode,
        "tracker": trk,
    }


@pytest.mark.asyncio
async def test_backup_v12_includes_all_sections(pool):
    """Format 1.2 carries every user-data section, even when empty."""
    data = await backup_all(pool)
    assert data["version"] == BACKUP_VERSION
    for key in (
        "memories",
        "relationships",
        "behaviors",
        "entities",
        "entity_mentions",
        "episodes",
        "episode_memories",
        "modes",
        "trackers",
        "workspaces",
        "workspace_members",
    ):
        assert key in data, f"backup missing section {key!r}"
        assert isinstance(data[key], list)


@pytest.mark.asyncio
async def test_full_inventory_roundtrip(pool):
    """Seed every backed-up table, backup, TRUNCATE everything, restore.

    The invariant: after restore the rows match what was backed up — ids,
    user_ids, jsonb payloads, intervals, embeddings, FK edges. This is
    the load-bearing test that 'I can recover from a bad write' actually
    holds for the full inventory of user data, not just memories.
    """
    seeded = await _seed_full_inventory(pool)
    data = await backup_all(pool)

    counts = data["counts"]
    assert counts["memories_count"] == 2
    assert counts["relationships_count"] == 1
    assert counts["behaviors_count"] == 1
    assert counts["entities_count"] == 1
    assert counts["entity_mentions_count"] == 1
    assert counts["episodes_count"] == 1
    assert counts["episode_memories_count"] == 1
    assert counts["modes_count"] == 1
    assert counts["trackers_count"] == 1
    assert counts["workspaces_count"] == 1
    assert counts["workspace_members_count"] == 2  # creator + added member

    await pool.execute(_TRUNCATE_ALL)
    assert await pool.fetchval("SELECT count(*) FROM memories") == 0
    assert await pool.fetchval("SELECT count(*) FROM behaviors") == 0
    assert await pool.fetchval("SELECT count(*) FROM trackers") == 0
    assert await pool.fetchval("SELECT count(*) FROM workspaces") == 0

    report = await restore_all(pool, data)
    assert report["errors"] == []
    assert report["memories_restored"] == 2
    assert report["relationships_restored"] == 1
    assert report["behaviors_restored"] == 1
    assert report["entities_restored"] == 1
    assert report["entity_mentions_restored"] == 1
    assert report["episodes_restored"] == 1
    assert report["episode_memories_restored"] == 1
    assert report["modes_restored"] == 1
    assert report["trackers_restored"] == 1
    assert report["workspaces_restored"] == 1
    assert report["workspace_members_restored"] == 2

    # Spot-check: every row carries the seeded user_id.
    rows = await pool.fetch(
        "SELECT user_id FROM memories UNION ALL "
        "SELECT user_id FROM behaviors UNION ALL "
        "SELECT user_id FROM entities UNION ALL "
        "SELECT user_id FROM episodes UNION ALL "
        "SELECT user_id FROM modes UNION ALL "
        "SELECT user_id FROM trackers"
    )
    assert {r["user_id"] for r in rows} == {_BACKUP_USER}

    # Tracker survives jsonb (state_history, context) + interval round-trip.
    # asyncpg returns jsonb as a JSON-encoded string by default; round-trip
    # invariant is "decoded value matches what we inserted."
    trk_row = await pool.fetchrow(
        "SELECT context::text AS ctx, nudge_interval, "
        "state_history::text AS hist FROM trackers WHERE id = $1",
        seeded["tracker"].id,
    )
    assert trk_row is not None
    assert json.loads(trk_row["ctx"])["shopping"] == ["milk", "eggs"]
    assert trk_row["nudge_interval"] == timedelta(hours=24)
    history = json.loads(trk_row["hist"])
    # create_tracker stamps a single creation event; non-empty after restore
    # means the JSONB array round-tripped intact.
    assert isinstance(history, list) and len(history) >= 1
    assert history[0]["to"] == "in_progress"

    # Mode survives jsonb (weights) round-trip.
    mode_row = await pool.fetchrow(
        "SELECT weights::text AS w FROM modes WHERE id = $1", seeded["mode"].id
    )
    assert mode_row is not None
    assert isinstance(json.loads(mode_row["w"]), dict)

    # Workspace member jsonb identity round-trips.
    member_rows = await pool.fetch(
        "SELECT member_identity::text AS mi FROM workspace_members "
        "WHERE workspace_id = $1",
        seeded["workspace"].id,
    )
    identities = [json.loads(r["mi"]) for r in member_rows]
    assert {i["user_id"] for i in identities} == {_BACKUP_USER, "second-member"}

    # FK edge: entity_mentions still references both entity + memory.
    em = await pool.fetchrow(
        "SELECT entity_id, memory_id FROM entity_mentions"
    )
    assert em["entity_id"] == seeded["entity"].id
    assert em["memory_id"] == seeded["mem_a"].id

    # Embedding survived
    emb_row = await pool.fetchrow(
        "SELECT embedding::text AS e FROM memories WHERE id = $1",
        seeded["mem_a"].id,
    )
    assert emb_row["e"].startswith("[")


@pytest.mark.asyncio
async def test_full_inventory_idempotent_restore(pool):
    """Running restore twice over the same data is a no-op the second time."""
    await _seed_full_inventory(pool)
    data = await backup_all(pool)

    # First restore against the populated DB — every row already exists,
    # everything skips.
    report = await restore_all(pool, data, skip_duplicates=True)
    assert report["errors"] == []
    assert report["memories_restored"] == 0
    assert report["memories_skipped"] == 2
    assert report["behaviors_skipped"] == 1
    assert report["trackers_skipped"] == 1
    assert report["workspace_members_skipped"] == 2

    # Second restore after wiping: should restore everything cleanly.
    await pool.execute(_TRUNCATE_ALL)
    report2 = await restore_all(pool, data)
    assert report2["errors"] == []
    assert report2["memories_restored"] == 2
    assert report2["behaviors_restored"] == 1
    assert report2["trackers_restored"] == 1


@pytest.mark.asyncio
async def test_legacy_v11_backup_still_restores(pool):
    """A v1.1 backup (memories+relationships only) restores without errors.

    The v1.1 → v1.2 transition added new sections; older backups don't
    carry them. ``restore_all`` must treat absent sections as empty.
    """
    await _seed_test_data(pool)
    data = await backup_all(pool)

    # Hand-craft a v1.1 backup by trimming the new sections.
    legacy = {
        "version": "1.1",
        "schema_version": data["schema_version"],
        "exported_at": data["exported_at"],
        "checksum": data["checksum"],
        "memory_count": data["memory_count"],
        "relationship_count": data["relationship_count"],
        "memories": data["memories"],
        "relationships": data["relationships"],
    }

    await pool.execute(_TRUNCATE_ALL)
    report = await restore_all(pool, legacy)
    assert report["errors"] == []
    assert report["memories_restored"] == 3
    assert report["relationships_restored"] == 2
    # New sections weren't in the legacy payload — counts stay zero.
    assert report["behaviors_restored"] == 0
    assert report["trackers_restored"] == 0
    assert report["workspaces_restored"] == 0


@pytest.mark.asyncio
async def test_dry_run_preserves_data_after_full_seed(pool):
    """Dry-run on a populated DB doesn't write anything new."""
    await _seed_full_inventory(pool)
    data = await backup_all(pool)
    counts_before = {
        t: await pool.fetchval(f"SELECT count(*) FROM {t}")
        for t in ("memories", "behaviors", "trackers", "workspaces", "modes")
    }

    report = await restore_all(pool, data, dry_run=True)
    assert report["errors"] == []

    counts_after = {
        t: await pool.fetchval(f"SELECT count(*) FROM {t}")
        for t in ("memories", "behaviors", "trackers", "workspaces", "modes")
    }
    assert counts_before == counts_after


def test_verify_v12_section_fk_dangling_entity_mention():
    """verify_backup catches an entity_mentions row pointing at a missing entity."""
    data = {
        "version": BACKUP_VERSION,
        "memories": [{"id": "m1", "type": "fact", "content": "x"}],
        "relationships": [],
        "entities": [],
        "entity_mentions": [{"entity_id": "ghost", "memory_id": "m1"}],
    }
    report = verify_backup(data)
    assert report["valid"] is False
    assert any("unknown entity_id" in i for i in report["issues"])


def test_verify_v12_section_fk_dangling_workspace_member():
    """workspace_members → workspaces FK is checked too."""
    data = {
        "version": BACKUP_VERSION,
        "memories": [],
        "relationships": [],
        "workspaces": [],
        "workspace_members": [{"workspace_id": "ghost", "member_identity": {}}],
    }
    report = verify_backup(data)
    assert report["valid"] is False
    assert any("unknown workspace_id" in i for i in report["issues"])
