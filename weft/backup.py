"""Full backup and restore for Weft memory data.

Exports ALL memories (including embeddings, metadata, and relationships)
as a self-contained JSON file that can be restored to any Postgres instance.

Backup and restore bypass RLS to operate on all rows across all user_ids.
This requires the database connection to use the table owner role (e.g.,
the Supabase service role key, not the anon key).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import asyncpg

from weft.db.migrations import MIGRATIONS

logger = logging.getLogger(__name__)

# Current backup format version — bump when the schema changes
BACKUP_VERSION = "1.1"


async def backup_all(pool: asyncpg.Pool) -> dict:
    """Export all memories, relationships, and metadata as a dict.

    The returned dict is fully self-contained and can be serialized to JSON.
    Embeddings are included as float lists so restore doesn't need re-embedding.

    Bypasses RLS to ensure ALL rows (across all user_ids) are exported.
    Requires the connection role to be the table owner.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Bypass RLS so backup sees all rows regardless of user_id
            await conn.execute("SET LOCAL row_security = off")

            rows = await conn.fetch(
                "SELECT * FROM memories ORDER BY created_at ASC"
            )

            memories = []
            for row in rows:
                m = {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "type": row["type"],
                    "topic": list(row["topic"]) if row["topic"] else [],
                    "content": row["content"],
                    "source": row["source"],
                    "confidence": float(row["confidence"]),
                    "token_count": row["token_count"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "accessed_at": row["accessed_at"].isoformat(),
                    "access_count": row["access_count"],
                    "project_id": row["project_id"],
                    "agent_id": row["agent_id"],
                    "status": row["status"],
                    "pinned": bool(row["pinned"]) if row.get("pinned") is not None else False,
                    "usefulness_score": float(row["usefulness_score"]) if row["usefulness_score"] is not None else 0.7,
                    "usefulness_count": row["usefulness_count"] if row["usefulness_count"] is not None else 0,
                    "review_after": row["review_after"].isoformat() if row.get("review_after") else None,
                    "embedding": row["embedding"],  # decoded to list[float] by pgvector codec
                }
                memories.append(m)

            # Fetch all relationships
            rel_rows = await conn.fetch(
                "SELECT * FROM memory_relationships ORDER BY created_at ASC"
            )
            relationships = [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "relation": r["relation"],
                    "user_id": r["user_id"],
                    "created_at": r["created_at"].isoformat(),
                }
                for r in rel_rows
            ]

    # Schema version = highest applied migration
    schema_version = max(v for v, _, _ in MIGRATIONS)

    # Compute content checksum for integrity verification
    content_hash = hashlib.sha256(
        json.dumps([m["id"] + m["content"] for m in memories], sort_keys=True).encode()
    ).hexdigest()

    return {
        "version": BACKUP_VERSION,
        "schema_version": schema_version,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "checksum": content_hash,
        "memory_count": len(memories),
        "relationship_count": len(relationships),
        "memories": memories,
        "relationships": relationships,
    }


async def restore_all(
    pool: asyncpg.Pool,
    data: dict,
    *,
    dry_run: bool = False,
    skip_duplicates: bool = True,
) -> dict:
    """Restore memories and relationships from a backup dict.

    Bypasses RLS to restore rows with their original user_id values.

    Args:
        pool: Target database connection pool (migrations must be applied).
        data: Backup dict (from backup_all or loaded from JSON file).
        dry_run: If True, report what would happen without writing.
        skip_duplicates: If True, skip memories whose IDs already exist.

    Returns:
        Report dict with counts of restored, skipped, and errored items.
    """
    if data.get("version") not in (BACKUP_VERSION, "1.0"):
        logger.warning(
            "Backup version mismatch: expected %s, got %s",
            BACKUP_VERSION, data.get("version"),
        )

    memories = data.get("memories", [])
    relationships = data.get("relationships", [])

    report = {
        "memories_restored": 0,
        "memories_skipped": 0,
        "relationships_restored": 0,
        "relationships_skipped": 0,
        "errors": [],
    }

    if dry_run:
        # Check which IDs already exist
        existing_ids = set()
        if memories:
            rows = await pool.fetch("SELECT id FROM memories")
            existing_ids = {r["id"] for r in rows}

        for m in memories:
            if m["id"] in existing_ids:
                report["memories_skipped"] += 1
            else:
                report["memories_restored"] += 1

        # For relationships, check if both ends exist (or will be restored)
        will_exist = existing_ids | {m["id"] for m in memories if m["id"] not in existing_ids}
        existing_rels = set()
        if relationships:
            rows = await pool.fetch(
                "SELECT source_id, target_id, relation FROM memory_relationships"
            )
            existing_rels = {(r["source_id"], r["target_id"], r["relation"]) for r in rows}

        for rel in relationships:
            key = (rel["source_id"], rel["target_id"], rel["relation"])
            if key in existing_rels:
                report["relationships_skipped"] += 1
            elif rel["source_id"] in will_exist and rel["target_id"] in will_exist:
                report["relationships_restored"] += 1
            else:
                report["relationships_skipped"] += 1

        return report

    # --- Actual restore ---

    # Get existing IDs to handle skip_duplicates
    existing_ids = set()
    if skip_duplicates and memories:
        rows = await pool.fetch("SELECT id FROM memories")
        existing_ids = {r["id"] for r in rows}

    # Restore memories in a transaction with RLS bypassed
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL row_security = off")

            for m in memories:
                if m["id"] in existing_ids:
                    report["memories_skipped"] += 1
                    continue

                try:
                    embedding = m.get("embedding")

                    await conn.execute(
                        """
                        INSERT INTO memories (
                            id, user_id, type, topic, content, source, confidence,
                            token_count, created_at, updated_at, accessed_at,
                            access_count, project_id, agent_id, embedding, status,
                            pinned, usefulness_score, usefulness_count, review_after
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7,
                            $8, $9, $10, $11,
                            $12, $13, $14, $15::vector, $16,
                            $17, $18, $19, $20
                        )
                        ON CONFLICT (id) DO NOTHING
                        """,
                        m["id"],
                        m.get("user_id"),
                        m["type"],
                        m.get("topic", []),
                        m["content"],
                        m.get("source", "conversation"),
                        m.get("confidence", 0.7),
                        m.get("token_count", 0),
                        datetime.fromisoformat(m["created_at"]),
                        datetime.fromisoformat(m["updated_at"]),
                        datetime.fromisoformat(m["accessed_at"]),
                        m.get("access_count", 0),
                        m.get("project_id"),
                        m.get("agent_id"),
                        embedding,
                        m.get("status", "active"),
                        m.get("pinned", False),
                        m.get("usefulness_score", 0.7),
                        m.get("usefulness_count", 0),
                        datetime.fromisoformat(m["review_after"]) if m.get("review_after") else None,
                    )
                    report["memories_restored"] += 1
                except Exception as e:
                    report["errors"].append(f"Memory {m['id']}: {e}")

    # Restore relationships (after memories so FKs are satisfied)
    if relationships:
        # Get IDs that now exist in the DB
        rows = await pool.fetch("SELECT id FROM memories")
        valid_ids = {r["id"] for r in rows}

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL row_security = off")

                for rel in relationships:
                    if rel["source_id"] not in valid_ids or rel["target_id"] not in valid_ids:
                        report["relationships_skipped"] += 1
                        continue

                    try:
                        result = await conn.execute(
                            """
                            INSERT INTO memory_relationships (
                                source_id, target_id, relation, user_id, created_at
                            ) VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (source_id, target_id, relation) DO NOTHING
                            """,
                            rel["source_id"],
                            rel["target_id"],
                            rel["relation"],
                            rel.get("user_id"),
                            datetime.fromisoformat(rel["created_at"]),
                        )
                        if result.split()[-1] != "0":
                            report["relationships_restored"] += 1
                        else:
                            report["relationships_skipped"] += 1
                    except Exception as e:
                        report["errors"].append(
                            f"Relationship {rel['source_id']}->{rel['target_id']}: {e}"
                        )

    return report


def verify_backup(data: dict) -> dict:
    """Verify a backup file's integrity without connecting to a database.

    Returns a report with validation results.
    """
    issues = []

    if "version" not in data:
        issues.append("Missing 'version' field")
    elif data["version"] not in (BACKUP_VERSION, "1.0"):
        issues.append(f"Version mismatch: expected {BACKUP_VERSION}, got {data['version']}")

    if "memories" not in data:
        issues.append("Missing 'memories' field")
    elif not isinstance(data["memories"], list):
        issues.append("'memories' is not a list")

    memories = data.get("memories", [])
    relationships = data.get("relationships", [])

    # Verify checksum if present
    if data.get("checksum") and memories:
        computed = hashlib.sha256(
            json.dumps([m["id"] + m["content"] for m in memories], sort_keys=True).encode()
        ).hexdigest()
        if computed != data["checksum"]:
            issues.append(f"Checksum mismatch: expected {data['checksum']}, computed {computed}")

    # Verify counts match
    if data.get("memory_count") is not None and len(memories) != data["memory_count"]:
        issues.append(
            f"Memory count mismatch: header says {data['memory_count']}, "
            f"actual {len(memories)}"
        )

    if data.get("relationship_count") is not None and len(relationships) != data["relationship_count"]:
        issues.append(
            f"Relationship count mismatch: header says {data['relationship_count']}, "
            f"actual {len(relationships)}"
        )

    # Verify required fields on memories
    required_fields = {"id", "type", "content"}
    for i, m in enumerate(memories):
        missing = required_fields - set(m.keys())
        if missing:
            issues.append(f"Memory [{i}] missing fields: {missing}")

    # Verify relationship references
    memory_ids = {m["id"] for m in memories}
    for i, rel in enumerate(relationships):
        if rel.get("source_id") not in memory_ids:
            issues.append(f"Relationship [{i}] references unknown source_id: {rel.get('source_id')}")
        if rel.get("target_id") not in memory_ids:
            issues.append(f"Relationship [{i}] references unknown target_id: {rel.get('target_id')}")

    # Count memories with embeddings
    with_embeddings = sum(1 for m in memories if m.get("embedding"))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "memory_count": len(memories),
        "relationship_count": len(relationships),
        "memories_with_embeddings": with_embeddings,
        "exported_at": data.get("exported_at"),
        "schema_version": data.get("schema_version"),
    }
