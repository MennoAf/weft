"""End-to-end validation: 10 memories, semantic search, relevance verification.

This is the Phase 1 capstone test — validates the complete pipeline:
text → embedding → store → vector search → ranked results.
"""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType, MemorySource
from weft.store import get_stats, search_by_vector, store_memory, list_memories

# 10 diverse memories spanning different types and topics
SEED_MEMORIES = [
    {
        "type": MemoryType.preference,
        "content": "Always use fastembed as the default embedding provider for local development",
        "topic": ["embeddings", "development"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
    {
        "type": MemoryType.architecture,
        "content": "store.py is the ONLY module that writes to Postgres — all other modules go through it",
        "topic": ["weft", "architecture", "postgres"],
        "confidence": 0.95,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.fact,
        "content": "pgvector extension version 0.8.1 supports cosine distance operator <=>",
        "topic": ["postgres", "pgvector"],
        "confidence": 0.9,
        "source": MemorySource.documentation,
    },
    {
        "type": MemoryType.pattern,
        "content": "Using testcontainers with session-scoped fixtures gives the best balance of isolation and speed",
        "topic": ["testing", "patterns"],
        "confidence": 0.8,
        "source": MemorySource.inference,
    },
    {
        "type": MemoryType.solution,
        "content": "The Ryuk reaper container from testcontainers can become stale — clean it up in conftest.py",
        "topic": ["testing", "docker"],
        "confidence": 0.9,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.relationship,
        "content": "Jason Bauman owns Weft, Loom, and Muttr projects",
        "topic": ["people", "projects"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
    {
        "type": MemoryType.architecture,
        "content": "Weft MCP server uses FastMCP with stdio transport for agent communication",
        "topic": ["weft", "mcp", "architecture"],
        "confidence": 0.95,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.fact,
        "content": "Redis cache uses 1 hour TTL for memories and 24 hour TTL for embeddings",
        "topic": ["redis", "caching"],
        "confidence": 0.9,
        "source": MemorySource.code,
    },
    {
        "type": MemoryType.pattern,
        "content": "Three-layer config (global YAML, project YAML, env vars) provides good flexibility without complexity",
        "topic": ["configuration", "patterns"],
        "confidence": 0.85,
        "source": MemorySource.inference,
    },
    {
        "type": MemoryType.preference,
        "content": "Use dedicated infrastructure for each project — do not share Postgres between Loom and Weft",
        "topic": ["infrastructure", "docker"],
        "confidence": 1.0,
        "source": MemorySource.conversation,
    },
]


@pytest.fixture
async def seeded_pool(pool):
    """Pool with 10 memories pre-seeded with embeddings."""
    provider = get_provider("fastembed")
    for mem_data in SEED_MEMORIES:
        create = MemoryCreate(**mem_data)
        embedding = await provider.embed(create.content)
        await store_memory(pool, create, embedding=embedding)
    return pool


async def test_ten_memories_stored(seeded_pool):
    """Verify all 10 memories were stored."""
    memories = await list_memories(seeded_pool, limit=20)
    assert len(memories) == 10


async def test_stats_reflect_seed(seeded_pool):
    """Stats should reflect the 10 seeded memories."""
    stats = await get_stats(seeded_pool)
    assert stats["total"] == 10
    assert stats["by_status"]["active"] == 10
    # We have 2 preferences, 2 architectures, 2 facts, 2 patterns, 1 solution, 1 relationship
    assert stats["by_type"]["preference"] == 2
    assert stats["by_type"]["architecture"] == 2


async def test_recall_database_topics(seeded_pool):
    """Query about databases should surface pgvector and postgres memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("database and postgres configuration")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 2
    top_contents = [r.memory.content for r in results[:3]]
    # pgvector or postgres-related memories should rank high
    assert any("postgres" in c.lower() or "pgvector" in c.lower() for c in top_contents)


async def test_recall_testing_topics(seeded_pool):
    """Query about testing should surface testing-related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("how to set up tests with containers")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 2
    top_contents = [r.memory.content for r in results[:3]]
    assert any("testcontainer" in c.lower() or "testing" in c.lower() for c in top_contents)


async def test_recall_infrastructure_topics(seeded_pool):
    """Query about infrastructure should surface infra-related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("docker infrastructure setup for projects")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 1
    top_contents = [r.memory.content for r in results[:3]]
    assert any("infrastructure" in c.lower() or "docker" in c.lower() for c in top_contents)


async def test_recall_people_and_ownership(seeded_pool):
    """Query about project ownership should surface the relationship memory."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("who owns this project")
    results = await search_by_vector(seeded_pool, query_emb, limit=3)

    assert len(results) >= 1
    top_contents = [r.memory.content for r in results[:3]]
    assert any("jason" in c.lower() or "owns" in c.lower() for c in top_contents)


async def test_similarity_scores_reasonable(seeded_pool):
    """Similarity scores should be between 0 and 1 and decrease with rank."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("embedding provider configuration")
    results = await search_by_vector(seeded_pool, query_emb, limit=10)

    assert len(results) == 10
    similarities = [r.similarity for r in results]

    # All between 0 and 1
    assert all(0 <= s <= 1 for s in similarities)

    # Sorted descending (highest similarity first)
    assert similarities == sorted(similarities, reverse=True)

    # Top result should have reasonable similarity
    assert similarities[0] > 0.5


async def test_threshold_filters_irrelevant(seeded_pool):
    """A high threshold should filter out weakly related memories."""
    provider = get_provider("fastembed")
    query_emb = await provider.embed("quantum physics and black holes")
    results = await search_by_vector(seeded_pool, query_emb, limit=10, threshold=0.7)

    # Nothing in our seed data is about quantum physics — should get very few or none
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Phase 1 user-scope capstone: single-user byte-identical recall & primer
# ---------------------------------------------------------------------------
#
# Invariant: in a single-user deployment, recall and primer output stays
# byte-identical across the user-scope migration + backfill. OR-NULL filtering
# guarantees the same row set via the IS-NULL branch (pre-backfill) and the
# `= uid` branch (post-backfill).


import hashlib
import json
import uuid as _uuid

from weft.behaviors import list_behaviors
from weft.config.user_identity import get_user_id
from weft.db.backfill_user_id import backfill_user_id
from weft.primer import build_primer


def _fingerprint_behaviors(behaviors) -> str:
    """Hash behavior list with time/identity-varying fields stripped.

    Excluded fields:
      user_id     — changes NULL → uid by design (backfill is the whole point)
      updated_at  — UPDATE bumps the row timestamp on backfill
    Included: everything else (id, trigger_pattern, action, confidence,
    scope, project_id, agent_id, priority, enabled, access_count,
    token_count, created_at).
    """
    stable = []
    for b in behaviors:
        d = b.model_dump(mode="json")
        d.pop("user_id", None)
        d.pop("updated_at", None)
        stable.append(d)
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _fingerprint_primer(result: dict) -> str:
    """Hash primer output with time-varying top-level fields stripped."""
    # Top-level clock-dependent fields. `changes_since` depends on wall time
    # from the last handoff and flips between calls separated by a backfill.
    skip_top = {"freshness_hours", "changes_since", "wellness_snapshot"}
    stable = {k: v for k, v in result.items() if k not in skip_top}

    # Handoff items carry an `age_hours` field tied to wall clock — strip it.
    if isinstance(stable.get("handoff"), list):
        stable["handoff"] = [
            {k: v for k, v in item.items() if k != "age_hours"}
            for item in stable["handoff"]
        ]
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


async def _seed_unscoped_behaviors(pool, n: int = 3) -> list[str]:
    """Insert N behaviors with user_id IS NULL (pre-migration / pre-backfill state).

    Uses raw SQL so we bypass store_behavior's RLS-derived user_id default.
    """
    ids: list[str] = []
    for i in range(n):
        row_id = _uuid.uuid4().hex
        await pool.execute(
            """
            INSERT INTO behaviors (
                id, trigger_pattern, action, confidence, scope,
                project_id, agent_id, user_id, priority, enabled,
                access_count, token_count, created_at, updated_at,
                status
            ) VALUES (
                $1, $2, $3, 0.9, 'global',
                NULL, NULL, NULL, $4, TRUE,
                0, 10, NOW(), NOW(),
                'active'
            )
            """,
            row_id,
            f"byte-identical-trigger-{i}",
            f"byte-identical-action-{i}",
            i,
        )
        ids.append(row_id)
    return ids


async def test_single_user_byte_identical_recall(pool):
    """list_behaviors output is byte-identical across backfill in a single-user deployment.

    Pre-backfill: user_id IS NULL → query with user_id=uid hits the IS-NULL branch.
    Post-backfill: user_id = uid  → query with user_id=uid hits the `= uid` branch.
    Result set (modulo user_id + updated_at) must be identical.
    """
    await _seed_unscoped_behaviors(pool, n=3)
    uid = get_user_id()

    pre = await list_behaviors(pool, user_id=uid, enabled=True, status="active")
    assert len(pre) >= 3, "seed failed — user_id IS NULL branch did not surface rows"
    pre_hash = _fingerprint_behaviors(pre)

    migrated = await backfill_user_id(pool)
    assert migrated >= 3

    post = await list_behaviors(pool, user_id=uid, enabled=True, status="active")
    assert len(post) == len(pre)
    post_hash = _fingerprint_behaviors(post)

    assert pre_hash == post_hash, (
        "recall output diverged across backfill — OR-NULL filtering is not "
        "preserving the pre-migration row set"
    )

    # Post-backfill user_id must be the config uid on every surfaced row.
    assert all(b.user_id == uid for b in post)


async def test_single_user_byte_identical_primer(pool):
    """build_primer output is byte-identical across backfill in a single-user deployment.

    Compares primer dicts with clock-dependent fields (freshness_hours,
    changes_since, wellness_snapshot, handoff age_hours) stripped. The
    remaining content — rules, behaviors, handoff body, issues, decisions,
    section_tokens, etc. — must hash-match before and after.
    """
    # Seed a handful of pinned rules + behaviors so the primer has work to do.
    from weft.models import MemoryCreate, MemorySource, MemoryType

    pinned_rules = [
        "Always use uv for Python package management",
        "store.py is the only module that writes to Postgres",
    ]
    for content in pinned_rules:
        await store_memory(pool, MemoryCreate(
            type=MemoryType.preference,
            content=content,
            topic=["byte-identical-primer"],
            confidence=1.0,
            source=MemorySource.conversation,
            pinned=True,
        ))

    await _seed_unscoped_behaviors(pool, n=2)

    pre_primer = await build_primer(pool, budget_tokens=1800, disclosure="full")
    pre_hash = _fingerprint_primer(pre_primer)

    await backfill_user_id(pool)

    post_primer = await build_primer(pool, budget_tokens=1800, disclosure="full")
    post_hash = _fingerprint_primer(post_primer)

    assert pre_hash == post_hash, (
        "primer output diverged across backfill — single-user byte-identical "
        "invariant violated"
    )
