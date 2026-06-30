"""RC2: memory embed-text composition (content + topics) + backfill.

Covers the PRD's V3 (shared helper, write/backfill parity), version stamping,
and AC5 (backfill convergence).
"""

from __future__ import annotations

import pytest

from weft.db.reembed import backfill_memory_composition
from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemoryType
from weft.store import (
    EMBED_COMPOSITION_VERSION,
    embed_text_for_memory,
    store_memory,
)


@pytest.fixture
def provider():
    return get_provider("fastembed")


# --- Pure-function composition (no DB) ---


def test_embed_text_includes_content_and_topics():
    assert embed_text_for_memory("hello world", ["alpha", "beta"]) == "hello world alpha beta"


def test_embed_text_empty_topics_is_content_plus_space():
    # Trailing space matches the SQL mirror (content || ' ' || '') so the
    # write path and the re-embed expression produce identical text.
    assert embed_text_for_memory("hello", []) == "hello "
    assert embed_text_for_memory("hello", None) == "hello "


def test_embed_text_is_deterministic():
    a = embed_text_for_memory("c", ["x", "y"])
    b = embed_text_for_memory("c", ["x", "y"])
    assert a == b == "c x y"


# --- Write-path version stamping ---


async def test_store_memory_stamps_current_version(pool, provider):
    mc = MemoryCreate(
        type=MemoryType.fact,
        content="composition version is stamped on write",
        topic=["rc2", "versioning"],
        confidence=0.9,
    )
    emb = await provider.embed(embed_text_for_memory(mc.content, mc.topic))
    mem = await store_memory(pool, mc, embedding=emb)
    ver = await pool.fetchval(
        "SELECT embed_composition_version FROM memories WHERE id = $1", mem.id
    )
    assert ver == EMBED_COMPOSITION_VERSION == 1


async def test_legacy_rows_default_to_version_zero(pool, provider):
    """A row written with the explicit legacy version stays at 0 (migration default)."""
    mc = MemoryCreate(
        type=MemoryType.fact, content="legacy content-only row", topic=["rc2"], confidence=0.8
    )
    emb = await provider.embed(mc.content)  # content-only, the old way
    mem = await store_memory(pool, mc, embedding=emb, embed_composition_version=0)
    ver = await pool.fetchval(
        "SELECT embed_composition_version FROM memories WHERE id = $1", mem.id
    )
    assert ver == 0


# --- Python/SQL parity (the silent-drift guard) ---


async def test_python_helper_matches_sql_expression(pool, provider):
    """embed_text_for_memory(content, topic) must byte-match the reembed SQL mirror.

    If these diverge, the write path and the re-embed backfill produce different
    vectors for the same row — the exact silent corpus-split RC2 must prevent.
    """
    mc = MemoryCreate(
        type=MemoryType.decision,
        content="parity between python and sql composition",
        topic=["loc-key", "code-library", "rc2"],
        confidence=0.9,
    )
    emb = await provider.embed(embed_text_for_memory(mc.content, mc.topic))
    mem = await store_memory(pool, mc, embedding=emb)

    sql_text = await pool.fetchval(
        "SELECT content || ' ' || array_to_string(COALESCE(topic, '{}'), ' ') "
        "FROM memories WHERE id = $1",
        mem.id,
    )
    py_text = embed_text_for_memory(mc.content, mc.topic)
    assert sql_text == py_text


async def test_parity_holds_for_empty_topics(pool, provider):
    mc = MemoryCreate(
        type=MemoryType.fact, content="no topics here", topic=[], confidence=0.8
    )
    emb = await provider.embed(embed_text_for_memory(mc.content, mc.topic))
    mem = await store_memory(pool, mc, embedding=emb)
    sql_text = await pool.fetchval(
        "SELECT content || ' ' || array_to_string(COALESCE(topic, '{}'), ' ') "
        "FROM memories WHERE id = $1",
        mem.id,
    )
    assert sql_text == embed_text_for_memory(mc.content, mc.topic) == "no topics here "


# --- Backfill (AC5) ---


async def test_backfill_bumps_stale_rows_and_reembeds(pool, provider):
    """A legacy (version 0) row is re-embedded with topics and bumped to current."""
    mc = MemoryCreate(
        type=MemoryType.fact,
        content="backfill should recompose this with its topics",
        topic=["distinctive-backfill-topic"],
        confidence=0.8,
    )
    content_only = await provider.embed(mc.content)  # legacy embedding
    mem = await store_memory(pool, mc, embedding=content_only, embed_composition_version=0)

    n = await backfill_memory_composition(pool, provider, batch_size=50)
    assert n >= 1

    ver = await pool.fetchval(
        "SELECT embed_composition_version FROM memories WHERE id = $1", mem.id
    )
    assert ver == EMBED_COMPOSITION_VERSION

    # The re-embedded vector must equal a fresh embed of the composed text
    # (proves the backfill used embed_text_for_memory, not content-only).
    composed = await provider.embed(embed_text_for_memory(mc.content, mc.topic))
    stored_vec = await pool.fetchval(
        "SELECT embedding::text FROM memories WHERE id = $1", mem.id
    )
    stored = [float(x) for x in stored_vec.strip("[]").split(",")]
    assert stored == pytest.approx(composed, abs=1e-5)


async def test_backfill_is_idempotent(pool, provider):
    """A second backfill run is a no-op once all rows are current (AC5: count<current = 0)."""
    mc = MemoryCreate(
        type=MemoryType.fact, content="idempotent backfill row", topic=["rc2"], confidence=0.8
    )
    await store_memory(
        pool, mc,
        embedding=await provider.embed(mc.content),
        embed_composition_version=0,
    )
    await backfill_memory_composition(pool, provider)
    stale = await pool.fetchval(
        "SELECT count(*) FROM memories WHERE embed_composition_version < $1",
        EMBED_COMPOSITION_VERSION,
    )
    assert stale == 0
    assert await backfill_memory_composition(pool, provider) == 0
