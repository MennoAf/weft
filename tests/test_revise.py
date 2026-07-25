"""Tests for version-aware memory updates (weft_revise)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.embeddings import get_provider
from weft.mcp.server import AppContext
from weft.models import (
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
    PreferenceMetadata,
    RelationType,
)
from weft.revise import revise_memory
from weft.store import get_memory, get_relationships, store_memory


_FAKE_EMBEDDING = [0.1] * 768


class _FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return list(_FAKE_EMBEDDING)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(_FAKE_EMBEDDING) for _ in texts]


def _make_ctx(pool) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbeddingProvider(),
        config=WeftConfig(),
    )
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@pytest.fixture
async def original_memory(pool):
    """Create an original memory to revise."""
    provider = get_provider("fastembed")
    create = MemoryCreate(
        type=MemoryType.fact,
        content="pgvector version 0.7.0 supports cosine distance",
        topic=["postgres", "pgvector"],
        source=MemorySource.documentation,
        confidence=0.9,
    )
    emb = await provider.embed(create.content)
    memory = await store_memory(pool, create, embedding=emb)
    return pool, memory, provider


async def test_revise_creates_new_memory(original_memory):
    """Revising should create a new memory with updated content."""
    pool, old, provider = original_memory
    new_content = "pgvector version 0.8.1 supports cosine distance operator <=>"
    emb = await provider.embed(new_content)

    new, archived_old = await revise_memory(pool, old.id, new_content, embedding=emb)

    assert new.id != old.id
    assert new.content == new_content
    assert new.type == old.type
    assert new.topic == old.topic
    assert new.source == old.source


async def test_revise_archives_old_memory(original_memory):
    """The old memory should be archived after revision."""
    pool, old, provider = original_memory
    emb = await provider.embed("updated content")

    _, archived = await revise_memory(pool, old.id, "updated content", embedding=emb)
    assert archived.status == MemoryStatus.archived

    # Verify in DB
    db_old = await get_memory(pool, old.id)
    assert db_old is not None
    assert db_old.status == MemoryStatus.archived


async def test_revise_creates_supersedes_relationship(original_memory):
    """A supersedes relationship should link new → old."""
    pool, old, provider = original_memory
    emb = await provider.embed("revised content")

    new, _ = await revise_memory(pool, old.id, "revised content", embedding=emb)

    rels = await get_relationships(pool, new.id, relation=RelationType.supersedes)
    assert len(rels) == 1
    assert rels[0].source_id == new.id
    assert rels[0].target_id == old.id


async def test_revise_with_new_confidence(original_memory):
    """Revise can update confidence."""
    pool, old, provider = original_memory
    emb = await provider.embed("high confidence update")

    new, _ = await revise_memory(
        pool, old.id, "high confidence update",
        embedding=emb, new_confidence=1.0,
    )
    assert new.confidence == 1.0


async def test_revise_with_new_topic(original_memory):
    """Revise can update topics."""
    pool, old, provider = original_memory
    emb = await provider.embed("updated with new topics")

    new, _ = await revise_memory(
        pool, old.id, "updated with new topics",
        embedding=emb, new_topic=["postgres", "pgvector", "extensions"],
    )
    assert "extensions" in new.topic


async def test_revise_chain(original_memory):
    """Multiple revisions should form a chain: v3 supersedes v2 supersedes v1."""
    pool, v1, provider = original_memory

    emb2 = await provider.embed("version 2 content")
    v2, _ = await revise_memory(pool, v1.id, "version 2 content", embedding=emb2)

    emb3 = await provider.embed("version 3 content")
    v3, _ = await revise_memory(pool, v2.id, "version 3 content", embedding=emb3)

    # v3 supersedes v2
    rels_v3 = await get_relationships(pool, v3.id, relation=RelationType.supersedes)
    assert any(r.target_id == v2.id for r in rels_v3)

    # v2 supersedes v1
    rels_v2 = await get_relationships(pool, v2.id, relation=RelationType.supersedes)
    assert any(r.target_id == v1.id for r in rels_v2)

    # v1 and v2 are archived
    assert (await get_memory(pool, v1.id)).status == MemoryStatus.archived
    assert (await get_memory(pool, v2.id)).status == MemoryStatus.archived
    assert v3.status == MemoryStatus.active


async def test_revise_nonexistent_raises():
    """Revising a nonexistent memory should raise ValueError."""
    # This test uses the pool fixture from conftest
    pass


async def test_revise_with_new_type(original_memory):
    """Revise can change the memory type."""
    pool, old, provider = original_memory
    assert old.type == MemoryType.fact

    emb = await provider.embed("retyped to handoff")
    new, _ = await revise_memory(
        pool, old.id, "## Session Handoff\n\n**Summary:** retyped",
        embedding=emb, new_type=MemoryType.handoff,
    )
    assert new.type == MemoryType.handoff
    assert new.id != old.id


async def test_revise_without_new_type_inherits(original_memory):
    """Revise without new_type inherits the original type."""
    pool, old, provider = original_memory
    emb = await provider.embed("same type update")

    new, _ = await revise_memory(pool, old.id, "same type update", embedding=emb)
    assert new.type == old.type


@pytest.fixture
async def preference_original(pool):
    provider = get_provider("fastembed")
    metadata = PreferenceMetadata(
        polarity="avoidance", strength="hard", subject="commute", value="true crime"
    )
    create = MemoryCreate(
        type=MemoryType.preference,
        content="I avoid true crime podcasts during my commute.",
        source=MemorySource.conversation,
        preference_metadata=metadata,
    )
    memory = await store_memory(pool, create, embedding=await provider.embed(create.content))
    return pool, memory, provider, metadata


async def test_revise_preference_metadata_omission_preserves(preference_original):
    pool, old, provider, metadata = preference_original
    new, _ = await revise_memory(pool, old.id, "I still avoid true crime podcasts.", embedding=await provider.embed("I still avoid true crime podcasts."))
    assert new.preference_metadata == metadata


async def test_revise_preference_metadata_explicit_clear(preference_original):
    pool, old, provider, _ = preference_original
    new, _ = await revise_memory(pool, old.id, "I no longer avoid true crime podcasts.", embedding=await provider.embed("I no longer avoid true crime podcasts."), preference_metadata=None)
    assert new.preference_metadata is None


async def test_revise_preference_metadata_replacement(preference_original):
    pool, old, provider, _ = preference_original
    replacement = PreferenceMetadata(polarity="positive", strength="soft", subject="commute", value="history")
    content = "I prefer history podcasts during my commute."
    new, _ = await revise_memory(pool, old.id, content, embedding=await provider.embed(content), preference_metadata=replacement)
    assert new.preference_metadata == replacement


async def test_revise_preference_retype_requires_explicit_clear(preference_original):
    pool, old, provider, _ = preference_original
    with pytest.raises(ValueError, match="requires explicit preference_metadata=null"):
        await revise_memory(pool, old.id, "A factual statement.", embedding=await provider.embed("A factual statement."), new_type=MemoryType.fact)


async def test_revise_with_new_project_id(original_memory):
    """Revise can reassign a memory to a different project."""
    pool, old, provider = original_memory
    assert old.project_id is None  # fixture doesn't set project_id

    emb = await provider.embed("fixed project assignment")
    new, _ = await revise_memory(
        pool, old.id, "fixed project assignment",
        embedding=emb, new_project_id="delphi",
    )
    assert new.project_id == "delphi"
    assert new.id != old.id

    # Verify in DB
    db_new = await get_memory(pool, new.id)
    assert db_new.project_id == "delphi"


async def test_revise_without_new_project_id_inherits(original_memory):
    """Revise without new_project_id inherits the original project_id."""
    pool, old, provider = original_memory
    emb = await provider.embed("same project update")

    new, _ = await revise_memory(pool, old.id, "same project update", embedding=emb)
    assert new.project_id == old.project_id


@pytest.mark.asyncio
async def test_revise_nonexistent(pool):
    """Revising a nonexistent memory should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await revise_memory(pool, "weft-nonexist", "new content")


# ---------------------------------------------------------------------------
# Pinned-state preservation (regression: weft-102dbe19, Shuttle 2026-05-07)
# ---------------------------------------------------------------------------


@pytest.fixture
async def pinned_original(pool):
    """An original memory with pinned=True and a non-null project_id."""
    provider = get_provider("fastembed")
    create = MemoryCreate(
        type=MemoryType.decision,
        content="canonical decision worth pinning",
        topic=["pinning", "regression"],
        source=MemorySource.conversation,
        confidence=0.95,
        pinned=True,
        project_id="warp",
    )
    emb = await provider.embed(create.content)
    memory = await store_memory(pool, create, embedding=emb)
    assert memory.pinned is True
    assert memory.project_id == "warp"
    return pool, memory, provider


async def test_revise_preserves_pinned_when_omitted(pinned_original):
    """Revising a pinned memory without new_pinned must inherit pinned=True."""
    pool, old, provider = pinned_original
    emb = await provider.embed("revised content")

    new, _ = await revise_memory(pool, old.id, "revised content", embedding=emb)

    assert new.pinned is True
    db_new = await get_memory(pool, new.id)
    assert db_new.pinned is True


async def test_revise_can_explicitly_unpin(pinned_original):
    """Revising with new_pinned=False unpins the new version."""
    pool, old, provider = pinned_original
    emb = await provider.embed("explicit unpin")

    new, _ = await revise_memory(
        pool, old.id, "explicit unpin",
        embedding=emb, new_pinned=False,
    )

    assert new.pinned is False
    db_new = await get_memory(pool, new.id)
    assert db_new.pinned is False


async def test_revise_can_explicitly_pin(original_memory):
    """Revising an unpinned memory with new_pinned=True pins the new version."""
    pool, old, provider = original_memory
    assert old.pinned is False

    emb = await provider.embed("explicit pin")
    new, _ = await revise_memory(
        pool, old.id, "explicit pin",
        embedding=emb, new_pinned=True,
    )

    assert new.pinned is True
    db_new = await get_memory(pool, new.id)
    assert db_new.pinned is True


# ---------------------------------------------------------------------------
# MCP-layer regression: omitted args must not be coerced to None and overwrite
# the predecessor's pinned/project_id state. The original bug (Shuttle's repro
# in weft-102dbe19) was that tools.py passed new_project_id=None unconditionally,
# defeating revise_memory's _UNSET sentinel.
# ---------------------------------------------------------------------------


async def test_mcp_revise_preserves_pinned_and_project_id(pinned_original, monkeypatch):
    """End-to-end: weft_revise without new_pinned/new_project_id inherits both."""
    from weft.mcp.tools import weft_revise

    pool, old, _ = pinned_original
    ctx = _make_ctx(pool)

    result = await weft_revise(ctx, memory_id=old.id, new_content="revised")

    assert "error" not in result, f"unexpected error: {result}"
    new = result["new"]
    assert new["pinned"] is True
    assert new["project_id"] == "warp"


async def test_mcp_revise_explicit_unpin_overrides(pinned_original):
    """weft_revise with new_pinned=False unpins."""
    from weft.mcp.tools import weft_revise

    pool, old, _ = pinned_original
    ctx = _make_ctx(pool)

    result = await weft_revise(
        ctx, memory_id=old.id, new_content="unpin me", new_pinned=False,
    )

    assert "error" not in result
    assert result["new"]["pinned"] is False
    assert result["new"]["project_id"] == "warp"  # still inherited
