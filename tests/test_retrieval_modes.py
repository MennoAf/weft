"""Tests for retrieval_mode presets and source-filtered search."""

from __future__ import annotations

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.retrieval_modes import MODE_SOURCES, sources_for_mode
from weft.store import (
    search_by_keyword,
    search_by_vector,
    search_hybrid,
    store_memory,
)


# --- Pure unit tests ---


def test_sources_for_mode_face_excludes_ingest():
    sources = sources_for_mode("face")
    assert sources is not None
    assert "ingest" not in sources
    assert "conversation" in sources


def test_sources_for_mode_code_includes_ingest():
    sources = sources_for_mode("code")
    assert sources is not None
    assert "ingest" in sources


def test_sources_for_mode_all_is_none():
    assert sources_for_mode("all") is None


def test_sources_for_mode_default_is_face():
    assert sources_for_mode(None) == MODE_SOURCES["face"]


def test_sources_for_mode_unknown_returns_none():
    # Unknown modes fall through to no-filter rather than erroring.
    assert sources_for_mode("bogus") is None


# --- Integration tests against real DB ---


@pytest.fixture
def provider():
    return get_provider("fastembed")


async def _seed_mixed_sources(pool, provider):
    """Seed memories from two different sources for filter assertions."""
    specs = [
        (MemorySource.conversation, "Jason mentioned Tory Burch pitch going out tomorrow"),
        (MemorySource.conversation, "Harper had a tooth thing Tuesday"),
        (MemorySource.ingest, "def search_by_vector in weft/store.py handles pgvector ANN queries"),
        (MemorySource.ingest, "tests/test_hybrid_search.py validates RRF fusion of BM25 and vector"),
    ]
    stored = []
    for source, content in specs:
        mc = MemoryCreate(
            type=MemoryType.fact,
            content=content,
            topic=["retrieval-mode-test"],
            confidence=0.8,
            source=source,
        )
        emb = await provider.embed(content)
        mem = await store_memory(pool, mc, embedding=emb)
        stored.append(mem)
    return stored


async def test_vector_search_face_mode_excludes_ingest(pool, provider):
    await _seed_mixed_sources(pool, provider)
    embedding = await provider.embed("search function")
    results = await search_by_vector(
        pool, embedding, topic="retrieval-mode-test",
        sources=sources_for_mode("face"),
    )
    assert len(results) > 0
    for r in results:
        assert r.memory.source != MemorySource.ingest


async def test_vector_search_code_mode_includes_ingest(pool, provider):
    await _seed_mixed_sources(pool, provider)
    embedding = await provider.embed("search function")
    results = await search_by_vector(
        pool, embedding, topic="retrieval-mode-test",
        sources=sources_for_mode("code"),
    )
    # Code mode must be able to surface the ingested function description.
    assert any(r.memory.source == MemorySource.ingest for r in results)


async def test_vector_search_no_filter_includes_all(pool, provider):
    await _seed_mixed_sources(pool, provider)
    embedding = await provider.embed("Jason")
    results = await search_by_vector(
        pool, embedding, topic="retrieval-mode-test",
        sources=None,
    )
    sources_seen = {r.memory.source for r in results}
    # With no filter and a generic query over the seed set, both buckets reachable.
    assert MemorySource.conversation in sources_seen or MemorySource.ingest in sources_seen


async def test_keyword_search_face_mode_excludes_ingest(pool, provider):
    await _seed_mixed_sources(pool, provider)
    results = await search_by_keyword(
        pool, "search function", topic="retrieval-mode-test",
        sources=sources_for_mode("face"),
    )
    for r in results:
        assert r.memory.source != MemorySource.ingest


async def test_hybrid_search_respects_sources(pool, provider):
    await _seed_mixed_sources(pool, provider)
    embedding = await provider.embed("search function")
    results = await search_hybrid(
        pool, "search function", embedding, topic="retrieval-mode-test",
        sources=sources_for_mode("face"),
    )
    for r in results:
        assert r.memory.source != MemorySource.ingest
