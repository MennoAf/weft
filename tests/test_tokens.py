"""Tests for token estimation module."""

from __future__ import annotations

import pytest

from weft.tokens import estimate_tokens, estimate_tokens_heuristic


def test_empty_string():
    assert estimate_tokens("") == 0
    assert estimate_tokens_heuristic("") == 0


def test_short_string():
    result = estimate_tokens_heuristic("hello world")
    assert result >= 1
    # "hello world" = 11 chars, ~2-3 tokens
    assert result == 11 // 4  # 2


def test_longer_string():
    text = "Using testcontainers with session-scoped fixtures gives the best balance"
    result = estimate_tokens_heuristic(text)
    assert result == len(text) // 4


def test_minimum_one_token():
    """Even very short strings should return at least 1."""
    assert estimate_tokens_heuristic("hi") == 1
    assert estimate_tokens_heuristic("a") == 1


def test_estimate_tokens_returns_int():
    result = estimate_tokens("This is a test sentence for token counting.")
    assert isinstance(result, int)
    assert result > 0


def test_heuristic_consistency():
    """Same input should always give same output."""
    text = "Weft is a persistent memory system for AI agents"
    assert estimate_tokens_heuristic(text) == estimate_tokens_heuristic(text)


@pytest.mark.asyncio
async def test_token_count_populated_on_store(pool):
    """Token count should be auto-populated when storing a memory."""
    from weft.models import MemoryCreate, MemoryType

    from weft.store import store_memory

    content = "store.py is the ONLY module that writes to Postgres"
    create = MemoryCreate(type=MemoryType.architecture, content=content)
    memory = await store_memory(pool, create)

    assert memory.token_count > 0
    assert memory.token_count == estimate_tokens(content)


@pytest.mark.asyncio
async def test_token_count_updated_on_content_change(pool):
    """Token count should update when content changes."""
    from weft.models import MemoryCreate, MemoryType

    from weft.store import store_memory, update_memory

    create = MemoryCreate(type=MemoryType.fact, content="short")
    memory = await store_memory(pool, create)
    original_count = memory.token_count

    new_content = "This is a much longer piece of content that should have more tokens"
    updated = await update_memory(pool, memory.id, content=new_content)

    assert updated is not None
    assert updated.token_count > original_count
    assert updated.token_count == estimate_tokens(new_content)
