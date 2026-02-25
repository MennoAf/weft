"""Tests for the user_model memory type."""

from __future__ import annotations

from weft.models import Memory, MemoryCreate, MemoryType
from weft.store import get_memory, list_memories, store_memory


async def test_create_user_model_via_memory_create():
    """A user_model memory can be created via MemoryCreate."""
    create = MemoryCreate(
        type=MemoryType.user_model,
        content="User prefers concise responses without emojis",
        topic=["preferences", "communication"],
        confidence=0.85,
    )
    assert create.type == MemoryType.user_model
    assert create.content == "User prefers concise responses without emojis"


async def test_store_and_retrieve_user_model(pool):
    """A user_model memory can be stored and retrieved via store_memory / get_memory."""
    create = MemoryCreate(
        type=MemoryType.user_model,
        content="User works primarily in Python and TypeScript",
        topic=["languages", "skills"],
        confidence=0.9,
    )
    mem = await store_memory(pool, create)
    assert mem.id.startswith("weft-")
    assert mem.type == MemoryType.user_model
    assert mem.confidence == 0.9

    fetched = await get_memory(pool, mem.id)
    assert fetched is not None
    assert fetched.type == MemoryType.user_model
    assert fetched.content == "User works primarily in Python and TypeScript"
    assert fetched.topic == ["languages", "skills"]


async def test_list_memories_filtered_by_user_model(pool):
    """A user_model memory appears correctly in list_memories filtered by memory_type."""
    # Create memories of different types
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Python 3.12 was released in 2023",
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.user_model,
        content="User prefers dark mode editors",
        topic=["preferences"],
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.user_model,
        content="User is a senior engineer with 10 years experience",
        topic=["background"],
    ))
    await store_memory(pool, MemoryCreate(
        type=MemoryType.pattern,
        content="This codebase uses async/await throughout",
    ))

    user_models = await list_memories(pool, memory_type=MemoryType.user_model)
    assert len(user_models) == 2
    assert all(m.type == MemoryType.user_model for m in user_models)

    # Other types should not be affected
    facts = await list_memories(pool, memory_type=MemoryType.fact)
    assert len(facts) == 1


async def test_to_dict_returns_user_model_string(pool):
    """The to_dict() method returns 'user_model' for the type field."""
    mem = await store_memory(pool, MemoryCreate(
        type=MemoryType.user_model,
        content="User prefers functional programming patterns",
        topic=["style"],
    ))
    d = mem.to_dict()
    assert d["type"] == "user_model"
    assert isinstance(d["type"], str)
