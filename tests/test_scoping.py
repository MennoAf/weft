"""Tests for orthogonal project_id + agent_id scoping in the store layer."""

import pytest

from weft.embeddings import get_provider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import list_memories, search_by_vector, store_memory


@pytest.fixture
def provider():
    return get_provider("fastembed")


@pytest.fixture
async def scoped_memories(pool, provider):
    """Create memories across different project/agent scopes."""
    cases = [
        ("proj-a", "agent-1", "proj-a agent-1 memory about testing"),
        ("proj-a", "agent-2", "proj-a agent-2 memory about deployment"),
        ("proj-a", None, "proj-a global-agent memory about config"),
        ("proj-b", "agent-1", "proj-b agent-1 memory about databases"),
        (None, "agent-1", "global-project agent-1 memory about caching"),
        (None, None, "fully global memory about architecture"),
    ]
    memories = []
    for project_id, agent_id, content in cases:
        create = MemoryCreate(
            type=MemoryType.fact,
            content=content,
            source=MemorySource.conversation,
            project_id=project_id,
            agent_id=agent_id,
        )
        emb = await provider.embed(content)
        m = await store_memory(pool, create, embedding=emb)
        memories.append(m)
    return memories


class TestListMemoriesScoping:
    async def test_brain_wide_no_filters(self, pool, scoped_memories):
        """No project_id or agent_id → returns everything."""
        results = await list_memories(pool)
        assert len(results) == 6

    async def test_project_scoped(self, pool, scoped_memories):
        """project_id only → memories for that project + global project."""
        results = await list_memories(pool, project_id="proj-a")
        contents = {m.content for m in results}
        assert "proj-a agent-1 memory about testing" in contents
        assert "proj-a agent-2 memory about deployment" in contents
        assert "proj-a global-agent memory about config" in contents
        assert "fully global memory about architecture" in contents
        assert "global-project agent-1 memory about caching" in contents
        # proj-b memories should NOT appear
        assert "proj-b agent-1 memory about databases" not in contents

    async def test_agent_scoped(self, pool, scoped_memories):
        """agent_id only → memories for that agent + global agent."""
        results = await list_memories(pool, agent_id="agent-1")
        contents = {m.content for m in results}
        assert "proj-a agent-1 memory about testing" in contents
        assert "proj-b agent-1 memory about databases" in contents
        assert "global-project agent-1 memory about caching" in contents
        assert "fully global memory about architecture" in contents
        assert "proj-a global-agent memory about config" in contents
        # agent-2 memories should NOT appear
        assert "proj-a agent-2 memory about deployment" not in contents

    async def test_project_and_agent_scoped(self, pool, scoped_memories):
        """Both axes → intersection of project+global and agent+global."""
        results = await list_memories(pool, project_id="proj-a", agent_id="agent-1")
        contents = {m.content for m in results}
        assert "proj-a agent-1 memory about testing" in contents
        assert "fully global memory about architecture" in contents
        assert "proj-a global-agent memory about config" in contents
        assert "global-project agent-1 memory about caching" in contents
        # Different project or different agent should not appear
        assert "proj-b agent-1 memory about databases" not in contents
        assert "proj-a agent-2 memory about deployment" not in contents


class TestSearchByVectorScoping:
    async def test_brain_wide_search(self, pool, scoped_memories, provider):
        """Unscoped vector search returns all matching memories."""
        emb = await provider.embed("memory")
        results = await search_by_vector(pool, emb, limit=10, threshold=0.0)
        assert len(results) == 6

    async def test_project_scoped_search(self, pool, scoped_memories, provider):
        """Project-scoped vector search excludes other projects."""
        emb = await provider.embed("memory")
        results = await search_by_vector(pool, emb, project_id="proj-b", limit=10, threshold=0.0)
        contents = {r.memory.content for r in results}
        assert "proj-b agent-1 memory about databases" in contents
        assert "fully global memory about architecture" in contents
        assert "proj-a agent-1 memory about testing" not in contents

    async def test_agent_scoped_search(self, pool, scoped_memories, provider):
        """Agent-scoped vector search excludes other agents."""
        emb = await provider.embed("memory")
        results = await search_by_vector(pool, emb, agent_id="agent-2", limit=10, threshold=0.0)
        contents = {r.memory.content for r in results}
        assert "proj-a agent-2 memory about deployment" in contents
        assert "fully global memory about architecture" in contents
        assert "proj-a agent-1 memory about testing" not in contents

    async def test_both_axes_search(self, pool, scoped_memories, provider):
        """Both axes in vector search → intersection."""
        emb = await provider.embed("memory")
        results = await search_by_vector(
            pool, emb, project_id="proj-a", agent_id="agent-1", limit=10, threshold=0.0
        )
        contents = {r.memory.content for r in results}
        assert "proj-a agent-1 memory about testing" in contents
        assert "fully global memory about architecture" in contents
        assert "proj-b agent-1 memory about databases" not in contents
        assert "proj-a agent-2 memory about deployment" not in contents
