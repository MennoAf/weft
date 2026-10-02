"""RC3A project-scope resolution contract tests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.config import WeftConfig
from weft.mcp.tools import ProjectResolution, _resolve_project_id, _resolve_project_scope


_UUID = "49d60a99-5a5f-4f02-a545-18f8a9bb51d5"


def _ctx(*, transport: str = "stdio", project_name: str = "default", roots=None):
    ctx = MagicMock()
    ctx.transport = transport
    ctx.request_context.lifespan_context = SimpleNamespace(
        config=WeftConfig(project_name=project_name)
    )
    ctx.list_roots = AsyncMock(return_value=roots or [])
    return ctx


def _root(name: str):
    return SimpleNamespace(uri=f"file:///workspace/{name}")


def test_project_resolution_is_structured_and_reports_resolution_state():
    resolved = ProjectResolution("weft", "configured")
    unresolved = ProjectResolution(None, "unresolved")

    assert resolved.resolved is True
    assert resolved.project_id == "weft"
    assert resolved.source == "configured"
    assert unresolved.resolved is False


@pytest.mark.asyncio
async def test_explicit_beats_configured_and_roots_without_calling_roots():
    ctx = _ctx(project_name="configured", roots=[_root("root-project")])

    result = await _resolve_project_scope(ctx, "explicit")

    assert result.project_id == "explicit"
    assert result.source == "explicit"
    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_uuid_rejects_before_configured_or_roots_fallback():
    ctx = _ctx(project_name="configured", roots=[_root("root-project")])

    with pytest.raises(ValueError, match="looks like a UUID"):
        await _resolve_project_scope(ctx, _UUID)

    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_configured_value_is_intentional_source_even_when_root_differs():
    ctx = _ctx(project_name="  Intentional-Project  ", roots=[_root("other-root")])

    result = await _resolve_project_scope(ctx, None)

    assert result.project_id == "Intentional-Project"
    assert result.source == "configured"
    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_configured_project_is_used_on_http_without_roots():
    ctx = _ctx(transport="streamable-http", project_name="configured")
    ctx.list_roots = AsyncMock(side_effect=AssertionError("roots RPC issued"))

    result = await _resolve_project_scope(ctx, None)

    assert result.project_id == "configured"
    assert result.source == "configured"
    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("sentinel", ["default", "DEFAULT", "  DeFaUlT  ", "", "   "])
async def test_default_sentinel_falls_back_to_stdio_roots(sentinel):
    ctx = _ctx(project_name=sentinel, roots=[_root("root-project")])

    result = await _resolve_project_scope(ctx, None)

    assert result.project_id == "root-project"
    assert result.source == "roots"
    ctx.list_roots.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("sentinel", ["default", "DEFAULT", "  DeFaUlT  ", "", "   "])
@pytest.mark.parametrize("transport", ["streamable-http", "streamable_http", "sse"])
async def test_default_sentinel_is_unresolved_on_http_without_roots(sentinel, transport):
    ctx = _ctx(transport=transport, project_name=sentinel)
    ctx.list_roots = AsyncMock(side_effect=AssertionError("roots RPC issued"))

    result = await _resolve_project_scope(ctx, None)

    assert result.project_id is None
    assert result.source == "unresolved"
    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_configured_stdio_root_fallback_is_bounded_and_unresolved():
    ctx = _ctx(project_name="default")

    async def never_returns():
        await asyncio.Event().wait()

    ctx.list_roots = never_returns
    result = await asyncio.wait_for(_resolve_project_scope(ctx, None), timeout=3.0)

    assert result.project_id is None
    assert result.source == "unresolved"


@pytest.mark.asyncio
async def test_stale_http_environment_cannot_enable_roots_for_http_aliases(monkeypatch):
    monkeypatch.setenv("WEFT_TRANSPORT", "stdio")
    for transport in ("streamable-http", "streamable_http", "sse"):
        ctx = _ctx(transport=transport, project_name="default", roots=[_root("wrong")])
        ctx.list_roots = AsyncMock(side_effect=AssertionError("roots RPC issued"))

        result = await _resolve_project_scope(ctx, None)

        assert result.project_id is None
        assert result.source == "unresolved"
        ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolved_scope_is_explicit_and_legacy_id_wrapper_is_none():
    ctx = _ctx(project_name="default", roots=[])

    result = await _resolve_project_scope(ctx, None)

    assert result.project_id is None
    assert result.source == "unresolved"
    assert await _resolve_project_id(ctx, None) is None


@pytest.mark.asyncio
async def test_legacy_id_wrapper_preserves_explicit_behavior():
    ctx = _ctx(project_name="configured", roots=[_root("root")])

    assert await _resolve_project_id(ctx, "explicit") == "explicit"
    ctx.list_roots.assert_not_awaited()


@pytest.mark.asyncio
async def test_loader_configured_value_is_used_when_lifespan_has_no_config():
    ctx = MagicMock()
    ctx.transport = "stdio"
    ctx.request_context.lifespan_context = SimpleNamespace()
    ctx.list_roots = AsyncMock(side_effect=AssertionError("roots RPC issued"))
    with patch("weft.mcp.tools.load_config", return_value=WeftConfig(project_name="loader-project")):
        result = await _resolve_project_scope(ctx, None)

    assert result.project_id == "loader-project"
    assert result.source == "configured"
    ctx.list_roots.assert_not_awaited()


# ---------------------------------------------------------------------------
# MCP boundary adoption contracts
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_acquire(_pool):
    yield SimpleNamespace()


class _FakeMemory:
    id = "memory-scope"
    write_provenance = None

    def __init__(self, project_id="resolved-project", memory_type=None):
        self.project_id = project_id
        self.type = memory_type

    def to_dict(self):
        return {"id": self.id, "project_id": self.project_id, "content": "body"}


class _FakeFocus:
    focused_memories = []

    def to_dict(self):
        return {"focused_memories": [], "count": 0}


def _tool_ctx(*, project_name="configured", transport="stdio", roots=None):
    app = SimpleNamespace(
        pool=object(),
        config=WeftConfig(project_name=project_name),
        embedding=SimpleNamespace(embed=AsyncMock(return_value=[0.1])),
        cache=SimpleNamespace(set_memory=AsyncMock(), invalidate_stats=AsyncMock()),
        _tasks=[],
    )

    def spawn(awaitable, *, name):
        task = asyncio.create_task(awaitable, name=name)
        app._tasks.append(task)
        return task

    app.spawn_background_task = spawn
    ctx = _ctx(transport=transport, project_name=project_name, roots=roots)
    ctx.request_context.lifespan_context = app
    return ctx, app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "project_name", "roots", "expected"),
    [
        ("stdio", "configured-project", [_root("root-project")], "configured-project"),
        ("stdio", "default", [_root("root-project")], "root-project"),
        ("streamable-http", "default", [_root("root-project")], None),
    ],
)
async def test_scope_adoption_for_remember_focus_handoff_and_count(
    transport, project_name, roots, expected, monkeypatch,
):
    from weft.mcp.tools import weft_count_occurrences, weft_focus, weft_handoff, weft_remember

    ctx, app = _tool_ctx(transport=transport, project_name=project_name, roots=roots)
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    async def fake_store(_pool, create, *, embedding=None):
        return _FakeMemory(create.project_id, create.type)

    monkeypatch.setattr("weft.mcp.tools.store_memory", fake_store)
    monkeypatch.setattr("weft.mcp.tools.check_dedup_on_store", AsyncMock(return_value=SimpleNamespace(is_duplicate=False)), raising=False)
    monkeypatch.setattr("weft.focus.build_focus", AsyncMock(return_value=_FakeFocus()))
    monkeypatch.setattr("weft.mcp.tools.recall_turns", AsyncMock(return_value=[]))
    monkeypatch.setattr("weft.mcp.tools.summarize_occurrences", lambda turns, basis: SimpleNamespace(count=0, occurrences=()))
    monkeypatch.setattr("weft.mcp.tools.list_memories", AsyncMock(return_value=[]))
    monkeypatch.setattr("weft.mcp.tools.list_episodes", AsyncMock(return_value=[]))
    monkeypatch.setattr("weft.mcp.tools.create_episode", AsyncMock(return_value=SimpleNamespace(id="episode-scope")))
    monkeypatch.setattr("weft.mcp.tools.add_memory_to_episode", AsyncMock())
    monkeypatch.setattr("weft.mcp.tools.boost_session_memories", AsyncMock(return_value={}))
    monkeypatch.setattr("weft.mcp.tools.boost_session_turns", AsyncMock(return_value={}))

    remember = await weft_remember(ctx, content="A meaningful scoped memory", check_contradictions=False)
    focus = await weft_focus(ctx, intent="scoped focus")
    count = await weft_count_occurrences(ctx, query="scoped event")
    handoff = await weft_handoff(ctx, summary="Scoped handoff")

    if expected:
        assert remember["project_id"] == expected
        assert focus == {"focused_memories": [], "count": 0}
        assert count["count"] == 0
        assert handoff["stored"] is True
        assert handoff["project_id"] == expected
    else:
        assert handoff["error"] == "Invalid input"
        assert handoff["tool"] == "weft_handoff"
        assert count["count"] == 0
    assert ctx.list_roots.await_count == (4 if project_name == "default" and transport == "stdio" else 0)
    await asyncio.gather(*app._tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_unresolved_prime_preserves_response_shape_and_scope_warning(monkeypatch):
    from weft.mcp.tools import weft_prime

    ctx, app = _tool_ctx(project_name="default", roots=[])
    monkeypatch.setattr("weft.primer.build_primer", AsyncMock(return_value={"rules": [], "handoff": []}))
    monkeypatch.setattr("weft.mcp.tools._owner_scoped_canary_health", AsyncMock(return_value=None))
    monkeypatch.setattr("weft.consolidation.consolidate_if_due", AsyncMock(return_value=None))
    result = await weft_prime(ctx)

    assert result["rules"] == []
    assert result["handoff"] == []
    assert result["project_resolution"]["resolved"] is False
    assert result["project_resolution"]["scope"] == "user-wide"
    assert ctx.list_roots.await_count == 1
    await asyncio.gather(*app._tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_recall_face_uses_resolved_facet_without_search_project_wall(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="configured-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    search = AsyncMock(return_value=[])
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", search)
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))
    result = await weft_recall(ctx, query="face scope", mode="keyword", retrieval_mode="face")

    assert result["results"] == []
    assert search.await_args.kwargs["project_id"] is None
    assert search.await_args.kwargs["facet_boost_project_id"] == "configured-project"
    assert ctx.list_roots.assert_not_awaited() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retrieval_mode", ["code", "all"])
async def test_recall_code_and_all_forward_resolved_project_to_search_and_count(retrieval_mode, monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="resolved-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    search = AsyncMock(return_value=[])
    count = AsyncMock(return_value=0)
    monkeypatch.setattr("weft.mcp.tools.search_hybrid", search)
    monkeypatch.setattr("weft.mcp.tools.count_by_vector", count)
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))
    result = await weft_recall(ctx, query="catalog scope", mode="hybrid", retrieval_mode=retrieval_mode)

    assert result["results"] == []
    assert search.await_args.kwargs["project_id"] == "resolved-project"
    assert count.await_args.kwargs["project_id"] == "resolved-project"
    assert search.await_args.kwargs["facet_boost_project_id"] is None
    assert ctx.list_roots.assert_not_awaited() is None


@pytest.mark.asyncio
async def test_recall_resolves_scope_once_and_forwards_it_to_turns_and_both(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="resolved-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    resolution = AsyncMock(return_value=ProjectResolution("resolved-project", "configured"))
    turns = AsyncMock(return_value={"query": "when did it happen", "tier": "turns", "count": 1, "turns": [{"id": "t1", "content": "when did it happen"}]})
    both = AsyncMock(return_value={"query": "remember", "tier": "both", "count": 0, "results": []})
    monkeypatch.setattr("weft.mcp.tools._resolve_project_scope", resolution)
    monkeypatch.setattr("weft.mcp.tools._weft_recall_turns", turns)
    monkeypatch.setattr("weft.mcp.tools._weft_recall_both", both)
    monkeypatch.setattr("weft.mcp.tools.search_hybrid", AsyncMock(return_value=[]))
    monkeypatch.setattr("weft.mcp.tools.count_by_vector", AsyncMock(return_value=0))
    monkeypatch.setattr("weft.mcp.tools.load_config", lambda **_: WeftConfig())
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))

    turn_result = await weft_recall(ctx, query="when did it happen", tier="turns")
    both_result = await weft_recall(ctx, query="do you remember this", tier="both")

    assert turn_result["tier"] == "turns"
    assert both_result["tier"] == "both"
    assert turns.await_args.kwargs["project_id"] == "resolved-project"
    assert both.await_args.kwargs["project_id"] == "resolved-project"
    assert resolution.await_count == 2


@pytest.mark.asyncio
async def test_recall_enumeration_forwards_resolved_scope_and_keeps_shape(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="resolved-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    search = AsyncMock(return_value=[])
    enum = AsyncMock(return_value=([], {"memories": [], "complete": True}))
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", search)
    monkeypatch.setattr("weft.mcp.tools.load_config", lambda **_: WeftConfig())
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))
    monkeypatch.setattr("weft.enumeration_router.detect_enumeration_intent", lambda query: (True, "things"))
    monkeypatch.setattr("weft.enumeration_router.gather_enumeration", enum)
    response = await weft_recall(ctx, query="list all things", mode="keyword", tier="belief")

    assert response["results"] == []
    assert enum.await_args.kwargs["project_id"] == "resolved-project"
    # Face-mode baseline search intentionally has no hard project wall;
    # enumeration is the scope-bearing branch under test above.
    assert search.await_args.kwargs["project_id"] is None
    assert ctx.list_roots.assert_not_awaited() is None


@pytest.mark.asyncio
async def test_recall_default_root_scope_resolves_once_and_reaches_hard_wall(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="", roots=[_root("Root-Derived")])
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    search = AsyncMock(return_value=[])
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", search)
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))

    response = await weft_recall(
        ctx, query="root scoped catalog", mode="keyword", retrieval_mode="code", tier="belief",
    )

    assert response["results"] == []
    assert search.await_args.kwargs["project_id"] == "root-derived"
    assert ctx.list_roots.await_count == 1
    await asyncio.gather(*app._tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_recall_query_telemetry_receives_resolved_project(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="telemetry-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", AsyncMock(return_value=[]))
    logged = AsyncMock(return_value=None)
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", logged)

    response = await weft_recall(ctx, query="telemetry scope", mode="keyword", tier="belief")
    await asyncio.sleep(0)

    assert response["results"] == []
    logged.assert_awaited_once()
    assert logged.await_args.kwargs["project_id"] == "telemetry-project"
    assert ctx.list_roots.assert_not_awaited() is None
    await asyncio.gather(*app._tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_turns_fallback_preserves_shape_and_reuses_resolved_scope(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="fallback-project")
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    turns = AsyncMock(return_value={"query": "before launch", "tier": "turns", "count": 0, "turns": []})
    search = AsyncMock(return_value=[])
    monkeypatch.setattr("weft.mcp.tools._weft_recall_turns", turns)
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", search)
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))

    response = await weft_recall(
        ctx, query="before launch", mode="keyword", tier="turns", retrieval_mode="code",
    )

    assert response["results"] == []
    assert response["tier_fallback"] == {"from": "turns", "reason": "empty_turns_result"}
    assert turns.await_args.kwargs["project_id"] == "fallback-project"
    assert search.await_args.kwargs["project_id"] == "fallback-project"
    assert ctx.list_roots.assert_not_awaited() is None
    await asyncio.gather(*app._tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_deterministic_recovery_snapshot_keeps_raw_and_resolved_scope(monkeypatch):
    from weft.mcp.tools import weft_recall

    ctx, app = _tool_ctx(project_name="", roots=[_root("Recovery-Root")])
    monkeypatch.setattr("weft.mcp.tools.acquire", _noop_acquire)
    monkeypatch.setattr("weft.mcp.tools.search_by_keyword", AsyncMock(return_value=[]))
    monkeypatch.setattr("weft.mcp.tools.log_recall_query", AsyncMock(return_value=None))
    captured = {}

    class _FakeRecoveryController:
        def __init__(self, **kwargs):
            pass

        async def recover(self, *args, **kwargs):
            captured["scope"] = kwargs["scope"]
            return SimpleNamespace(
                supported=False,
                to_public_dict=lambda **_: {"supported": False, "not_attempted": True},
            )

    monkeypatch.setattr("weft.retrieval_recovery.RecoveryController", _FakeRecoveryController)
    response = await weft_recall(
        ctx, query="recovery scope", mode="keyword", tier="belief", recovery_mode="deterministic",
    )

    scope = captured["scope"]
    assert response["recovery"] == {"supported": False, "not_attempted": True}
    assert scope.requested_project_id is None
    assert scope.resolved_project_id == "recovery-root"
    assert ctx.list_roots.await_count == 1
    await asyncio.gather(*app._tasks, return_exceptions=True)
