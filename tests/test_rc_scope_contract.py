"""RC3A project-scope resolution contract tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.config import WeftConfig
from weft.mcp.tools import _resolve_project_id, _resolve_project_scope


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
