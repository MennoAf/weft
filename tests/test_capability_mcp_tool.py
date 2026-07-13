"""Tests for the capability-registry MCP coordinator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_ctx(pool: object, project_name: str = "weft") -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = SimpleNamespace(pool=pool)
    ctx.list_roots = AsyncMock(
        return_value=[SimpleNamespace(uri=f"file:///workspace/{project_name}")]
    )
    return ctx


@pytest.mark.asyncio
async def test_capability_lookup_delegates_and_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from weft.mcp import tools

    pool = object()
    ctx = _make_ctx(pool)
    results = [{"memory_id": "weft-1", "parsed": {"repo": "muttr"}}]
    lookup = AsyncMock(return_value=results)
    formatter = MagicMock(return_value="[repo:muttr] crawl/escalation.py :: LazyEscalationPolicy")
    monkeypatch.setattr(tools, "lookup_capabilities", lookup)
    monkeypatch.setattr(tools, "format_lookup_results", formatter)

    response = await tools.weft_capability_lookup(
        ctx, query="bot blocked crawler", limit=7
    )

    assert response == "[repo:muttr] crawl/escalation.py :: LazyEscalationPolicy"
    lookup.assert_awaited_once_with("bot blocked crawler", pool, "weft", 7)
    formatter.assert_called_once_with(results)


@pytest.mark.asyncio
async def test_capability_lookup_reports_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from weft.mcp import tools

    lookup = AsyncMock(return_value=[])
    formatter = MagicMock()
    monkeypatch.setattr(tools, "lookup_capabilities", lookup)
    monkeypatch.setattr(tools, "format_lookup_results", formatter)

    response = await tools.weft_capability_lookup(
        _make_ctx(object()), query="unknown capability"
    )

    assert response == (
        "No capability entries found for query: unknown capability. "
        "Try indexing repos with capability_registry/ingest.py first."
    )
    formatter.assert_not_called()
