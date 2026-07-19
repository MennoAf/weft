"""Streamable HTTP release-gate tests for registered ``weft_prime``.

These tests use the production FastMCP singleton and tool registry. Only the
resource lifespan is replaced so testcontainers resources are not duplicated
or closed by production scheduler teardown.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports.http import StreamableHttpTransport

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext, mcp


class _FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimensions for _ in texts]


@asynccontextmanager
async def _running_prime_app(pool):
    app_context = AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=_FakeEmbeddingProvider(),
        config=WeftConfig(),
    )

    @asynccontextmanager
    async def test_lifespan(_server):
        yield app_context

    original_lifespan = mcp._lifespan
    mcp._lifespan = test_lifespan
    app = mcp.http_app(path="/mcp", transport="streamable-http")
    try:
        # Enter and exit in the calling test task. Streamable HTTP's AnyIO
        # cancel scope cannot be split across pytest fixture setup/teardown tasks.
        async with app.router.lifespan_context(app):
            yield app
    finally:
        mcp._lifespan = original_lifespan
        mcp._lifespan_result = None
        mcp._lifespan_result_set = False
        mcp._started.clear()


def _transport_for(app) -> StreamableHttpTransport:
    def client_factory(**kwargs):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=kwargs.get("headers"),
            auth=kwargs.get("auth"),
            follow_redirects=kwargs.get("follow_redirects", True),
            timeout=kwargs.get("timeout"),
        )

    return StreamableHttpTransport(
        "http://testserver/mcp",
        httpx_client_factory=client_factory,
    )


def _structured(result) -> dict:
    data = result.structured_content
    assert isinstance(data, dict), result
    return data


async def _silent_roots(_context):
    await asyncio.Event().wait()


@pytest.mark.parametrize(
    ("roots", "case"),
    [
        (None, "absent"),
        ([], "empty"),
        (_silent_roots, "silent"),
    ],
)
async def test_prime_omitted_project_cannot_hang_across_roots_states(
    pool, roots, case,
):
    async with _running_prime_app(pool) as app:
        transport = _transport_for(app)
        with patch(
            "weft.consolidation.consolidate_if_due", new_callable=AsyncMock,
        ), patch(
            "weft.mcp.tools.log_memory_access", new_callable=AsyncMock,
        ):
            async with Client(transport, roots=roots, timeout=6) as client:
                tools = {tool.name for tool in await client.list_tools()}
                assert "weft_prime" in tools
                result = await asyncio.wait_for(
                    client.call_tool(
                        "weft_prime",
                        {"budget_tokens": 500, "disclosure": "progressive"},
                    ),
                    timeout=5,
                )

    data = _structured(result)
    assert data["project_resolution"]["resolved"] is False, case
    assert data["total_tokens"] <= data["budget_tokens"] == 500


@pytest.mark.parametrize("disclosure", ["progressive", "full"])
async def test_explicit_project_prime_is_scoped_and_budgeted_over_transport(
    pool, disclosure,
):
    async with _running_prime_app(pool) as app:
        transport = _transport_for(app)
        with patch(
            "weft.consolidation.consolidate_if_due", new_callable=AsyncMock,
        ), patch(
            "weft.mcp.tools.log_memory_access", new_callable=AsyncMock,
        ):
            async with Client(transport, roots=_silent_roots, timeout=3) as client:
                result = await asyncio.wait_for(
                    client.call_tool(
                        "weft_prime",
                        {
                            "project_id": "transport-project",
                            "budget_tokens": 500,
                            "disclosure": disclosure,
                        },
                    ),
                    timeout=3,
                )

    data = _structured(result)
    assert "project_resolution" not in data
    assert data["total_tokens"] <= data["budget_tokens"] == 500
