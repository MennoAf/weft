"""Streamable HTTP release-gate tests for registered ``weft_prime``.

These tests use the production FastMCP singleton and tool registry. Only the
resource lifespan is replaced so testcontainers resources are not duplicated
or closed by production scheduler teardown.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import uvicorn
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
    try:
        app = mcp.http_app(path="/mcp", transport="streamable-http")
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


@asynccontextmanager
async def _running_tcp_server(app):
    """Serve the ASGI app on a real loopback socket and stop it cleanly."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server = None
    task = None
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="off"),
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        for _ in range(100):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        if server is not None:
            server.should_exit = True
        try:
            if task is not None:
                await asyncio.wait_for(asyncio.shield(task), timeout=3)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            listener.close()


async def test_tcp_server_closes_listener_when_setup_fails(monkeypatch):
    listeners = []
    real_socket = socket.socket

    def tracked_socket(*args, **kwargs):
        listener = real_socket(*args, **kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(socket, "socket", tracked_socket)
    monkeypatch.setattr(
        uvicorn,
        "Config",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        async with _running_tcp_server(object()):
            pytest.fail("server must not start")

    assert len(listeners) == 1
    assert listeners[0].fileno() == -1


@pytest.mark.parametrize(
    ("roots", "case"),
    [
        (None, "absent"),
        ([], "empty"),
        (_silent_roots, "silent"),
    ],
)
async def test_prime_omitted_project_cannot_hang_across_roots_states(
    pool, roots, case, monkeypatch,
):
    monkeypatch.setenv("WEFT_TRANSPORT", "streamable-http")
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
    assert data["project_resolution"]["scope"] == "user-wide", case
    assert "project filters" in data["project_resolution"]["warning"], case
    assert data["total_tokens"] <= data["budget_tokens"] == 500


async def test_silent_roots_returns_and_server_shuts_down_over_real_tcp(
    pool, monkeypatch,
):
    """Regression for cancellation coupling hidden by in-process ASGI clients."""
    monkeypatch.setenv("WEFT_TRANSPORT", "streamable-http")
    async with _running_prime_app(pool) as app:
        with patch(
            "weft.consolidation.consolidate_if_due", new_callable=AsyncMock,
        ), patch(
            "weft.mcp.tools.log_memory_access", new_callable=AsyncMock,
        ):
            async with _running_tcp_server(app) as url:
                async with Client(
                    StreamableHttpTransport(url),
                    roots=_silent_roots,
                    timeout=4,
                ) as client:
                    result = await asyncio.wait_for(
                        client.call_tool(
                            "weft_prime",
                            {"budget_tokens": 500, "disclosure": "progressive"},
                        ),
                        timeout=4,
                    )

    data = _structured(result)
    assert data["project_resolution"]["resolved"] is False
    assert data["project_resolution"]["scope"] == "user-wide"
    assert "project filters" in data["project_resolution"]["warning"]
    assert data["total_tokens"] <= data["budget_tokens"] == 500


@pytest.mark.parametrize("disclosure", ["progressive", "full"])
async def test_explicit_project_prime_is_scoped_and_budgeted_over_transport(
    pool, disclosure, monkeypatch,
):
    monkeypatch.setenv("WEFT_TRANSPORT", "streamable-http")
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
