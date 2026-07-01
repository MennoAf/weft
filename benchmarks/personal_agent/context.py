#!/usr/bin/env python3
"""
context.py — In-process AppContext + MCP ctx construction for PAAH.

PAAH's whole premise is asserting on the *agent-facing* response, so it must
call the real ``weft_remember`` / ``weft_recall`` tool functions — which expect
an MCP ``Context``. We build a MagicMock ctx whose ``lifespan_context`` is a
real ``AppContext`` wired to the testcontainers pool and a REAL embedding
provider (fastembed 768-dim). Fake vectors would make the stochastic candidate
recall path unable to match seeded rows — the same degenerate meter the
enumeration_eval harness warns about — so embeddings are not optional.

Pattern mirrors ``tests/test_recall_both.py`` (``_make_ctx`` + ``app`` fixture).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import asyncpg

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


async def build_app_context(pool: asyncpg.Pool) -> AppContext:
    """Construct an AppContext over ``pool`` with a real fastembed provider.

    Initializes the pgvector codec on every pooled connection first (mirrors the
    ``app`` fixture in test_recall_both.py) so direct list[float] embeddings
    encode without asyncpg type errors.
    """
    from weft.db.connection import _pgvector_codec_init
    from weft.embeddings import get_provider

    conns = [await pool.acquire() for _ in range(pool.get_size())]
    try:
        for c in conns:
            await _pgvector_codec_init(c)
    finally:
        for c in conns:
            await pool.release(c)

    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=get_provider("fastembed", dimensions=768),
        config=WeftConfig(),
    )


def make_ctx(app: AppContext) -> MagicMock:
    """Wrap an AppContext in a MagicMock MCP Context.

    ``list_roots`` returns [] so project-id auto-detection is a no-op — PAAH
    always passes ``project_id`` explicitly.
    """
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx
