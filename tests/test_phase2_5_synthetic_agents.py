"""Synthetic two-agent end-to-end smoke for Phase 2.5 credential auth.

Drives two distinct bearer tokens (different users, different caller
modes) through the real chain — UserIdentityMiddleware →
``acquire(pool)`` → ``SET LOCAL app.user_id`` → ``store_memory`` /
``list_memories`` — to prove that:

  1. Each token resolves to its bound (user_id, caller_mode) on
     the contextvars the rest of the stack reads from.
  2. Bob's escalation header (``X-Weft-Caller-Mode: supervisor``)
     is ignored — the agent floor holds.
  3. ``acquire()`` issues the right ``SET LOCAL app.user_id`` —
     the Postgres GUC matches the token's bound user inside a
     handler, even with the test pool's session-level default.
  4. Inserts get the right ``user_id`` column value (the row's
     identity matches the credential, not the request header or
     the pool default).
  5. Write provenance is stamped from the resolved caller_mode,
     not the request header — Bob's writes are tagged ``agent``
     even when he claims supervisor in the header.
  6. Revoking Bob's token causes his next call to 401 while
     Alice's still works.

What this test deliberately does NOT assert: cross-user RLS
isolation at the SELECT layer. The testcontainer Postgres role is
superuser, which bypasses RLS regardless of policies — see the
preamble of ``tests/test_rls_e2e.py``. Cross-user SELECT isolation
is verified at deploy time against the non-superuser app role.
What this test gives us instead is the *upstream* guarantee: the
``app.user_id`` GUC and the row's ``user_id`` column carry the
right value, so when RLS is in force in prod it has the right
input to filter on.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from weft.auth import current_caller_mode, current_user_id
from weft.credentials import issue_token, revoke_token
from weft.db.connection import acquire
from weft.mcp.server import UserIdentityMiddleware
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import list_memories, store_memory


def _build_app(pool) -> Starlette:
    """Starlette app exposing three /mcp routes that hit the real
    ``acquire(pool)`` path so RLS is in force."""

    async def whoami(request: Request) -> JSONResponse:
        return JSONResponse({
            "user_id": current_user_id.get(),
            "caller_mode": current_caller_mode.get(),
        })

    async def write(request: Request) -> JSONResponse:
        body = await request.json()
        async with acquire(pool):
            mem = await store_memory(
                pool,
                MemoryCreate(
                    type=MemoryType.fact,
                    content=body["content"],
                    topic=body.get("topic", []),
                    source=MemorySource.conversation,
                ),
            )
        return JSONResponse({
            "id": mem.id,
            "write_provenance": mem.write_provenance,
            "review_status": mem.review_status,
        })

    async def listing(request: Request) -> JSONResponse:
        async with acquire(pool):
            mems = await list_memories(pool, limit=100)
        return JSONResponse({
            "count": len(mems),
            "ids": [m.id for m in mems],
            "contents": [m.content for m in mems],
        })

    async def session_user(request: Request) -> JSONResponse:
        """Reflect ``current_setting('app.user_id')`` from inside an
        ``acquire()`` scope — proves SET LOCAL fired with the right
        value for this request."""
        async with acquire(pool) as conn:
            uid = await conn.fetchval(
                "SELECT current_setting('app.user_id', true)"
            )
        return JSONResponse({"app_user_id": uid})

    app = Starlette(routes=[
        Route("/mcp/whoami", whoami),
        Route("/mcp/write", write, methods=["POST"]),
        Route("/mcp/list", listing),
        Route("/mcp/session_user", session_user),
    ])
    app.add_middleware(
        UserIdentityMiddleware,
        pool_getter=lambda: pool,
        auth_required=True,
        oauth_enabled=False,
    )
    return app


def _client(app: Starlette, token: str, *, mode_header: str | None = None):
    headers = {"authorization": f"Bearer {token}"}
    if mode_header is not None:
        headers["x-weft-caller-mode"] = mode_header
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://testserver", headers=headers,
    )


@pytest.mark.asyncio
async def test_two_synthetic_agents_full_chain(pool):
    """The full Phase 2.5 promise, end-to-end through the ASGI stack."""
    alice_token, alice_row = await issue_token(
        pool, user_id="alice", caller_mode="supervisor", label="alice-cli",
    )
    bob_token, bob_row = await issue_token(
        pool, user_id="bob", caller_mode="agent", label="bob-runtime",
    )
    app = _build_app(pool)

    # 1. Each token resolves to the bound identity ----------------------
    async with _client(app, alice_token) as alice:
        resp = await alice.get("/mcp/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "alice", "caller_mode": "supervisor"}

    async with _client(app, bob_token) as bob:
        resp = await bob.get("/mcp/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "bob", "caller_mode": "agent"}

    # 2. Bob's escalation header is ignored — agent floor holds ----------
    async with _client(app, bob_token, mode_header="supervisor") as bob_lying:
        resp = await bob_lying.get("/mcp/whoami")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "bob", "caller_mode": "agent"}, (
        "ESCALATION DETECTED: agent token claimed supervisor and was honoured"
    )

    # 3. Alice writes; the row is tagged to her, supervisor-provenance ---
    async with _client(app, alice_token) as alice:
        resp = await alice.post("/mcp/write", json={
            "content": "alice-secret-canary-value",
            "topic": ["alice-only"],
        })
    assert resp.status_code == 200
    alice_write = resp.json()
    assert alice_write["write_provenance"] == "supervisor"
    assert alice_write["review_status"] == "active"

    # Verify the user_id column directly (bypasses RLS via raw pool exec)
    alice_row_uid = await pool.fetchval(
        "SELECT user_id FROM memories WHERE id = $1", alice_write["id"],
    )
    assert alice_row_uid == "alice", (
        f"USER_ID STAMPING WRONG: alice's row stamped {alice_row_uid!r}"
    )

    # 4. SET LOCAL fired with the right user under each token ------------
    # Cross-user SELECT isolation depends on RLS, which the testcontainer
    # superuser bypasses (see test_rls_e2e.py preamble). Instead, prove
    # the GUC the RLS policy reads from carries the right value — that's
    # the upstream guarantee RLS depends on.
    async with _client(app, alice_token) as alice:
        resp = await alice.get("/mcp/session_user")
    assert resp.json()["app_user_id"] == "alice"

    async with _client(app, bob_token, mode_header="supervisor") as bob_lying:
        resp = await bob_lying.get("/mcp/session_user")
    assert resp.json()["app_user_id"] == "bob", (
        "GUC MISMATCH: app.user_id did not match the token's bound user"
    )

    # 5. Bob's writes are stamped 'agent' regardless of the header.
    #    Use innocuous content to avoid Layer 3 instruction-quarantine.
    async with _client(app, bob_token, mode_header="supervisor") as bob_lying:
        resp = await bob_lying.post("/mcp/write", json={
            "content": "bob plain observational note about the weather",
            "topic": ["bob-only"],
        })
    assert resp.status_code == 200
    bob_write = resp.json()
    assert bob_write["write_provenance"] == "agent", (
        "PROVENANCE BYPASS: agent-token write tagged supervisor"
    )

    bob_row_uid = await pool.fetchval(
        "SELECT user_id FROM memories WHERE id = $1", bob_write["id"],
    )
    assert bob_row_uid == "bob", (
        f"USER_ID STAMPING WRONG: bob's row stamped {bob_row_uid!r}"
    )

    # 6. Each user's row carries that user's id in the user_id column.
    # (The DB also still has the rows visible via raw pool query — the
    # superuser-bypass means we can read both rows directly here even
    # though the application path goes through acquire().)
    rows = await pool.fetch(
        "SELECT id, user_id FROM memories WHERE id = ANY($1::text[])",
        [alice_write["id"], bob_write["id"]],
    )
    by_id = {r["id"]: r["user_id"] for r in rows}
    assert by_id[alice_write["id"]] == "alice"
    assert by_id[bob_write["id"]] == "bob", (
        "USER_ID STAMPING WRONG: bob's row not tagged 'bob'"
    )

    # 7. Revoking Bob's token kills his access; Alice unaffected ---------
    flipped = await revoke_token(pool, bob_row.token_hash)
    assert flipped is True

    async with _client(app, bob_token) as bob_revoked:
        resp = await bob_revoked.get("/mcp/whoami")
    assert resp.status_code == 401, (
        "REVOCATION INEFFECTIVE: revoked token still authorised"
    )

    async with _client(app, alice_token) as alice:
        resp = await alice.get("/mcp/whoami")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "alice"
