"""Weft board — localhost overlay server (weft-board-epic Task 9).

LOCALHOST DEV SURFACE — NOT a production-exposed server. This is a thin ASGI
front for the board triage loop so a local agent/UI can `GET /board` and
`POST /act` without going through the MCP transport. It has no auth, no
CSRF protection, and no rate limiting, so `run()` below refuses to bind
anywhere other than localhost/127.0.0.1 — never 0.0.0.0.

Both routes are thin wrappers around the existing `weft.board` functions;
this module reimplements neither the fan-out (`assemble_board`) nor the
dispatch/allowlist/event-append (`act`) — see their docstrings for the
`weft-board-epic Task 9` cross-reference. `POST /act` re-checks the tool
against `ACT_ALLOWLIST` BEFORE calling `act()` so the rejection is a clean
4xx with no write attempted at all, even though `act()` itself would also
reject an unlisted tool (PRD §Critical Implementation Notes: the allowlist
is the security boundary, checked before dispatch).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from weft.auth import current_user_id
from weft.board import ACT_ALLOWLIST, act, assemble_board
from weft.board_page import BOARD_HTML
from weft.db.connection import acquire

logger = logging.getLogger(__name__)


def _owner_user_id() -> str | None:
    """Resolve the single owner this localhost overlay acts as.

    Unlike the MCP server, the overlay has no auth middleware to bind a caller
    identity — but triage writes NEED one: `board_triage_events.user_id` is
    `NOT NULL DEFAULT` the `app.user_id` GUC, and the tracker/alert tables are
    RLS-scoped. Without an identity every write fails (NULL user_id / RLS). The
    overlay is single-user by design, so bind the deployment owner:
    `WEFT_DEFAULT_USER_ID` if set, else this installation's canonical id.
    """
    uid = os.environ.get("WEFT_DEFAULT_USER_ID")
    if uid:
        return uid
    try:
        from weft.config.user_identity import get_user_id

        return get_user_id()
    except Exception:  # noqa: BLE001 - identity is best-effort here
        return None


@asynccontextmanager
async def _as_owner(pool: asyncpg.Pool):
    """Bind the owner identity and open an RLS-scoped connection for a write.

    Sets `current_user_id` (so `acquire()` issues `SET LOCAL app.user_id`) and
    holds one scoped connection, which every nested `get_db(pool)` /
    `acquire(pool)` inside `act()` reuses — mirroring how the MCP tool handlers
    wrap their bodies. This is what lets the tracker UPDATE and the
    `board_triage_events` INSERT both see a non-empty `app.user_id`.
    """
    token = current_user_id.set(_owner_user_id())
    try:
        async with acquire(pool):
            yield
    finally:
        current_user_id.reset(token)

# Fields `act()` requires besides `tool` (PRD §Interfaces `POST /act` body).
_ACT_REQUIRED_FIELDS = (
    "args", "item_id", "source", "kind",
    "urgency_at_surface", "age_days_at_surface", "verb",
)


async def _get_page(request: Request) -> HTMLResponse:
    """GET / — the triage dashboard (a single self-contained static page).

    Holds no business logic: it fetches `GET /board` and posts item actions to
    `/act`, speaking only the `weft_board` contract (weft-board-epic Goal #4).
    """
    return HTMLResponse(BOARD_HTML)


def _int_param(request: Request, name: str, default: int) -> int:
    """Parse a positive-int query param, falling back on absent/garbage."""
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


async def _get_board(request: Request) -> JSONResponse:
    """GET /board — read-only fan-out via `assemble_board` (no writes).

    Honors `?days=` (urgency horizon) and `?include_snoozed=` so the dashboard's
    horizon selector and the PRD-named board params are actually wired through,
    rather than always assembling with defaults.
    """
    pool: asyncpg.Pool = request.app.state.pool
    days = _int_param(request, "days", 7)
    include_snoozed = request.query_params.get("include_snoozed", "").lower() in (
        "1", "true", "yes",
    )
    # Bind the owner identity so the board is RLS-scoped to the owner (and the
    # "no identity resolved" warning doesn't fire), rather than relying on the
    # connection role's RLS behavior.
    token = current_user_id.set(_owner_user_id())
    try:
        result = await assemble_board(
            pool, days=days, include_snoozed=include_snoozed,
        )
    finally:
        current_user_id.reset(token)
    return JSONResponse(result)


async def _post_act(request: Request) -> JSONResponse:
    """POST /act — validate against the write-tool allowlist, then dispatch
    through `board.act()`.

    Body: `{tool, args, item_id, source, kind, urgency_at_surface,
    age_days_at_surface, verb, snooze_duration_days?}` — mirrors the
    `weft_board_act` MCP tool's parameters (`weft/mcp/tools.py`) so the two
    callers of `act()` stay in sync.
    """
    pool: asyncpg.Pool = request.app.state.pool

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    tool = payload.get("tool")

    # SECURITY BOUNDARY (Epic Critical Implementation Note): reject any tool
    # not on the write-tool allowlist BEFORE dispatch — no write, no
    # triage-event append. `act()` enforces this too, but checking here
    # first means a disallowed tool never even reaches the dispatch call.
    if tool not in ACT_ALLOWLIST:
        return JSONResponse(
            {
                "error": (
                    f"tool {tool!r} is not on the triage write-tool "
                    f"allowlist ({sorted(ACT_ALLOWLIST)})"
                ),
            },
            status_code=400,
        )

    missing = [f for f in _ACT_REQUIRED_FIELDS if f not in payload]
    if missing:
        return JSONResponse(
            {"error": f"missing required fields: {missing}"}, status_code=400,
        )

    try:
        # Bind the owner identity + a scoped connection so the write tool AND
        # record_triage_event both run with app.user_id set (else the triage
        # event's NOT NULL user_id / the tables' RLS fail → 500).
        async with _as_owner(pool):
            result = await act(
                pool,
                tool=tool,
                args=payload["args"],
                item_id=payload["item_id"],
                source=payload["source"],
                kind=payload["kind"],
                urgency_at_surface=payload["urgency_at_surface"],
                age_days_at_surface=payload["age_days_at_surface"],
                verb=payload["verb"],
                snooze_duration_days=payload.get("snooze_duration_days"),
            )
    except ValueError as exc:
        # act() also enforces the allowlist internally (belt-and-suspenders)
        # and raises ValueError for other bad-arg cases — surface as a 4xx,
        # never a 500, and no write has occurred at this point.
        return JSONResponse({"error": str(exc)}, status_code=400)
    except KeyError as exc:
        return JSONResponse({"error": f"missing arg: {exc}"}, status_code=400)

    return JSONResponse({"result": result})


_ROUTES = [
    Route("/", _get_page, methods=["GET"]),
    Route("/board", _get_board, methods=["GET"]),
    Route("/act", _post_act, methods=["POST"]),
]


def create_app(pool: asyncpg.Pool) -> Starlette:
    """Build the board overlay ASGI app bound to an already-created `pool`.

    Returns a plain Starlette app (no lifespan) so tests can drive it over
    `httpx.ASGITransport` in the same event loop that built the pool. The CLI
    entrypoint uses `_serving_app` instead — see the loop note there.
    """
    app = Starlette(routes=_ROUTES)
    app.state.pool = pool
    return app


def _serving_app(config) -> Starlette:
    """Build the app the CLI serves under uvicorn.

    The pool MUST be created inside the server's event loop: asyncpg binds
    connections to the loop they were opened in, and `uvicorn.run()` starts a
    fresh loop. Creating the pool beforehand (e.g. via `asyncio.run(...)`, whose
    loop is then closed) makes every query raise InterfaceError/RuntimeError
    against the dead loop. So the pool is opened in a Starlette lifespan, which
    uvicorn runs in its own loop, and closed on shutdown.
    """
    from contextlib import asynccontextmanager

    from weft.db.connection import create_pool, register_pgvector_codec
    from weft.db.migrations import run_migrations

    @asynccontextmanager
    async def lifespan(app: Starlette):
        pool = await create_pool(config)
        await run_migrations(pool)
        await register_pgvector_codec(pool)
        app.state.pool = pool
        try:
            yield
        finally:
            await pool.close()

    return Starlette(routes=_ROUTES, lifespan=lifespan)


def run(config, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the overlay with uvicorn — LOCALHOST ONLY.

    Takes a config (not a pre-built pool): the pool is opened in-loop via the
    lifespan (see `_serving_app`). Refuses any host other than
    localhost/127.0.0.1/::1: this overlay has no auth, so binding it to 0.0.0.0
    (or any other interface) would expose an unauthenticated triage-write
    endpoint to the network.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"board_server.run() only binds to localhost/127.0.0.1 — "
            f"refusing host={host!r}"
        )
    import uvicorn

    uvicorn.run(_serving_app(config), host=host, port=port)


def main() -> None:
    """CLI entrypoint: `uv run python -m weft.board_server`.

    Loads the deployment's normal config and serves the overlay on 127.0.0.1
    only. Migrations/codec run in the lifespan, inside the serving loop.
    """
    from weft.config import load_config

    run(load_config())


if __name__ == "__main__":
    main()
