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

import asyncpg
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from weft.board import ACT_ALLOWLIST, act, assemble_board
from weft.board_page import BOARD_HTML

logger = logging.getLogger(__name__)

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
    result = await assemble_board(pool, days=days, include_snoozed=include_snoozed)
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


def create_app(pool: asyncpg.Pool) -> Starlette:
    """Build the board overlay ASGI app bound to `pool`.

    Returns a plain Starlette app (no lifespan/startup wiring) so tests can
    drive it directly over `httpx.ASGITransport` without a real socket.
    """
    app = Starlette(
        routes=[
            Route("/", _get_page, methods=["GET"]),
            Route("/board", _get_board, methods=["GET"]),
            Route("/act", _post_act, methods=["POST"]),
        ],
    )
    app.state.pool = pool
    return app


def run(pool: asyncpg.Pool, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the overlay with uvicorn — LOCALHOST ONLY.

    Refuses any host other than localhost/127.0.0.1/::1: this overlay has no
    auth, so binding it to 0.0.0.0 (or any other interface) would expose an
    unauthenticated triage-write endpoint to the network. This is a dev
    surface for a local agent/UI, not a production deployment target.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"board_server.run() only binds to localhost/127.0.0.1 — "
            f"refusing host={host!r}"
        )
    import uvicorn

    uvicorn.run(create_app(pool), host=host, port=port)


def main() -> None:
    """CLI entrypoint: `uv run python -m weft.board_server`.

    Loads the deployment's normal config/pool (same path the MCP server
    uses) and serves the overlay on 127.0.0.1 only.
    """
    import asyncio

    from weft.config import load_config
    from weft.db.connection import create_pool, register_pgvector_codec
    from weft.db.migrations import run_migrations

    async def _bootstrap() -> asyncpg.Pool:
        config = load_config()
        pool = await create_pool(config)
        await run_migrations(pool)
        await register_pgvector_codec(pool)
        return pool

    pool = asyncio.run(_bootstrap())
    run(pool)


if __name__ == "__main__":
    main()
