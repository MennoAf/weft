"""Real local gateway for the faithful LongMemEval benchmark.

The gateway calls the same async MCP tool functions used by the public server.
It is deliberately lazy: importing this module performs no provider, database,
or network work.  Construction is allowed only after strict local disposable
DSN validation and an identity query against the connected database.
"""
from __future__ import annotations

import asyncio
import inspect
from contextvars import Token
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


class GatewayError(RuntimeError):
    """Raised when the local public-tool integration cannot be made safely."""


_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}
_ALLOWED_TOOLS = {"weft_remember", "weft_recall", "weft_prime", "weft_handoff"}


def validate_local_dsn(dsn: str) -> dict[str, Any]:
    """Validate a disposable local PostgreSQL URL without opening a socket.

    A hostname prefix is intentionally insufficient: ``localhost.evil`` and
    omitted ports are rejected.  The database and user must visibly identify a
    disposable LongMemEval sandbox; this prevents accidentally pointing the
    benchmark at an ordinary local Weft database.
    """
    if not isinstance(dsn, str) or not dsn.strip():
        raise GatewayError("LONGMEMEVAL_DATABASE_URL is required")
    parsed = urlsplit(dsn)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise GatewayError("benchmark DSN must use postgresql:// or postgres://")
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise GatewayError("benchmark DSN host must be exactly localhost, 127.0.0.1, or ::1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise GatewayError("benchmark DSN must include an explicit valid local port") from exc
    if port is None or not (1 <= port <= 65535):
        raise GatewayError("benchmark DSN must include an explicit valid local port")
    database = unquote(parsed.path.lstrip("/"))
    user = unquote(parsed.username or "")
    if not database or not user:
        raise GatewayError("benchmark DSN must include database and user identity")
    marker = f"{database} {user}".lower()
    if not any(part in marker for part in ("longmemeval", "lme_bench", "benchmark")):
        raise GatewayError("database/user must identify a disposable LongMemEval benchmark sandbox")
    if any(part in database.lower() for part in ("prod", "production", "weft")):
        raise GatewayError("refusing a production or ordinary Weft database")
    return {"host": parsed.hostname, "port": port, "database": database, "user": user}


@dataclass(slots=True)
class FaithfulGateway:
    """Bound public MCP calls to one isolated owner/project/agent."""

    app: Any
    ctx: Any
    owner_id: str
    project_id: str
    agent_id: str = "faithful-s36"
    _pool: Any = None
    calls: list[dict[str, Any]] | None = None

    background_disclosure: str = (
        "public weft_prime/weft_recall retain production background tasks; "
        "this gateway does not silently suppress them."
    )
    lifecycle_timeout_seconds: float = 10.0

    def bind_project(self, project_id: str) -> None:
        """Reject scope mutation; create a new gateway for another case."""
        if project_id != self.project_id:
            raise GatewayError("project scope is immutable; create a new gateway")

    async def _replay_pending(self) -> int | None:
        """Read pending replay depth when the isolated DB exposes it."""
        if self._pool is None:
            return None
        try:
            return int(await self._pool.fetchval("SELECT count(*) FROM replay_queue WHERE status = 'pending'"))
        except Exception as exc:
            raise GatewayError(f"unable to verify replay queue state: {exc}") from exc

    async def _settle_tasks(self, before: set[asyncio.Task]) -> dict[str, Any]:
        """Drain request-created tasks, including tasks outside AppContext registry."""
        current = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
        owned = [task for task in current - before if not task.done()]
        if owned:
            done, pending = await asyncio.wait(owned, timeout=self.lifecycle_timeout_seconds)
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise GatewayError("request-triggered background task did not settle")
        errors = []
        for task in owned:
            if task.cancelled():
                continue
            try:
                error = task.exception()
            except Exception as exc:
                error = exc
            if error is not None:
                errors.append(f"{task.get_name()}: {error}")
        if errors:
            raise GatewayError("request-triggered task failed: " + "; ".join(errors))
        return {"created": len(owned), "settled": len(owned)}

    async def call(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Invoke one public tool and verify all request-triggered work settles."""
        if name not in _ALLOWED_TOOLS:
            raise GatewayError(f"public tool is not permitted by faithful gateway: {name}")
        if not isinstance(arguments, Mapping):
            raise GatewayError("tool arguments must be an object")
        allowed = {
            "weft_remember": {"content", "type", "topic", "source", "confidence", "project_id", "agent_id", "workspace_id", "check_contradictions", "pinned", "review_after", "project_facets", "preference_metadata"},
            "weft_recall": {"query", "topic", "type", "status", "project_id", "agent_id", "limit", "threshold", "mode", "retrieval_mode", "user_id", "tier"},
            "weft_prime": {"project_id", "agent_id", "budget_tokens", "query", "disclosure", "mode"},
            "weft_handoff": {"summary", "in_progress", "next_steps", "open_questions", "project_id", "agent_id"},
        }[name]
        unknown = set(arguments) - allowed
        if unknown:
            raise GatewayError(f"unknown arguments for {name}: {sorted(unknown)}")
        args = dict(arguments)
        self._scope(args, "project_id", self.project_id)
        self._scope(args, "agent_id", self.agent_id)
        if name == "weft_recall":
            self._scope(args, "user_id", self.owner_id)
        if name == "weft_remember":
            self._scope(args, "workspace_id", None, reject_non_null=True)
            args.setdefault("check_contradictions", True)
            args.setdefault("source", "conversation")
        pending_before = await self._replay_pending()
        if pending_before:
            raise GatewayError("pending replay_queue rows make faithful execution unsafe")
        before = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
        # Prime's consolidation path can reach the Anthropic detector.  Guard
        # the actual lazy constructor, let normal empty-queue behavior run,
        # then restore it only after every request-created task has settled.
        import weft.views.belief_detector as belief_detector
        attempted: list[str] = []
        original_get_client = belief_detector._get_client
        def forbidden_client():
            attempted.append("belief_detector._get_client")
            raise GatewayError("Anthropic replay detector is forbidden in faithful execution")
        belief_detector._get_client = forbidden_client
        from weft.auth import current_user_id
        from weft.mcp import tools
        identity_token = current_user_id.set(self.owner_id)
        try:
            function = getattr(tools, name)
            result = function(self.ctx, **args)
            if inspect.isawaitable(result):
                result = await result
            lifecycle = await self._settle_tasks(before)
            pending_after = await self._replay_pending()
            if pending_after:
                raise GatewayError("public tool left pending replay_queue rows")
            if attempted:
                raise GatewayError("forbidden Anthropic detector was reached")
        finally:
            current_user_id.reset(identity_token)
            belief_detector._get_client = original_get_client
        if not isinstance(result, Mapping):
            raise GatewayError(f"{name} returned a non-object result")
        if self.calls is None:
            self.calls = []
        self.calls.append({"name": name, "arguments": dict(args), "result": dict(result)})
        return result

    def _scope(self, args: dict[str, Any], key: str, expected: Any, *, reject_non_null: bool = False) -> None:
        if key in args:
            value = args[key]
            if reject_non_null and value is not None:
                raise GatewayError(f"{key} is not allowed in the faithful gateway")
            if not reject_non_null and value not in (None, expected):
                raise GatewayError(f"{key} cannot override the faithful run scope")
        args[key] = expected

    async def close(self) -> None:
        """Close only resources created by this gateway."""
        close = getattr(self._pool, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


async def create_local_gateway(
    dsn: str,
    *,
    owner_id: str,
    project_id: str,
    agent_id: str = "faithful-s36",
    embedding: Any | None = None,
) -> FaithfulGateway:
    """Construct a real gateway after DSN, identity, and provider preflight.

    Optional embedding is intended for offline tests.  In execution, FastEmbed
    is constructed by the caller before this function or supplied explicitly;
    this function never fabricates a fake database or provider.
    """
    identity = validate_local_dsn(dsn)
    if not owner_id.strip() or not project_id.strip():
        raise GatewayError("owner_id and project_id must be non-empty")
    # Fail closed before creating any identity-scoped connection:
    # weft.db.connection skips the ``SET LOCAL app.user_id`` GUC for
    # identities it cannot validate and weft.store stamps row user_id from
    # that GUC, so a rejected owner would write NULL-user_id rows in RLS
    # global scope instead of failing.
    if not owner_id.replace("-", "").replace("_", "").isalnum():
        raise GatewayError(f"owner_id must be an [A-Za-z0-9_-] identity; refused: {owner_id!r}")
    import asyncpg
    from weft.cache import NullCache
    from weft.config import WeftConfig
    from weft.db.connection import create_pool
    from weft.embeddings import get_provider
    from weft.mcp.server import AppContext

    config = WeftConfig()
    config.database.url = dsn
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    config.retrieval.recovery_mode = "off"
    pool = await create_pool(config)
    try:
        row = await pool.fetchrow("SELECT current_database() AS database, current_user AS user")
        if row is None or row["database"] != identity["database"] or row["user"] != identity["user"]:
            raise GatewayError("connected database identity does not match disposable DSN")
        if embedding is None:
            embedding = get_provider(config.embedding.provider, model_name=config.embedding.model, dimensions=config.embedding.dimensions)
        app = AppContext(pool=pool, cache=NullCache(), embedding=embedding, config=config)
        request_context = SimpleNamespace(lifespan_context=app)
        ctx = SimpleNamespace(request_context=request_context, transport="stdio")
        return FaithfulGateway(app=app, ctx=ctx, owner_id=owner_id, project_id=project_id, agent_id=agent_id, _pool=pool)
    except Exception:
        await pool.close()
        raise


__all__ = ["FaithfulGateway", "GatewayError", "create_local_gateway", "validate_local_dsn"]
