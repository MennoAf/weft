"""Weft MCP server — FastMCP with stdio/HTTP transport."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import asyncpg
import redis.asyncio as aioredis
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from weft.auth import (
    current_caller_mode,
    current_user_id,
    extract_user_id_from_header,
    parse_caller_mode_header,
)
from weft.cache import Cache, NullCache
from weft.config import WeftConfig, load_config
from weft.db.connection import create_pool
from weft.db.migrations import run_migrations
from weft.db.schema import ensure_vector_dimensions
from weft.embeddings import get_provider
from weft.embeddings.base import EmbeddingProvider
from weft.mcp.oauth_consent import handle_consent
from weft.mcp.oauth_metadata import handle_authorization_server_metadata
from weft.mcp.slack_commands import handle_slash_checkin
from weft.mcp.tool_usage import ToolUsageMiddleware
from weft.cost_enforcement import cost_enforcement_loop
from weft.scheduler import (
    canary_audit_loop,
    daily_brief_loop,
    discord_bot_loop,
    loom_awareness_loop,
    memory_hygiene_loop,
    quarantine_review_loop,
    reask_feedback_loop,
    scheduler_loop,
    slack_sync_loop,
    trigger_evaluation_loop,
)
from weft.seed import seed_memories

logger = logging.getLogger(__name__)

FALLBACK_PATH = Path.home() / ".weft" / "fallback.md"

# Pool health check interval in seconds
_KEEPALIVE_INTERVAL = 300  # 5 minutes
# Fallback snapshot refresh interval in seconds
_FALLBACK_REFRESH_INTERVAL = 1800  # 30 minutes
# Startup retry config
_STARTUP_MAX_RETRIES = 5
_STARTUP_BASE_DELAY = 1.0  # seconds, doubles each retry

# Valid outbound connector values
_VALID_OUTBOUND_CONNECTORS = {"slack", "discord", "none", ""}


class UserIdentityMiddleware(BaseHTTPMiddleware):
    """Extract user identity + caller mode from Authorization header.

    Phase 2.5 (commit-bound caller mode): the Authorization header is
    resolved through :func:`weft.credentials.lookup_token` and the
    resolved row is the authoritative source for both ``user_id`` and
    ``caller_mode``. The ``X-Weft-Caller-Mode`` header is no longer
    trusted on its own — an agent holding a valid token can no longer
    forge ``caller_mode=supervisor`` and bypass the Phase 2 poisoning
    defense.

    Two auth paths on ``/mcp``, in order:

    1. **Token row lookup.** sha256 of the bearer is matched against
       ``weft_tokens``. Hit → row owns the request: ``current_user_id``
       comes from ``row.user_id``, ``current_caller_mode`` from
       ``row.caller_mode``. Legacy ``WEFT_API_KEY`` clients land here
       too via the L3 bootstrap row.
    2. **Supabase JWT (when ``oauth_enabled`` is True).** If no row
       matches, try JWKS-verified Supabase JWT decode (same as today).
       On success, ``current_user_id = sub`` and
       ``current_caller_mode = 'supervisor'`` until a future change
       adds a scope claim that distinguishes agent-issued tokens.

    Header narrowing: ``X-Weft-Caller-Mode`` is honoured **only** when
    the resolved credential mode is ``supervisor`` — Jason can downgrade
    himself to ``agent`` locally for testing without minting a real
    agent token. When the credential mode is ``agent``, the header is
    ignored. This is the change that closes the escalation path.

    Health checks and other non-``/mcp`` endpoints are unauthenticated.
    On those, JWT extraction is best-effort: missing or invalid tokens
    silently fall through with ``current_user_id=None``.

    On 401, a ``WWW-Authenticate`` header points the client at our
    protected-resource metadata document so MCP clients can discover
    the Supabase authorization server (RFC 9728).
    """

    def __init__(
        self,
        app,
        oauth_enabled: bool = False,
        supabase_url: str | None = None,
        pool_getter: "Callable[[], asyncpg.Pool | None] | None" = None,
        auth_required: bool = True,
    ):
        super().__init__(app)
        self._oauth_enabled = oauth_enabled
        self._supabase_url = (supabase_url or "").rstrip("/")
        self._pool_getter = pool_getter
        # ``auth_required`` mirrors the prior ``api_key or oauth_enabled``
        # gate: in local-dev mode (``WEFT_ENV != production`` and OAuth
        # off) anonymous traffic is still allowed through unauthenticated.
        self._auth_required = auth_required

    def _unauthorized(self, message: str) -> JSONResponse:
        headers: dict[str, str] = {}
        if self._oauth_enabled:
            # Point the MCP client at the protected-resource metadata
            # document. Per RFC 9728 the client follows that to discover
            # the authorization server (Supabase) and start the OAuth
            # dance.
            headers["WWW-Authenticate"] = (
                'Bearer realm="weft", '
                'resource_metadata="/.well-known/oauth-protected-resource"'
            )
        return JSONResponse({"error": message}, status_code=401, headers=headers)

    def _resolve_caller_mode(
        self, credential_mode: str, request: Request
    ) -> str:
        """Apply the L4 header-narrowing rule.

        * Credential mode is the floor. An agent credential **always**
          resolves to ``agent`` regardless of any header — that's the
          rule that closes the X-Weft-Caller-Mode escalation.
        * For supervisor credentials, the header is allowed to
          downgrade the request (supervisor → agent) so Jason can
          locally test agent code paths without minting an agent
          token.
        """
        if credential_mode != "supervisor":
            return credential_mode
        return parse_caller_mode_header(
            request.headers.get("x-weft-caller-mode"),
        )

    async def dispatch(self, request: Request, call_next):
        from weft.credentials import lookup_token

        is_mcp_path = request.url.path.startswith("/mcp")

        if self._auth_required and is_mcp_path:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return self._unauthorized("missing authorization header")
            bearer = auth_header[7:]

            pool = self._pool_getter() if self._pool_getter else None
            row = None
            if pool is not None:
                row = await lookup_token(pool, bearer)

            if row is not None:
                user_id = row.user_id
                effective_mode = self._resolve_caller_mode(
                    row.caller_mode, request,
                )
            elif self._oauth_enabled:
                # Supabase-issued OAuth access token. Falls back to None
                # on any failure (signature, expiry, audience). OAuth
                # tokens currently always resolve as supervisor — a
                # follow-up will add a scope claim that distinguishes
                # agent-issued OAuth tokens.
                user_id = extract_user_id_from_header(auth_header)
                if not user_id:
                    return self._unauthorized("invalid token")
                effective_mode = self._resolve_caller_mode(
                    "supervisor", request,
                )
            else:
                return self._unauthorized("invalid token")

            ctx_token = current_user_id.set(user_id)
            mode_token = current_caller_mode.set(effective_mode)
            try:
                return await call_next(request)
            finally:
                current_caller_mode.reset(mode_token)
                current_user_id.reset(ctx_token)

        # Non-/mcp paths and unauthenticated mode: best-effort identity
        # extraction from a JWT if present, header-driven caller mode
        # (no credential to clamp against).
        auth_header = request.headers.get("authorization")
        user_id = extract_user_id_from_header(auth_header)
        caller_mode = parse_caller_mode_header(
            request.headers.get("x-weft-caller-mode"),
        )
        token = current_user_id.set(user_id)
        mode_token = current_caller_mode.set(caller_mode)
        try:
            return await call_next(request)
        finally:
            current_caller_mode.reset(mode_token)
            current_user_id.reset(token)


@dataclass
class AppContext:
    pool: asyncpg.Pool
    cache: Cache | NullCache
    embedding: EmbeddingProvider
    config: WeftConfig
    # Episode-tier embedder. Resolved once at startup via
    # ``resolve_episode_embedder`` (honours ``WEFT_EPISODE_EMBEDDER`` env
    # override, falls back to ``embedding``). Stored on AppContext so the
    # write hooks reuse a single provider instance instead of paying the
    # construction cost on every episode write.
    episode_embedding: EmbeddingProvider | None = None
    _keepalive_task: asyncio.Task | None = field(default=None, repr=False)
    _fallback_task: asyncio.Task | None = field(default=None, repr=False)
    _scheduler_task: asyncio.Task | None = field(default=None, repr=False)


async def _connect_with_retry(
    connect_fn,
    label: str,
    max_retries: int = _STARTUP_MAX_RETRIES,
    base_delay: float = _STARTUP_BASE_DELAY,
):
    """Call *connect_fn* with exponential backoff on failure.

    Returns the result of *connect_fn()* on success.
    Raises the last exception after exhausting retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await connect_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s connection failed (attempt %d/%d): %s — retrying in %.1fs",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s connection failed after %d attempts: %s",
                    label, max_retries + 1, exc,
                )
    raise last_exc  # type: ignore[misc]


async def _pool_keepalive(ctx: AppContext) -> None:
    """Periodically ping the pool; recreate it if connections are stale."""
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        try:
            async with ctx.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            logger.warning("Pool health check failed: %s — recreating pool", exc)
            old_pool = ctx.pool
            try:
                new_pool = await create_pool(ctx.config)
                ctx.pool = new_pool
                logger.info("Pool recreated successfully")
            except Exception as create_exc:
                logger.error("Failed to recreate pool: %s", create_exc)
                continue
            # Best-effort close of the old pool
            try:
                await old_pool.close()
            except Exception as e:
                logger.debug("Old pool close failed: %s", e, exc_info=True)


async def _redis_keepalive(ctx: AppContext) -> None:
    """Periodically ping Redis; recreate the client if the connection is stale."""
    while True:
        await asyncio.sleep(_KEEPALIVE_INTERVAL)
        try:
            await ctx.cache._redis.ping()
        except Exception as exc:
            logger.warning("Redis health check failed: %s — reconnecting", exc)
            old_redis = ctx.cache._redis
            try:
                new_redis = aioredis.from_url(
                    ctx.config.redis.url, decode_responses=True,
                )
                await new_redis.ping()
                ctx.cache._redis = new_redis
                logger.info("Redis reconnected successfully")
            except Exception as create_exc:
                logger.error("Failed to reconnect Redis: %s", create_exc)
                continue
            try:
                await old_redis.aclose()
            except Exception as e:
                logger.debug("Old Redis close failed: %s", e, exc_info=True)


async def _refresh_fallback(ctx: AppContext) -> None:
    """Periodically re-export the fallback snapshot."""
    while True:
        await asyncio.sleep(_FALLBACK_REFRESH_INTERVAL)
        await _write_fallback_snapshot(ctx.pool)


async def _write_fallback_snapshot(pool: asyncpg.Pool) -> None:
    """Export active memories to the fallback markdown file."""
    try:
        from weft.exporter import export_memories

        content = await export_memories(pool, format="md")
        FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_PATH.write_text(content, encoding="utf-8")
        logger.info("Fallback snapshot written to %s", FALLBACK_PATH)
    except Exception as e:
        logger.warning("Failed to write fallback snapshot: %s", e)


def _validate_outbound_connector_env() -> None:
    """Validate WEFT_OUTBOUND_CONNECTOR env var. Raises ValueError on invalid values.

    Valid values (case-insensitive, whitespace-trimmed): slack, discord, none, or unset.
    """
    raw = os.environ.get("WEFT_OUTBOUND_CONNECTOR")
    connector = (raw or "").strip().lower()
    if connector not in _VALID_OUTBOUND_CONNECTORS:
        raise ValueError(
            f"WEFT_OUTBOUND_CONNECTOR must be one of {sorted(_VALID_OUTBOUND_CONNECTORS - {''})} or unset "
            f"(got: {raw!r})"
        )


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize database, Redis, and embedding provider."""
    from weft.correlation import CorrelationFilter

    config = load_config()

    # Validate WEFT_OUTBOUND_CONNECTOR at startup
    _validate_outbound_connector_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Filter must be on the HANDLER (not the logger) so it applies to
    # propagated records from child loggers (mcp, uvicorn, etc.).
    corr_filter = CorrelationFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(corr_filter)

    # Database (with retry)
    pool = await _connect_with_retry(
        lambda: create_pool(config),
        "Postgres",
    )
    await run_migrations(pool)
    # Recreate pool so ALL connections get the pgvector codec via init
    # callback. The first pool's connections were created before migrations
    # installed the vector extension, so their codec registration silently
    # failed.
    await pool.close()
    pool = await _connect_with_retry(
        lambda: create_pool(config),
        "Postgres",
    )

    # Phase 2.5 L3: bootstrap a credential row for the legacy
    # WEFT_API_KEY env var so existing clients keep working when L4
    # flips middleware to lookup_token. Idempotent across restarts.
    from weft.credentials import bootstrap_legacy_api_key

    await bootstrap_legacy_api_key(
        pool,
        api_key=config.api_key,
        default_user_id=os.environ.get("WEFT_DEFAULT_USER_ID") or None,
    )

    # Self-heal vector dimensions if config changed since last run
    migrated_tables = await ensure_vector_dimensions(pool, config.embedding.dimensions)

    # Export fallback snapshot
    await _write_fallback_snapshot(pool)

    # Redis (optional — use NullCache if not configured)
    r: aioredis.Redis | None = None
    cache: Cache | NullCache
    _redis_url = config.redis.url
    if _redis_url:
        try:
            async def _connect_redis():
                client = aioredis.from_url(_redis_url, decode_responses=True)
                await client.ping()
                return client

            r = await _connect_with_retry(_connect_redis, "Redis", max_retries=1, base_delay=0.5)
            cache = Cache(r)
        except Exception as exc:
            logger.warning("Redis unavailable, using NullCache: %s", exc)
            r = None
            cache = NullCache()
    else:
        logger.info("No Redis URL configured, using NullCache")
        cache = NullCache()

    # Embedding provider (validate eagerly to catch config errors at startup)
    embedding = get_provider(
        config.embedding.provider,
        model_name=config.embedding.model,
        dimensions=config.embedding.dimensions,
    )
    try:
        test_vec = await embedding.embed("startup validation")
        logger.info(
            "Embedding provider %s validated (%d dims)",
            embedding.provider_name, len(test_vec),
        )
    except Exception as exc:
        logger.error(
            "Embedding provider %s failed validation: %s. "
            "Check API keys and model configuration.",
            embedding.provider_name, exc,
        )
        raise

    # Re-embed rows nulled by dimension migration (best-effort)
    if migrated_tables:
        try:
            from weft.db.reembed import auto_reembed
            results = await auto_reembed(pool, embedding, migrated_tables)
            total = sum(results.values())
            if total:
                logger.info("Auto re-embedded %d rows across %d tables", total, len(results))
        except Exception as exc:
            logger.warning("Auto re-embed failed (will retry on next query): %s", exc)

    # P2.1: backfill episodes.embedding for rows added before the column
    # existed (or rows whose embedding generation failed previously).
    # ``reembed_table`` is idempotent — once every row has an embedding,
    # this becomes a cheap empty SELECT on subsequent boots. Uses the
    # episode-tier embedder helper so a future ``WEFT_EPISODE_EMBEDDER``
    # override picks up automatically. The same provider instance is then
    # threaded onto ``AppContext.episode_embedding`` so the P2.2 write
    # hooks (create/close/graduate) embed inline without rebuilding it.
    episode_embedder: EmbeddingProvider | None = None
    try:
        from weft.db.reembed import (
            reembed_table,
            resolve_episode_embedder,
        )
        episode_embedder = resolve_episode_embedder(config)
        backfilled = await reembed_table(pool, "episodes", episode_embedder)
        if backfilled:
            logger.info("Backfilled episode embeddings for %d rows", backfilled)
    except Exception as exc:
        logger.warning("Episode embedding backfill failed (non-fatal): %s", exc)

    # Seed memories on fresh installs (best-effort, never blocks startup)
    try:
        seeded = await seed_memories(pool, embedding)
        if seeded:
            logger.info("Seeded %d starter memories", seeded)
    except Exception as exc:
        logger.warning("Seed bootstrapping failed (non-fatal): %s", exc)

    ctx = AppContext(
        pool=pool,
        cache=cache,
        embedding=embedding,
        config=config,
        episode_embedding=episode_embedder,
    )
    _app_ctx_ref.ctx = ctx

    # Start background tasks
    ctx._keepalive_task = asyncio.create_task(_pool_keepalive(ctx))
    _redis_task = asyncio.create_task(_redis_keepalive(ctx)) if r else None
    ctx._fallback_task = asyncio.create_task(_refresh_fallback(ctx))
    ctx._scheduler_task = asyncio.create_task(
        scheduler_loop(
            pool,
            interval=config.alert.poll_interval,
            batch_size=config.alert.batch_size,
        )
    )
    ctx._slack_sync_task = asyncio.create_task(
        slack_sync_loop(
            pool,
            embedding,
            interval=config.slack_sync.interval,
            smart_ingest=config.slack_sync.smart_ingest,
        )
    )
    ctx._daily_brief_task = asyncio.create_task(
        daily_brief_loop(
            pool,
            brief_time=config.daily_brief.time,
            brief_tz=config.daily_brief.timezone,
            brief_channel=config.daily_brief.channel,
        )
    )
    ctx._discord_bot_task = asyncio.create_task(discord_bot_loop(pool))
    ctx._loom_awareness_task = asyncio.create_task(
        loom_awareness_loop(pool)
    )
    ctx._memory_hygiene_task = asyncio.create_task(
        memory_hygiene_loop(pool)
    )
    ctx._trigger_eval_task = asyncio.create_task(
        trigger_evaluation_loop(pool)
    )
    ctx._reask_feedback_task = asyncio.create_task(
        reask_feedback_loop(pool)
    )
    ctx._canary_audit_task = asyncio.create_task(
        canary_audit_loop(pool, embedding)
    )
    ctx._quarantine_review_task = None
    if config.quarantine_review.enabled:
        ctx._quarantine_review_task = asyncio.create_task(
            quarantine_review_loop(
                pool,
                interval=config.quarantine_review.interval,
                limit=config.quarantine_review.limit,
                concurrency=config.quarantine_review.concurrency,
                model=config.quarantine_review.model,
            )
        )
    ctx._cost_enforcement_task = None
    if config.cost_enforcement.enabled and config.cost_enforcement.daily_limit_usd > 0:
        ctx._cost_enforcement_task = asyncio.create_task(
            cost_enforcement_loop(pool, config=config.cost_enforcement)
        )

    # OAuth in the new architecture is delegated to Supabase's OAuth 2.1
    # server. Weft hosts only the consent page (see weft/mcp/oauth_consent.py)
    # and validates incoming Supabase-issued tokens via the existing JWKS
    # path in weft.auth. There's no per-process state to install here.
    if config.oauth_enabled:
        logger.info(
            "OAuth 2.1 enabled — Supabase as authorization server "
            "(supabase_url=%s)",
            config.supabase_url or "<unset>",
        )

    try:
        yield ctx
    finally:
        _app_ctx_ref.ctx = None
        for task in (ctx._keepalive_task, _redis_task, ctx._fallback_task, ctx._scheduler_task, ctx._slack_sync_task, ctx._daily_brief_task, ctx._discord_bot_task, ctx._loom_awareness_task, ctx._memory_hygiene_task, ctx._trigger_eval_task, ctx._reask_feedback_task, ctx._canary_audit_task, ctx._quarantine_review_task, ctx._cost_enforcement_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await pool.close()
        if r is not None:
            await r.aclose()


class _AppCtxRef:
    """Module-level holder for the AppContext, set during lifespan."""
    ctx: AppContext | None = None


_app_ctx_ref = _AppCtxRef()

_config_for_middleware = load_config()

# ---------------------------------------------------------------------------
# Middleware wiring (module scope).
# ---------------------------------------------------------------------------
# Phase 2.5: every Authorization header is resolved through
# ``weft.credentials.lookup_token``. Legacy ``WEFT_API_KEY`` clients still
# work because :func:`bootstrap_legacy_api_key` (called from lifespan)
# inserts a row matching the env-var on cold start. OAuth JWTs remain a
# fallback when no token row matches and ``oauth_enabled`` is True.
# ---------------------------------------------------------------------------


def _middleware_pool_getter() -> asyncpg.Pool | None:
    """Return the live pool, or None if lifespan hasn't installed it.

    Pre-lifespan request paths (a startup health probe slipping in
    before ``run_migrations`` finishes) get None here, which makes the
    middleware skip the credential lookup. With ``auth_required=True``
    that resolves to a 401 — fail-closed — until the pool comes up."""
    ctx = _app_ctx_ref.ctx
    if ctx is None:
        return None
    return ctx.pool


# Auth is required in production OR whenever OAuth is on. Local dev with
# neither stays unauthenticated so ``uv run python -m weft`` keeps
# working without setting a key.
_auth_required = (
    _config_for_middleware.is_production or _config_for_middleware.oauth_enabled
)
logger.info(
    "user_identity_middleware: auth_required=%s oauth_enabled=%s is_production=%s",
    _auth_required,
    _config_for_middleware.oauth_enabled,
    _config_for_middleware.is_production,
)
user_identity_middleware = Middleware(
    UserIdentityMiddleware,
    oauth_enabled=_config_for_middleware.oauth_enabled,
    supabase_url=_config_for_middleware.supabase_url,
    pool_getter=_middleware_pool_getter,
    auth_required=_auth_required,
)

mcp = FastMCP("weft", lifespan=lifespan)
mcp.add_middleware(ToolUsageMiddleware(_middleware_pool_getter))


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    """Health check endpoint — unauthenticated, used by Fly.io."""
    ctx = _app_ctx_ref.ctx
    if ctx is None:
        return JSONResponse(
            {"status": "unhealthy", "error": "server starting up"},
            status_code=503,
        )
    try:
        async with ctx.pool.acquire(timeout=3.0) as conn:
            await conn.fetchval("SELECT 1")
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return JSONResponse({"status": "unhealthy", "error": str(exc)}, status_code=503)


@mcp.custom_route("/slack/commands", methods=["POST"])
async def slack_commands(request: Request) -> JSONResponse:
    """Slack slash command handler — /checkin mood 3 sleep 7 energy 4."""
    ctx = _app_ctx_ref.ctx
    if ctx is None:
        return JSONResponse({"text": "Server starting up, try again shortly."}, status_code=200)
    return await handle_slash_checkin(request, ctx.pool)


@mcp.custom_route("/mcp/", methods=["GET", "POST", "DELETE"])
async def mcp_trailing_slash(request: Request) -> Response:
    """Redirect /mcp/ → /mcp with correct scheme behind TLS-terminating proxies."""
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    url = request.url.replace(scheme=scheme, path="/mcp")
    return Response(status_code=307, headers={"Location": str(url)})


# ---------------------------------------------------------------------------
# OAuth 2.1 — Supabase OAuth Server architecture.
# ---------------------------------------------------------------------------
# In this architecture Supabase hosts the entire authorization server
# (authorize/token/register/JWKS endpoints all live at
# ``<project>.supabase.co/auth/v1/...``). Weft only:
#
#   * Publishes RFC 9728 protected-resource metadata pointing the MCP
#     client at Supabase as the auth server.
#   * Hosts the consent UI page Supabase redirects users to after they
#     reach ``/auth/v1/oauth/authorize``. The Supabase dashboard's
#     "Authorization URL Path" must be set to ``/oauth/consent``.
#
# When ``WEFT_OAUTH_ENABLED=0`` neither route is registered — the path
# stays byte-identical to the API-key-only deployment.
# ---------------------------------------------------------------------------

if _config_for_middleware.oauth_enabled:

    @mcp.custom_route(
        "/.well-known/oauth-protected-resource", methods=["GET"],
    )
    async def _oauth_resource_metadata(request: Request) -> Response:
        from weft.config import load_config

        cfg = load_config()
        # Resource = our public origin. Authorization server = Supabase's
        # GoTrue mount, which is reached at ``<project>.supabase.co/auth/v1``.
        # RFC 8414 clients fetch
        # ``<authorization_servers[i]>/.well-known/oauth-authorization-server``,
        # so the URL we publish must include the /auth/v1 suffix —
        # Supabase does NOT mount discovery at the project root.
        resource = (cfg.oauth_issuer or "").rstrip("/")
        if not resource:
            base = request.base_url
            resource = f"{base.scheme}://{base.netloc}".rstrip("/")
        supabase_root = (cfg.supabase_url or "").rstrip("/")
        auth_servers = (
            [f"{supabase_root}/auth/v1"] if supabase_root else []
        )
        return JSONResponse({
            "resource": resource,
            "authorization_servers": auth_servers,
            "bearer_methods_supported": ["header"],
            "scopes_supported": ["openid", "email"],
        })

    @mcp.custom_route(
        "/.well-known/oauth-authorization-server", methods=["GET"],
    )
    async def _oauth_authorization_server_metadata(
        request: Request,
    ) -> Response:
        # Mirror of Supabase's RFC 8414 metadata at our origin — works
        # around MCP OAuth clients that skip RFC 9728's protected-resource
        # redirection and look for auth-server metadata at the resource
        # origin directly. See weft/mcp/oauth_metadata.py for the why.
        return await handle_authorization_server_metadata(request)

    @mcp.custom_route("/oauth/consent", methods=["GET"])
    async def _oauth_consent(request: Request) -> Response:
        return await handle_consent(request)
