"""Stable, read-only diagnostics for the Weft installation.

Doctor deliberately does not use the application startup path: that path may
start infrastructure, apply migrations, bootstrap credentials, or construct a
paid provider.  The default probes are bounded and read-only, and every probe
is replaceable through :class:`DoctorDependencies` for deterministic tests.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

CHECK_IDS = ("D001", "D002", "D003", "D004", "D005", "D006", "D007", "D008", "D009", "D010")
_CHECK_NAMES = {
    "D001": "configuration precedence",
    "D002": "configuration safety and secret handling",
    "D003": "runtime engine/backend",
    "D004": "database connectivity and role",
    "D005": "Redis connectivity",
    "D006": "migration ledger",
    "D007": "conflicting configuration",
    "D008": "provider credential prerequisite",
    "D009": "MCP/client environment",
    "D010": "filesystem/resource/package wiring",
}

_SECRET_KEYS = {
    "token", "api_key", "access_token", "secret", "password", "credential",
    "authorization", "body", "response", "raw", "dsn", "url", "path", "error",
}
_URL_RE = re.compile(r"(?:postgres(?:ql)?|redis(?:s)?|https?)://[^\s\"']+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|private|tmp|var|opt|srv|Volumes|workspace|workspaces)(?:/[^\s\"']*)?", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\b(?:sk|pk|tok|key|secret|bearer)[-_][A-Za-z0-9._~-]{6,}\b", re.IGNORECASE)


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    name: str
    status: str
    severity: str
    remedy: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "remedy": self.remedy,
            "details": _redact(self.details),
        }


@dataclass
class DoctorReport:
    checks: list[DoctorCheck]
    unexpected_error: bool = False
    invalid_configuration: bool = False

    @property
    def exit_code(self) -> int:
        if self.unexpected_error:
            return 3
        if self.invalid_configuration:
            return 2
        if any(check.status == "fail" for check in self.checks):
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "checks": [check.to_dict() for check in self.checks],
            "exit_code": self.exit_code,
            "read_only": True,
        }


@dataclass
class DoctorDependencies:
    """Probe seam for Doctor.

    A dependency returns a mapping (or an awaitable mapping).  Values are
    intentionally small facts, not raw exceptions or provider responses.
    ``config_loader`` may return the real WeftConfig or a test fixture mapping.
    """

    config_loader: Callable[[], Any] = field(default_factory=lambda: _default_config_loader)
    engine_probe: Callable[[], Any] = field(default_factory=lambda: _default_engine_probe)
    database_probe: Callable[[Any], Any] = field(default_factory=lambda: _default_database_probe)
    redis_probe: Callable[[Any], Any] = field(default_factory=lambda: _default_redis_probe)
    migration_probe: Callable[[Any], Any] = field(default_factory=lambda: _default_migration_probe)
    provider_probe: Callable[[Any], Any] = field(default_factory=lambda: _default_provider_probe)
    client_probe: Callable[[Any], Any] = field(default_factory=lambda: _default_client_probe)
    resource_probe: Callable[[], Any] = field(default_factory=lambda: _default_resource_probe)
    conflict_probe: Callable[[], Any] = field(default_factory=lambda: _default_conflict_probe)


def load_doctor_dependencies() -> DoctorDependencies:
    """Return production probes. Kept as a function so CLI tests can replace it."""
    return DoctorDependencies()


def _default_config_loader() -> Any:
    from weft.config import load_config
    return load_config()


def _default_engine_probe() -> dict[str, Any]:
    # Discovery only; never invoke Docker/Podman or ask it to start anything.
    found = [name for name in ("docker", "podman") if shutil.which(name)]
    return {"available": bool(found), "backend": found[0] if found else None}


def _endpoint(value: str, default_port: int) -> tuple[str, int]:
    parsed = urlsplit(value)
    host = parsed.hostname
    if not host:
        raise ValueError("endpoint has no host")
    return host, parsed.port or default_port


def _default_database_probe(config: Any) -> Any:
    async def probe() -> dict[str, Any]:
        try:
            import asyncpg
            url = getattr(getattr(config, "database", None), "url", None)
            if not url:
                return {"reachable": False, "reason": "database endpoint is not configured"}
            timeout = getattr(getattr(config, "database", None), "command_timeout", None) or 3.0
            conn = await asyncpg.connect(url, timeout=min(float(timeout), 5.0), statement_cache_size=0)
            try:
                role = await conn.fetchval("SELECT current_user")
                return {"reachable": True, "role": str(role) if role else None}
            finally:
                await conn.close()
        except Exception as exc:
            return {"reachable": False, "reason": _safe_reason(exc)}
    return probe()


def _default_redis_probe(config: Any) -> Any:
    async def probe() -> dict[str, Any]:
        client = None
        try:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                return {"reachable": False, "reason": "Redis client dependency is unavailable"}
            url = getattr(getattr(config, "redis", None), "url", None)
            if not url:
                return {"reachable": False, "reason": "Redis endpoint is not configured"}
            client = aioredis.from_url(url, socket_connect_timeout=3.0, socket_timeout=3.0)
            await client.ping()
            return {"reachable": True}
        except Exception as exc:
            return {"reachable": False, "reason": _safe_reason(exc)}
        finally:
            if client is not None:
                close = getattr(client, "aclose", None) or getattr(client, "close", None)
                if close:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
    return probe()


def _default_migration_probe(config: Any) -> Any:
    async def probe() -> dict[str, Any]:
        try:
            import asyncpg
            from weft.db.migrations import MIGRATIONS
            url = getattr(getattr(config, "database", None), "url", None)
            if not url:
                return {"reachable": False}
            conn = await asyncpg.connect(url, timeout=5.0, statement_cache_size=0)
            try:
                rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
                applied = {int(row["version"]) for row in rows}
                expected = {int(version) for version, _, _ in MIGRATIONS}
                return {
                    "reachable": True,
                    "consistent": applied == expected,
                    "pending": len(expected - applied),
                }
            except Exception:
                # A missing ledger is a diagnosed schema condition, not a raw DB dump.
                return {"reachable": True, "consistent": False, "reason": "migration ledger unavailable"}
            finally:
                await conn.close()
        except Exception as exc:
            return {"reachable": False, "reason": _safe_reason(exc)}
    return probe()


def _default_provider_probe(config: Any) -> dict[str, Any]:
    # Check the configured generation provider because Doctor must identify a
    # missing paid-provider prerequisite without constructing or calling it.
    generation = getattr(config, "text_generation", None)
    provider = str(getattr(generation, "provider", "anthropic"))
    required = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    variable = required.get(provider)
    if variable is None:
        return {"configured": False, "provider": provider, "credential_required": True}
    config_key = getattr(config, "api_key", None) if provider == "anthropic" else None
    present = bool(os.environ.get(variable) or config_key)
    return {
        "configured": present,
        "provider": provider,
        "credential_required": True,
        "credential_present": present,
        "credential_variable": variable,
    }


def _default_client_probe(config: Any) -> dict[str, Any]:
    # Only inspect names and executable availability. Never print env values.
    command = shutil.which("weft") or sys.executable
    transport = os.environ.get("WEFT_TRANSPORT", "stdio").strip().lower()
    compatible = transport in {"stdio", "streamable-http", "sse", "http"}
    return {"compatible": compatible, "transport": transport, "command_available": bool(command)}


def _default_resource_probe() -> dict[str, Any]:
    try:
        from weft.resources import compose_file_path
        with compose_file_path() as path:
            return {"available": path.is_file(), "package": "installed"}
    except Exception:
        return {"available": False, "package": "unavailable"}


def _default_conflict_probe() -> dict[str, Any]:
    # DATABASE_URL and WEFT_DATABASE_URL are aliases; differing values are an
    # actionable conflict, but values themselves must never leave this function.
    first = os.environ.get("DATABASE_URL")
    second = os.environ.get("WEFT_DATABASE_URL")
    return {"conflict": bool(first and second and first != second)}


def _safe_reason(exc: BaseException) -> str:
    # Exception text is not evidence-safe (asyncpg errors can include DSNs).
    name = type(exc).__name__.lower()
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "connection timed out"
    if isinstance(exc, (ConnectionRefusedError, ConnectionError, OSError)):
        return "connection unavailable"
    return "probe failed"


def _redact(value: Any, key: str | None = None) -> Any:
    if key and any(part in key.lower() for part in _SECRET_KEYS):
        if isinstance(value, (str, bytes)):
            return "[redacted]"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact(item, key) for item in value]
    if isinstance(value, Path):
        return "[redacted-path]"
    if isinstance(value, bytes):
        return "[redacted]"
    if isinstance(value, str):
        value = _URL_RE.sub("[redacted-url]", value)
        value = _PATH_RE.sub("[redacted-path]", value)
        value = _TOKEN_RE.sub("[redacted-token]", value)
        return value
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"value": value}


def _status(result: Mapping[str, Any], positive: str, *, warning: bool = False) -> str:
    if result.get(positive) is True:
        return "pass"
    return "warn" if warning else "fail"


def _run(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _check(id_: str, status: str, severity: str, remedy: str, details: Mapping[str, Any]) -> DoctorCheck:
    return DoctorCheck(id_, _CHECK_NAMES[id_], status, severity, remedy, dict(details))


def run_doctor(*, dependencies: DoctorDependencies | None = None) -> DoctorReport:
    """Run all ten checks without changing infrastructure, config, or data."""
    deps = dependencies or load_doctor_dependencies()
    checks: list[DoctorCheck] = []
    try:
        config = _run(deps.config_loader())
    except (ValueError, TypeError):
        reason = "invalid configuration"
        checks.extend([
            _check("D001", "fail", "error", "Fix the configuration syntax and values, then rerun Doctor.", {"valid": False, "reason": reason}),
            _check("D002", "skip", "info", "Resolve the configuration error before checking secret handling.", {"blocked_by": "D001"}),
        ])
        for id_ in CHECK_IDS[2:]:
            checks.append(_check(id_, "skip", "info", "Resolve invalid configuration first, then rerun Doctor.", {"blocked_by": "D001"}))
        return DoctorReport(checks, invalid_configuration=True)
    except Exception:
        return DoctorReport([_check("D001", "fail", "error", "Inspect configuration loading and rerun Doctor.", {"valid": False})], unexpected_error=True)

    try:
        config_map = _mapping(config)
        d001 = _mapping(config_map)
        valid = d001.get("valid", True)
        checks.append(_check("D001", "pass" if valid else "fail", "error" if not valid else "info", "Fix configuration sources and values, then rerun Doctor." if not valid else "No action required.", {"valid": bool(valid), "sources": d001.get("sources", ["defaults", "environment"])}))
        checks.append(_check("D002", "pass", "info", "Keep credentials in the environment or a secret manager; do not place them in receipts.", {"redaction": "enabled", "credential_values": "not reported"}))

        engine = _mapping(_run(deps.engine_probe()))
        checks.append(_check("D003", _status(engine, "available", warning=True), "warning", "Install or start an approved runtime separately; Doctor never starts it.", {"available": bool(engine.get("available")), "backend": engine.get("backend", "not detected")}))

        database = _mapping(_run(deps.database_probe(config)))
        db_status = _status(database, "reachable")
        if db_status == "pass" and database.get("role") not in (None, "weft_app"):
            db_status = "fail"
        checks.append(_check("D004", db_status, "error" if db_status == "fail" else "info", "Verify the database endpoint, TLS, credentials, and restricted runtime role." if db_status == "fail" else "No action required.", {"reachable": bool(database.get("reachable")), "role": database.get("role", "not reported")}))

        redis = _mapping(_run(deps.redis_probe(config)))
        checks.append(_check("D005", _status(redis, "reachable", warning=True), "warning", "Start or configure Redis separately if caching is enabled; Doctor never starts it.", {"reachable": bool(redis.get("reachable"))}))

        if db_status != "pass":
            checks.append(_check("D006", "skip", "info", "Restore read-only database connectivity, then rerun Doctor.", {"blocked_by": "D004"}))
        else:
            migration = _mapping(_run(deps.migration_probe(config)))
            migration_status = "pass" if migration.get("consistent") is True else "fail"
            checks.append(_check("D006", migration_status, "error" if migration_status == "fail" else "info", "Run the separately authorized owner migration command; Doctor never applies migrations." if migration_status == "fail" else "No action required.", {"consistent": bool(migration.get("consistent")), "pending": int(migration.get("pending", 0) or 0)}))

        conflict = _mapping(_run(deps.conflict_probe()))
        conflict_status = "fail" if conflict.get("conflict") is True else "pass"
        checks.append(_check("D007", conflict_status, "error" if conflict_status == "fail" else "info", "Remove conflicting duplicate configuration values and keep one authoritative source." if conflict_status == "fail" else "No action required.", {"conflict": bool(conflict.get("conflict"))}))

        provider = _mapping(_run(deps.provider_probe(config)))
        provider_status = "pass" if provider.get("configured", True) and provider.get("credential_present", True) else "fail"
        checks.append(_check("D008", provider_status, "error" if provider_status == "fail" else "info", "Configure the selected provider and its credential, or select the local provider." if provider_status == "fail" else "No action required.", {"provider": provider.get("provider", "configured"), "credential_required": bool(provider.get("credential_required", False)), "credential_present": bool(provider.get("credential_present", not provider.get("credential_required", False)))}))

        client = _mapping(_run(deps.client_probe(config)))
        client_status = "pass" if client.get("compatible", True) else "fail"
        checks.append(_check("D009", client_status, "error" if client_status == "fail" else "info", "Use a supported MCP transport and ensure the client can launch the `weft mcp` command." if client_status == "fail" else "No action required.", {"compatible": bool(client.get("compatible", True)), "transport": client.get("transport", "not reported")}))

        resource = _mapping(_run(deps.resource_probe()))
        resource_status = "pass" if resource.get("available") is True else "fail"
        checks.append(_check("D010", resource_status, "error" if resource_status == "fail" else "info", "Reinstall Weft with package resources, then rerun Doctor." if resource_status == "fail" else "No action required.", {"available": bool(resource.get("available")), "package": resource.get("package", "not reported")}))
        return DoctorReport(checks)
    except (ValueError, TypeError):
        return DoctorReport(checks, invalid_configuration=True)
    except Exception:
        return DoctorReport(checks, unexpected_error=True)


__all__ = ["CHECK_IDS", "DoctorCheck", "DoctorDependencies", "DoctorReport", "load_doctor_dependencies", "run_doctor"]
