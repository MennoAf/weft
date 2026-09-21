#!/usr/bin/env python3
"""Future-use hosted RC smoke; plan-only unless explicitly authorized.

No endpoint is contacted by default. Execution requires all of ``--execute``,
``--approve-hosted``, and ``--approve-destructive``, plus an HTTPS endpoint whose
hostname is explicitly allowlisted. Credentials are referenced by environment
variable name and are never printed or persisted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SCHEMA = "rc-fl-21.hosted-smoke.v1"
DEFAULT_TIMEOUT = 10.0
MAX_TIMEOUT = 60.0
RECOVERY_LIMIT = 10
_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class SmokeError(RuntimeError):
    """A bounded smoke assertion or authorization failure."""


def _timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be numeric") from exc
    if not 0 < result <= MAX_TIMEOUT:
        raise argparse.ArgumentTypeError(f"timeout must be >0 and <= {MAX_TIMEOUT:g} seconds")
    return result


def _allowed_hostnames(allowed_hosts: list[str]) -> set[str]:
    normalized: set[str] = set()
    for host in allowed_hosts:
        value = host.strip().lower().rstrip(".")
        if not value or ":" in value or "/" in value or "@" in value:
            raise SmokeError("allowlisted hosts must be bare DNS hostnames")
        normalized.add(value)
    return normalized


def validate_endpoint(endpoint: str, allowed_hosts: list[str]) -> urllib.parse.SplitResult:
    """Return the one canonical MCP URL accepted by this smoke.

    The path is intentionally exact. In particular, trailing slash variants are
    rejected instead of silently rewritten, so the URL used for auth and every
    MCP request is auditable and identical.
    """
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise SmokeError("endpoint URL is malformed") from exc
    if parsed.scheme.lower() != "https":
        raise SmokeError("hosted smoke endpoint must use https")
    if not hostname or parsed.username is not None or parsed.password is not None:
        raise SmokeError("endpoint must have a hostname and no embedded credentials")
    if port is not None and port != 443:
        raise SmokeError("endpoint must not specify a non-standard port")
    if hostname.lower().rstrip(".") not in _allowed_hostnames(allowed_hosts):
        raise SmokeError("endpoint hostname is not explicitly allowlisted")
    if parsed.path != "/mcp":
        raise SmokeError("endpoint must be exactly the /mcp MCP path (no trailing slash)")
    if parsed.query or parsed.fragment:
        raise SmokeError("endpoint must not contain query or fragment")
    if parsed.scheme != "https" or parsed.hostname != hostname:
        raise SmokeError("endpoint scheme/hostname must use canonical lowercase form")
    # Reject spelling differences (trailing-dot host and explicit :443) rather
    # than making a second URL that could diverge from the requested endpoint.
    if endpoint != f"https://{hostname}/mcp":
        raise SmokeError("endpoint must be the canonical https://host/mcp URL")
    return urllib.parse.urlsplit(endpoint)


def _canonical_health_endpoint(parsed: urllib.parse.SplitResult) -> str:
    """Construct health URL from validated components, never string input."""
    return urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", "/healthz", "", ""))


def build_plan(endpoint: str, allowed_hosts: list[str], timeout: float) -> dict[str, Any]:
    parsed = validate_endpoint(endpoint, allowed_hosts)
    canonical_endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", "/mcp", "", ""))
    return {
        "schema": SCHEMA,
        "mode": "plan",
        "endpoint": canonical_endpoint,
        "health_endpoint": _canonical_health_endpoint(parsed),
        "allowlisted_hostname": parsed.hostname,
        "timeout_seconds": timeout,
        "synthetic_scope": {
            "project_id": "fresh UUID per execution",
            "user_id": "authenticated bearer owner (fresh credential scope per execution)",
            "remember_contract": "weft_remember has no user_id argument; owner comes from Authorization credential",
            "recall_contract": "weft_recall uses the same authenticated bearer owner and explicit project_id",
        },
        "steps": [
            "GET /healthz and require HTTP 200 with status=ok",
            "POST /mcp JSON-RPC initialize and require result plus MCP-Session-Id",
            "tools/call weft_remember with one synthetic fact and explicit project_id (owner from bearer auth)",
            "tools/call weft_recall with the same explicit project_id and bearer owner scope",
            "finally: tools/call weft_forget hard=true for every tracked memory id under the bearer owner scope",
        ],
        "authorization": {
            "required_flags": ["--execute", "--approve-hosted", "--approve-destructive"],
            "credential_reference": "--credential-env NAME (environment value never logged or persisted)",
            "network": "HTTPS endpoint only; exact /mcp path; redirects rejected and bearer is never forwarded",
        },
        "cleanup": "mandatory finally cleanup; cleanup failure makes receipt failed",
    }


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: Any


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from forwarding Authorization to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibTransport:
    """Small real transport used only after explicit authorization."""

    def __init__(self, token: str, timeout: float):
        self._token = token
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def request(
        self,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        request_headers = {"Accept": "application/json", **(headers or {})}
        data = None
        method = "GET" if payload is None else "POST"
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme == "https" and parsed.path == "/mcp" and not parsed.query and not parsed.fragment:
            request_headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read(1_000_000)
                final_url = response.geturl()
                if final_url != url:
                    raise SmokeError("HTTP redirect or URL rewrite rejected")
                body = json.loads(raw.decode("utf-8")) if raw else None
                return HttpResponse(response.status, dict(response.REDACTEDitems()), body)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise SmokeError("HTTP redirects are not accepted") from exc
            raise SmokeError(f"HTTP request failed with status {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SmokeError(f"bounded HTTP request failed: {type(exc).__name__}") from exc


def _tool_payload(response: HttpResponse, tool: str) -> dict[str, Any]:
    if response.status != 200 or not isinstance(response.body, dict):
        raise SmokeError(f"{tool} returned an invalid HTTP/JSON response")
    if "error" in response.body:
        raise SmokeError(f"{tool} returned a JSON-RPC error")
    result = response.body.get("result")
    if not isinstance(result, dict):
        raise SmokeError(f"{tool} returned no JSON-RPC result")
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                try:
                    decoded = json.loads(item["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    return decoded
    return result


def _call_tool(
    transport: Any,
    endpoint: str,
    name: str,
    arguments: dict[str, Any],
    request_id: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    response = transport.request(
        endpoint,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
        headers,
    )
    return _tool_payload(response, name)


def _synthetic_memory_id(value: Any) -> str | None:
    """Return an ID safe to use for compensation, never coercing arbitrary data."""
    if isinstance(value, str) and _MEMORY_ID_RE.fullmatch(value):
        return value
    return None


def _memory_records(value: Any) -> list[dict[str, Any]]:
    """Extract only documented flat/nested memory records from recall output."""
    if not isinstance(value, dict):
        return []
    results = value.get("results")
    if not isinstance(results, list):
        return []
    records: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        nested = item.get("memory")
        if isinstance(nested, dict):
            records.append(nested)
            continue
        payload = item.get("payload")
        if item.get("kind") == "memory" and isinstance(payload, dict):
            records.append(payload)
        elif "id" in item:
            records.append(item)
    return records


def _exact_recovery_ids(recalled: dict[str, Any], project_id: str, content: str) -> list[str]:
    """Select candidates only when ID, project, and full sentinel content match."""
    ids: list[str] = []
    for record in _memory_records(recalled):
        memory_id = _synthetic_memory_id(record.get("id"))
        if (
            memory_id is not None
            and record.get("project_id") == project_id
            and record.get("content") == content
        ):
            ids.append(memory_id)
    return list(dict.fromkeys(ids))


def run_smoke(
    endpoint: str,
    allowed_hosts: list[str],
    timeout: float,
    transport_factory: Callable[[str, float], Any],
    credential: str,
) -> dict[str, Any]:
    parsed = validate_endpoint(endpoint, allowed_hosts)
    canonical_endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.hostname or "", "/mcp", "", ""))
    health_endpoint = _canonical_health_endpoint(parsed)
    run_id = uuid.uuid4().hex
    project_id = "rc-fl-21-project-" + run_id
    content = "RC-FL-21 hosted smoke synthetic sentinel " + run_id
    tracked_ids: list[str] = []
    checks: dict[str, str] = {}
    errors: list[str] = []
    cleanup_errors: list[str] = []
    cleanup_recovered_ids: list[str] = []
    cleanup_forgotten_ids: list[str] = []
    cleanup_recovery_attempted = False
    remember_attempted = False
    transport = transport_factory(credential, timeout)
    headers: dict[str, str] = {}
    request_id = 1
    status = "failed"
    try:
        health = transport.request(health_endpoint)
        if health.status != 200 or not isinstance(health.body, dict) or health.body.get("status") != "ok":
            raise SmokeError("health check did not return HTTP 200 status=ok")
        checks["health"] = "passed"
        init = transport.request(
            canonical_endpoint,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "weft-rc-hosted-smoke", "version": "1"},
                },
            },
            {"Accept": "application/json"},
        )
        if init.status != 200 or not isinstance(init.body, dict) or "result" not in init.body:
            raise SmokeError("MCP initialize did not return a result")
        session_id = next((value for key, value in init.REDACTEDitems() if key.lower() == "mcp-session-id"), None)
        if not session_id:
            raise SmokeError("MCP initialize did not return MCP-Session-Id")
        headers = {"MCP-Session-Id": session_id, "MCP-Protocol-Version": "2025-06-18"}
        checks["mcp_initialize"] = "passed"
        request_id += 1
        # remember has no user_id parameter. Its owner is the authenticated
        # bearer credential held by the transport; project_id isolates the run.
        remember_attempted = True
        remembered = _call_tool(
            transport,
            canonical_endpoint,
            "weft_remember",
            {
                "content": content,
                "type": "fact",
                "topic": ["rc-fl-21-smoke"],
                "project_id": project_id,
                "check_contradictions": False,
            },
            request_id,
            headers,
        )
        # Capture a syntactically safe ID before validating any other response
        # fields. A malformed response must not hide a write we can compensate.
        memory_id = _synthetic_memory_id(remembered.get("id"))
        if memory_id is not None:
            tracked_ids.append(memory_id)
        if memory_id is None or remembered.get("content") != content or remembered.get("project_id") != project_id:
            raise SmokeError("remember did not return the tracked synthetic memory")
        checks["remember"] = "passed"
        request_id += 1
        try:
            recalled = _call_tool(
                transport,
                canonical_endpoint,
                "weft_recall",
                {
                    "query": content,
                    "project_id": project_id,
                    "limit": 5,
                    "mode": "keyword",
                },
                request_id,
                headers,
            )
        except Exception as exc:
            raise SmokeError(f"recall failed: {type(exc).__name__}") from exc
        if memory_id not in _exact_recovery_ids(recalled, project_id, content):
            raise SmokeError("recall did not return the exact synthetic memory")
        checks["recall"] = "passed"
        status = "passed"
    except Exception as exc:  # noqa: BLE001 - receipt contains only bounded error classes
        if isinstance(exc, SmokeError):
            errors.append(str(exc))
        else:
            errors.append(type(exc).__name__ + ": bounded smoke assertion failed")
    finally:
        # If remember may have succeeded but did not expose a usable ID, make
        # one bounded, exact-marker lookup for compensation. We never delete a
        # result that lacks the exact project and complete sentinel content.
        if remember_attempted and not tracked_ids:
            cleanup_recovery_attempted = True
            try:
                request_id += 1
                recovered = _call_tool(
                    transport,
                    canonical_endpoint,
                    "weft_recall",
                    {
                        "query": content,
                        "project_id": project_id,
                        "limit": RECOVERY_LIMIT,
                        "mode": "keyword",
                    },
                    request_id,
                    headers,
                )
                cleanup_recovered_ids = _exact_recovery_ids(recovered, project_id, content)
                if not cleanup_recovered_ids:
                    cleanup_errors.append("cleanup_incomplete: no exact synthetic memory recovered")
            except Exception:
                cleanup_errors.append("cleanup_incomplete: recovery recall failed")
        tracked_ids.extend(memory_id for memory_id in cleanup_recovered_ids if memory_id not in tracked_ids)
        for memory_id in tracked_ids:
            try:
                request_id += 1
                deleted = _call_tool(
                    transport,
                    canonical_endpoint,
                    "weft_forget",
                    {"memory_id": memory_id, "hard": True},
                    request_id,
                    headers,
                )
                if (
                    deleted.get("memory_id") != memory_id
                    or deleted.get("deleted") is not True
                    or deleted.get("hard") is not True
                ):
                    raise SmokeError("forget did not confirm hard deletion of the tracked memory")
                cleanup_forgotten_ids.append(memory_id)
            except Exception:
                cleanup_errors.append("cleanup_incomplete: synthetic forget failed")
        if cleanup_errors:
            errors.extend(cleanup_errors)
            status = "failed"
        checks["cleanup"] = "passed" if not cleanup_errors else "failed"
    return {
        "schema": SCHEMA,
        "status": status,
        "run_id": run_id,
        "endpoint": canonical_endpoint,
        "health_endpoint": health_endpoint,
        "endpoint_host": parsed.hostname,
        "synthetic_project_id": project_id,
        "synthetic_user_scope": "authenticated bearer owner; remember has no user_id argument",
        "checks": checks,
        "tracked_count": len(tracked_ids),
        "cleanup": {
            "attempted": True,
            "recovery_attempted": cleanup_recovery_attempted,
            "recovered_ids": cleanup_recovered_ids,
            "forgotten_ids": cleanup_forgotten_ids,
            "uncertainty": "incomplete" if cleanup_errors else "none",
            "errors": cleanup_errors,
        },
        "errors": errors,
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--allow-host", action="append", required=True)
    parser.add_argument("--credential-env", default="WEFT_HOSTED_SMOKE_TOKEN", help="name of env var; value is never logged")
    parser.add_argument("--timeout", type=_timeout, default=DEFAULT_TIMEOUT)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-hosted", action="store_true")
    parser.add_argument("--approve-destructive", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.endpoint, args.allow_host, args.timeout)
        if not (args.execute or args.approve_hosted or args.approve_destructive):
            output: dict[str, Any] = plan
            if args.receipt:
                _write_receipt(args.receipt, output)
            print(json.dumps(output, indent=2, sort_keys=True))
            return 0
        if not (args.execute and args.approve_hosted and args.approve_destructive):
            raise SmokeError("execution requires --execute, --approve-hosted, and --approve-destructive")
        if not args.credential_env.isidentifier() or not args.credential_env.isupper():
            raise SmokeError("credential env reference must be an uppercase environment variable name")
        credential = os.environ.get(args.credential_env)
        if not credential:
            raise SmokeError("credential environment reference is unset")
        receipt = run_smoke(args.endpoint, args.allow_host, args.timeout, UrllibTransport, credential)
        if args.receipt:
            _write_receipt(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "passed" else 1
    except SmokeError as exc:
        print(f"RC-FL-21 hosted smoke refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
