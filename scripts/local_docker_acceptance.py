#!/usr/bin/env python3
"""Run the bounded local Docker RC acceptance contract.

This runner starts the separately-scoped local Compose app, calls the actual
image CMD through streamable HTTP, and records a redacted machine-readable
receipt. It uses a unique Compose project and synthetic project scope; cleanup
is attempted in a ``finally`` block and never targets the normal ``weft``
Compose project.

Author:  Jason Bauman
Version: 0.1.0
Date:    2026-09-12
Python:  >=3.9

Dependencies:
    Standard library only; the Docker CLI and candidate image are runtime
    prerequisites. FastEmbed is loaded by the app image, not by this runner.

Usage:
    See bottom of file for run commands.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_COMPOSE_FILE = "docker-compose.local.yml"
DEFAULT_IMAGE = "weft-rc-local:1.0.0rc1"
MCP_PROTOCOL_VERSION = "2025-03-26"


@dataclass
class AcceptanceReceipt:
    """Machine-readable run evidence with secrets excluded by construction."""

    run_id: str
    project_name: str
    synthetic_project: str
    owner_user_id: str
    image: str
    compose_file: str
    started_at: float
    status: str = "running"
    checks: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe receipt data, omitting runtime secrets."""
        return {
            "schema": "weft.local-docker-acceptance.v1",
            "run_id": self.run_id,
            "project_name": self.project_name,
            "synthetic_project": self.synthetic_project,
            "owner_user_id": self.owner_user_id,
            "image": self.image,
            "compose_file": self.compose_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "checks": self.checks,
            "warnings": self.warnings,
            "error": self.error,
            "cleanup_scope": {
                "compose_project": self.project_name,
                "volumes": "unique Compose project only",
                "host_ports": "app loopback only; postgres/redis unpublished",
            },
        }


class AcceptanceFailure(RuntimeError):
    """Raised when a required acceptance assertion fails."""


def _free_port() -> int:
    """Reserve no port; return an currently-unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _docker_compose_prefix() -> list[str]:
    """Return the portable Docker Compose CLI prefix."""
    docker = shutil.which("docker")
    if docker is None:
        raise AcceptanceFailure("Docker CLI not found; acceptance requires Docker Desktop/Engine")
    return [docker, "compose"]


def _run_compose(
    prefix: list[str],
    compose_file: Path,
    project_name: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run one Compose command with bounded output and no shell interpolation."""
    command = [*prefix, "-f", str(compose_file), "-p", project_name, *args]
    return subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _write_receipt(path: Path, receipt: AcceptanceReceipt) -> None:
    """Write the receipt atomically so interrupted runs cannot leave valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{receipt.run_id}.tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _decode_http_payload(raw: bytes) -> dict[str, Any]:
    """Decode a JSON or Server-Sent Events MCP response."""
    text = raw.decode("utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        data_lines = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        if not data_lines:
            raise AcceptanceFailure(f"MCP response was neither JSON nor SSE: {text[:500]!r}")
        try:
            payload = json.loads(data_lines[-1])
        except json.JSONDecodeError as exc:
            raise AcceptanceFailure(f"MCP SSE data was not JSON: {data_lines[-1]!r}") from exc
    if not isinstance(payload, dict):
        raise AcceptanceFailure(f"MCP response must be an object, got {type(payload).__name__}")
    if "error" in payload:
        raise AcceptanceFailure(f"MCP JSON-RPC error: {payload['error']}")
    return payload


def _http_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """POST one MCP JSON-RPC request and return payload plus response REDACTED"""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode_http_payload(response.read()), dict(response.REDACTEDitems())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AcceptanceFailure(f"HTTP {exc.code} from {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise AcceptanceFailure(f"HTTP request to {url} failed: {exc.reason}") from exc


def _wait_for_health(base_url: str, timeout: float) -> None:
    """Wait for the app's unauthenticated DB-backed readiness endpoint."""
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/healthz", timeout=3) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise AcceptanceFailure(f"Timed out waiting for /healthz: {last_error}")


def _structured_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract FastMCP structured tool output and fail on tool errors."""
    result = payload.get("result")
    if not isinstance(result, dict):
        raise AcceptanceFailure(f"MCP response has no result object: {payload}")
    if result.get("isError"):
        raise AcceptanceFailure(f"MCP tool returned error: {result}")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            try:
                decoded = json.loads(item.get("text", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
    raise AcceptanceFailure(f"MCP tool response had no structured content: {result}")


def _mcp_initialize(
    base_url: str, bearer: str, timeout: float
) -> tuple[str, dict[str, str]]:
    """Initialize one authenticated streamable HTTP MCP session."""
    payload, response_headers = _http_json(
        f"{base_url}/mcp",
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "weft-local-rc", "version": "0.1.0"},
            },
        },
        {"Authorization": f"Bearer {bearer}"},
        timeout,
    )
    if "result" not in payload:
        raise AcceptanceFailure(f"MCP initialize did not return result: {payload}")
    session_id = next(
        (value for key, value in response_REDACTEDitems() if key.lower() == "mcp-session-id"),
        None,
    )
    if not session_id:
        raise AcceptanceFailure("MCP initialize response did not provide MCP-Session-Id")
    return session_id, {"MCP-Session-Id": session_id, "MCP-Protocol-Version": MCP_PROTOCOL_VERSION}


def _mcp_call(
    base_url: str,
    headers: dict[str, str],
    bearer: str,
    tool: str,
    arguments: dict[str, Any],
    request_id: int,
    timeout: float,
) -> dict[str, Any]:
    """Call one MCP tool over the initialized authenticated session."""
    request_headers = {**headers, "Authorization": f"Bearer {bearer}"}
    payload, _ = _http_json(
        f"{base_url}/mcp",
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
        request_headers,
        timeout,
    )
    return _structured_tool_result(payload)


def _assert_memory(
    result: dict[str, Any],
    sentinel: str,
    synthetic_project: str,
) -> str:
    """Validate stored memory identity, content, and explicit project scope."""
    memory_id = result.get("id")
    if not isinstance(memory_id, str) or not memory_id:
        raise AcceptanceFailure(f"remember did not return a memory id: {result}")
    if result.get("content") != sentinel:
        raise AcceptanceFailure(f"remember returned unexpected content: {result}")
    if result.get("project_id") != synthetic_project:
        raise AcceptanceFailure(
            "remember omitted or changed the synthetic project scope: "
            f"{result}"
        )
    return memory_id


def _result_entries(response: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    """Return actual memory result entries, rejecting empty/invalid responses."""
    entries = response.get("results")
    if not isinstance(entries, list):
        raise AcceptanceFailure(f"{operation} did not return a results list: {response}")
    return [
        entry for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def _assert_recall(
    response: dict[str, Any],
    memory_id: str,
    sentinel: str,
    synthetic_project: str,
    operation: str,
) -> None:
    """Require the tracked memory in actual results, not echoed query text."""
    entries = _result_entries(response, operation)
    match = next((entry for entry in entries if entry.get("id") == memory_id), None)
    if match is None:
        raise AcceptanceFailure(f"{operation} omitted tracked memory {memory_id}: {response}")
    if match.get("content") != sentinel:
        raise AcceptanceFailure(f"{operation} returned wrong memory content: {match}")
    if match.get("project_id") != synthetic_project:
        raise AcceptanceFailure(f"{operation} omitted tracked project scope: {match}")


def _assert_not_recalled(
    response: dict[str, Any],
    memory_id: str,
    sentinel: str,
    operation: str,
) -> None:
    """Require an isolation response has no actual matching memory entry."""
    entries = _result_entries(response, operation)
    if any(
        entry.get("id") == memory_id or entry.get("content") == sentinel
        for entry in entries
    ):
        raise AcceptanceFailure(f"{operation} exposed isolated memory: {entries}")


def _assert_prime_handoff(
    response: dict[str, Any],
    handoff_id: str,
    handoff_summary: str,
    synthetic_project: str,
    operation: str,
) -> None:
    """Require the authoritative handoff ID/content and project scope in prime."""
    entries = response.get("handoff")
    if not isinstance(entries, list):
        raise AcceptanceFailure(f"{operation} did not return a handoff list: {response}")
    match = next((entry for entry in entries if isinstance(entry, dict) and entry.get("id") == handoff_id), None)
    if match is None or handoff_summary not in str(match.get("content", "")):
        raise AcceptanceFailure(f"{operation} omitted authoritative handoff {handoff_id}: {response}")
    # Current prime entries intentionally omit project_id; the requested project
    # is proven by the wrong-project negative check in the workflow. If a future
    # serializer includes it, reject any contradictory scope.
    returned_project = match.get("project_id")
    if returned_project is not None and returned_project != synthetic_project:
        raise AcceptanceFailure(f"{operation} returned wrong handoff project scope: {match}")


def _assert_not_primed(
    response: dict[str, Any],
    handoff_id: str,
    handoff_summary: str,
    operation: str,
) -> None:
    """Require another owner/project cannot surface the authoritative handoff."""
    entries = response.get("handoff")
    if not isinstance(entries, list):
        raise AcceptanceFailure(f"{operation} did not return a handoff list: {response}")
    if any(
        isinstance(entry, dict)
        and (entry.get("id") == handoff_id or handoff_summary in str(entry.get("content", "")))
        for entry in entries
    ):
        raise AcceptanceFailure(f"{operation} exposed isolated handoff {handoff_id}: {response}")


def _mark_cleanup_failed(receipt: AcceptanceReceipt, message: str) -> None:
    """Make cleanup failure terminal and retain the diagnostic warning."""
    receipt.status = "failed"
    receipt.warnings.append(message)
    if receipt.error is None:
        receipt.error = message


def _cleanup_resources(
    prefix: list[str],
    compose_file: Path,
    project_name: str,
    environment: dict[str, str],
    timeout: float,
    bearer: str,
    tracked_memory_ids: list[str],
    receipt: AcceptanceReceipt,
) -> None:
    """Best-effort memory deletion plus mandatory unique-project teardown."""
    try:
        session_id, mcp_headers = _mcp_initialize(
            f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}", bearer, min(timeout, 15)
        )
        for index, memory_id in enumerate(tracked_memory_ids, start=1000):
            try:
                _mcp_call(
                    f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}",
                    mcp_headers,
                    bearer,
                    "weft_forget",
                    {"memory_id": memory_id, "hard": True},
                    index,
                    min(timeout, 15),
                )
            except Exception as exc:
                _mark_cleanup_failed(receipt, f"synthetic memory cleanup failed for tracked id: {exc}")
    except Exception as exc:
        _mark_cleanup_failed(receipt, f"MCP synthetic cleanup unavailable: {exc}")

    try:
        result = _run_compose(
            prefix,
            compose_file,
            project_name,
            ["down", "--volumes", "--remove-orphans"],
            environment,
            timeout,
        )
        if result.returncode:
            _mark_cleanup_failed(receipt, f"Compose cleanup failed: {result.stderr[-1000:]}")
        else:
            receipt.checks["unique_resource_cleanup"] = "passed"
    except Exception as exc:
        _mark_cleanup_failed(receipt, f"Compose cleanup raised: {exc}")


def run_acceptance(args: argparse.Namespace) -> AcceptanceReceipt:
    """Execute the complete bounded acceptance workflow."""
    run_id = uuid.uuid4().hex[:12]
    project_name = f"weft-rc-{run_id}"
    synthetic_project = f"local-docker-rc-{run_id}"
    owner_user_id = str(uuid.uuid4())
    image = args.image
    receipt = AcceptanceReceipt(
        run_id=run_id,
        project_name=project_name,
        synthetic_project=synthetic_project,
        owner_user_id=owner_user_id,
        image=image,
        compose_file=str(args.compose_file),
        started_at=time.time(),
    )
    prefix = _docker_compose_prefix()
    environment = os.environ.copy()
    environment.update(
        {
            "WEFT_LOCAL_IMAGE": image,
            "WEFT_LOCAL_API_KEY": secrets.token_urlsafe(32),
            "WEFT_LOCAL_USER_ID": owner_user_id,
            "WEFT_LOCAL_PORT": str(_free_port()),
        }
    )
    base_url = f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}"
    tracked_memory_ids: list[str] = []
    # Namespace selection itself makes teardown safe, even if Compose partially starts.
    cleanup_eligible = True
    try:
        result = _run_compose(
            prefix, args.compose_file, project_name, ["up", "-d"], environment, args.timeout
        )
        if result.returncode:
            raise AcceptanceFailure(f"Compose up failed: {result.stderr[-1000:]}")
        receipt.checks["compose_up"] = "passed"
        _wait_for_health(base_url, args.timeout)
        receipt.checks["healthz_readiness"] = "passed"

        bearer = environment["WEFT_LOCAL_API_KEY"]
        session_id, mcp_headers = _mcp_initialize(base_url, bearer, args.timeout)
        receipt.checks["image_cmd_mcp_initialize"] = "passed"
        request_id = 2
        sentinel = f"Local Docker RC sentinel {run_id}"
        remember = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_remember",
            {
                "content": sentinel,
                "type": "fact",
                "topic": ["local-docker-rc", run_id],
                "project_id": synthetic_project,
                "source": "conversation",
                "confidence": 1.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        tracked_memory_ids.append(_assert_memory(remember, sentinel, synthetic_project))
        receipt.checks["mcp_memory_write_fastembed"] = "passed"

        recall = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": sentinel,
                "project_id": synthetic_project,
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_recall(
            recall,
            tracked_memory_ids[-1],
            sentinel,
            synthetic_project,
            "recall",
        )
        receipt.checks["mcp_memory_recall"] = "passed"

        handoff_summary = f"Acceptance handoff {run_id}"
        handoff = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_handoff",
            {
                "summary": handoff_summary,
                "in_progress": "local Docker acceptance",
                "next_steps": "verify persistence",
                "open_questions": "none",
                "project_id": synthetic_project,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        handoff_id = handoff.get("id")
        if not isinstance(handoff_id, str) or not handoff.get("stored"):
            raise AcceptanceFailure(f"handoff was not stored: {handoff}")
        tracked_memory_ids.append(handoff_id)
        receipt.checks["mcp_handoff"] = "passed"
        prime = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_prime",
            {
                "project_id": synthetic_project,
                "budget_tokens": 1200,
                "disclosure": "progressive",
                "query": "Acceptance handoff",
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_prime_handoff(
            prime,
            handoff_id,
            handoff_summary,
            synthetic_project,
            "prime",
        )
        receipt.checks["mcp_prime"] = "passed"

        invalid_request = urllib.request.Request(
            f"{base_url}/mcp",
            data=b"{}",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(invalid_request, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise AcceptanceFailure(f"invalid auth returned HTTP {exc.code}, expected 401")
        else:
            raise AcceptanceFailure("invalid auth unexpectedly reached MCP")
        receipt.checks["negative_auth"] = "passed"

        # Project wall: the same owner must not see the synthetic handoff under another project.
        wrong_project_prime = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_prime",
            {
                "project_id": f"{synthetic_project}-wrong",
                "budget_tokens": 1200,
                "disclosure": "progressive",
                "query": "Acceptance handoff",
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_not_primed(
            wrong_project_prime,
            handoff_id,
            handoff_summary,
            "prime_wrong_project",
        )

        # Owner wall: issue a separate token and prove it cannot read this owner's memory.
        distinct_owner = str(uuid.uuid4())
        issued = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_token_issue",
            {
                "user_id": distinct_owner,
                "caller_mode": "supervisor",
                "label": f"local-rc-isolation-{run_id}",
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        distinct_bearer = issued.get("token")
        if not isinstance(distinct_bearer, str) or not distinct_bearer:
            raise AcceptanceFailure(f"token issue did not return a distinct owner token: {issued}")
        _, distinct_headers = _mcp_initialize(base_url, distinct_bearer, args.timeout)
        isolated_recall = _mcp_call(
            base_url,
            distinct_headers,
            distinct_bearer,
            "weft_recall",
            {
                "query": sentinel,
                "project_id": synthetic_project,
                "retrieval_mode": "code",
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_not_recalled(
            isolated_recall,
            tracked_memory_ids[0],
            sentinel,
            "recall_distinct_owner",
        )
        isolated_prime = _mcp_call(
            base_url,
            distinct_headers,
            distinct_bearer,
            "weft_prime",
            {
                "project_id": synthetic_project,
                "budget_tokens": 1200,
                "disclosure": "progressive",
                "query": "Acceptance handoff",
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_not_primed(
            isolated_prime,
            handoff_id,
            handoff_summary,
            "prime_distinct_owner",
        )
        receipt.checks["owner_project_isolation"] = "passed"

        for service in ("postgres", "redis", "app"):
            result = _run_compose(
                prefix, args.compose_file, project_name, ["restart", service], environment, args.timeout
            )
            if result.returncode:
                raise AcceptanceFailure(f"Compose restart {service} failed: {result.stderr[-1000:]}")
            _wait_for_health(base_url, args.timeout)
        receipt.checks["restart_postgres_redis_app"] = "passed"

        session_id, mcp_headers = _mcp_initialize(base_url, bearer, args.timeout)
        recalled_after_restart = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": sentinel,
                "project_id": synthetic_project,
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        _assert_recall(
            recalled_after_restart,
            tracked_memory_ids[0],
            sentinel,
            synthetic_project,
            "recall_after_restart",
        )
        request_id += 1
        prime_after_restart = _mcp_call(
            base_url,
            mcp_headers,
            bearer,
            "weft_prime",
            {
                "project_id": synthetic_project,
                "budget_tokens": 1200,
                "disclosure": "progressive",
                "query": "Acceptance handoff",
            },
            request_id,
            args.timeout,
        )
        _assert_prime_handoff(
            prime_after_restart,
            handoff_id,
            handoff_summary,
            synthetic_project,
            "prime_after_restart",
        )
        receipt.checks["memory_handoff_persistence"] = "passed"
        receipt.status = "passed"
    except Exception as exc:
        receipt.status = "failed"
        receipt.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Namespace selection makes this teardown safe after partial Compose startup.
        if cleanup_eligible:
            _cleanup_resources(
                prefix,
                args.compose_file,
                project_name,
                environment,
                args.timeout,
                environment["WEFT_LOCAL_API_KEY"],
                tracked_memory_ids,
                receipt,
            )
        receipt.finished_at = time.time()
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bounded acceptance CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Candidate image tag")
    parser.add_argument(
        "--compose-file", type=Path, default=Path(DEFAULT_COMPOSE_FILE),
        help="Standalone local RC Compose file",
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("artifacts/local-docker-acceptance-receipt.json"),
        help="Machine-readable receipt path",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-operation timeout seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run acceptance, write receipt, and return a CI-friendly exit code."""
    args = parse_args(argv)
    receipt: AcceptanceReceipt
    try:
        receipt = run_acceptance(args)
    except Exception as exc:
        receipt = AcceptanceReceipt(
            run_id="unstarted",
            project_name="none",
            synthetic_project="none",
            owner_user_id="none",
            image=args.image,
            compose_file=str(args.compose_file),
            started_at=time.time(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
    _write_receipt(args.receipt, receipt)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0 if receipt.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ═══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ═══════════════════════════════════════════════════════════════
#
# 1. Install dependencies:
#    No Python package dependencies; install Docker Desktop or Docker Engine.
#
# 2. Build the candidate image (from repository root):
#    docker build --tag weft-rc-local:1.0.0rc1 .
#
# 3. Run the bounded acceptance (actual image CMD + HTTP MCP):
#    python scripts/local_docker_acceptance.py \
#      --image weft-rc-local:1.0.0rc1 \
#      --receipt artifacts/local-docker-rc-20260912/local-docker-acceptance.json
#
# 4. Expected output:
#    JSON with status=passed and checks for readiness, FastEmbed memory write,
#    recall, prime, handoff, negative auth, restart persistence, and cleanup.
#
# ═══════════════════════════════════════════════════════════════
