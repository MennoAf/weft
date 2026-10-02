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
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import shutil
import signal
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
MAX_OUTPUT_BYTES = 64 * 1024
MAX_PHASES = 128
HTTP_READ_CHUNK_SIZE = 8192
# HTTP error bodies are diagnostics, not protocol payloads.  Keep their raw
# retained bytes within the existing output ceiling; once full, stop reading
# rather than draining an unbounded or adversarial response.
HTTP_ERROR_BODY_MAX_BYTES = MAX_OUTPUT_BYTES
TERM_GRACE_SECONDS = 2.0
DRAIN_SECONDS = 1.0
# v3 aggregate budget policy.  ``--timeout`` remains the per-operation cap.
DEFAULT_TOTAL_TIMEOUT = 2400.0
DEFAULT_FINALIZATION_RESERVE = 240.0
DEFAULT_COMPOSE_CLEANUP_RESERVE = 120.0
DEFAULT_PUBLICATION_RESERVE = 60.0
DEFAULT_COMPOSE_CLEANUP_TIMEOUT = 90.0
DEFAULT_CLEANUP_KILL_GRACE = 1.0
CLEANUP_DRAIN_REAP_ALLOWANCE = 20.0
SIGNAL_COUNT_CAP = 16
# This is intentionally a finite grammar, not an attempt to detect arbitrary
# secret-looking text.  It covers the credential forms this harness generates
# or sends: labelled env/header values, Bearer headers, and DSN URLs.  Secrets
# generated at runtime are additionally registered in ``_REDACTION_VALUES``.
_CREDENTIAL_LABEL = re.compile(
    r"(?P<label>WEFT_[A-Z0-9_]*(?:KEY|PASSWORD|TOKEN)|DATABASE_URL|"
    r"API_KEY|X-API-KEY|API-KEY)(?P<separator>[=:]\s*)"
    r"(?P<value>[^\s,;\"'}]+)"
    r"|(?P<bearer>AUTHORIZATION\s*:\s*BEARER\s+)"
    r"(?P<bearer_value>[^\s,;\"'}]+)",
    re.IGNORECASE,
)
_DSN_URL = re.compile(
    r"(?P<dsn>(?:postgres(?:ql)?|redis)://[^\s,;\"'}]+)",
    re.IGNORECASE,
)
_REDACTION_VALUES: set[str] = set()
_REDACTION_BOUNDARY = 512
# ``verify-runtime`` emits this fixed object contract.  Keep each field bounded
# independently so a verbose but valid probe cannot hide a decisive denial or
# skip behind a trailing stdout cap.
_RUNTIME_VERIFIER_SUMMARY_FIELDS = (
    "current_user",
    "rolsuper",
    "rolbypassrls",
    "rolcreatedb",
    "rolcreaterole",
    "rolreplication",
    "rolinherit",
    "rolconfig",
    "memories_rls",
    "memories_owner",
    "schema_migrations_write",
    "schema_migrations_read",
    "explicit_runtime_tables",
    "dormant_oauth_tables_denied",
    "sequence_setval",
    "schema_ddl",
    "database_name",
    "schema_name",
    "role_configuration_reset",
    "public_acl_reset",
)
_RUNTIME_VERIFIER_PROBE_FIELDS = {
    "schema_ddl_probe",
    "sequence_setval_probe",
    "future_table_select_probe",
    "future_sequence_setval_probe",
}
_RUNTIME_VERIFIER_EXPECTED_TEXT = {
    "current_user": "weft_app",
    "database_name": "weft",
    "schema_name": "public",
}
_RUNTIME_VERIFIER_EXPECTED_FLAGS = {
    "rolsuper": False,
    "rolbypassrls": False,
    "rolcreatedb": False,
    "rolcreaterole": False,
    "rolreplication": False,
    "rolinherit": False,
    "memories_rls": True,
    "schema_migrations_write": False,
    "schema_migrations_read": True,
    "dormant_oauth_tables_denied": True,
    "sequence_setval": False,
    "schema_ddl": False,
    "role_configuration_reset": True,
    "public_acl_reset": True,
}
_RUNTIME_VERIFIER_EXPECTED_TABLES = {
    "alert_state", "alerts", "autonomy_overrides", "autonomy_policies", "behaviors",
    "belief_claims", "board_feedback_proposals", "board_triage_events", "calibration_records",
    "check_ins", "cost_enforcement_state", "cost_entries", "degradation_policies", "entities",
    "entity_mentions", "episode_memories", "episode_turns", "episodes", "memories",
    "memory_access_log", "memory_relationships", "modes", "policy_calibration_events",
    "recall_canary", "recall_canary_audit", "replay_queue", "shuttle_claims", "topic_digests",
    "topic_resolution_aliases", "trackers", "triggers", "turn_access_log", "weft_counters",
    "weft_metadata", "weft_recall_queries", "weft_recovery_attempts", "weft_tokens",
    "weft_tool_usage_coverage", "weft_tool_usage_daily", "workspace_members", "workspaces",
}
_RUNTIME_VERIFIER_DENIAL_STATUSES = {"denied_and_rollback_verified"}
_RUNTIME_VERIFIER_SKIP_STATUSES = {"skipped_no_allowlisted_sequence"}
_RUNTIME_VERIFIER_BOOLEAN_FIELDS = {
    "rolsuper",
    "rolbypassrls",
    "rolcreatedb",
    "rolcreaterole",
    "rolreplication",
    "rolinherit",
    "memories_rls",
    "schema_migrations_write",
    "schema_migrations_read",
    "dormant_oauth_tables_denied",
    "sequence_setval",
    "schema_ddl",
    "role_configuration_reset",
    "public_acl_reset",
}


def _redact_match(match: re.Match[str]) -> str:
    """Replace one credential grammar match while retaining its label."""
    label = match.group("label")
    if label is not None:
        return f"{label}{match.group('separator')}[REDACTED]"
    return f"{match.group('bearer')}[REDACTED]"


def _sanitize_text(value: str, limit: int = 1000) -> str:
    """Redact known credentials, then return a bounded diagnostic string.

    Unknown arbitrary secrets are deliberately outside the contract; callers
    must register generated values in ``_REDACTION_VALUES``.  Sanitizing before
    truncating prevents a credential at a cap boundary from being persisted.
    """
    text = str(value)
    for secret in sorted(_REDACTION_VALUES, key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _DSN_URL.sub("[REDACTED]", text)
    text = _CREDENTIAL_LABEL.sub(_redact_match, text)
    return text[-limit:]


def _sanitize_value(value: Any, limit: int = 1000, depth: int = 0) -> Any:
    """Recursively sanitize durable receipt values with bounded containers."""
    if depth > 8:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return _sanitize_text(value, limit)
    if isinstance(value, dict):
        return {
            _sanitize_text(str(key), limit): _sanitize_value(item, limit, depth + 1)
            for key, item in list(value.items())[:MAX_PHASES]
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, limit, depth + 1) for item in list(value)[-MAX_PHASES:]]
    return value


class AcceptanceInterrupted(KeyboardInterrupt):
    """Raised by the runner's signal handler so normal cleanup still runs."""


class AcceptanceTimeout(TimeoutError):
    """A process-group operation exceeded its deadline."""


_ACTIVE_RECEIPT: AcceptanceReceipt | None = None
_ACTIVE_CHECKPOINT: Path | None = None
_FINALIZING = False
_FINALIZATION_RUNNING = False
_SIGNALS_FROZEN = False
_PENDING_SIGNALS: list[str] = []


@dataclass
class AcceptanceReceipt:
    """Canonical v3 evidence; publication metadata is deliberately external."""

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
    phases: list[dict[str, Any]] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    finished_at: float | None = None
    checkpoint_path: str | None = None
    generation: str = field(default_factory=lambda: str(uuid.uuid4()))
    lifecycle_status: str = "accepting"
    acceptance_status: str = "not_started"
    acceptance_rc: int | None = None
    cleanup_status: str = "not_attempted"
    cleanup_rc: int | None = None
    first_signal: str | None = None
    signal_count: int = 0
    selected_return_code: int | None = None
    primary_error: str | None = None
    cleanup_errors: list[str] = field(default_factory=list)
    cleanup_registration: dict[str, Any] = field(default_factory=dict)
    budget_ledger: dict[str, Any] = field(default_factory=dict)
    # Direct helper tests may intentionally construct filesystem-only receipts.
    # The owner CLI marks its receipts explicitly so initialization failures
    # cannot fall through the legacy empty-ledger compatibility path.
    publication_owner: bool = False
    budget_initialized: bool = False
    original_return_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return bounded v3 payload; marker digests never appear in payload."""
        document = {
            "schema_version": 2,
            "schema": "weft.local-docker-acceptance.v2",
            "run_id": self.run_id,
            "generation": self.generation,
            "project_name": self.project_name,
            "synthetic_project": self.synthetic_project,
            "owner_user_id": self.owner_user_id,
            "image": self.image,
            "compose_file": self.compose_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "lifecycle_status": self.lifecycle_status,
            "acceptance_status": self.acceptance_status,
            "acceptance_rc": self.acceptance_rc,
            "cleanup_status": self.cleanup_status,
            "cleanup_rc": self.cleanup_rc,
            "first_signal": self.first_signal,
            "signal_count": min(self.signal_count, SIGNAL_COUNT_CAP),
            "selected_return_code": self.selected_return_code,
            "primary_error": self.primary_error or self.error,
            "cleanup_errors": self.cleanup_errors,
            "checks": self.checks,
            "warnings": self.warnings,
            "phases": self.phases[-MAX_PHASES:],
            "verifier": self.verifier,
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
            "cleanup_registration": self.cleanup_registration,
            "budget_ledger": self.budget_ledger,
            "cleanup_scope": {
                "compose_project": self.project_name,
                "volumes": "unique Compose project only",
                "host_ports": "app loopback only; postgres/redis unpublished",
            },
            "persistence_evidence": {
                "mcp_container_persistence": "required",
                "installed_library_persistence": "not_run_host_only",
            },
        }
        return _sanitize_value(document, _REDACTION_BOUNDARY)


class AcceptanceFailure(RuntimeError):
    """Raised when a required acceptance assertion fails."""


def _free_port() -> int:
    """Reserve no port; return an currently-unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _docker_compose_prefix(executable: str | Path | None = None) -> list[str]:
    """Return the validated Docker Compose CLI prefix.

    An explicit executable is authoritative and may have any basename.  The
    discovery path is retained only for direct CLI compatibility.
    """
    if executable is None:
        docker = shutil.which("docker")
    else:
        candidate = Path(executable)
        if not candidate.is_absolute() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise AcceptanceFailure("Docker executable must be an absolute executable file")
        docker = str(candidate)
    if docker is None:
        raise AcceptanceFailure("Docker CLI not found; acceptance requires Docker Desktop/Engine")
    return [docker, "compose"]


class _OutputRedactor:
    """Redact credentials even when a pipe splits the label/value pair."""

    def __init__(self) -> None:
        self._pending = ""
        self._emitted = 0

    def _bounded(self, text: str) -> str:
        remaining = max(0, MAX_OUTPUT_BYTES - self._emitted)
        visible = _sanitize_text(text, remaining)
        visible = visible[:remaining]
        self._emitted += len(visible)
        return visible

    def feed(self, chunk: bytes) -> str:
        text = self._pending + chunk.decode("utf-8", errors="replace")
        # Keep a suffix so a credential marker split across reads is retained.
        safe_end = max(0, len(text) - 256)
        visible, self._pending = text[:safe_end], text[safe_end:]
        return self._bounded(visible)

    def finish(self) -> str:
        visible = self._bounded(self._pending)
        self._pending = ""
        return visible


def _run_compose_process(
    prefix: list[str],
    compose_file: Path,
    project_name: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
    *,
    kill_grace: float = TERM_GRACE_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run Compose in a bounded, owned POSIX process group.

    ``start_new_session`` makes the child process group owned by this runner.
    Cleanup targets only that group; a child which calls ``setsid`` escapes this
    containment boundary and is intentionally not discovered or killed here.
    """
    command = [*prefix, "-f", str(compose_file), "-p", project_name, *args]
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector: selectors.BaseSelector | None = None
    streams = (process.stdout, process.stderr)
    output = {"stdout": [], "stderr": []}
    output_bytes = {"stdout": 0, "stderr": 0}
    redactors = {"stdout": _OutputRedactor(), "stderr": _OutputRedactor()}
    timed_out = False
    lifecycle_error: BaseException | None = None

    def close_stream(stream: Any) -> None:
        try:
            if stream is not None and not stream.closed:
                stream.close()
        except OSError:
            pass

    def send_group(sig: signal.Signals) -> str | None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return None
        except OSError as exc:
            # Permission and other OS errors are diagnostic only: bounded wait,
            # pipe close, and reap must still run below.
            return f"killpg({sig.name}) failed: {type(exc).__name__}: {exc}"
        return None

    def reap(limit: float) -> int | None:
        while process.poll() is None and time.monotonic() < limit:
            try:
                process.wait(timeout=min(0.05, max(0.0, limit - time.monotonic())))
            except subprocess.TimeoutExpired:
                continue
            except OSError as exc:
                nonlocal lifecycle_error
                lifecycle_error = lifecycle_error or exc
                break
        return process.poll()

    def drain_until(limit: float) -> None:
        if selector is None:
            return
        while selector.get_map() and time.monotonic() < limit:
            try:
                events = selector.select(max(0.0, limit - time.monotonic()))
            except (OSError, ValueError) as exc:
                nonlocal lifecycle_error
                lifecycle_error = lifecycle_error or exc
                break
            for key, _ in events:
                stream = key.fileobj
                try:
                    chunk = stream.read1(8192) if hasattr(stream, "read1") else stream.read(8192)
                except (OSError, ValueError) as exc:
                    lifecycle_error = lifecycle_error or exc
                    try:
                        selector.unregister(stream)
                    except (KeyError, ValueError, OSError):
                        pass
                    close_stream(stream)
                    continue
                if not chunk:
                    try:
                        selector.unregister(stream)
                    except (KeyError, ValueError, OSError):
                        pass
                    close_stream(stream)
                    continue
                name = key.data
                remaining = MAX_OUTPUT_BYTES - output_bytes[name]
                if remaining > 0:
                    kept = chunk[:remaining]
                    output_bytes[name] += len(kept)
                    output[name].append(redactors[name].feed(kept))

    try:
        # Selector construction and registration are guarded just like the
        # workflow: a failure after Popen still enters group cleanup/reap.
        selector = selectors.DefaultSelector()
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            if stream is None:
                raise OSError(f"Compose {name} pipe was not created")
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + timeout
        drain_until(deadline)
        if selector.get_map() or process.poll() is None:
            timed_out = True
            diagnostic = send_group(signal.SIGTERM)
            if diagnostic:
                output["stderr"].append(diagnostic)
            drain_until(time.monotonic() + max(0.0, kill_grace))
            if process.poll() is None:
                diagnostic = send_group(signal.SIGKILL)
                if diagnostic:
                    output["stderr"].append(diagnostic)
                drain_until(time.monotonic() + DRAIN_SECONDS)
        # Drain/reap are bounded overhead and must be included in the caller's
        # cleanup reserve; never add an unbudgeted last-chance wait.
        returncode = reap(time.monotonic() + 0.5)
        if returncode is None:
            # A bounded wait always occurs even after killpg errors. The parent
            # may be gone while descendants hold pipes; close pipes and retain a
            # deterministic signal-style result rather than waiting forever.
            returncode = -signal.SIGKILL
    except BaseException as exc:
        lifecycle_error = lifecycle_error or exc
        diagnostic = send_group(signal.SIGKILL)
        if diagnostic:
            output["stderr"].append(diagnostic)
        reap(time.monotonic() + 0.5)
        raise
    finally:
        if selector is not None:
            try:
                selector.close()
            except (OSError, ValueError):
                pass
        for stream in streams:
            close_stream(stream)
    for name, redactor in redactors.items():
        output[name].append(redactor.finish())
    stdout = "".join(output["stdout"])[-MAX_OUTPUT_BYTES:]
    stderr = "".join(output["stderr"])[-MAX_OUTPUT_BYTES:]
    if lifecycle_error is not None and not timed_out:
        stderr = (stderr + "\\n" + _safe_text(str(lifecycle_error)))[-MAX_OUTPUT_BYTES:]
    result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
    return result


def _active_operation_timeout(cap: float) -> float:
    """Admit an operation against the currently selected aggregate reserve."""
    if _ACTIVE_RECEIPT is None:
        return cap
    deadline = _ACTIVE_RECEIPT.budget_ledger.get("operation_deadline")
    if deadline is None:
        return cap
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise AcceptanceTimeout("aggregate acceptance deadline exhausted before operation")
    return min(cap, remaining)


def _run_compose(
    prefix: list[str],
    compose_file: Path,
    project_name: str,
    args: list[str],
    env: dict[str, str],
    timeout: float,
    *,
    kill_grace: float = TERM_GRACE_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one named Compose phase, preserving the legacy test seam."""
    def operation() -> subprocess.CompletedProcess[str]:
        # Admission belongs at the external process boundary.  In the
        # checkpointed path this callable runs only after _phase has written
        # its running checkpoint, so bookkeeping cannot leave a stale positive
        # timeout behind.
        effective_timeout = _active_operation_timeout(timeout)
        return _run_compose_process(
            prefix, compose_file, project_name, args, env, effective_timeout,
            kill_grace=kill_grace,
        )

    if _ACTIVE_RECEIPT is None or _ACTIVE_CHECKPOINT is None:
        return operation()
    name = "compose_" + "_".join(args[:4]).replace("-", "_")
    return _phase(_ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT, name, operation)


def _write_receipt(path: Path, receipt: AcceptanceReceipt) -> None:
    """Write evidence atomically and never expose a partially-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{receipt.run_id}.tmp")
    temporary.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _admit_publication(receipt: AcceptanceReceipt, boundary: str) -> None:
    """Admit one publication write against this owner's logical deadline.

    Focused legacy helpers construct receipts without a budget ledger; those
    calls retain their historical filesystem-only behavior.  An actual runner
    receipt always has a ledger, so a missing or malformed deadline fails
    closed rather than silently bypassing publication admission.
    """
    ledger = receipt.budget_ledger
    if receipt.publication_owner:
        if not receipt.budget_initialized:
            raise AcceptanceTimeout(f"publication budget ledger unavailable at {boundary}")
        if not isinstance(ledger, dict) or not ledger:
            raise AcceptanceTimeout(f"publication deadline unavailable at {boundary}")
    if not ledger:
        return
    deadline = ledger.get("deadline")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
        raise AcceptanceTimeout(f"publication deadline unavailable at {boundary}")
    if time.monotonic() >= float(deadline):
        raise AcceptanceTimeout(f"publication deadline exhausted at {boundary}")


def _write_checkpoint(path: Path, receipt: AcceptanceReceipt) -> None:
    """Persist bounded progress before and after every externally-visible phase."""
    _write_receipt(path, receipt)


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Replace one output atomically, without allowing a partial target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _status_projection(receipt: AcceptanceReceipt) -> str:
    """Render the legacy status projection from the canonical scalar result."""
    def value(item: Any, fallback: str = "unknown") -> str:
        return fallback if item is None else str(item)
    return "".join(
        f"{key}={value(item)}\n" for key, item in (
            ("run_id", receipt.run_id),
            ("project", receipt.project_name),
            ("image", receipt.image),
            ("acceptance_rc", receipt.acceptance_rc),
            ("signal_status", receipt.first_signal or "none"),
            ("signal_exit_code", ({"SIGINT": 130, "SIGTERM": 143}.get(receipt.first_signal, 0))),
            ("cleanup_rc", receipt.cleanup_rc),
            ("exit_code", receipt.selected_return_code),
        )
    )


def _publish_terminal(receipt: AcceptanceReceipt, receipt_path: Path, status_path: Path | None = None) -> Path:
    """Publish frozen v3 evidence and verify the marker's exact byte digests."""
    _admit_publication(receipt, "terminal publication entry")
    receipt.lifecycle_status = "terminal" if receipt.status == "passed" else "incomplete"
    frozen = receipt.to_dict()
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode("utf-8")
    checkpoint = Path(receipt.checkpoint_path or receipt_path.with_name(f".{receipt_path.name}.checkpoint.json"))
    marker = receipt_path.with_name(f".{receipt_path.name}.complete.json")
    targets = [checkpoint, receipt_path, marker] + ([status_path] if status_path is not None else [])
    if len({str(path.resolve()) for path in targets}) != len(targets):
        raise FileExistsError("terminal publication targets must be distinct")
    for path in targets:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise FileExistsError(f"terminal publication target is not a regular file: {path}")
        if path.exists() and path != checkpoint:
            raise FileExistsError(f"terminal publication target already exists: {path}")
    if checkpoint.exists():
        try:
            provisional = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"checkpoint is not this run's provisional output: {checkpoint}") from exc
        if (
            not isinstance(provisional, dict)
            or provisional.get("run_id") != receipt.run_id
            or provisional.get("generation") != receipt.generation
            or provisional.get("lifecycle_status") != "accepting"
        ):
            raise FileExistsError(f"checkpoint is not this run's provisional output: {checkpoint}")
    _admit_publication(receipt, "terminal checkpoint write")
    _atomic_bytes(checkpoint, encoded)
    _admit_publication(receipt, "terminal receipt write")
    _atomic_bytes(receipt_path, encoded)
    digests = {
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    if status_path is not None:
        _admit_publication(receipt, "terminal status write")
        _atomic_bytes(status_path, _status_projection(receipt).encode("utf-8"))
        digests["status_sha256"] = hashlib.sha256(status_path.read_bytes()).hexdigest()
    marker_document = {
        "publication_schema": "weft.local-docker-acceptance.commit.v1",
        "run_id": receipt.run_id,
        "generation": receipt.generation,
        "receipt_sha256": digests["receipt_sha256"],
        "checkpoint_sha256": digests["checkpoint_sha256"],
        **({"status_sha256": digests["status_sha256"]} if status_path is not None else {}),
        "selected_return_code": receipt.selected_return_code,
        "terminal_complete": True,
    }
    marker_bytes = (json.dumps(marker_document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _admit_publication(receipt, "completion marker write")
    _atomic_bytes(marker, marker_bytes)
    _admit_publication(receipt, "completion marker readback")
    observed = json.loads(marker.read_text(encoding="utf-8"))
    required_digests = [(observed["receipt_sha256"], receipt_path), (observed["checkpoint_sha256"], checkpoint)]
    if status_path is not None:
        required_digests.append((observed["status_sha256"], status_path))
    if observed != marker_document or any(
        digest != hashlib.sha256(path.read_bytes()).hexdigest() for digest, path in required_digests
    ):
        raise AcceptanceFailure("terminal publication digest verification failed")
    return marker


def _phase(
    receipt: AcceptanceReceipt,
    checkpoint: Path,
    name: str,
    operation: Any,
) -> Any:
    """Record phase start/end, preserving failures and checkpointing each edge."""
    now = time.time()
    entry = {"name": name, "status": "running", "started_at": now}
    receipt.phases.append(entry)
    _write_checkpoint(checkpoint, receipt)
    try:
        result = operation()
    except BaseException as exc:
        entry.update({"status": "failed", "finished_at": time.time(), "error": _safe_text(str(exc))})
        _write_checkpoint(checkpoint, receipt)
        raise
    if isinstance(result, subprocess.CompletedProcess) and result.returncode:
        entry.update(
            {
                "status": "failed",
                "finished_at": time.time(),
                "error": _safe_text(result.stderr or f"return code {result.returncode}"),
            }
        )
        _write_checkpoint(checkpoint, receipt)
        return result
    entry.update({"status": "passed", "finished_at": time.time()})
    _write_checkpoint(checkpoint, receipt)
    return result


def _safe_text(value: str, limit: int = 1000) -> str:
    """Compatibility seam for bounded durable diagnostic sanitization."""
    return _sanitize_text(value, limit)


def _sanitize_structured_text(value: str, limit: int = _REDACTION_BOUNDARY) -> str:
    """Redact a structured field before retaining its meaningful prefix."""
    text = str(value)
    for secret in sorted(_REDACTION_VALUES, key=len, reverse=True):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _DSN_URL.sub("[REDACTED]", text)
    text = _CREDENTIAL_LABEL.sub(_redact_match, text)
    return text[:limit]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys instead of silently taking the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_runtime_verifier_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Parse and retain the fixed JSON contract emitted by verify-runtime.

    Successful output is deliberately parsed before any diagnostic cap is
    applied.  The bootstrap command has a closed result shape, so accepting an
    arbitrary object would make a malformed probe look like a passed check.
    """
    if not isinstance(stdout, str) or not stdout.strip():
        raise AcceptanceFailure("runtime DB role verifier returned empty output")
    try:
        payload = json.loads(
            stdout,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AcceptanceFailure("runtime DB role verifier output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AcceptanceFailure("runtime DB role verifier output must be an object")
    expected_keys = set(_RUNTIME_VERIFIER_SUMMARY_FIELDS) | {"probe_details"}
    if set(payload) != expected_keys:
        raise AcceptanceFailure("runtime DB role verifier output has the wrong shape")

    for verifier_field in _RUNTIME_VERIFIER_BOOLEAN_FIELDS:
        if type(payload[verifier_field]) is not bool:
            raise AcceptanceFailure(f"runtime DB role verifier field {verifier_field} must be boolean")
    for verifier_field in ("current_user", "memories_owner", "database_name", "schema_name"):
        if not isinstance(payload[verifier_field], str):
            raise AcceptanceFailure(f"runtime DB role verifier field {verifier_field} must be text")
    for verifier_field, expected in _RUNTIME_VERIFIER_EXPECTED_TEXT.items():
        if payload[verifier_field] != expected:
            raise AcceptanceFailure(f"runtime DB role verifier field {verifier_field} contradicts the canonical contract")
    if payload["memories_owner"] == _RUNTIME_VERIFIER_EXPECTED_TEXT["current_user"]:
        raise AcceptanceFailure("runtime DB role verifier memories_owner contradicts runtime ownership")
    for verifier_field, expected in _RUNTIME_VERIFIER_EXPECTED_FLAGS.items():
        if payload[verifier_field] is not expected:
            raise AcceptanceFailure(f"runtime DB role verifier field {verifier_field} contradicts the canonical contract")
    if payload["rolconfig"] is not None:
        raise AcceptanceFailure("runtime DB role verifier field rolconfig must be null")
    tables = payload["explicit_runtime_tables"]
    if not isinstance(tables, list) or any(not isinstance(item, str) for item in tables):
        raise AcceptanceFailure("runtime DB role verifier field explicit_runtime_tables has the wrong shape")
    if tables != sorted(tables) or len(tables) != len(set(tables)):
        raise AcceptanceFailure("runtime DB role verifier field explicit_runtime_tables is not canonical")
    if set(tables) != _RUNTIME_VERIFIER_EXPECTED_TABLES:
        raise AcceptanceFailure("runtime DB role verifier field explicit_runtime_tables contradicts the canonical contract")
    probe_details = payload["probe_details"]
    if not isinstance(probe_details, dict) or set(probe_details) != _RUNTIME_VERIFIER_PROBE_FIELDS:
        raise AcceptanceFailure("runtime DB role verifier probe_details has the wrong shape")
    if any(not isinstance(value, str) for value in probe_details.values()):
        raise AcceptanceFailure("runtime DB role verifier probe_details values must be text")
    for probe_name, probe_status in probe_details.items():
        allowed = (
            _RUNTIME_VERIFIER_DENIAL_STATUSES | _RUNTIME_VERIFIER_SKIP_STATUSES
            if probe_name == "sequence_setval_probe"
            else _RUNTIME_VERIFIER_DENIAL_STATUSES
        )
        if probe_status not in allowed:
            raise AcceptanceFailure(f"runtime DB role verifier probe {probe_name} has an invalid status")

    summary = {
        summary_field: (
            _sanitize_structured_text(payload[summary_field])
            if summary_field in {"current_user", "memories_owner", "database_name", "schema_name"}
            else [_sanitize_structured_text(item) for item in payload[summary_field][:MAX_PHASES]]
            if summary_field == "explicit_runtime_tables"
            else payload[summary_field]
        )
        for summary_field in _RUNTIME_VERIFIER_SUMMARY_FIELDS
    }
    retained_details = {
        key: _sanitize_structured_text(value)
        for key, value in probe_details.items()
    }
    skips = [
        _sanitize_structured_text(f"{key}={value}")
        for key, value in retained_details.items()
        if value.startswith("skipped_")
    ]
    return {
        "status": "passed",
        "summary": summary,
        "probe_details": retained_details,
        "stderr": _safe_text(stderr),
        "skips": skips,
    }


def _record_runtime_verifier_failure(
    receipt: AcceptanceReceipt, stdout: Any, stderr: Any, error: Exception
) -> None:
    """Preserve bounded failure evidence without allowing it to pass checks."""
    receipt.verifier = {
        "status": "failed",
        "stdout": _safe_text(str(stdout)),
        "stderr": _safe_text(str(stderr)),
        "skips": [],
        "error": _safe_text(str(error)),
    }


def _interrupt_handler(signum: int, _frame: Any) -> None:
    """Queue finalization observations until the CLI freeze owns recording."""
    name = signal.Signals(signum).name
    if _FINALIZING and not _SIGNALS_FROZEN:
        if len(_PENDING_SIGNALS) < SIGNAL_COUNT_CAP:
            _PENDING_SIGNALS.append(name)
        if _ACTIVE_RECEIPT is not None:
            _ACTIVE_RECEIPT.status = "failed"
            warning = f"interrupted during finalization: {name}"
            if warning not in _ACTIVE_RECEIPT.warnings:
                _ACTIVE_RECEIPT.warnings.append(warning)
            if _ACTIVE_RECEIPT.error is None:
                _ACTIVE_RECEIPT.error = warning
        return
    raise AcceptanceInterrupted(name)


def _validate_signal_platform() -> None:
    """Require the POSIX signal-mask primitives used by the CLI contract."""
    if os.name != "posix" or not callable(getattr(signal, "pthread_sigmask", None)):
        raise AcceptanceFailure("POSIX pthread_sigmask is required for CLI acceptance")
    if not callable(getattr(signal, "sigpending", None)):
        raise AcceptanceFailure("POSIX sigpending is required for CLI acceptance")


def _freeze_cli_signals(receipt: AcceptanceReceipt) -> None:
    """Block INT/TERM and merge all pre-cutoff observations exactly once."""
    global _SIGNALS_FROZEN, _PENDING_SIGNALS
    if _SIGNALS_FROZEN:
        return
    _validate_signal_platform()
    blocked = {signal.SIGINT, signal.SIGTERM}
    signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    pending = signal.sigpending()
    queued = tuple(_PENDING_SIGNALS)
    _PENDING_SIGNALS.clear()
    _SIGNALS_FROZEN = True
    # Handler observations precede the POSIX pending set.  sigpending() reports
    # coalesced standard signals, so each member is one observation at most.
    for name in queued:
        _record_signal(receipt, name)
    for sig in blocked.intersection(pending):
        _record_signal(receipt, signal.Signals(sig).name)
    # No caller or candidate mutation occurs after this point before return.


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


def _read_http_body(response: Any, timeout: float) -> bytes:
    """Read a successful HTTP body in bounded chunks under the deadline.

    ``urllib`` applies ``timeout`` to socket operations, not to the complete
    response body.  Re-admitting before every bounded read and immediately
    after EOF prevents a continuously dribbling response from being accepted
    after the aggregate cutoff.  A blocked kernel read remains subject to the
    socket-level limitation documented by the acceptance contract.
    """
    chunks: list[bytes] = []
    while True:
        _active_operation_timeout(timeout)
        chunk = response.read(HTTP_READ_CHUNK_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    _active_operation_timeout(timeout)
    return b"".join(chunks)


def _read_http_error_body(response: Any, timeout: float) -> tuple[bytes, bool]:
    """Read only the bounded diagnostic prefix of an HTTP error body.

    Error bodies are never protocol payloads, so retaining more than the
    existing process-output ceiling provides no useful evidence.  The reader
    stops as soon as that cap is filled: it does not perform an unbounded
    drain, and the truncation marker is added before secret sanitation by the
    caller.  Deadline admission remains identical to the normal body reader;
    a blocked kernel read is still subject to the socket-level limitation.
    """
    chunks: list[bytes] = []
    retained = 0
    while retained < HTTP_ERROR_BODY_MAX_BYTES:
        _active_operation_timeout(timeout)
        chunk = response.read(min(HTTP_READ_CHUNK_SIZE, HTTP_ERROR_BODY_MAX_BYTES - retained))
        if not chunk:
            _active_operation_timeout(timeout)
            return b"".join(chunks), False
        kept = chunk[: HTTP_ERROR_BODY_MAX_BYTES - retained]
        chunks.append(kept)
        retained += len(kept)
        if len(kept) < len(chunk) or retained == HTTP_ERROR_BODY_MAX_BYTES:
            # The cap is a logical body-completion boundary.  Admit the
            # retained prefix before returning, but never read/drain further.
            _active_operation_timeout(timeout)
            return b"".join(chunks), True
    # The loop always returns at the cap, but retain a fail-closed fallback if
    # the cap is ever changed to an unusual value.
    return b"".join(chunks), True


def _protect_truncated_secret_boundary(text: str) -> str:
    """Hide a registered secret prefix cut by the raw error-body cap."""
    for secret in sorted(_REDACTION_VALUES, key=len, reverse=True):
        if not secret:
            continue
        max_overlap = min(len(secret) - 1, _REDACTION_BOUNDARY, len(text))
        for overlap in range(max_overlap, 0, -1):
            if text.endswith(secret[:overlap]):
                return text[:-overlap] + "[REDACTED]"
    return text


def _format_http_error_message(code: int, url: str, body: str) -> str:
    """Keep HTTP context visible while bounding each sanitized component."""
    status_prefix = f"HTTP {code} from "
    separator = ": "
    url_budget = max(0, 1000 - len(status_prefix) - len(separator) - 1)
    safe_url = _safe_text(url, url_budget) if url_budget else ""
    context = f"{status_prefix}{safe_url}{separator}"
    body_budget = max(0, 1000 - len(context))
    safe_body = _safe_text(body, body_budget) if body_budget else ""
    return f"{context}{safe_body}"


def _negative_auth_probe(base_url: str, timeout: float) -> None:
    """Prove unauthenticated MCP access is rejected after phase bookkeeping."""
    invalid_request = urllib.request.Request(
        f"{base_url}/mcp",
        data=b"{}",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(
            invalid_request,
            timeout=_active_operation_timeout(timeout),
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise AcceptanceFailure(f"invalid auth returned HTTP {exc.code}, expected 401")
    else:
        raise AcceptanceFailure("invalid auth unexpectedly reached MCP")


def _http_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """POST one MCP JSON-RPC request and return payload plus response headers."""
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
    effective_timeout = _active_operation_timeout(timeout)
    try:
        with urllib.request.urlopen(request, timeout=effective_timeout) as response:
            body = _read_http_body(response, timeout)
            return _decode_http_payload(body), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raw_body, truncated = _read_http_error_body(exc, timeout)
        body = raw_body.decode("utf-8", errors="replace")
        if truncated:
            # The cap is applied to raw bytes before sanitation.  If a
            # registered secret crosses that boundary, replace its retained
            # prefix before appending the explicit marker so no partial secret
            # can survive the final diagnostic truncation.
            body = _protect_truncated_secret_boundary(body)
            body = f"{body} [TRUNCATED]"
        raise AcceptanceFailure(_format_http_error_message(exc.code, url, body)) from exc
    except urllib.error.URLError as exc:
        raise AcceptanceFailure(_safe_text(f"HTTP request to {url} failed: {exc.reason}")) from exc


def _wait_for_health(base_url: str, timeout: float) -> None:
    """Wait for readiness without exceeding the aggregate acceptance cutoff."""
    deadline = time.monotonic() + _active_operation_timeout(timeout)
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            request_timeout = min(3.0, max(0.001, deadline - time.monotonic()))
            with urllib.request.urlopen(f"{base_url}/healthz", timeout=request_timeout) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(2.0, remaining))
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


def _mcp_initialize_raw(
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
        (value for key, value in response_headers.items() if key.lower() == "mcp-session-id"),
        None,
    )
    if not session_id:
        raise AcceptanceFailure("MCP initialize response did not provide MCP-Session-Id")
    return session_id, {"MCP-Session-Id": session_id, "MCP-Protocol-Version": MCP_PROTOCOL_VERSION}


def _mcp_initialize(
    base_url: str, bearer: str, timeout: float
) -> tuple[str, dict[str, str]]:
    """Checkpoint an authenticated MCP initialize operation."""
    if _ACTIVE_RECEIPT is None or _ACTIVE_CHECKPOINT is None:
        return _mcp_initialize_raw(base_url, bearer, timeout)
    return _phase_http(
        _ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT, "http_initialize",
        lambda: _mcp_initialize_raw(base_url, bearer, timeout),
    )


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


def _mcp_call_checkpointed(
    base_url: str,
    headers: dict[str, str],
    bearer: str,
    tool: str,
    arguments: dict[str, Any],
    request_id: int,
    timeout: float,
) -> dict[str, Any]:
    """Checkpoint one MCP tool call while retaining the legacy helper seam."""
    if _ACTIVE_RECEIPT is None or _ACTIVE_CHECKPOINT is None:
        return _mcp_call(base_url, headers, bearer, tool, arguments, request_id, timeout)
    return _phase_http(
        _ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT, f"http_tool_{tool}",
        lambda: _mcp_call(base_url, headers, bearer, tool, arguments, request_id, timeout),
    )


def _phase_http(
    receipt: AcceptanceReceipt,
    checkpoint: Path,
    name: str,
    operation: Any,
) -> Any:
    """Checkpoint non-Compose health/HTTP operations with bounded errors."""
    return _phase(receipt, checkpoint, name, operation)


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


def _assert_preference(
    response: dict[str, Any],
    preference_content: str,
    preference_metadata: dict[str, Any],
    synthetic_project: str,
) -> str:
    """Validate a preference write's real identity, scope, and metadata."""
    memory_id = response.get("id")
    if not isinstance(memory_id, str) or not memory_id:
        raise AcceptanceFailure(f"preference did not return a memory id: {response}")
    if response.get("content") != preference_content:
        raise AcceptanceFailure(f"preference returned unexpected content: {response}")
    if response.get("type") != "preference":
        raise AcceptanceFailure(f"preference returned wrong type: {response}")
    if response.get("project_id") != synthetic_project:
        raise AcceptanceFailure(f"preference omitted synthetic project scope: {response}")
    if response.get("preference_metadata") != preference_metadata:
        raise AcceptanceFailure(
            f"preference metadata was not persisted faithfully: {response}"
        )
    return memory_id


def _assert_revision(
    response: dict[str, Any],
    old_id: str,
    old_content: str,
    new_content: str,
    synthetic_project: str,
    preference_metadata: dict[str, Any],
) -> tuple[str, str]:
    """Validate both versions returned by ``weft_revise`` before lineage checks."""
    new = response.get("new")
    superseded = response.get("superseded")
    if not isinstance(new, dict) or not isinstance(superseded, dict):
        raise AcceptanceFailure(f"revision omitted new/superseded records: {response}")
    new_id = new.get("id")
    superseded_id = superseded.get("id")
    if not isinstance(new_id, str) or not new_id or new_id == old_id:
        raise AcceptanceFailure(f"revision did not return a distinct successor id: {response}")
    if superseded_id != old_id:
        raise AcceptanceFailure(f"revision returned the wrong predecessor: {response}")
    if new.get("content") != new_content or new.get("project_id") != synthetic_project:
        raise AcceptanceFailure(f"revision successor content/scope mismatch: {new}")
    if superseded.get("content") != old_content:
        raise AcceptanceFailure(f"revision predecessor content mismatch: {superseded}")
    if new.get("type") != "preference" or superseded.get("type") != "preference":
        raise AcceptanceFailure(f"revision changed preference type: {response}")
    if new.get("preference_metadata") != preference_metadata or superseded.get("preference_metadata") != preference_metadata:
        raise AcceptanceFailure(f"revision did not retain preference metadata: {response}")
    if new.get("status") != "active" or superseded.get("status") != "archived":
        raise AcceptanceFailure(f"revision status did not supersede old version: {response}")
    return new_id, superseded_id


def _assert_revision_lineage(
    response: dict[str, Any], new_id: str, old_id: str, operation: str
) -> None:
    """Require the persisted ``supersedes`` edge, not only a revise success flag."""
    relationships = response.get("relationships")
    if not isinstance(relationships, list):
        raise AcceptanceFailure(f"{operation} did not return relationships: {response}")
    if not any(
        isinstance(item, dict)
        and item.get("source_id") == new_id
        and item.get("target_id") == old_id
        and item.get("relation") == "supersedes"
        for item in relationships
    ):
        raise AcceptanceFailure(f"{operation} omitted successor lineage {new_id}->{old_id}: {response}")


def _assert_deleted(response: dict[str, Any], memory_id: str) -> None:
    """Require a successful hard deletion for exactly the tracked memory."""
    if response.get("memory_id") != memory_id or response.get("deleted") is not True:
        raise AcceptanceFailure(f"forget did not delete tracked memory {memory_id}: {response}")
    if response.get("hard") is not True:
        raise AcceptanceFailure(f"forget did not perform the requested hard deletion: {response}")


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
    """Make cleanup failure terminal while preserving a primary error."""
    safe_message = _safe_text(message)
    receipt.status = "failed"
    receipt.cleanup_status = "failed"
    receipt.cleanup_rc = 1
    receipt.cleanup_errors.append(safe_message)
    receipt.warnings.append(safe_message)
    if receipt.error is None:
        receipt.error = safe_message


def _record_signal(receipt: AcceptanceReceipt, name: str) -> None:
    """Record bounded first-observation signal state."""
    receipt.signal_count = min(SIGNAL_COUNT_CAP, receipt.signal_count + 1)
    if receipt.first_signal is None:
        receipt.first_signal = name


def _select_return_code(receipt: AcceptanceReceipt) -> int:
    """Apply the v3 scalar precedence table before terminal publication."""
    if receipt.acceptance_rc not in (None, 0):
        return receipt.acceptance_rc
    if receipt.first_signal in {"SIGINT", "SIGTERM"}:
        return {"SIGINT": 130, "SIGTERM": 143}[receipt.first_signal]
    if receipt.cleanup_rc not in (None, 0):
        return 1
    if receipt.status != "passed" or receipt.acceptance_rc != 0 or receipt.cleanup_status != "succeeded":
        return 1
    return 0


def _cleanup_resources(
    prefix: list[str],
    compose_file: Path,
    project_name: str,
    environment: dict[str, str],
    timeout: float,
    bearer: str | None,
    tracked_memory_ids: list[str],
    receipt: AcceptanceReceipt,
    *,
    cleanup_deadline: float | None = None,
    down_timeout: float | None = None,
    cleanup_kill_grace: float = DEFAULT_CLEANUP_KILL_GRACE,
) -> None:
    """Delete tracked IDs within memory reserve, then perform one exact down."""
    cleanup_errors: list[str] = receipt.cleanup_errors
    memory_deadline = cleanup_deadline
    ledger = receipt.budget_ledger
    original_operation_deadline = ledger.get("operation_deadline")
    receipt.cleanup_status = "running"
    ledger["operation_deadline"] = memory_deadline
    if bearer and (memory_deadline is None or time.monotonic() < memory_deadline):
        try:
            _session_id, mcp_headers = _mcp_initialize(
                f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}", bearer,
                min(timeout, 15, max(0.001, (memory_deadline - time.monotonic()) if memory_deadline else timeout)),
            )
            for index, memory_id in enumerate(tracked_memory_ids, start=1000):
                if memory_deadline is not None and time.monotonic() >= memory_deadline:
                    receipt.cleanup_status = "not_run_due_budget"
                    receipt.cleanup_rc = 1
                    receipt.cleanup_errors.append(f"synthetic memory cleanup stopped before id {memory_id}")
                    receipt.warnings.append("synthetic memory cleanup stopped at reserved deadline")
                    break
                try:
                    operation_timeout = min(timeout, 15)
                    if memory_deadline is not None:
                        operation_timeout = min(operation_timeout, max(0.001, memory_deadline - time.monotonic()))
                    deleted = _mcp_call_checkpointed(
                        f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}", mcp_headers, bearer,
                        "weft_forget", {"memory_id": memory_id, "hard": True}, index, operation_timeout,
                    )
                    _assert_deleted(deleted, memory_id)
                except Exception as exc:
                    message = f"synthetic memory cleanup failed for tracked id: {exc}"
                    cleanup_errors.append(_safe_text(message))
        except Exception as exc:
            cleanup_errors.append(_safe_text(f"MCP synthetic cleanup unavailable: {exc}"))
    elif tracked_memory_ids:
        receipt.cleanup_status = "not_run_due_budget"
        receipt.cleanup_rc = 1
        receipt.cleanup_errors.append("synthetic memory cleanup not run: reserved memory budget exhausted")
        receipt.warnings.append("synthetic memory cleanup not run: reserved memory budget exhausted")
    try:
        # The down command must fit before publication reserve, including the
        # actual TERM/KILL/drain/reap overhead used by _run_compose_process.
        down_deadline = ledger.get("compose_cleanup_deadline")
        ledger["operation_deadline"] = down_deadline
        configured_cap = down_timeout or timeout
        overhead = cleanup_kill_grace + DRAIN_SECONDS + 0.5
        remaining = (float(down_deadline) - time.monotonic()) if down_deadline is not None else configured_cap
        if remaining <= overhead:
            receipt.original_return_code = None
            receipt.cleanup_status = "timed_out"
            receipt.cleanup_rc = 124
            receipt.cleanup_errors.append("Compose cleanup not attempted: reserve exhausted before teardown overhead")
            receipt.status = "failed"
            receipt.warnings.append("Compose cleanup not attempted before publication reserve")
        else:
            admitted_down = min(configured_cap, remaining - overhead)
            result = _run_compose(
                prefix,
                compose_file,
                project_name,
                ["down", "--volumes", "--remove-orphans"],
                {**environment, "COMPOSE_INTERACTIVE_NO_CLI": "1"},
                admitted_down,
                kill_grace=cleanup_kill_grace,
            )
            receipt.original_return_code = result.returncode
            if result.returncode:
                _mark_cleanup_failed(receipt, f"Compose cleanup failed: {result.stderr[-1000:]}")
            elif cleanup_errors or receipt.cleanup_rc not in (None, 0):
                receipt.cleanup_status = "failed"
                receipt.cleanup_rc = 1
            else:
                receipt.cleanup_status = "succeeded"
                receipt.cleanup_rc = 0
                receipt.checks["unique_resource_cleanup"] = "passed"
    except subprocess.TimeoutExpired as exc:
        receipt.original_return_code = None
        receipt.cleanup_status = "timed_out"
        receipt.cleanup_rc = 124
        receipt.cleanup_errors.append(_safe_text(f"Compose cleanup deadline exceeded: {exc}"))
        receipt.status = "failed"
        receipt.warnings.append("Compose cleanup deadline exceeded")
    except Exception as exc:
        _mark_cleanup_failed(receipt, f"Compose cleanup raised: {exc}")
    finally:
        ledger["operation_deadline"] = original_operation_deadline


def _validate_budget(total: float, finalization: float, compose: float, publication: float) -> None:
    """Validate the aggregate v3 envelope and its positive memory slice."""
    values = (total, finalization, compose, publication)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise AcceptanceFailure("aggregate timeout and reserves must be finite and positive")
    if not 0 < finalization < total:
        raise AcceptanceFailure("finalization reserve must be less than total timeout")
    if finalization <= compose + publication:
        raise AcceptanceFailure("finalization reserve must exceed cleanup plus publication reserves")


def _budget_ledger(args: argparse.Namespace, started: float) -> dict[str, Any]:
    """Create the logical deadline ledger; values are policy, not wall-clock SLA."""
    total = float(getattr(args, "total_timeout", DEFAULT_TOTAL_TIMEOUT))
    finalization = float(getattr(args, "finalization_reserve", DEFAULT_FINALIZATION_RESERVE))
    compose = float(getattr(args, "compose_cleanup_reserve", DEFAULT_COMPOSE_CLEANUP_RESERVE))
    publication = float(getattr(args, "publication_reserve", DEFAULT_PUBLICATION_RESERVE))
    _validate_budget(total, finalization, compose, publication)
    cleanup_timeout = float(getattr(args, "compose_cleanup_timeout", DEFAULT_COMPOSE_CLEANUP_TIMEOUT))
    cleanup_grace = float(getattr(args, "cleanup_kill_grace", DEFAULT_CLEANUP_KILL_GRACE))
    if cleanup_timeout + cleanup_grace + CLEANUP_DRAIN_REAP_ALLOWANCE >= compose:
        raise AcceptanceFailure("cleanup command cap, kill grace, and drain/reap allowance must fit compose reserve")
    monotonic_started = time.monotonic()
    acceptance_deadline = monotonic_started + total - finalization
    memory_cleanup_deadline = monotonic_started + total - compose - publication
    compose_cleanup_deadline = monotonic_started + total - publication
    return {
        "total_timeout": total,
        "finalization_reserve": finalization,
        "compose_cleanup_reserve": compose,
        "publication_reserve": publication,
        "started_at": started,
        "monotonic_started_at": monotonic_started,
        "deadline": monotonic_started + total,
        "acceptance_deadline": acceptance_deadline,
        "memory_cleanup_deadline": memory_cleanup_deadline,
        "compose_cleanup_deadline": compose_cleanup_deadline,
        "operation_deadline": acceptance_deadline,
        "memory_window": finalization - compose - publication,
        "memory_operation_cap": 15.0,
        "compose_cleanup_timeout": cleanup_timeout,
        "cleanup_kill_grace": cleanup_grace,
        "drain_reap_allowance": CLEANUP_DRAIN_REAP_ALLOWANCE,
        "teardown_overhead": cleanup_grace + DRAIN_SECONDS + 0.5,
    }


def _operation_timeout(args: argparse.Namespace, cap: float, deadline: float | None = None) -> float:
    """Admit one operation against the logical deadline without resetting caps."""
    remaining = deadline - time.monotonic() if deadline is not None else cap
    return max(0.001, min(float(getattr(args, "timeout", 120.0)), cap, remaining))


def run_acceptance(args: argparse.Namespace) -> AcceptanceReceipt:
    """Execute the complete bounded acceptance workflow."""
    global _ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT, _FINALIZING, _FINALIZATION_RUNNING, _SIGNALS_FROZEN
    global _PENDING_SIGNALS
    run_id = getattr(args, "run_id", None) or uuid.uuid4().hex[:12]
    project_name = getattr(args, "project_name", None) or f"weft-rc-{run_id}"
    synthetic_project = f"local-docker-rc-{run_id}"
    owner_user_id = str(uuid.uuid4())
    image = args.image
    checkpoint = args.receipt.with_name(f".{args.receipt.name}.checkpoint.json")
    started_at = time.time()
    receipt = AcceptanceReceipt(
        run_id=run_id,
        project_name=project_name,
        synthetic_project=synthetic_project,
        owner_user_id=owner_user_id,
        image=image,
        compose_file=str(args.compose_file),
        started_at=started_at,
        checkpoint_path=str(checkpoint),
        publication_owner=True,
    )
    receipt.budget_ledger = _budget_ledger(args, started_at)
    receipt.budget_initialized = True
    receipt.cleanup_registration = {
        "run_id": run_id,
        "generation": receipt.generation,
        "compose_project": project_name,
        "compose_file": str(args.compose_file),
        "image": image,
        "receipt_path": str(args.receipt),
        "checkpoint_path": str(checkpoint),
        "compose_invocation_attempted": False,
    }
    _ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT = receipt, checkpoint
    _FINALIZING = False
    _FINALIZATION_RUNNING = False
    _SIGNALS_FROZEN = False
    _PENDING_SIGNALS = []
    environment: dict[str, str] = {}
    tracked_memory_ids: list[str] = []
    prefix: list[str] = []
    # Registration is immutable and precedes all potentially effective external work.
    # The exact cleanup eligibility bit flips immediately before ``up``.
    cleanup_eligible = False
    try:
        # Identity/checkpoint, Compose discovery, and port setup are all inside
        # the guarded workflow so setup interruption still gets a final receipt.
        _write_checkpoint(checkpoint, receipt)
        docker_executable = getattr(args, "docker_executable", None)
        prefix = _docker_compose_prefix(docker_executable) if docker_executable is not None else _docker_compose_prefix()
        environment = os.environ.copy()
        environment.update(
            {
                "WEFT_LOCAL_IMAGE": image,
                "WEFT_LOCAL_API_KEY": secrets.token_urlsafe(32),
                "WEFT_LOCAL_USER_ID": owner_user_id,
                "WEFT_LOCAL_PORT": str(_free_port()),
            }
        )
        _REDACTION_VALUES.add(environment["WEFT_LOCAL_API_KEY"])
        base_url = f"http://127.0.0.1:{environment['WEFT_LOCAL_PORT']}"
        receipt.cleanup_registration["compose_invocation_attempted"] = True
        cleanup_eligible = True
        result = _run_compose(
            prefix, args.compose_file, project_name, ["up", "-d"], environment, args.timeout
        )
        receipt.original_return_code = result.returncode
        if result.returncode:
            receipt.acceptance_rc = 1
            raise AcceptanceFailure(f"Compose up failed: {result.stderr[-1000:]}")
        receipt.checks["compose_up"] = "passed"
        _phase(receipt, checkpoint, "healthz_readiness", lambda: _wait_for_health(base_url, args.timeout))
        receipt.checks["healthz_readiness"] = "passed"

        # Exercise the persistent-role path in this same isolated acceptance
        # attempt.  The owner-only bootstrap service first taints an existing
        # role with privilege-bearing attributes and a membership; the second
        # invocation must reset it idempotently before the app is trusted.
        taint = _run_compose(
            prefix,
            args.compose_file,
            project_name,
            [
                "run", "--rm", "--no-deps", "bootstrap",
                "/app/.venv/bin/python", "-m", "weft.local_bootstrap", "taint-runtime",
            ],
            environment,
            args.timeout,
        )
        if taint.returncode:
            raise AcceptanceFailure("runtime-role taint setup failed: " + taint.stderr[-1000:])
        reprovision = _run_compose(
            prefix,
            args.compose_file,
            project_name,
            [
                "run", "--rm", "--no-deps", "bootstrap",
                "/app/.venv/bin/python", "-m", "weft.local_bootstrap", "provision",
            ],
            environment,
            args.timeout,
        )
        if reprovision.returncode:
            raise AcceptanceFailure("runtime-role reprovision failed: " + reprovision.stderr[-1000:])
        receipt.checks["persistent_role_reprovision"] = "passed"
        future_probes = _run_compose(
            prefix,
            args.compose_file,
            project_name,
            [
                "run", "--rm", "--no-deps", "bootstrap",
                "/app/.venv/bin/python", "-m", "weft.local_bootstrap", "create-future-probes",
            ],
            environment,
            args.timeout,
        )
        if future_probes.returncode:
            raise AcceptanceFailure(
                "future ACL probe setup failed: " + future_probes.stderr[-1000:]
            )
        receipt.checks["future_default_acl_probe_setup"] = "passed"

        # Prove the effective app connection is the restricted non-owner role,
        # not merely that Compose declared a runtime-looking DSN.  The probe
        # audits the explicit relation/sequence allowlist and executes DDL,
        # setval, OAuth, and schema-ledger denial checks.
        role_probe = _run_compose(
            prefix,
            args.compose_file,
            project_name,
            ["exec", "--interactive=false", "-T", "app", "/app/.venv/bin/python", "-m", "weft.local_bootstrap", "verify-runtime"],
            environment,
            args.timeout,
        )
        if role_probe.returncode:
            failure = AcceptanceFailure(
                "runtime DB role verification failed: " + _safe_text(role_probe.stderr)
            )
            _record_runtime_verifier_failure(receipt, role_probe.stdout, role_probe.stderr, failure)
            raise failure
        try:
            receipt.verifier = _parse_runtime_verifier_output(role_probe.stdout, role_probe.stderr)
        except AcceptanceFailure as verifier_error:
            _record_runtime_verifier_failure(receipt, role_probe.stdout, role_probe.stderr, verifier_error)
            raise
        receipt.checks["runtime_db_role_isolation"] = "passed"
        receipt.checks["runtime_privilege_contract"] = "passed"

        bearer = environment["WEFT_LOCAL_API_KEY"]
        session_id, mcp_headers = _phase_http(
            receipt, checkpoint, "http_initialize", lambda: _mcp_initialize(base_url, bearer, args.timeout)
        )
        receipt.checks["image_cmd_mcp_initialize"] = "passed"
        request_id = 2
        sentinel = f"Local Docker RC sentinel {run_id}"
        remember = _mcp_call_checkpointed(
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
        returned_memory_id = remember.get("id")
        if isinstance(returned_memory_id, str) and returned_memory_id:
            tracked_memory_ids.append(returned_memory_id)
        _assert_memory(remember, sentinel, synthetic_project)
        receipt.checks["mcp_memory_write_fastembed"] = "passed"

        recall = _mcp_call_checkpointed(
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

        # Exercise preference capture and version-aware lifecycle through actual
        # MCP calls.  These values are synthetic and provider-free; no paid model
        # route is involved in this journey.
        preference_content = f"Local Docker RC preference {run_id} uses concise evidence"
        preference_metadata = {
            "schema_version": 1,
            "polarity": "positive",
            "strength": "hard",
            "subject": "local acceptance",
            "value": "concise evidence",
            "context": ["rc", "mcp-journey"],
        }
        preference = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_remember",
            {
                "content": preference_content,
                "type": "preference",
                "topic": ["local-docker-rc", "preference", run_id],
                "project_id": synthetic_project,
                "source": "conversation",
                "confidence": 1.0,
                "preference_metadata": preference_metadata,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        preference_id = preference.get("id")
        if isinstance(preference_id, str) and preference_id:
            tracked_memory_ids.append(preference_id)
        preference_id = _assert_preference(
            preference, preference_content, preference_metadata, synthetic_project
        )
        receipt.checks["mcp_preference_capture"] = "passed"
        preference_recall = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": preference_content,
                "project_id": synthetic_project,
                "type": "preference",
                "tier": "belief",
                "mode": "keyword",
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_recall(
            preference_recall,
            preference_id,
            preference_content,
            synthetic_project,
            "preference_recall",
        )
        receipt.checks["mcp_preference_recall"] = "passed"

        revised_content = f"Local Docker RC preference {run_id} uses verified concise evidence"
        revision = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_revise",
            {
                "memory_id": preference_id,
                "new_content": revised_content,
                "new_confidence": 1.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        # Register every returned synthetic ID before validating the payload so
        # partial/broken responses still enter the single normal cleanup path.
        for key in ("new", "superseded"):
            returned = revision.get(key)
            returned_id = returned.get("id") if isinstance(returned, dict) else None
            if isinstance(returned_id, str) and returned_id and returned_id not in tracked_memory_ids:
                tracked_memory_ids.append(returned_id)
        successor_id, predecessor_id = _assert_revision(
            revision, preference_id, preference_content, revised_content, synthetic_project,
            preference_metadata,
        )
        lineage = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_relate",
            {
                "action": "get",
                "memory_id": successor_id,
                "relation": "supersedes",
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_revision_lineage(lineage, successor_id, predecessor_id, "revision_lineage")
        receipt.checks["mcp_revision_supersession"] = "passed"
        successor_recall = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": revised_content,
                "project_id": synthetic_project,
                "tier": "belief",
                "mode": "keyword",
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_recall(
            successor_recall,
            successor_id,
            revised_content,
            synthetic_project,
            "revised_recall",
        )
        old_recall = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": preference_content,
                "project_id": synthetic_project,
                "tier": "belief",
                "mode": "keyword",
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_not_recalled(old_recall, predecessor_id, preference_content, "superseded_recall")
        receipt.checks["mcp_old_version_excluded"] = "passed"
        deleted = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_forget",
            {"memory_id": successor_id, "hard": True},
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_deleted(deleted, successor_id)
        # Do not ask normal cleanup to hard-delete an ID already deleted by the
        # journey; the predecessor remains tracked for the one cleanup pass.
        tracked_memory_ids[:] = [
            memory_id for memory_id in tracked_memory_ids if memory_id != successor_id
        ]
        post_delete = _mcp_call_checkpointed(
            base_url,
            mcp_headers,
            bearer,
            "weft_recall",
            {
                "query": revised_content,
                "project_id": synthetic_project,
                "tier": "belief",
                "mode": "keyword",
                "limit": 10,
                "threshold": 0.0,
            },
            request_id,
            args.timeout,
        )
        request_id += 1
        _assert_not_recalled(post_delete, successor_id, revised_content, "deleted_recall")
        receipt.checks["mcp_deletion_exclusion"] = "passed"

        handoff_summary = f"Acceptance handoff {run_id}"
        handoff = _mcp_call_checkpointed(
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
        if isinstance(handoff_id, str) and handoff_id:
            tracked_memory_ids.append(handoff_id)
        if not isinstance(handoff_id, str) or not handoff.get("stored"):
            raise AcceptanceFailure(f"handoff was not stored: {handoff}")
        receipt.checks["mcp_handoff"] = "passed"
        prime = _mcp_call_checkpointed(
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

        _phase_http(
            receipt,
            checkpoint,
            "http_negative_auth",
            lambda: _negative_auth_probe(base_url, args.timeout),
        )
        receipt.checks["negative_auth"] = "passed"

        # Project wall: the same owner must not see the synthetic handoff under another project.
        wrong_project_prime = _mcp_call_checkpointed(
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
        issued = _mcp_call_checkpointed(
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
        isolated_recall = _mcp_call_checkpointed(
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
        isolated_prime = _mcp_call_checkpointed(
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
            _phase_http(
                receipt, checkpoint, f"healthz_after_restart_{service}",
                lambda: _wait_for_health(base_url, args.timeout),
            )
        receipt.checks["restart_postgres_redis_app"] = "passed"

        previous_session_id = session_id
        session_id, mcp_headers = _mcp_initialize(base_url, bearer, args.timeout)
        if not session_id or session_id == previous_session_id:
            raise AcceptanceFailure(
                "process replacement did not provide a new MCP session identity"
            )
        receipt.checks["process_replacement_new_mcp_session"] = "passed"
        recalled_after_restart = _mcp_call_checkpointed(
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
        prime_after_restart = _mcp_call_checkpointed(
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
        receipt.acceptance_status = "succeeded"
        receipt.acceptance_rc = 0
        receipt.status = "passed"
    except AcceptanceInterrupted as exc:
        _record_signal(receipt, str(exc))
        receipt.acceptance_status = "failed"
        receipt.acceptance_rc = 130 if str(exc) == "SIGINT" else 143
        receipt.status = "failed"
        receipt.error = f"{type(exc).__name__}: interrupted"
    except Exception as exc:
        receipt.acceptance_status = "failed"
        receipt.acceptance_rc = 1
        receipt.status = "failed"
        receipt.error = f"{type(exc).__name__}: {exc}"
    finally:
        # Finalization is reentrancy-guarded. TERM/INT received here is recorded
        # by _interrupt_handler and cannot skip scoped cleanup or final writes.
        _FINALIZING = True
        if _FINALIZATION_RUNNING:
            return receipt
        _FINALIZATION_RUNNING = True
        try:
            # Namespace selection makes this teardown safe after partial Compose
            # startup. Missing setup fields mean no external resources exist yet.
            if cleanup_eligible:
                try:
                    _cleanup_resources(
                        prefix,
                        args.compose_file,
                        project_name,
                        environment,
                        args.timeout,
                        environment["WEFT_LOCAL_API_KEY"],
                        tracked_memory_ids,
                        receipt,
                        cleanup_deadline=receipt.budget_ledger.get("memory_cleanup_deadline"),
                        down_timeout=min(
                            float(getattr(args, "compose_cleanup_timeout", DEFAULT_COMPOSE_CLEANUP_TIMEOUT)),
                            max(0.001, receipt.budget_ledger.get("compose_cleanup_deadline", time.monotonic()) - time.monotonic()),
                        ),
                        cleanup_kill_grace=float(getattr(args, "cleanup_kill_grace", DEFAULT_CLEANUP_KILL_GRACE)),
                    )
                except BaseException as cleanup_exc:
                    _mark_cleanup_failed(receipt, f"cleanup failed: {_safe_text(str(cleanup_exc))}")
            receipt.finished_at = time.time()
            try:
                _admit_publication(receipt, "final checkpoint write")
                _write_checkpoint(checkpoint, receipt)
            except BaseException as checkpoint_exc:
                _mark_cleanup_failed(receipt, f"final checkpoint failed: {checkpoint_exc}")
        finally:
            _ACTIVE_RECEIPT, _ACTIVE_CHECKPOINT = None, None
            _FINALIZATION_RUNNING = False
    return receipt


def _positive_timeout(value: str) -> float:
    """Parse a finite positive timeout suitable for selector/HTTP deadlines."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than zero")
    return parsed


def _nonnegative_timeout(value: str) -> float:
    """Parse a finite non-negative cleanup grace value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("cleanup grace must be finite and non-negative")
    return parsed


def _safe_identity(value: str, label: str) -> str:
    """Reject shell/path metacharacters in externally supplied identities."""
    pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}" if label == "project" else r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}"
    if not re.fullmatch(pattern, value) or value.startswith(".") or ".." in value:
        raise argparse.ArgumentTypeError(f"unsafe {label}")
    return value


def _validate_output_targets(args: argparse.Namespace) -> None:
    """Reject collisions, symlinks, and non-regular terminal output targets."""
    receipt = Path(args.receipt)
    checkpoint = receipt.with_name(f".{receipt.name}.checkpoint.json")
    marker = receipt.with_name(f".{receipt.name}.complete.json")
    status = Path(args.status) if getattr(args, "status", None) is not None else None
    targets = [receipt, checkpoint, marker] + ([status] if status is not None else [])
    if len({str(path.resolve()) for path in targets}) != len(targets):
        raise AcceptanceFailure("receipt, checkpoint, marker, and status paths must be distinct")
    for path in targets:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise AcceptanceFailure(f"output target is not a regular non-symlink file: {path}")
        if path.exists():
            raise AcceptanceFailure(f"refusing pre-existing output target: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bounded acceptance CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, type=lambda value: _safe_identity(value, "image"), help="Candidate image tag")
    parser.add_argument("--project-name", default=None, type=lambda value: _safe_identity(value, "project"), help="Wrapper-owned Compose project identity")
    parser.add_argument("--run-id", default=None, type=lambda value: _safe_identity(value, "run"), help="Wrapper-owned run identity")
    parser.add_argument(
        "--compose-file", type=Path, default=Path(DEFAULT_COMPOSE_FILE),
        help="Standalone local RC Compose file",
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("artifacts/local-docker-acceptance-receipt.json"),
        help="Machine-readable receipt path",
    )
    parser.add_argument("--timeout", type=_positive_timeout, default=120.0, help="Per-operation timeout seconds")
    parser.add_argument("--total-timeout", type=_positive_timeout, default=DEFAULT_TOTAL_TIMEOUT)
    parser.add_argument("--finalization-reserve", type=_positive_timeout, default=DEFAULT_FINALIZATION_RESERVE)
    parser.add_argument("--compose-cleanup-reserve", type=_positive_timeout, default=DEFAULT_COMPOSE_CLEANUP_RESERVE)
    parser.add_argument("--publication-reserve", type=_positive_timeout, default=DEFAULT_PUBLICATION_RESERVE)
    parser.add_argument("--compose-cleanup-timeout", type=_positive_timeout, default=DEFAULT_COMPOSE_CLEANUP_TIMEOUT)
    parser.add_argument("--cleanup-kill-grace", type=_nonnegative_timeout, default=DEFAULT_CLEANUP_KILL_GRACE)
    parser.add_argument("--docker-executable", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None, help="Optional legacy status projection")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run acceptance, persist interruption evidence, and return a CI exit code."""
    args = parse_args(argv)
    try:
        _validate_output_targets(args)
        if args.docker_executable is not None:
            _docker_compose_prefix(args.docker_executable)
        _validate_budget(args.total_timeout, args.finalization_reserve, args.compose_cleanup_reserve, args.publication_reserve)
        _validate_signal_platform()
    except (AcceptanceFailure, argparse.ArgumentTypeError) as exc:
        print(f"FAIL: {_safe_text(str(exc))}", file=sys.stderr)
        return 2
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, _interrupt_handler)
    receipt: AcceptanceReceipt
    try:
        try:
            receipt = run_acceptance(args)
        except AcceptanceInterrupted as exc:
            receipt = AcceptanceReceipt(
                run_id="unstarted",
                project_name="none",
                synthetic_project="none",
                owner_user_id="none",
                image=args.image,
                compose_file=str(args.compose_file),
                started_at=time.time(),
                status="failed",
                error=f"{type(exc).__name__}: interrupted",
                primary_error=f"{type(exc).__name__}: interrupted",
                publication_owner=True,
                finished_at=time.time(),
            )
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
                primary_error=f"{type(exc).__name__}: {exc}",
                publication_owner=True,
                finished_at=time.time(),
            )
        # This is the sole CLI freeze event.  The mask remains blocked through
        # publication, serialization, and process return; run_acceptance itself
        # deliberately does not change the caller's signal mask.
        try:
            _freeze_cli_signals(receipt)
        except BaseException as exc:
            receipt.status = "failed"
            receipt.acceptance_rc = receipt.acceptance_rc or 1
            receipt.primary_error = receipt.primary_error or f"signal cutoff failed: {_safe_text(str(exc))}"
        receipt.selected_return_code = _select_return_code(receipt)
        receipt.lifecycle_status = "terminal"
        receipt.finished_at = receipt.finished_at or time.time()
        try:
            _publish_terminal(receipt, args.receipt, args.status)
        except BaseException as exc:
            if receipt.selected_return_code == 0:
                receipt.selected_return_code = 1
            receipt.status = "failed"
            receipt.lifecycle_status = "incomplete"
            receipt.primary_error = receipt.primary_error or f"terminal publication failed: {_safe_text(str(exc))}"
            print(f"FAIL: {_safe_text(str(exc))}", file=sys.stderr)
        print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
        return int(receipt.selected_return_code if receipt.selected_return_code is not None else 1)
    finally:
        # Restore handlers even when final receipt serialization/output is
        # interrupted. SIGKILL cannot be handled; wrapper exact-project cleanup
        # remains the independent fallback and is not a SIGKILL guarantee.
        for sig, handler in previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


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
