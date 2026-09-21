#!/usr/bin/env python3
"""Verify the RC hosted boundary without contacting a hosted service.

The verifier is deliberately source-bound and non-destructive. It records the
local candidate's content fingerprint and the sanitized Fly contract, but an
external deployment pin/ref/trigger is never inferred from Git or a local
configuration file. This leaf is therefore normally run in BLOCKED mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ACCEPTANCE = "RC-FL-21"
SCHEMA = "rc-fl-21.hosted-boundary.v1"
STATUS_CHOICES = ("READY", "HOLD", "BLOCKED")
BLOCKED_REASON = "external status not yet supplied"
HASH_ALGORITHM = "sha256"

SOURCE_FILES = (
    "pyproject.toml",
    "deploy/examples/fly/fly.example.toml",
    "docs/configuration.md",
    "docs/database-schema.md",
    "weft/db/migrations/_runner.py",
    "weft/db/migrations/__init__.py",
    "weft/health_check.py",
    "weft/mcp/server.py",
    "weft/mcp/tools.py",
    "scripts/hosted_rc_smoke.py",
    "tests/test_rc_hosted_boundary.py",
)
OPTIONAL_FLY_CONFIG = "fly.toml"

PIN_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|[0-9a-f]{7,40})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{16,})")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:ENCRYPTED |RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"\b[A-Z][A-Z0-9_-]*(?:API|ACCESS|AUTH|TOKEN|PASSWORD|PASSWD|"
    r"SECRET|PRIVATE[_-]?KEY|CREDENTIAL|KEY)[A-Z0-9_-]*\b\s*[:=]\s*"
    r"([^\s|`]+)"
)
_DSN_RE = re.compile(r"(?i)\b(?:postgres(?:ql)?|redis)://[^\s`\"']+")
_GENERIC_HOSTS = {"localhost", "127.0.0.1", "::1", "destination.example"}
_PLACEHOLDER_WORDS = {
    "change_me",
    "changeme",
    "example",
    "example_key",
    "example_token",
    "dummy",
    "none",
    "null",
    "pending",
    "placeholder",
    "placeholder_token",
    "replace_me",
    "test",
    "testing",
    "todo",
    "tbd",
    "unset",
    "your_api_key",
    "your_token",
    "0",
    "1",
    "true",
    "false",
}


def _is_generic_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host in _GENERIC_HOSTS or host.endswith(
        (".example", ".example.com", ".invalid", ".test")
    )


def _is_safe_documented_value(value: str) -> bool:
    value = value.strip("\\\"'),.;")
    if not value:
        return True
    lowered = value.lower().replace("-", "_")
    if lowered in _PLACEHOLDER_WORDS or lowered.startswith(("<", "{", "$")):
        return True
    if "localhost" in lowered or "127.0.0.1" in lowered or "::1" in lowered:
        return True
    if lowered.startswith(("https://", "http://")):
        try:
            return _is_generic_host(urlsplit(lowered).hostname or "")
        except ValueError:
            return False
    return False


def _has_credential_material(text: str) -> bool:
    bearer = _BEARER_RE.search(text)
    if bearer and not _is_safe_documented_value(bearer.group(1)):
        return True
    if _PRIVATE_KEY_RE.search(text):
        return True
    for match in _SENSITIVE_ASSIGNMENT_RE.finditer(text):
        if not _is_safe_documented_value(match.group(1)):
            return True
    for match in _DSN_RE.finditer(text):
        parsed = urlsplit(match.group(0).rstrip(".,;`"))
        host = (parsed.hostname or "").lower().rstrip(".")
        if not _is_generic_host(host):
            return True
        # Generic local/example fixtures are deliberately portable. A real
        # credential on a real host remains disallowed by the branch above.
        if (parsed.username or parsed.password) and not _is_generic_host(host):
            return True
    return False


def canonical(value: Any) -> str:
    """Canonical JSON used for every receipt digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def _safe_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ValueError(f"source path is not repository-relative: {relative!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {relative}") from exc
    if (root / relative).is_symlink():
        raise ValueError(f"source path is symlinked: {relative}")
    return path


def source_hashes(root: Path, paths: tuple[str, ...] = SOURCE_FILES) -> dict[str, str]:
    """Hash current bytes; never use HEAD or timestamps as provenance."""
    if tuple(paths) != SOURCE_FILES:
        raise ValueError("source path set does not match the canonical boundary")
    result: dict[str, str] = {}
    for relative in paths:
        path = _safe_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"source boundary file is missing: {relative}")
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        if _has_credential_material(text):
            raise ValueError(f"credential-shaped material in source boundary: {relative}")
        result[relative] = sha256_bytes(data)
    return result


def _redacted_config(path: Path) -> dict[str, Any]:
    """Parse config only after rejecting credential-bearing values."""
    text = path.read_text(encoding="utf-8")
    if _has_credential_material(text):
        raise ValueError("Fly configuration contains credential/DSN-shaped material")
    parsed = tomllib.loads(text)

    def shape(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): shape(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [shape(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    return shape(parsed)


def config_comparison(root: Path) -> dict[str, Any]:
    example = _safe_path(root, "deploy/examples/fly/fly.example.toml")
    if not example.is_file():
        raise FileNotFoundError("sanitized Fly example is missing")
    example_data = _redacted_config(example)
    current = _safe_path(root, OPTIONAL_FLY_CONFIG)
    if not current.is_file():
        return {
            "current_config": OPTIONAL_FLY_CONFIG,
            "current_present": False,
            "observation": "private fly.toml not present; no production topology inferred",
            "sanitized_example_sha256": sha256_bytes(example.read_bytes()),
            "differences": "not-comparable-without-private-config",
        }
    current_data = _redacted_config(current)
    differences = sorted(_config_differences(current_data, example_data))
    return {
        "current_config": OPTIONAL_FLY_CONFIG,
        "current_present": True,
        "sanitized_example_sha256": sha256_bytes(example.read_bytes()),
        "differences": differences,
    }


def _config_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_config_differences(left[key], right[key], child))
        return result
    return [prefix] if left != right else []


def _assert_status(status: str, reason: str | None) -> None:
    if status not in STATUS_CHOICES:
        raise ValueError(f"status must be one of {STATUS_CHOICES}")
    if status == "BLOCKED" and reason != BLOCKED_REASON:
        raise ValueError(f"BLOCKED requires exact reason {BLOCKED_REASON!r}")
    if status != "BLOCKED" and reason:
        raise ValueError(f"{status} cannot carry a BLOCKED reason")


def _external_evidence(status: str) -> dict[str, Any]:
    if status in {"READY", "HOLD"}:
        raise ValueError(f"{status} requires external pin, trigger, ref, and deployed pin")
    return {
        "candidate_pin": None,
        "production_pin": None,
        "deployment_trigger": None,
        "deployment_ref": None,
        "deployed_pin": None,
        "supplied": False,
    }


def _grounded_assumptions() -> dict[str, str]:
    return {
        "migration_owner_path": "weft migrate with WEFT_OWNER_DATABASE_URL; owner applies DDL before restricted startup",
        "runtime_migration_mode": "WEFT_MIGRATION_MODE=verify performs read-only ledger/runtime checks",
        "runtime_role": "restricted weft_app; source checks reject SUPERUSER/BYPASSRLS/public ownership/role memberships",
        "rls": "memories RLS and scoped policies are required by verify_runtime_invariants",
        "tls": "sslmode=require uses Python default TLS context with certificate and hostname verification; CA settings do not activate TLS alone",
        "health": "GET /healthz is unauthenticated and returns status=ok only when the pool responds",
        "mcp": "streamable HTTP /mcp requires bearer auth; initialize precedes tools/call",
    }


def _future_smoke() -> dict[str, Any]:
    return {
        "command": "uv run python scripts/hosted_rc_smoke.py --endpoint https://<allowlisted-host>/mcp --allow-host <allowlisted-host> --credential-env WEFT_HOSTED_SMOKE_TOKEN --execute --approve-hosted --approve-destructive --receipt <receipt.json>",
        "expected": [
            "GET /healthz returns HTTP 200 and {status: ok}",
            "MCP initialize returns a result and MCP-Session-Id",
            "weft_remember stores one synthetic unique-ID fact in an explicit synthetic project under the authenticated bearer owner scope (remember has no user_id argument)",
            "weft_recall returns that exact synthetic content/id under the same authenticated bearer owner and project scope",
            "if remember returns no safe id, one bounded keyword recall with the exact sentinel project/content recovers only exact candidates for cleanup",
            "weft_forget hard-deletes every tracked or conservatively recovered synthetic ID under the same authenticated bearer owner scope",
        ],
        "cleanup": "mandatory finally cleanup; safe remember IDs are tracked before response validation; missing/invalid IDs trigger one bounded exact-marker recovery; no exact candidate or recall/forget failure is explicit cleanup_incomplete",
        "approval": "separate operator approval is required; this leaf does not execute the command",
    }


def _leaf_flags() -> dict[str, bool]:
    return {
        "deploy": False,
        "migrate": False,
        "restart": False,
        "secret_rotation": False,
        "hosted_write": False,
        "network_calls": False,
        "future_smoke_executed": False,
    }


def build_report(root: Path, *, mode: str, reason: str | None) -> dict[str, Any]:
    status = mode.upper()
    _assert_status(status, reason)
    hashes = source_hashes(root)
    source_digest = sha256_json(hashes)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "acceptance": ACCEPTANCE,
        "status": status,
        "reason": reason,
        "hash_algorithm": HASH_ALGORITHM,
        "source": {
            "candidate_pin": f"source-bound:{source_digest}",
            "production_pin": None,
            "source_file_sha256": hashes,
            "source_files_digest": source_digest,
            "pin_comparison": "not-comparable: external production pin absent",
        },
        "external_evidence": _external_evidence(status),
        "fly_config": config_comparison(root),
        "grounded_assumptions": _grounded_assumptions(),
        "future_smoke": _future_smoke(),
        "leaf_flags": _leaf_flags(),
    }
    # The digest is an integrity checksum, not authorization or authenticity.
    report["receipt_digest"] = sha256_json(report)
    return report


def render_receipt(report: dict[str, Any]) -> str:
    payload = canonical(report)
    return "\n".join(
        [
            "# RC-FL-21 Hosted Boundary",
            "",
            f"**Status:** `{report['status']}`",
            f"**Acceptance:** `{report['acceptance']}`",
            f"**Reason:** `{report['reason']}`",
            "",
            "This is deterministic source evidence only. No Fly, Supabase, network, deployment, migration, restart, secret rotation, or hosted write was performed.",
            "",
            "## Machine receipt",
            "",
            "```json",
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            "```",
            "",
            f"Receipt digest (sha256 over canonical JSON above): `{report['receipt_digest']}`",
            "",
            "## Interpretation",
            "",
            "`BLOCKED` is the successful result for this leaf: external status was not supplied. READY/HOLD must never be inferred from source, Git HEAD, or a local sanitized example. The public digest is only an integrity checksum, never authorization or authenticity.",
            "",
            "## Future smoke boundary",
            "",
            "The exact command, expected health/MCP/remember/recall/cleanup results, endpoint allowlisting, explicit multi-factor approval, and credential-reference rules are recorded in the machine receipt. The future script defaults to plan mode and is not run here.",
            "",
        ]
    )


def _expect_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} has unexpected keys: expected {sorted(expected)}, got {actual}")


def _expect_string(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")


def _validate_exact_schema(report: dict[str, Any], root: Path) -> dict[str, str]:
    """Validate every deterministic field before checking the public digest."""
    top = {"schema", "acceptance", "status", "reason", "hash_algorithm", "source", "external_evidence", "fly_config", "grounded_assumptions", "future_smoke", "leaf_flags", "receipt_digest"}
    _expect_keys(report, top, "receipt")
    if report["schema"] != SCHEMA or report["acceptance"] != ACCEPTANCE:
        raise ValueError("receipt schema or acceptance mismatch")
    if report["status"] != "BLOCKED" or report["reason"] != BLOCKED_REASON:
        raise ValueError("this leaf only accepts the exact BLOCKED status/reason")
    if report["hash_algorithm"] != HASH_ALGORITHM:
        raise ValueError("receipt hash algorithm mismatch")
    _expect_string(report["reason"], "reason")
    _expect_string(report["receipt_digest"], "receipt_digest")
    if not SHA256_RE.fullmatch(report["receipt_digest"]):
        raise ValueError("receipt digest must be lowercase sha256")

    hashes = source_hashes(root)
    source = report["source"]
    _expect_keys(source, {"candidate_pin", "production_pin", "source_file_sha256", "source_files_digest", "pin_comparison"}, "source")
    expected_source_digest = sha256_json(hashes)
    if source["candidate_pin"] != f"source-bound:{expected_source_digest}":
        raise ValueError("candidate pin is not source-bound to the canonical source set")
    if source["production_pin"] is not None or source["pin_comparison"] != "not-comparable: external production pin absent":
        raise ValueError("production comparison is not the blocked deterministic value")
    if source["source_file_sha256"] != hashes or source["source_files_digest"] != expected_source_digest:
        raise ValueError("source provenance is stale or tampered")

    external = report["external_evidence"]
    _expect_keys(external, {"candidate_pin", "production_pin", "deployment_trigger", "deployment_ref", "deployed_pin", "supplied"}, "external_evidence")
    if external != _external_evidence("BLOCKED"):
        raise ValueError("blocked receipt cannot contain external evidence")
    if report["fly_config"] != config_comparison(root):
        raise ValueError("Fly configuration comparison is stale or tampered")
    if report["grounded_assumptions"] != _grounded_assumptions():
        raise ValueError("grounded assumptions are stale or tampered")
    future = report["future_smoke"]
    _expect_keys(future, {"command", "expected", "cleanup", "approval"}, "future_smoke")
    if future != _future_smoke():
        raise ValueError("future smoke contract is stale or tampered")
    flags = report["leaf_flags"]
    _expect_keys(flags, set(_leaf_flags()), "leaf_flags")
    if flags != _leaf_flags():
        raise ValueError("blocked receipt has unsafe or non-canonical execution flags")
    return hashes


def validate_receipt(report: dict[str, Any], root: Path) -> None:
    # Structural and source claims are checked first; only then is the public
    # digest recomputed. A matching digest cannot authorize altered claims.
    _validate_exact_schema(report, root)
    unsigned = dict(report)
    del unsigned["receipt_digest"]
    if report["receipt_digest"] != sha256_json(unsigned):
        raise ValueError("receipt digest mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("ready", "hold", "blocked"))
    parser.add_argument("--reason", required=True)
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    receipt = args.receipt if args.receipt.is_absolute() else root / args.receipt
    report = build_report(root, mode=args.mode, reason=args.reason)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(render_receipt(report), encoding="utf-8")
    print(f"{ACCEPTANCE} hosted boundary: {report['status']} ({report['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
