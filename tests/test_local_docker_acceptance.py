"""Host-only safety tests for the local Docker RC contract.

These tests never invoke Docker, mutate the shared environment, or contact a
provider. Runtime acceptance remains the responsibility of
``scripts/local_docker_acceptance.py`` on a disposable Docker installation.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from weft.config import load_config


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.local.yml"
RUNNER = ROOT / "scripts/local_docker_acceptance.py"


_spec = importlib.util.spec_from_file_location("local_docker_acceptance", RUNNER)
assert _spec is not None and _spec.loader is not None
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


def test_local_compose_has_private_dependencies_and_loopback_app_port() -> None:
    """Compose exposes only the app loopback endpoint and health-gates services."""
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = document["services"]
    assert set(services) == {"postgres", "redis", "app"}
    assert "ports" not in services["postgres"]
    assert "ports" not in services["redis"]
    assert services["app"]["ports"] == ["127.0.0.1:${WEFT_LOCAL_PORT:-18000}:8000"]
    assert services["app"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["app"]["depends_on"]["redis"]["condition"] == "service_healthy"
    environment = services["app"]["environment"]
    assert environment["WEFT_DATABASE_URL"].endswith("@postgres:5432/weft")
    assert environment["WEFT_REDIS_URL"] == "redis://redis:6379"
    assert environment["WEFT_MIGRATION_MODE"] == "apply"
    assert environment["WEFT_OUTBOUND_CONNECTOR"] == "none"
    assert environment["WEFT_QUARANTINE_REVIEW_ENABLED"] == "0"
    assert "WEFT_LOCAL_API_KEY:?" in environment["WEFT_API_KEY"]
    assert "WEFT_LOCAL_USER_ID:?" in environment["WEFT_DEFAULT_USER_ID"]


def test_local_compose_has_no_literal_secret_or_production_override() -> None:
    """The local file references runtime secrets and leaves Dockerfile untouched."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "weft_local" in text  # development DB password, overridable at runtime
    assert "WEFT_LOCAL_API_KEY:?" in text
    assert "WEFT_API_KEY: local" not in text
    assert "ANTHROPIC_API_KEY" not in text
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "WEFT_MIGRATION_MODE" not in dockerfile


def test_dockerignore_blocks_private_and_nested_workspace_inputs() -> None:
    """The normal build context cannot copy secrets, artifacts, or worktrees."""
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".env", ".loom/", ".polytoken/", "artifacts/", ".ci-rc-worktree/", ".rc-candidate-worktree/"):
        assert entry in ignored


def test_acceptance_runner_is_stdlib_only_and_redacts_secrets() -> None:
    """Runner imports no provider/client SDK and receipt omits runtime key values."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    assert imports <= {
        "argparse", "dataclasses", "json", "os", "secrets", "shutil", "socket",
        "subprocess", "sys", "time", "urllib", "uuid", "pathlib", "typing",
    }
    assert "ANTHROPIC_API_KEY" not in RUNNER.read_text(encoding="utf-8")
    assert '"WEFT_LOCAL_API_KEY"' not in "\n".join(
        line for line in RUNNER.read_text(encoding="utf-8").splitlines()
        if "to_dict" in line or "return {" in line
    )
    receipt_source = RUNNER.read_text(encoding="utf-8")
    assert '"cleanup_scope"' in receipt_source
    assert '"unique_resource_cleanup"' in receipt_source


def test_quarantine_env_override_is_local_and_preserves_explicit_text_provider(
    tmp_path: Path, monkeypatch,
) -> None:
    """The new local-only switch disables paid review without changing providers."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[text_generation]\nprovider = "openai"\n', encoding="utf-8")
    monkeypatch.setattr("weft.config.CONFIG_PATH", config_path)
    monkeypatch.setenv("WEFT_QUARANTINE_REVIEW_ENABLED", "0")
    monkeypatch.delenv("WEFT_ENV", raising=False)
    config = load_config()
    assert config.quarantine_review.enabled is False
    assert config.text_generation.provider == "openai"

    monkeypatch.setenv("WEFT_QUARANTINE_REVIEW_ENABLED", "1")
    enabled = load_config()
    assert enabled.quarantine_review.enabled is True
    assert enabled.text_generation.provider == "openai"


def test_receipt_schema_is_json_and_never_mentions_bearer_fields() -> None:
    """A representative receipt remains machine-readable and secret-free."""
    receipt = {
        "schema": "weft.local-docker-acceptance.v1",
        "status": "failed",
        "checks": {},
        "warnings": [],
        "cleanup_scope": {"compose_project": "weft-rc-example"},
    }
    encoded = json.dumps(receipt)
    assert json.loads(encoded)["schema"].endswith(".v1")
    assert "Authorization" not in encoded
    assert "WEFT_LOCAL_API_KEY" not in encoded


def test_recall_rejects_echoed_query_without_actual_evidence() -> None:
    """An echoed query cannot satisfy the tracked result assertion."""
    with pytest.raises(acceptance.AcceptanceFailure, match="omitted tracked memory"):
        acceptance._assert_recall(
            {"query": "sentinel", "results": []},
            "weft-memory",
            "sentinel",
            "local-project",
            "recall",
        )


def test_prime_rejects_empty_response_without_authoritative_handoff() -> None:
    """A dict-shaped or empty prime response cannot satisfy continuity."""
    with pytest.raises(acceptance.AcceptanceFailure, match="omitted authoritative handoff"):
        acceptance._assert_prime_handoff(
            {"handoff": []},
            "weft-handoff",
            "Acceptance handoff",
            "local-project",
            "prime",
        )


def test_memory_assertion_rejects_missing_project_ownership_evidence() -> None:
    """Remember must return explicit project evidence before tracking an ID."""
    with pytest.raises(acceptance.AcceptanceFailure, match="synthetic project scope"):
        acceptance._assert_memory(
            {"id": "weft-memory", "content": "sentinel"},
            "sentinel",
            "local-project",
        )


def test_cleanup_failure_forces_failed_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleanup errors cannot coexist with a passing receipt."""
    receipt = acceptance.AcceptanceReceipt(
        run_id="cleanup-test",
        project_name="weft-rc-cleanup-test",
        synthetic_project="local-project",
        owner_user_id="owner",
        image="image",
        compose_file="docker-compose.local.yml",
        started_at=0.0,
        status="passed",
    )
    compose_calls: list[list[str]] = []

    def fail_mcp(*args, **kwargs):
        raise acceptance.AcceptanceFailure("app unavailable")

    def fail_down(prefix, compose_file, project_name, args, env, timeout):
        compose_calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "down failed")

    monkeypatch.setattr(acceptance, "_mcp_initialize", fail_mcp)
    monkeypatch.setattr(acceptance, "_run_compose", fail_down)
    acceptance._cleanup_resources(
        ["docker", "compose"],
        Path("docker-compose.local.yml"),
        receipt.project_name,
        {"WEFT_LOCAL_PORT": "18000"},
        1.0,
        "bearer",
        [],
        receipt,
    )
    assert receipt.status == "failed"
    assert receipt.warnings
    assert "unique_resource_cleanup" not in receipt.checks
    assert compose_calls == [["down", "--volumes", "--remove-orphans"]]


def test_partial_startup_still_attempts_unique_project_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception from Compose up still invokes down for the unique namespace."""
    compose_calls: list[list[str]] = []

    def compose(prefix, compose_file, project_name, args, env, timeout):
        compose_calls.append(args)
        if args == ["up", "-d"]:
            raise subprocess.TimeoutExpired(args, timeout)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(acceptance, "_docker_compose_prefix", lambda: ["docker", "compose"])
    monkeypatch.setattr(acceptance, "_run_compose", compose)
    args = Namespace(
        image="image",
        compose_file=Path("docker-compose.local.yml"),
        receipt=Path("receipt.json"),
        timeout=1.0,
    )
    receipt = acceptance.run_acceptance(args)
    assert receipt.status == "failed"
    assert compose_calls == [["up", "-d"], ["down", "--volumes", "--remove-orphans"]]
    assert receipt.checks["unique_resource_cleanup"] == "passed"
