"""RC-FL-14 host-safe lifecycle and isolation contract tests.

These tests deliberately exercise deterministic acceptance seams only.  They do
not start Docker, call HTTP, connect to Postgres, or delete resources.  The
single real-container probe is owned by the RC-FL-14 execution command.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.local.yml"
RUNNER = ROOT / "scripts/local_docker_acceptance.py"
_spec = importlib.util.spec_from_file_location("local_docker_acceptance_lifecycle_v4", RUNNER)
assert _spec and _spec.loader
acceptance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acceptance
_spec.loader.exec_module(acceptance)


def _receipt(**overrides):
    values = dict(
        run_id="v4-run",
        project_name="weft-rc-v4",
        synthetic_project="local-docker-rc-v4",
        owner_user_id="owner-v4",
        image="safe:tag",
        compose_file="docker-compose.local.yml",
        started_at=0.0,
    )
    values.update(overrides)
    return acceptance.AcceptanceReceipt(**values)


def test_compose_keeps_owner_only_migration_and_bootstrap_separate_from_app() -> None:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    migrate = services["migrate"]["environment"]
    bootstrap = services["bootstrap"]["environment"]
    app = services["app"]["environment"]

    assert "WEFT_OWNER_DATABASE_URL" in migrate
    assert "WEFT_DATABASE_URL" in bootstrap
    assert app["WEFT_DATABASE_URL"].startswith("postgresql://weft_app:")
    assert "WEFT_OWNER_DATABASE_URL" not in app
    assert app["WEFT_MIGRATION_MODE"] == "verify"
    assert services["bootstrap"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["app"]["depends_on"]["bootstrap"]["condition"] == (
        "service_completed_successfully"
    )


def test_prime_isolation_rejects_either_id_or_content_leak() -> None:
    response = {
        "handoff": [
            {"id": "other", "content": "unrelated"},
            {"id": "target", "content": "Acceptance handoff v4"},
        ]
    }
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated handoff"):
        acceptance._assert_not_primed(response, "target", "Acceptance handoff v4", "prime")

    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated handoff"):
        acceptance._assert_not_primed(
            {"handoff": [{"id": "other", "content": "Acceptance handoff v4"}]},
            "target",
            "Acceptance handoff v4",
            "prime",
        )


def test_recall_isolation_rejects_id_or_content_leak() -> None:
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated memory"):
        acceptance._assert_not_recalled(
            {"results": [{"id": "tracked", "content": "safe"}]},
            "tracked",
            "safe",
            "recall_distinct_owner",
        )
    with pytest.raises(acceptance.AcceptanceFailure, match="exposed isolated memory"):
        acceptance._assert_not_recalled(
            {"results": [{"id": "different", "content": "safe"}]},
            "tracked",
            "safe",
            "recall_distinct_owner",
        )


def _runtime_verifier_payload(**overrides):
    payload = {
        "current_user": "weft_app",
        "rolsuper": False,
        "rolbypassrls": False,
        "rolcreatedb": False,
        "rolcreaterole": False,
        "rolreplication": False,
        "rolinherit": False,
        "rolconfig": None,
        "memories_rls": True,
        "memories_owner": "weft",
        "schema_migrations_write": False,
        "schema_migrations_read": True,
        "explicit_runtime_tables": sorted(acceptance._RUNTIME_VERIFIER_EXPECTED_TABLES),
        "dormant_oauth_tables_denied": True,
        "sequence_setval": False,
        "schema_ddl": False,
        "database_name": "weft",
        "schema_name": "public",
        "role_configuration_reset": True,
        "public_acl_reset": True,
        "probe_details": {
            "schema_ddl_probe": "denied_and_rollback_verified",
            "sequence_setval_probe": "skipped_no_allowlisted_sequence",
            "future_table_select_probe": "denied_and_rollback_verified",
            "future_sequence_setval_probe": "denied_and_rollback_verified",
        },
    }
    payload.update(overrides)
    return payload


def test_role_probe_stdout_observes_structured_contract_and_retains_decisive_fields() -> None:
    secret = "registered-verifier-secret"
    acceptance._REDACTION_VALUES.add(secret)
    payload = _runtime_verifier_payload(
        explicit_runtime_tables=sorted(acceptance._RUNTIME_VERIFIER_EXPECTED_TABLES),
        probe_details={
            "schema_ddl_probe": "denied_and_rollback_verified",
            "sequence_setval_probe": "skipped_no_allowlisted_sequence",
            "future_table_select_probe": "denied_and_rollback_verified",
            "future_sequence_setval_probe": "denied_and_rollback_verified",
        },
    )
    raw = json.dumps(payload)
    assert len(raw) > 1000

    verifier = acceptance._parse_runtime_verifier_output(raw, "")

    assert verifier["status"] == "passed"
    assert verifier["probe_details"]["schema_ddl_probe"].startswith("denied_and_rollback_verified")
    assert verifier["probe_details"]["sequence_setval_probe"] == "skipped_no_allowlisted_sequence"
    assert verifier["skips"] == ["sequence_setval_probe=skipped_no_allowlisted_sequence"]
    assert verifier["summary"]["explicit_runtime_tables"] == sorted(acceptance._RUNTIME_VERIFIER_EXPECTED_TABLES)
    assert len(verifier["summary"]["database_name"]) <= acceptance._REDACTION_BOUNDARY
    assert secret not in json.dumps(verifier)
    assert len(verifier["probe_details"]["schema_ddl_probe"]) <= acceptance._REDACTION_BOUNDARY

    sanitized = acceptance._sanitize_structured_text(
        f"DATABASE_URL=postgresql://user:{secret}@db/weft " + "x" * 1200
    )
    assert secret not in sanitized
    assert "[REDACTED]" in sanitized
    assert len(sanitized) <= acceptance._REDACTION_BOUNDARY


@pytest.mark.parametrize(
    "raw",
    ["", "not-json", "[]", json.dumps({"probe_details": {}})],
)
def test_role_probe_invalid_output_fails_closed(raw: str) -> None:
    with pytest.raises(acceptance.AcceptanceFailure):
        acceptance._parse_runtime_verifier_output(raw, "")


@pytest.mark.parametrize(
    "field,value",
    [
        ("current_user", "evil"),
        ("database_name", "other"),
        ("schema_name", "private"),
        ("rolsuper", True),
        ("rolbypassrls", True),
        ("rolcreatedb", True),
        ("rolcreaterole", True),
        ("rolreplication", True),
        ("rolinherit", True),
        ("memories_rls", False),
        ("schema_migrations_write", True),
        ("schema_migrations_read", False),
        ("dormant_oauth_tables_denied", False),
        ("sequence_setval", True),
        ("schema_ddl", True),
        ("role_configuration_reset", False),
        ("public_acl_reset", False),
        ("memories_owner", "weft_app"),
        ("rolconfig", ["unsafe=on"]),
        ("explicit_runtime_tables", ["memories"]),
    ],
)
def test_role_probe_semantic_contradictions_fail_closed(field: str, value) -> None:
    payload = _runtime_verifier_payload(**{field: value})

    with pytest.raises(acceptance.AcceptanceFailure, match="canonical contract|ownership|null"):
        acceptance._parse_runtime_verifier_output(json.dumps(payload), "")


def test_role_probe_wrong_nested_shape_fails_closed() -> None:
    payload = _runtime_verifier_payload()
    payload["probe_details"]["sequence_setval_probe"] = {"status": "skipped"}

    with pytest.raises(acceptance.AcceptanceFailure, match="probe_details"):
        acceptance._parse_runtime_verifier_output(json.dumps(payload), "")


def test_role_probe_duplicate_json_keys_fail_closed() -> None:
    payload = json.dumps(_runtime_verifier_payload())
    raw = payload.replace(
        '"current_user": "weft_app",',
        '"current_user": "weft_app", "current_user": "evil",',
        1,
    )

    with pytest.raises(acceptance.AcceptanceFailure, match="valid JSON"):
        acceptance._parse_runtime_verifier_output(raw, "")


def test_role_probe_duplicate_nested_object_keys_fail_closed() -> None:
    payload = _runtime_verifier_payload()
    payload["probe_details"] = {
        "schema_ddl_probe": "denied_and_rollback_verified",
        "sequence_setval_probe": "skipped_no_allowlisted_sequence",
        "future_table_select_probe": "denied_and_rollback_verified",
        "future_sequence_setval_probe": "denied_and_rollback_verified",
    }
    raw = json.dumps(payload).replace(
        '"probe_details": {',
        '"probe_details": {"nested": {"status": "ok", "status": "bad"},',
        1,
    )
    with pytest.raises(acceptance.AcceptanceFailure, match="valid JSON"):
        acceptance._parse_runtime_verifier_output(raw, "")


def test_role_probe_accepts_sequence_denial_variant() -> None:
    payload = _runtime_verifier_payload()
    payload["probe_details"]["sequence_setval_probe"] = "denied_and_rollback_verified"

    verifier = acceptance._parse_runtime_verifier_output(json.dumps(payload), "")

    assert verifier["skips"] == []
    assert verifier["probe_details"]["sequence_setval_probe"] == "denied_and_rollback_verified"


def test_role_probe_invalid_status_fails_closed() -> None:
    payload = _runtime_verifier_payload()
    payload["probe_details"]["schema_ddl_probe"] = "allowed"

    with pytest.raises(acceptance.AcceptanceFailure, match="invalid status"):
        acceptance._parse_runtime_verifier_output(json.dumps(payload), "")


def test_invalid_role_probe_runs_existing_cleanup_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def compose(*args, **kwargs):
        operation = list(args[3])
        calls.append(operation)
        if operation[-1:] == ["verify-runtime"]:
            return subprocess.CompletedProcess(args, 0, "malformed", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(acceptance, "_docker_compose_prefix", lambda *_: ["docker", "compose"])
    monkeypatch.setattr(acceptance, "_run_compose", compose)
    monkeypatch.setattr(acceptance, "_wait_for_health", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acceptance, "_mcp_initialize", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no app")))
    monkeypatch.setattr(acceptance, "_free_port", lambda: 18001)
    args = type("Args", (), {
        "image": "safe:tag",
        "compose_file": COMPOSE,
        "receipt": tmp_path / "receipt.json",
        "timeout": 1.0,
        "total_timeout": 2400.0,
        "finalization_reserve": 240.0,
        "compose_cleanup_reserve": 120.0,
        "publication_reserve": 60.0,
        "compose_cleanup_timeout": 90.0,
        "cleanup_kill_grace": 1.0,
        "docker_executable": None,
    })()

    receipt = acceptance.run_acceptance(args)

    assert receipt.status == "failed"
    assert receipt.verifier["status"] == "failed"
    assert receipt.verifier["stdout"] == "malformed"
    assert "runtime_db_role_isolation" not in receipt.checks
    assert calls[-1] == ["down", "--volumes", "--remove-orphans"]


def test_cleanup_uses_one_scoped_down_without_destructive_global_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def compose(*args, **kwargs):
        calls.append(list(args[3]))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(acceptance, "_run_compose", compose)
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: 100.0)
    receipt = _receipt(budget_ledger={"compose_cleanup_deadline": 200.0})
    acceptance._cleanup_resources(
        ["docker", "compose"],
        COMPOSE,
        receipt.project_name,
        {"WEFT_LOCAL_PORT": "18000"},
        20.0,
        None,
        [],
        receipt,
        cleanup_deadline=150.0,
        down_timeout=20.0,
    )
    assert calls == [["down", "--volumes", "--remove-orphans"]]
    assert receipt.cleanup_status == "succeeded"
    assert receipt.checks["unique_resource_cleanup"] == "passed"


def test_acceptance_source_contains_runtime_verification_and_both_isolation_walls() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    source = RUNNER.read_text(encoding="utf-8")
    assert '"verify-runtime"' in source
    assert '"prime_wrong_project"' in source
    assert '"recall_distinct_owner"' in source
    assert '"prime_distinct_owner"' in source
    assert '"unique_resource_cleanup"' in source
    # Keep this test source-level and deterministic; it must not accidentally
    # acquire Docker/network behavior through a fixture or subprocess.
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"check_call", "check_output"}
        for node in ast.walk(tree)
    )
