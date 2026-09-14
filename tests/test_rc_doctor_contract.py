"""Focused RC-FL-06 contract tests for the read-only Doctor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from weft.cli import cli
from weft.doctor import (
    CHECK_IDS,
    DoctorDependencies,
    DoctorReport,
    run_doctor,
)


@pytest.fixture
def healthy_deps():
    return DoctorDependencies(
        config_loader=lambda: {"valid": True, "sources": ["defaults", "environment"]},
        engine_probe=lambda: {"available": True, "backend": "docker"},
        database_probe=lambda config: {"reachable": True, "role": "weft_app"},
        redis_probe=lambda config: {"reachable": True},
        migration_probe=lambda config: {"consistent": True, "pending": 0},
        provider_probe=lambda config: {"configured": True, "credential_present": True},
        client_probe=lambda config: {"compatible": True, "transport": "stdio"},
        resource_probe=lambda: {"available": True, "package": "installed"},
        conflict_probe=lambda: {"conflict": False},
    )


def test_d001_to_d010_are_stable_and_json_schema_is_complete(healthy_deps):
    report = run_doctor(dependencies=healthy_deps)
    payload = report.to_dict()

    assert [check["id"] for check in payload["checks"]] == list(CHECK_IDS)
    assert set(payload) >= {"schema_version", "checks", "exit_code", "read_only"}
    assert payload["schema_version"] == "1"
    assert payload["exit_code"] == 0
    assert payload["read_only"] is True
    for check in payload["checks"]:
        assert set(check) >= {"id", "name", "status", "severity", "remedy", "details"}
        assert check["status"] in {"pass", "warn", "fail", "skip"}
        assert isinstance(check["details"], dict)


def test_json_output_is_deterministic_and_cli_supports_json(healthy_deps, monkeypatch):
    monkeypatch.setattr("weft.cli.load_doctor_dependencies", lambda: healthy_deps)
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(
        runner.invoke(cli, ["doctor", "--json"]).output
    )


def test_redaction_never_emits_secrets_urls_paths_or_provider_body():
    deps = DoctorDependencies(
        config_loader=lambda: {
            "valid": False,
            "error": "postgresql://user:super-secret@example.invalid:5432/db",
            "token": "REDACTED",
            "path": "/Users/private/person/.weft/config.toml",
        },
        engine_probe=lambda: {"available": False},
        database_probe=lambda config: {
            "reachable": False,
            "error": "redis://user:password@example.invalid:6379/0",
        },
        redis_probe=lambda config: {"reachable": False, "error": "provider response body"},
        migration_probe=lambda config: {"consistent": False},
        provider_probe=lambda config: {"configured": False, "error": "provider response body"},
        client_probe=lambda config: {"compatible": False},
        resource_probe=lambda: {"available": False, "path": "/private/source"},
        conflict_probe=lambda: {"conflict": True},
    )
    text = json.dumps(run_doctor(dependencies=deps).to_dict())
    for secret in (
        "super-secret",
        "REDACTED",
        "password@example",
        "/Users/private",
        "provider response body",
    ):
        assert secret not in text
    assert "redacted" in text


def test_unreachable_prerequisites_are_actionable_without_starting_or_mutating(monkeypatch):
    calls = []
    deps = DoctorDependencies(
        config_loader=lambda: {"valid": True},
        engine_probe=lambda: {"available": False},
        database_probe=lambda config: {"reachable": False, "error": "connection refused"},
        redis_probe=lambda config: {"reachable": False, "error": "connection refused"},
        migration_probe=lambda config: {"consistent": False, "pending": 2},
        provider_probe=lambda config: {"configured": False, "credential_present": False},
        client_probe=lambda config: {"compatible": False},
        resource_probe=lambda: {"available": False},
        conflict_probe=lambda: {"conflict": True},
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append((a, k)))
    before = {p: p.read_bytes() for p in (Path("weft/cli.py"),)}
    report = run_doctor(dependencies=deps)
    assert report.exit_code == 1
    assert not calls
    statuses = {check.id: check.status for check in report.checks}
    assert statuses["D001"] == "pass"
    assert statuses["D003"] == "warn"
    assert statuses["D004"] == "fail"
    assert statuses["D005"] == "warn"
    assert before[Path("weft/cli.py")] == Path("weft/cli.py").read_bytes()


def test_exit_semantics_for_warning_invalid_config_and_unexpected_error(healthy_deps):
    warning = DoctorDependencies(**{**healthy_deps.__dict__, "engine_probe": lambda: {"available": False}})
    assert run_doctor(dependencies=warning).exit_code == 0

    invalid = DoctorDependencies(**{**healthy_deps.__dict__, "config_loader": lambda: (_ for _ in ()).throw(ValueError("bad config"))})
    assert run_doctor(dependencies=invalid).exit_code == 2

    broken = DoctorDependencies(**{**healthy_deps.__dict__, "engine_probe": lambda: (_ for _ in ()).throw(RuntimeError("boom"))})
    assert run_doctor(dependencies=broken).exit_code == 3


def test_report_exit_code_is_derived_from_statuses():
    assert DoctorReport([]).exit_code == 0
