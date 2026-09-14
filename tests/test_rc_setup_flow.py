"""RC-FL-07 lifecycle safety contracts."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from weft.cli import cli


def _use_config(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / ".weft" / "config.toml"
    monkeypatch.setattr("weft.config.CONFIG_PATH", path)
    monkeypatch.setattr("weft.cli.CONFIG_PATH", path)
    return path


def test_init_is_guided_rerunnable_and_does_not_rotate_or_start(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    calls: list[object] = []
    monkeypatch.setattr("weft.cli.subprocess.run", lambda *args, **kwargs: calls.append(args))

    runner = CliRunner()
    first = runner.invoke(cli, ["init"])
    assert first.exit_code == 0, first.output
    assert path.exists()
    original = path.read_bytes()
    assert "api_key" not in path.read_text()
    assert not calls

    second = runner.invoke(cli, ["init"])
    assert second.exit_code == 0, second.output
    assert "already configured" in second.output.lower()
    assert path.read_bytes() == original
    assert not calls


def test_init_never_overwrites_existing_configuration(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('project_name = "preserve-me"\napi_key = "credential-never-rotate"\n')

    result = CliRunner().invoke(cli, ["init"])

    assert result.exit_code == 0, result.output
    assert "already configured" in result.output.lower()
    assert path.read_text() == 'project_name = "preserve-me"\napi_key = "credential-never-rotate"\n'
    assert "credential-never-rotate" not in result.output


def test_repair_is_strict_read_only_diagnostic(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('project_name = "before"\n')
    before = path.read_bytes()
    observed = []

    class Report:
        exit_code = 1

        def to_dict(self):
            return {"schema_version": "1", "checks": [], "exit_code": 1, "read_only": True}

    monkeypatch.setattr("weft.cli.load_doctor_dependencies", lambda: observed.append("deps") or object())
    monkeypatch.setattr("weft.cli.run_doctor", lambda dependencies: observed.append(dependencies) or Report())

    result = CliRunner().invoke(cli, ["repair", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.output)["read_only"] is True
    assert observed[0] == "deps"
    assert path.read_bytes() == before


def test_reconfigure_defaults_to_plan_without_mutating(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('project_name = "before"\n')
    before = path.read_bytes()

    result = CliRunner().invoke(cli, ["reconfigure", "--set", "project_name=after"])

    assert result.exit_code == 0, result.output
    assert "plan" in result.output.lower()
    assert "project_name" in result.output
    assert path.read_bytes() == before
    assert not path.with_suffix(".toml.bak").exists()


def test_reconfigure_requires_confirm_backup_and_stage_then_changes_atomically(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('project_name = "before"\n')

    runner = CliRunner()
    for flags in (["--confirm"], ["--confirm", "--backup"], ["--confirm", "--stage"]):
        result = runner.invoke(cli, ["reconfigure", "--set", "project_name=after", *flags])
        assert result.exit_code != 0
        assert path.read_text() == 'project_name = "before"\n'

    result = runner.invoke(
        cli,
        ["reconfigure", "--set", "project_name=after", "--confirm", "--backup", "--stage"],
    )
    assert result.exit_code == 0, result.output
    assert 'project_name = "after"' in path.read_text()
    assert path.with_suffix(".toml.bak").read_text() == 'project_name = "before"\n'


def test_reconfigure_redacts_explicit_credential_values(monkeypatch, tmp_path):
    path = _use_config(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli, ["reconfigure", "--set", "api_key=secret-value"])
    assert result.exit_code == 0, result.output
    assert "secret-value" not in result.output
    assert "redacted" in result.output.lower()
    assert not path.exists()


def test_upgrade_requires_explicit_confirmation_and_separates_actions(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr("weft.cli.run_package_update", lambda: calls.append("package") or {"status": "updated"})
    monkeypatch.setattr("weft.cli.run_schema_upgrade", lambda: calls.append("schema") or {"status": "upgraded"})
    runner = CliRunner()

    implicit = runner.invoke(cli, ["upgrade"])
    assert implicit.exit_code != 0
    assert calls == []

    unconfirmed = runner.invoke(cli, ["upgrade", "--package-update"])
    assert unconfirmed.exit_code != 0
    assert calls == []

    both = runner.invoke(cli, ["upgrade", "--confirm", "--package-update", "--schema-upgrade"])
    assert both.exit_code != 0
    assert calls == []

    package = runner.invoke(cli, ["upgrade", "--confirm", "--package-update"])
    assert package.exit_code == 0, package.output
    assert calls == ["package"]
    assert "schema" not in package.output.lower()

    schema = runner.invoke(cli, ["upgrade", "--confirm", "--schema-upgrade"])
    assert schema.exit_code == 0, schema.output
    assert calls == ["package", "schema"]
    assert "package" not in schema.output.lower()
