"""Tests for TOML config file persistence and CLI config commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from weft.cli import cli
from weft.config import (
    WeftConfig,
    load_config,
    load_config_file,
    save_config_value,
)


@pytest.fixture
def toml_path(tmp_path: Path) -> Path:
    """Return a temporary path for a TOML config file."""
    return tmp_path / "config.toml"


def test_load_config_file_missing(toml_path: Path):
    """Loading a non-existent TOML file returns an empty dict."""
    result = load_config_file(toml_path)
    assert result == {}


def test_save_and_load_string_value(toml_path: Path):
    """Save a string value and read it back."""
    save_config_value("database.url", "postgresql://custom:5432/db", path=toml_path)

    data = load_config_file(toml_path)
    assert data["database"]["url"] == "postgresql://custom:5432/db"


def test_save_and_load_int_value(toml_path: Path):
    """Integer values are coerced and stored correctly."""
    save_config_value("database.pool_min_size", "5", path=toml_path)

    data = load_config_file(toml_path)
    assert data["database"]["pool_min_size"] == 5


def test_save_and_load_float_value(toml_path: Path):
    """Float values are coerced and stored correctly."""
    save_config_value("retrieval.similarity_threshold", "0.75", path=toml_path)

    data = load_config_file(toml_path)
    assert data["retrieval"]["similarity_threshold"] == pytest.approx(0.75)


def test_save_and_load_bool_value(toml_path: Path):
    """Boolean values are coerced and stored correctly."""
    save_config_value("decay.enabled", "false", path=toml_path)

    data = load_config_file(toml_path)
    assert data["decay"]["enabled"] is False


def test_save_top_level_key(toml_path: Path):
    """Top-level keys (not in a section) are stored correctly."""
    save_config_value("project_name", "my-project", path=toml_path)

    data = load_config_file(toml_path)
    assert data["project_name"] == "my-project"


def test_save_preserves_existing_keys(toml_path: Path):
    """Saving a new key does not overwrite existing keys."""
    save_config_value("database.url", "postgres://a", path=toml_path)
    save_config_value("database.pool_min_size", "4", path=toml_path)

    data = load_config_file(toml_path)
    assert data["database"]["url"] == "postgres://a"
    assert data["database"]["pool_min_size"] == 4


def test_save_overwrites_existing_key(toml_path: Path):
    """Saving the same key twice updates the value."""
    save_config_value("log_level", "DEBUG", path=toml_path)
    save_config_value("log_level", "WARNING", path=toml_path)

    data = load_config_file(toml_path)
    assert data["log_level"] == "WARNING"


def test_toml_file_content_is_valid(toml_path: Path):
    """Written TOML can be parsed back by tomllib."""
    import tomllib

    save_config_value("project_name", "test-proj", path=toml_path)
    save_config_value("database.url", "postgresql://x", path=toml_path)
    save_config_value("decay.enabled", "true", path=toml_path)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    assert data["project_name"] == "test-proj"
    assert data["database"]["url"] == "postgresql://x"
    assert data["decay"]["enabled"] is True


def test_load_config_with_toml(tmp_path: Path, monkeypatch):
    """load_config() integrates TOML file values."""
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    save_config_value("log_level", "DEBUG", path=toml_path)

    # Point CONFIG_PATH to our temp file
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)

    # Clear env vars that might interfere
    for var in ["WEFT_DATABASE_URL", "WEFT_REDIS_URL", "WEFT_EMBEDDING_PROVIDER",
                "WEFT_EMBEDDING_MODEL", "WEFT_LOG_LEVEL"]:
        monkeypatch.delenv(var, raising=False)

    config = load_config()
    assert config.log_level == "DEBUG"


def test_env_vars_override_toml(tmp_path: Path, monkeypatch):
    """Environment variables take precedence over TOML file."""
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    save_config_value("log_level", "DEBUG", path=toml_path)

    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_LOG_LEVEL", "ERROR")

    config = load_config()
    assert config.log_level == "ERROR"


def test_migration_mode_env_override(tmp_path: Path, monkeypatch):
    """Hosted runtimes can verify schema without receiving DDL ownership."""
    from weft.config import MigrationMode

    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_MIGRATION_MODE", "verify")

    assert load_config().migration_mode is MigrationMode.verify


def test_production_requires_explicit_migration_mode(tmp_path: Path, monkeypatch):
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_ENV", "production")
    monkeypatch.delenv("WEFT_MIGRATION_MODE", raising=False)

    with pytest.raises(ValueError, match="must be explicitly set in production"):
        load_config()


def test_production_accepts_explicit_verify_mode(tmp_path: Path, monkeypatch):
    from weft.config import MigrationMode

    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_ENV", "production")
    monkeypatch.setenv("WEFT_MIGRATION_MODE", "verify")

    assert load_config().migration_mode is MigrationMode.verify


def test_invalid_migration_mode_fails_config_load(tmp_path: Path, monkeypatch):
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_MIGRATION_MODE", "silently-ignore")

    with pytest.raises(ValueError):
        load_config()


def test_production_container_uses_frozen_virtualenv_directly():
    """Runtime startup must not reconcile packages inside Fly's health gate."""
    repo_root = Path(__file__).resolve().parents[1]
    dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    fly_config = (repo_root / "fly.toml").read_text(encoding="utf-8")

    assert 'CMD ["/app/.venv/bin/python", "-m", "weft.mcp"]' in dockerfile
    assert 'CMD ["uv", "run"' not in dockerfile
    assert 'WEFT_MIGRATION_MODE = "verify"' in fly_config
    assert 'grace_period = "60s"' in fly_config


def test_db_timeout_defaults_are_bounded():
    """Pool must ship with finite timeouts so a stalled query/drained pool
    fails fast instead of hanging (intermittent-hang regression guard)."""
    from weft.config import DatabaseConfig

    db = DatabaseConfig()
    assert db.command_timeout is not None and db.command_timeout > 0
    assert db.acquire_timeout is not None and db.acquire_timeout > 0
    # Pool must have headroom over a single prime's ~16-way section fan-out.
    assert db.pool_max_size >= 16


def test_db_pool_and_timeout_env_overrides(tmp_path: Path, monkeypatch):
    """Pool sizing and timeouts are tunable in prod without a redeploy."""
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_DB_POOL_MAX_SIZE", "25")
    monkeypatch.setenv("WEFT_DB_COMMAND_TIMEOUT", "12.5")
    monkeypatch.setenv("WEFT_DB_ACQUIRE_TIMEOUT", "7")

    config = load_config()
    assert config.database.pool_max_size == 25
    assert config.database.command_timeout == 12.5
    assert config.database.acquire_timeout == 7.0


# --- CLI tests ---


def test_config_show_runs(monkeypatch, tmp_path: Path):
    """'weft config show' runs without error."""
    toml_path = tmp_path / ".weft" / "config.toml"
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    # Also patch in cli module since it imports CONFIG_PATH at module level
    monkeypatch.setattr("weft.cli.CONFIG_PATH", toml_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "Weft Configuration" in result.output
    assert "database.url" in result.output


def test_config_set_and_show(monkeypatch, tmp_path: Path):
    """'weft config set' persists a value, visible in 'weft config show'."""
    toml_path = tmp_path / ".weft" / "config.toml"
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setattr("weft.cli.CONFIG_PATH", toml_path)

    runner = CliRunner()

    # Set a value
    result = runner.invoke(cli, ["config", "set", "log_level", "DEBUG"])
    assert result.exit_code == 0
    assert "Saved" in result.output

    # Verify file was created
    assert toml_path.exists()

    # Show should reflect the value
    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "DEBUG" in result.output


def test_config_set_invalid_key():
    """'weft config set' rejects invalid keys."""
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "set", "invalid.key.name", "value"])
    assert result.exit_code != 0
    assert "Unknown config key" in result.output


# --- Primer config ---


def test_primer_config_defaults():
    """PrimerConfig has sensible defaults (no sections disabled)."""
    from weft.config import PrimerConfig

    pc = PrimerConfig()
    assert pc.disabled_sections == []


def test_primer_config_in_weft_config():
    """WeftConfig includes a primer config section."""
    config = WeftConfig()
    assert hasattr(config, "primer")
    assert config.primer.disabled_sections == []


def test_primer_config_from_toml(tmp_path: Path, monkeypatch):
    """PrimerConfig.disabled_sections loads from TOML."""
    import tomllib

    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)
    toml_path.write_text('[primer]\ndisabled_sections = ["triggers", "cost"]\n')

    # Verify TOML is valid
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    assert data["primer"]["disabled_sections"] == ["triggers", "cost"]

    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    for var in ["WEFT_DATABASE_URL", "WEFT_REDIS_URL", "WEFT_LOG_LEVEL"]:
        monkeypatch.delenv(var, raising=False)

    config = load_config()
    assert config.primer.disabled_sections == ["triggers", "cost"]


# --- Daily Brief Config ---


def test_daily_brief_config_channel_type_default():
    """DailyBriefConfig.channel_type has a default value of 'slack'."""
    from weft.config import DailyBriefConfig

    dbc = DailyBriefConfig()
    assert dbc.channel_type == "slack"


def test_daily_brief_config_channel_type_from_env(monkeypatch):
    """DailyBriefConfig.channel_type loads from WEFT_DAILY_BRIEF_CHANNEL_TYPE env var."""
    toml_path = None
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path or Path.home() / ".weft" / "config.toml")
    monkeypatch.setenv("WEFT_DAILY_BRIEF_CHANNEL_TYPE", "discord")

    config = load_config()
    assert config.daily_brief.channel_type == "discord"


def test_daily_brief_config_channel_type_env_precedence(tmp_path: Path, monkeypatch):
    """Environment variable overrides default for channel_type."""
    toml_path = tmp_path / ".weft" / "config.toml"
    toml_path.parent.mkdir(parents=True)

    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    monkeypatch.setenv("WEFT_DAILY_BRIEF_CHANNEL_TYPE", "none")

    config = load_config()
    assert config.daily_brief.channel_type == "none"
