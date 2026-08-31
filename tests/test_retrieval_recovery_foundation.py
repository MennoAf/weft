"""Provider-free and focused contract tests for retrieval-recovery foundations."""

from __future__ import annotations

from pathlib import Path

import pytest

from weft.config import RetrievalConfig, load_config
from weft.db.migrations import MIGRATIONS
from weft.store import log_recall_query


def _clear_recovery_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WEFT_RETRIEVAL_RECOVERY_MODE",
        "WEFT_RETRIEVAL_RECOVERY_PLANNER_ENABLED",
        "WEFT_ENV",
        "WEFT_MIGRATION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_retrieval_recovery_defaults_are_off_and_provider_free() -> None:
    config = RetrievalConfig()

    assert config.recovery_mode == "off"
    assert config.recovery_planner_enabled is False


def test_retrieval_recovery_precedence_and_nested_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    toml_path = tmp_path / "config.toml"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("weft.config.CONFIG_PATH", toml_path)
    _clear_recovery_env(monkeypatch)

    toml_path.write_text(
        '[retrieval]\nrecovery_mode = "off"\ncontext_budget_tokens = 1111\n'
        'similarity_threshold = 0.11\n',
        encoding="utf-8",
    )
    _write_yaml(
        home / ".weft" / "config.yaml",
        "retrieval:\n  recovery_mode: deterministic\n  context_budget_tokens: 2222\n",
    )
    _write_yaml(
        project / ".weft" / "config.yaml",
        "retrieval:\n  recovery_planner_enabled: true\n  context_budget_tokens: 3333\n",
    )

    config = load_config(project)

    assert config.retrieval.recovery_mode == "deterministic"
    assert config.retrieval.recovery_planner_enabled is True
    assert config.retrieval.context_budget_tokens == 3333
    # Sparse YAML sections must retain lower-layer fields.
    assert config.retrieval.similarity_threshold == pytest.approx(0.11)

    monkeypatch.setenv("WEFT_RETRIEVAL_RECOVERY_MODE", "off")
    monkeypatch.setenv("WEFT_RETRIEVAL_RECOVERY_PLANNER_ENABLED", "0")
    config = load_config(project)
    assert config.retrieval.recovery_mode == "off"
    assert config.retrieval.recovery_planner_enabled is False
    assert config.retrieval.context_budget_tokens == 3333


def test_retrieval_recovery_env_mode_preserves_literal_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("weft.config.CONFIG_PATH", tmp_path / "missing.toml")
    monkeypatch.setenv("WEFT_RETRIEVAL_RECOVERY_MODE", "DETERMINISTIC")
    monkeypatch.delenv("WEFT_ENV", raising=False)
    monkeypatch.delenv("WEFT_MIGRATION_MODE", raising=False)

    with pytest.raises(ValueError):
        load_config()


@pytest.mark.asyncio
async def test_log_recall_query_returns_caller_supplied_id(pool) -> None:
    supplied_id = "rq-caller-supplied-a1"

    effective_id = await log_recall_query(
        pool,
        tool_name="recall",
        query_text="caller supplied query id",
        query_id=supplied_id,
    )

    assert effective_id == supplied_id
    row = await pool.fetchrow(
        "SELECT query_id FROM weft_recall_queries WHERE query_text = $1",
        "caller supplied query id",
    )
    assert row["query_id"] == supplied_id


def test_v73_token_rls_migration_is_registered_and_narrow() -> None:
    migrations = [migration for migration in MIGRATIONS if migration[0] == 73]
    assert len(migrations) == 1
    sql = migrations[0][2].lower()
    assert "weft_app" in sql
    assert "current_user" in sql
    assert "with check" in sql


def test_v72_recovery_attempts_is_registered_and_bounded() -> None:
    migrations = [migration for migration in MIGRATIONS if migration[0] == 72]
    assert len(migrations) == 1
    sql = migrations[0][2]

    assert "CREATE TABLE IF NOT EXISTS weft_recovery_attempts" in sql
    assert "user_id TEXT NOT NULL DEFAULT nullif(current_setting('app.user_id', true), '')" in sql
    assert "CHECK (jsonb_array_length(result_ids) <= 24)" in sql
    assert "CHECK (octet_length(coverage::text) <= 4096)" in sql
    assert "CHECK (provider_calls BETWEEN 0 AND 1)" in sql
    assert "idx_recovery_attempts_user_time" in sql
    assert "idx_recovery_attempts_parent_time" in sql
    for operation in ("select", "insert", "update", "delete"):
        assert f"weft_recovery_attempts_{operation}" in sql


@pytest.mark.asyncio
async def test_v72_migration_is_idempotent_and_has_exact_rls_schema(pool) -> None:
    sql = next(sql for version, _, sql in MIGRATIONS if version == 72)

    # The pool fixture already applied v72; executing it twice proves the
    # migration remains safe both after discovery and on an explicit rerun.
    await pool.execute(sql)
    await pool.execute(sql)

    column = await pool.fetchrow(
        """
        SELECT is_nullable, column_default, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'weft_recovery_attempts'
          AND column_name = 'user_id'
        """
    )
    assert column["is_nullable"] == "NO"
    assert "current_setting" in column["column_default"]
    assert "app.user_id" in column["column_default"]
    assert column["data_type"] == "text"

    policies = await pool.fetch(
        """
        SELECT policyname
        FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'weft_recovery_attempts'
        """
    )
    assert {row["policyname"] for row in policies} == {
        "weft_recovery_attempts_select",
        "weft_recovery_attempts_insert",
        "weft_recovery_attempts_update",
        "weft_recovery_attempts_delete",
    }
