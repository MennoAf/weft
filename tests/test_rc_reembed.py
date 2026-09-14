"""RC-FL-09: embedding profile and resumable re-embed contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from weft.cli import cli

from weft.db.migrations import MIGRATIONS, verify_migration_ledger
from weft.db.reembed import ReembedProfile, run_reembed, resume_reembed


class LocalProvider:
    provider_name = "local-fake"
    model_name = "rc-fake-v1"
    dimensions = 768

    def __init__(self, *, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    async def embed(self, text: str) -> list[float]:
        return [float(len(text))] + [1.0] * 767

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("local provider interruption")
        return [[float(len(text))] + [1.0] * 767 for text in texts]


@pytest.fixture(autouse=True)
async def clean_reembed_state(pool):
    await pool.execute(
        "TRUNCATE embedding_profile_vectors, embedding_reembed_runs, "
        "embedding_profile_state, embedding_profiles CASCADE"
    )


async def _seed(pool, count: int = 3):
    for i in range(count):
        await pool.execute(
            "INSERT INTO memories (id, type, content, embedding) "
            "VALUES ($1, 'fact', $2, $3::vector)",
            f"rc-reembed-{i}", f"memory {i}", [9.0] * 768,
        )


@pytest.mark.asyncio
async def test_v75_is_discovered_and_ledger_matches(pool):
    assert max(version for version, _, _ in MIGRATIONS) == 75
    assert sum(version == 75 for version, _, _ in MIGRATIONS) == 1
    await verify_migration_ledger(pool)
    columns = await pool.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'embedding_profiles'"
    )
    assert {row["column_name"] for row in columns} >= {
        "profile_id", "provider", "model", "dimensions", "composition", "state"
    }


@pytest.mark.asyncio
async def test_profile_identity_state_and_atomic_promotion(pool):
    await _seed(pool)
    provider = LocalProvider()
    report = await run_reembed(
        pool,
        provider,
        tables=["memories"],
        batch_size=2,
        composition={"version": 4, "text": "content"},
    )
    assert report["status"] == "promoted"
    assert report["completed"] is True
    profile = await pool.fetchrow(
        "SELECT provider, model, dimensions, composition, state "
        "FROM embedding_profiles WHERE profile_id = $1", report["target_profile_id"]
    )
    assert profile["provider"] == "local-fake"
    assert profile["model"] == "rc-fake-v1"
    assert profile["dimensions"] == 768
    composition = profile["composition"]
    if isinstance(composition, str):
        import json
        composition = json.loads(composition)
    assert composition == {"version": 4, "text": "content"}
    assert profile["state"] == "active"
    state = await pool.fetchrow(
        "SELECT active_profile_id, target_profile_id FROM embedding_profile_state"
    )
    assert state["active_profile_id"] == report["target_profile_id"]
    assert state["target_profile_id"] is None
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE embedding_profile_id = $1",
        report["target_profile_id"],
    ) == 3


@pytest.mark.asyncio
async def test_interruption_restart_preserves_old_reads_until_promotion(pool):
    await _seed(pool, 4)
    old = await pool.fetchval("SELECT embedding::text FROM memories WHERE id = 'rc-reembed-0'")
    provider = LocalProvider()
    partial = await run_reembed(
        pool, provider, tables=["memories"], batch_size=2, max_batches=1,
    )
    assert partial["status"] == "interrupted"
    assert partial["completed"] is False
    assert partial["cursor"] == {"memories": 2}
    assert await pool.fetchval(
        "SELECT embedding::text FROM memories WHERE id = 'rc-reembed-0'"
    ) == old
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE embedding_target IS NOT NULL"
    ) == 2

    resumed = await resume_reembed(pool, provider, partial["run_id"])
    assert resumed["status"] == "promoted"
    assert resumed["completed"] is True
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE embedding_profile_id = $1",
        resumed["target_profile_id"],
    ) == 4
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE embedding_target IS NOT NULL"
    ) == 0


@pytest.mark.asyncio
async def test_incomplete_provider_run_is_not_success(pool):
    await _seed(pool, 4)
    report = await run_reembed(
        pool, LocalProvider(fail_after=1), tables=["memories"], batch_size=2,
    )
    assert report["status"] == "failed"
    assert report["completed"] is False
    assert report["run_id"]
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM memories WHERE embedding_profile_id IS NOT NULL"
    ) == 0
    with pytest.raises(ValueError, match="not complete"):
        await resume_reembed(pool, LocalProvider(), report["run_id"], require_complete=True)


def test_profile_identity_is_explicit():
    profile = ReembedProfile.from_provider(
        SimpleNamespace(provider_name="p", model_name="m", dimensions=8),
        composition={"version": 1},
    )
    assert profile.provider == "p"
    assert profile.model == "m"
    assert profile.dimensions == 8
    assert profile.composition == {"version": 1}


@pytest.mark.parametrize(
    ("status", "completed", "exit_code", "expected_text"),
    [
        ("interrupted", False, 1, "Re-embed incomplete"),
        ("promoted", True, 0, "Re-embedded 3 rows"),
    ],
)
def test_reembed_cli_receipt_exit_reflects_promotion_state(
    monkeypatch, status, completed, exit_code, expected_text
):
    """CLI success is reserved for an atomically promoted run."""
    provider = SimpleNamespace(provider_name="local-fake", dimensions=768)
    pool = SimpleNamespace(close=AsyncMock())
    config = SimpleNamespace(
        embedding=SimpleNamespace(provider="local-fake", model="rc-fake", dimensions=768)
    )
    monkeypatch.setattr("weft.cli.load_config", Mock(return_value=config))
    monkeypatch.setattr("weft.db.connection.create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr("weft.embeddings.get_provider", Mock(return_value=provider))
    monkeypatch.setattr("weft.db.migrations.run_migrations", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "weft.db.reembed.run_reembed",
        AsyncMock(
            return_value={
                "run_id": "reembed-test",
                "target_profile_id": "emb-test",
                "status": status,
                "completed": completed,
                "cursor": {"memories": 3},
                "embedded_rows": 3,
                "total_rows": 3,
            }
        ),
    )

    result = CliRunner().invoke(cli, ["re-embed", "--table", "memories"])

    assert result.exit_code == exit_code, result.output
    assert expected_text in result.output
    pool.close.assert_awaited_once()
