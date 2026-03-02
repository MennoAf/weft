"""Tests for seed memory bootstrapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from weft.models import MemoryType
from weft.seed import load_seeds, seed_memories
from weft.store import list_memories


@pytest.fixture
def provider():
    from weft.embeddings import get_provider

    return get_provider("fastembed")


# --- YAML loading ---


def test_load_seeds_returns_list():
    """load_seeds returns a non-empty list from the bundled YAML."""
    seeds = load_seeds()
    assert isinstance(seeds, list)
    assert len(seeds) > 0


def test_seed_entries_have_required_fields():
    """Every seed entry has the required fields."""
    seeds = load_seeds()
    for i, entry in enumerate(seeds):
        assert "content" in entry, f"seed {i} missing content"
        assert "type" in entry, f"seed {i} missing type"
        assert "topic" in entry, f"seed {i} missing topic"
        assert "confidence" in entry, f"seed {i} missing confidence"


def test_seed_types_are_valid():
    """Every seed type is a valid MemoryType."""
    seeds = load_seeds()
    valid_types = {t.value for t in MemoryType}
    for i, entry in enumerate(seeds):
        assert entry["type"] in valid_types, f"seed {i} has invalid type: {entry['type']}"


def test_seed_confidence_values_valid():
    """All confidence values are between 0 and 1."""
    seeds = load_seeds()
    for i, entry in enumerate(seeds):
        c = entry["confidence"]
        assert 0.0 <= c <= 1.0, f"seed {i} has out-of-range confidence: {c}"


def test_seed_content_not_empty():
    """No seed has empty or whitespace-only content."""
    seeds = load_seeds()
    for i, entry in enumerate(seeds):
        assert entry["content"].strip(), f"seed {i} has empty content"


def test_load_seeds_missing_file():
    """Returns empty list when seeds.yaml is not found."""
    with patch("weft.seed.importlib.resources.files", side_effect=FileNotFoundError):
        result = load_seeds()
    assert result == []


def test_load_seeds_invalid_yaml():
    """Returns empty list when YAML is not a list."""
    mock_ref = type("Ref", (), {"read_text": lambda self, **kw: "not_a_list: true"})()
    mock_files = lambda _pkg: type("Files", (), {"joinpath": lambda self, _name: mock_ref})()
    with patch("weft.seed.importlib.resources.files", mock_files):
        result = load_seeds()
    assert result == []


# --- seed_memories integration ---


async def test_seed_memories_inserts_when_empty(pool, provider):
    """Seeds are inserted when the memory store is empty."""
    count = await seed_memories(pool, provider)
    seeds = load_seeds()
    assert count == len(seeds)

    mems = await list_memories(pool)
    assert len(mems) == len(seeds)


async def test_seed_memories_skips_when_not_empty(pool, provider):
    """Seeds are skipped when the store already has memories."""
    # Seed once
    first = await seed_memories(pool, provider)
    assert first > 0

    # Second call should skip
    second = await seed_memories(pool, provider)
    assert second == 0

    # Count unchanged
    mems = await list_memories(pool)
    assert len(mems) == first


async def test_seed_memories_force_bypasses_check(pool, provider):
    """force=True seeds even when memories already exist."""
    first = await seed_memories(pool, provider)
    assert first > 0

    forced = await seed_memories(pool, provider, force=True)
    seeds = load_seeds()
    assert forced == len(seeds)

    # Both batches stored
    mems = await list_memories(pool)
    assert len(mems) == first + forced


async def test_seed_memories_returns_zero_when_no_seeds(pool, provider):
    """Returns 0 when seed file yields no entries."""
    with patch("weft.seed.load_seeds", return_value=[]):
        count = await seed_memories(pool, provider)
    assert count == 0


async def test_seed_continues_on_single_failure(pool, provider):
    """A failing entry doesn't stop the rest from being seeded."""
    original_store = __import__("weft.store", fromlist=["store_memory"]).store_memory

    call_count = 0

    async def fail_second(p, create, embedding=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated failure")
        return await original_store(p, create, embedding=embedding)

    with patch("weft.seed.store_memory", side_effect=fail_second):
        count = await seed_memories(pool, provider)

    seeds = load_seeds()
    assert count == len(seeds) - 1  # one failed


async def test_seeded_memories_have_correct_source(pool, provider):
    """Seeded memories have source='seed'."""
    await seed_memories(pool, provider)
    mems = await list_memories(pool)
    for m in mems:
        assert m.source.value == "seed"


async def test_seeded_memories_have_embeddings(pool, provider):
    """Seeded memories have non-null embeddings (searchable)."""
    await seed_memories(pool, provider)

    # Verify by doing a vector search
    from weft.store import search_by_vector

    vec = await provider.embed("how to use weft")
    results = await search_by_vector(pool, vec, limit=3)
    assert len(results) > 0


async def test_seeded_pinned_memories(pool, provider):
    """Seeds with pinned=true are stored as pinned."""
    await seed_memories(pool, provider)
    mems = await list_memories(pool, pinned=True)

    seeds = load_seeds()
    expected_pinned = sum(1 for s in seeds if s.get("pinned", False))
    assert len(mems) == expected_pinned
    assert expected_pinned > 0  # sanity: we have some pinned seeds


# --- CLI command ---


def test_cli_seed_command_exists():
    """The seed CLI command is registered."""
    from click.testing import CliRunner

    from weft.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["seed", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output


def test_cli_seed_invokes_seed_memories():
    """CLI seed command calls seed_memories and reports the count."""
    from click.testing import CliRunner

    from weft.cli import cli

    runner = CliRunner()

    with patch("weft.cli.asyncio.run", return_value=5) as mock_run:
        result = runner.invoke(cli, ["seed"])
    assert result.exit_code == 0
    assert "5" in result.output
    mock_run.assert_called_once()


def test_cli_seed_reports_skip():
    """CLI seed reports when no memories were seeded."""
    from click.testing import CliRunner

    from weft.cli import cli

    runner = CliRunner()

    with patch("weft.cli.asyncio.run", return_value=0):
        result = runner.invoke(cli, ["seed"])
    assert result.exit_code == 0
    assert "No memories seeded" in result.output
