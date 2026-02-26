"""Tests for the import CLI command and import_memories function."""

from __future__ import annotations

import pytest

from weft.importer import ImportReport, import_memories, parse_memory_md_text
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import list_memories, store_memory


@pytest.fixture
def provider():
    from weft.embeddings import get_provider

    return get_provider("fastembed")


# --- ImportReport dataclass ---


def test_import_report_dataclass():
    """ImportReport has correct default values."""
    report = ImportReport()
    assert report.stored == 0
    assert report.skipped_duplicate == 0
    assert report.skipped_empty == 0
    assert report.errors == []


# --- import_memories function ---


async def test_import_memories_stores_new(pool, provider):
    """import_memories stores new memories with embeddings."""
    creates = [
        MemoryCreate(
            type=MemoryType.fact,
            content="Weft uses pgvector for semantic search",
            topic=["weft"],
        ),
        MemoryCreate(
            type=MemoryType.pattern,
            content="Always run migrations before running tests",
            topic=["testing"],
        ),
    ]

    report = await import_memories(pool, provider, creates)
    assert report.stored == 2
    assert report.skipped_duplicate == 0
    assert report.skipped_empty == 0
    assert report.errors == []

    # Verify memories are in the database
    mems = await list_memories(pool)
    assert len(mems) == 2


async def test_import_memories_dedup_skips_duplicate(pool, provider):
    """Importing the same content twice skips duplicates on second import."""
    creates = [
        MemoryCreate(
            type=MemoryType.fact,
            content="Redis is used as a caching layer in Weft",
            topic=["architecture"],
        ),
    ]

    # First import — should store
    report1 = await import_memories(pool, provider, creates)
    assert report1.stored == 1
    assert report1.skipped_duplicate == 0

    # Second import of same content — should skip as duplicate
    report2 = await import_memories(pool, provider, creates)
    assert report2.stored == 0
    assert report2.skipped_duplicate == 1

    # Only one memory in the database
    mems = await list_memories(pool)
    assert len(mems) == 1


async def test_import_memories_dry_run(pool, provider):
    """dry_run=True reports counts but does not store memories."""
    creates = [
        MemoryCreate(
            type=MemoryType.fact,
            content="Dry run should not persist anything to the database",
            topic=["testing"],
        ),
    ]

    report = await import_memories(pool, provider, creates, dry_run=True)
    assert report.stored == 1  # Counted as "would store"
    assert report.skipped_duplicate == 0
    assert report.errors == []

    # Nothing actually stored
    mems = await list_memories(pool)
    assert len(mems) == 0


async def test_import_memories_with_project_id(pool, provider):
    """project_id overrides the create's project_id."""
    creates = [
        MemoryCreate(
            type=MemoryType.fact,
            content="This memory should belong to project-override",
            topic=["testing"],
            project_id="original-project",
        ),
    ]

    report = await import_memories(
        pool, provider, creates, project_id="override-project"
    )
    assert report.stored == 1

    mems = await list_memories(pool, project_id="override-project")
    assert len(mems) == 1
    assert mems[0].project_id == "override-project"

    # Original project_id should have nothing
    mems_orig = await list_memories(pool, project_id="original-project")
    assert len(mems_orig) == 0


async def test_import_memories_empty_content_skipped(pool, provider):
    """Empty content creates are skipped."""
    creates = [
        MemoryCreate(
            type=MemoryType.fact,
            content="",
            topic=["testing"],
        ),
        MemoryCreate(
            type=MemoryType.fact,
            content="   ",
            topic=["testing"],
        ),
        MemoryCreate(
            type=MemoryType.fact,
            content="This one has real content",
            topic=["testing"],
        ),
    ]

    report = await import_memories(pool, provider, creates)
    assert report.skipped_empty == 2
    assert report.stored == 1
    assert report.errors == []


# --- CLI command ---


def test_cli_import_command_exists():
    """The import CLI command can be invoked via click runner."""
    from click.testing import CliRunner

    from weft.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["import", "--help"])
    assert result.exit_code == 0
    assert "Import memories from a MEMORY.md file" in result.output
    assert "--dry-run" in result.output
    assert "--project-id" in result.output
