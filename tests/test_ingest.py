"""Tests for weft.ingest — codebase ingestion pipeline."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.ingest import (
    _MAX_ARCHITECTURE_EVIDENCE_CHARS,
    _MAX_FILE_CHARS,
    build_file_tree,
    discover_files,
    filter_files,
    generate_architecture_overview,
    run_ingest,
    summarize_file,
)
from weft.text_generation import GenerationResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@test.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@test.com",
}


def _make_mock_client(summary_text: str = "Test summary"):
    """Return a fully mocked AsyncAnthropic client."""
    client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=summary_text)]
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo with a few tracked files."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("\n".join(f"line {i}" for i in range(20)))
    (src / "__init__.py").write_text("")
    (tmp_path / "README.md").write_text("# Test")

    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
    )
    subprocess.run(
        ["git", "add", "."], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# TestDiscoverFiles
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    """Test discover_files: git ls-files wrapper."""

    async def test_returns_tracked_files(self, git_repo):
        files = discover_files(git_repo)
        assert isinstance(files, list)
        assert len(files) > 0
        assert "src/main.py" in files
        assert "src/__init__.py" in files
        assert "README.md" in files

    async def test_raises_for_non_git_directory(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not appear to be a git repository"):
            discover_files(tmp_path)


# ---------------------------------------------------------------------------
# TestFilterFiles
# ---------------------------------------------------------------------------


class TestFilterFiles:
    """Test filter_files: skip binary, config, lock, hidden, tiny files etc."""

    @pytest.mark.parametrize(
        "rel_path, should_keep",
        [
            ("src/__init__.py", False),
            ("migrations/001.py", False),
            ("uv.lock", False),
            (".hidden/file.py", False),
            ("icon.png", False),
            ("config.yaml", False),
            ("src/main.py", True),
        ],
        ids=[
            "init_filtered",
            "migration_filtered",
            "lockfile_filtered",
            "hidden_dir_filtered",
            "binary_ext_filtered",
            "config_ext_filtered",
            "source_kept",
        ],
    )
    async def test_filter_decision(self, tmp_path, rel_path, should_keep):
        """Parametrized: verify each file is filtered or kept as expected."""
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        # Write enough lines for source files that should be kept
        if should_keep:
            full.write_text("\n".join(f"line {i}" for i in range(20)))
        else:
            # Still create the file so filter_files can inspect it
            full.write_text("short\n")

        result = filter_files([rel_path], tmp_path)
        if should_keep:
            assert rel_path in result, f"{rel_path} should be kept"
        else:
            assert rel_path not in result, f"{rel_path} should be filtered"

    async def test_tiny_file_filtered(self, tmp_path):
        """Files with fewer than 10 lines are filtered out."""
        (tmp_path / "tiny.py").write_text("\n".join(f"line {i}" for i in range(5)))
        result = filter_files(["tiny.py"], tmp_path)
        assert "tiny.py" not in result

    async def test_file_at_threshold_filtered(self, tmp_path):
        """A file with exactly 9 lines is below the 10-line minimum."""
        (tmp_path / "nine.py").write_text("\n".join(f"line {i}" for i in range(9)))
        result = filter_files(["nine.py"], tmp_path)
        assert "nine.py" not in result

    async def test_file_at_threshold_kept(self, tmp_path):
        """A file with exactly 10 lines meets the minimum."""
        (tmp_path / "ten.py").write_text("\n".join(f"line {i}" for i in range(10)))
        result = filter_files(["ten.py"], tmp_path)
        assert "ten.py" in result


# ---------------------------------------------------------------------------
# TestBuildFileTree
# ---------------------------------------------------------------------------


class TestBuildFileTree:
    """Test build_file_tree: ASCII directory tree rendering."""

    async def test_simple_tree(self):
        files = ["src/main.py", "src/utils.py", "tests/test_main.py"]
        tree = build_file_tree(files)
        assert isinstance(tree, str)
        assert "src" in tree
        assert "main.py" in tree
        assert "tests" in tree
        assert "test_main.py" in tree

    async def test_output_contains_ascii_connectors(self):
        files = ["a.py", "b.py"]
        tree = build_file_tree(files)
        # Should use standard tree connectors
        assert "├──" in tree or "└──" in tree

    async def test_nested_tree(self):
        files = ["a/b/c/d.py"]
        tree = build_file_tree(files)
        assert "a" in tree
        assert "b" in tree
        assert "c" in tree
        assert "d.py" in tree

    async def test_empty_list(self):
        tree = build_file_tree([])
        assert tree == ""


# ---------------------------------------------------------------------------
# TestSummarizeFile
# ---------------------------------------------------------------------------


class TestSummarizeFile:
    """Test summarize_file: LLM call for per-file summary."""

    async def test_injected_provider_can_supply_summary(self):
        class FakeProvider:
            async def generate(self, request):
                assert request.model == "claude-haiku-4-5-20251001"
                return GenerationResponse(text="provider summary", model=request.model)

        result = await summarize_file(
            "src/auth.py", "\n".join(f"line {i}" for i in range(20)),
            generation_provider=FakeProvider(),
        )

        assert result == "provider summary"

    async def test_returns_summary_text(self):
        client = _make_mock_client("This module handles user authentication.")
        content = "\n".join(f"line {i}" for i in range(20))

        result = await summarize_file("src/auth.py", content, client)

        assert result == "This module handles user authentication."
        client.messages.create.assert_awaited_once()

    async def test_truncates_long_files(self):
        client = _make_mock_client("Summary of long file")
        # Create content with 600 lines (exceeds _MAX_LINES = 500)
        content = "\n".join(f"line {i}" for i in range(600))

        await summarize_file("src/big.py", content, client)

        # Inspect what was sent to the API
        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "truncated at 500 lines out of 600" in user_msg

    async def test_short_file_not_truncated(self):
        client = _make_mock_client("Short file summary")
        content = "\n".join(f"line {i}" for i in range(50))

        await summarize_file("src/small.py", content, client)

        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "truncated" not in user_msg

    async def test_long_lines_are_character_bounded(self):
        client = _make_mock_client("Summary")
        content = "\n".join(["x" * (_MAX_FILE_CHARS // 2)] * 20)

        await summarize_file("src/long-lines.py", content, client)

        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert f"truncated at {_MAX_FILE_CHARS} characters" in user_msg
        assert len(user_msg) < _MAX_FILE_CHARS + 500


# ---------------------------------------------------------------------------
# TestGenerateArchitectureOverview
# ---------------------------------------------------------------------------


class TestGenerateArchitectureOverview:
    """Test generate_architecture_overview: LLM call for project overview."""

    async def test_injected_provider_can_supply_overview(self):
        class FakeProvider:
            async def generate(self, request):
                assert request.model == "claude-haiku-4-5-20251001"
                return GenerationResponse(text="provider overview", model=request.model)

        result = await generate_architecture_overview(
            "└── app.py", {"app.py": "application"}, generation_provider=FakeProvider()
        )

        assert result == "provider overview"

    async def test_returns_overview_text(self):
        client = _make_mock_client("This project is an API server.")
        tree = "├── src\n│   └── main.py\n└── tests\n    └── test_main.py"
        summaries = {"src/main.py": "Entry point", "tests/test_main.py": "Tests"}

        result = await generate_architecture_overview(tree, summaries, client)

        assert result == "This project is an API server."
        client.messages.create.assert_awaited_once()

    async def test_includes_tree_and_summaries_in_prompt(self):
        client = _make_mock_client("overview")
        tree = "└── app.py"
        summaries = {"app.py": "Flask application"}

        await generate_architecture_overview(tree, summaries, client)

        call_args = client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "app.py" in user_msg
        assert "Flask application" in user_msg
        assert "File Tree" in user_msg

    async def test_empty_summaries(self):
        client = _make_mock_client("Architecture with no summaries")
        tree = "└── app.py"

        result = await generate_architecture_overview(tree, {}, client)
        assert result == "Architecture with no summaries"

    async def test_aggregate_evidence_is_bounded_and_reports_omissions(self):
        client = _make_mock_client("Bounded overview")
        summaries = {
            f"module_{i:03d}.py": "s" * 2_000
            for i in range(100)
        }

        await generate_architecture_overview("└── src", summaries, client)

        user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
        marker = "## File Summaries\n"
        evidence = user_msg.split(marker, 1)[1]
        assert len(evidence) <= _MAX_ARCHITECTURE_EVIDENCE_CHARS + 150
        assert "omitted" in evidence
        assert "summary truncated" in evidence


# ---------------------------------------------------------------------------
# TestRunIngest (integration — real DB, mocked LLM)
# ---------------------------------------------------------------------------


class TestRunIngest:
    """Integration tests for run_ingest with real Postgres, mocked Anthropic."""

    @pytest.fixture
    def ingest_repo(self, tmp_path):
        """Git repo with files that pass filtering (>= 10 lines of source)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("\n".join(f"# line {i}" for i in range(20)))
        (src / "utils.py").write_text("\n".join(f"# util {i}" for i in range(15)))
        # This file should be filtered (init)
        (src / "__init__.py").write_text("")
        # This file should be filtered (too short)
        (tmp_path / "README.md").write_text("# Readme")

        subprocess.run(
            ["git", "init"], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        return tmp_path

    async def test_full_ingest_returns_correct_counts(self, pool, ingest_repo):
        client = _make_mock_client("Mock file summary")
        result = await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client,
        )

        assert result["files_discovered"] > 0
        assert result["files_summarized"] == 2  # main.py + utils.py
        assert result["architecture_stored"] is True

    async def test_architecture_memory_stored(self, pool, ingest_repo):
        client = _make_mock_client("Architecture overview text")
        await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client,
        )

        rows = await pool.fetch(
            "SELECT * FROM memories WHERE topic @> $1", ["codebase-map"],
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "Architecture overview text"
        assert rows[0]["project_id"] == "test-project"
        assert rows[0]["type"] == "architecture"
        assert rows[0]["source"] == "ingest"

    async def test_file_summaries_stored(self, pool, ingest_repo):
        client = _make_mock_client("File summary content")
        await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client,
        )

        rows = await pool.fetch(
            "SELECT * FROM memories WHERE topic @> $1", ["test-project"],
        )
        # Architecture + 2 file summaries = 3 total for this project
        assert len(rows) == 3

        file_rows = [r for r in rows if r["type"] == "fact"]
        assert len(file_rows) == 2
        # Each file summary should have a file: topic
        for row in file_rows:
            topics = list(row["topic"])
            file_topics = [t for t in topics if t.startswith("file:")]
            assert len(file_topics) == 1

    async def test_file_summaries_have_review_after(self, pool, ingest_repo):
        client = _make_mock_client("Summary with review date")
        await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client,
        )

        file_rows = await pool.fetch(
            "SELECT * FROM memories WHERE type = 'fact' AND topic @> $1",
            ["test-project"],
        )
        assert len(file_rows) > 0

        now = datetime.now(timezone.utc)
        for row in file_rows:
            assert row["review_after"] is not None
            delta = row["review_after"] - now
            # review_after should be approximately 30 days from now
            assert 29 <= delta.days <= 31

    async def test_idempotent_no_duplicates(self, pool, ingest_repo):
        """Calling run_ingest twice should upsert, not duplicate memories."""
        client = _make_mock_client("First pass summary")
        await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client,
        )

        first_count = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE topic @> $1", ["test-project"],
        )

        # Second ingest with different summary text (simulates updated content)
        client2 = _make_mock_client("Second pass summary")
        await run_ingest(
            ingest_repo, "test-project", depth="full", pool=pool, client=client2,
        )

        second_count = await pool.fetchval(
            "SELECT COUNT(*) FROM memories WHERE topic @> $1", ["test-project"],
        )
        assert second_count == first_count, "Second ingest should not create duplicates"

        # Verify content was updated to the new value
        arch_rows = await pool.fetch(
            "SELECT * FROM memories WHERE topic @> $1 AND type = 'architecture'",
            ["codebase-map"],
        )
        assert len(arch_rows) == 1
        assert arch_rows[0]["content"] == "Second pass summary"


# ---------------------------------------------------------------------------
# TestRunIngestArchitectureOnly
# ---------------------------------------------------------------------------


class TestRunIngestArchitectureOnly:
    """Integration tests for run_ingest with depth != 'full' (architecture only)."""

    @pytest.fixture
    def arch_repo(self, tmp_path):
        """Git repo for architecture-only ingestion."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("\n".join(f"# line {i}" for i in range(20)))

        subprocess.run(
            ["git", "init"], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, capture_output=True, env=_GIT_ENV, check=True,
        )
        return tmp_path

    async def test_no_file_summaries_created(self, pool, arch_repo):
        client = _make_mock_client("Architecture only overview")
        result = await run_ingest(
            arch_repo, "arch-project", depth="architecture", pool=pool, client=client,
        )

        assert result["files_summarized"] == 0
        assert result["architecture_stored"] is True

        # No file summary memories should exist
        file_rows = await pool.fetch(
            "SELECT * FROM memories WHERE type = 'fact' AND topic @> $1",
            ["arch-project"],
        )
        assert len(file_rows) == 0

    async def test_architecture_overview_stored(self, pool, arch_repo):
        client = _make_mock_client("High-level architecture text")
        await run_ingest(
            arch_repo, "arch-project", depth="architecture", pool=pool, client=client,
        )

        rows = await pool.fetch(
            "SELECT * FROM memories WHERE topic @> $1 AND type = 'architecture'",
            ["codebase-map"],
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "High-level architecture text"

    async def test_files_still_discovered(self, pool, arch_repo):
        """Even in architecture mode, files_discovered reflects git ls-files."""
        client = _make_mock_client("overview")
        result = await run_ingest(
            arch_repo, "arch-project", depth="architecture", pool=pool, client=client,
        )
        assert result["files_discovered"] > 0
