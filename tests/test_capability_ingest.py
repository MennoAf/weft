"""Tests for Capability Registry ingestion and upsert helpers."""

from __future__ import annotations

import contextlib
import hashlib
import io
import sys
from pathlib import Path
import tempfile

import pytest

from capability_registry.config import RepoConfig, RegistryConfig
from capability_registry.ingest import (
    format_dry_run_report,
    run_ingestion,
    scan_and_classify,
    write_entry_to_weft,
)
from capability_registry.models import CapabilityEntry
from capability_registry.upsert import (
    check_stale_files,
    extract_file_hash_from_content,
    upsert_entry,
)


def test_dry_run_report_includes_entry_details() -> None:
    entries = [
        CapabilityEntry(
            repo_slug="muttr",
            file_path="crawl/esc.py",
            symbol_name="LazyEscalationPolicy",
            symbol_kind="class",
            file_hash="abc123",
            capability_slugs=["bot-block-hardening"],
        ),
        CapabilityEntry(
            repo_slug="muttr",
            file_path="utils/retry.py",
            symbol_name="retry_fetch",
            symbol_kind="function",
            file_hash="def456",
            capability_slugs=["retry-logic"],
        ),
    ]

    report = format_dry_run_report(entries)

    assert "crawl/esc.py" in report
    assert "utils/retry.py" in report
    assert "symbol:LazyEscalationPolicy" in report
    assert "capability:retry-logic" in report


def test_scan_and_classify_applies_manual_slugs_and_filters_noise(
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_root = Path(temp_dir)
        (repo_root / "manual.py").write_text(
            '"""Plain manual capability."""\nVALUE = 1\n',
            encoding="utf-8",
        )
        (repo_root / "auto.py").write_text(
            '"""Cache helper."""\ndef get_cached_value():\n    return None\n',
            encoding="utf-8",
        )
        (repo_root / "noise.py").write_text("VALUE = 1\n", encoding="utf-8")
        repo = RepoConfig(
            slug="example",
            root=str(repo_root),
            manual_capability_slugs={"manual.py": ["manual-capability"]},
        )

        entries = scan_and_classify(repo)

    module_by_file = {
        entry.file_path: entry for entry in entries if entry.symbol_kind == "module"
    }
    assert module_by_file["manual.py"].capability_slugs == ["manual-capability"]
    assert "caching" in module_by_file["auto.py"].capability_slugs
    assert "noise.py" not in module_by_file


def test_extract_file_hash_from_content() -> None:
    content = "CAPABILITY: crawler\nFILE_HASH: abc123def456\n"

    assert extract_file_hash_from_content(content) == "abc123def456"


def test_extract_file_hash_missing() -> None:
    assert extract_file_hash_from_content("CAPABILITY: crawler\n") is None


def test_check_stale_files_missing() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="/nonexistent/path/foo.py",
        file_hash="abc123",
    )

    result = check_stale_files([entry])

    assert result == [
        {
            "status": "missing",
            "repo_slug": "muttr",
            "file_path": "/nonexistent/path/foo.py",
        }
    ]


def test_check_stale_files_changed() -> None:
    with tempfile.NamedTemporaryFile() as temp_file:
        path = Path(temp_file.name)
        path.write_bytes(b"current")
        entry = CapabilityEntry(
            repo_slug="muttr",
            file_path=str(path),
            file_hash="wronghash",
        )

        result = check_stale_files([entry])

    assert len(result) == 1
    assert result[0]["status"] == "changed"
    assert result[0]["actual_hash"] == hashlib.sha256(b"current").hexdigest()


def test_check_stale_files_ok() -> None:
    with tempfile.NamedTemporaryFile() as temp_file:
        path = Path(temp_file.name)
        path.write_bytes(b"current")
        entry = CapabilityEntry(
            repo_slug="muttr",
            file_path=str(path),
            file_hash=hashlib.sha256(b"current").hexdigest(),
        )

        result = check_stale_files([entry])

    assert result == []


@pytest.mark.asyncio
async def test_write_entry_to_weft_calls_store_memory() -> None:
    calls: list[object] = []

    async def fake_store_memory(pool: object, create: object) -> object:
        calls.extend([pool, create])
        return type("MemoryStub", (), {"id": "weft-test"})()

    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/esc.py",
        file_hash="abc123",
        capability_slugs=["bot-block-hardening"],
    )

    memory_id = await write_entry_to_weft(
        entry,
        pool="pool",
        project_id="project",
        store_memory_fn=fake_store_memory,
    )

    assert memory_id == "weft-test"
    assert calls[0] == "pool"
    create = calls[1]
    assert create.content == entry.content
    assert create.topic == entry.topics
    assert create.project_id == "project"


@pytest.mark.asyncio
async def test_run_ingestion_dry_run_prints_without_writes(
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_root = Path(temp_dir)
        (repo_root / "cache.py").write_text(
            '"""Cache helper."""\ndef get_cached_value():\n    return None\n',
            encoding="utf-8",
        )
        config = RegistryConfig(
            repos=[RepoConfig(slug="example", root=str(repo_root))],
            dry_run=True,
            weft_project_id="weft",
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = await run_ingestion(config)
        output = stdout.getvalue()

    assert result["mode"] == "dry_run"
    assert result["entry_count"] >= 1
    assert "cache.py" in output


@pytest.mark.asyncio
async def test_run_ingestion_write_requires_pool() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_root = Path(temp_dir)
        (repo_root / "cache.py").write_text('"""Cache helper."""\n', encoding="utf-8")
        config = RegistryConfig(
            repos=[RepoConfig(slug="example", root=str(repo_root))],
            dry_run=False,
            weft_project_id="weft",
        )

        with pytest.raises(ValueError, match="pool required"):
            await run_ingestion(config)


@pytest.mark.asyncio
async def test_upsert_entry_dry_run_actions() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/esc.py",
        symbol_name="LazyEscalationPolicy",
        file_hash="abc123",
        capability_slugs=["bot-block-hardening"],
    )

    created = await upsert_entry(
        entry,
        pool="pool",
        project_id="project",
        dry_run=True,
        find_existing_memories_fn=lambda *_args, **_kwargs: [],
    )
    skipped = await upsert_entry(
        entry,
        pool="pool",
        project_id="project",
        dry_run=True,
        find_existing_memories_fn=lambda *_args, **_kwargs: [
            {"id": "old", "content": "FILE_HASH: abc123"}
        ],
    )
    updated = await upsert_entry(
        entry,
        pool="pool",
        project_id="project",
        dry_run=True,
        find_existing_memories_fn=lambda *_args, **_kwargs: [
            {"id": "old", "content": "FILE_HASH: def456"}
        ],
    )

    assert created["action"] == "create"
    assert skipped["action"] == "skip"
    assert updated["action"] == "update"
    assert updated["existing_ids"] == ["old"]
    assert updated["review_reason"] == "hash-changed"


@pytest.mark.asyncio
async def test_upsert_entry_update_writes_review_record() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/esc.py",
        symbol_name="LazyEscalationPolicy",
        file_hash="abc123",
        capability_slugs=["bot-block-hardening"],
    )
    writes: list[tuple[str, object]] = []

    async def fake_write_entry(*args: object, **kwargs: object) -> str:
        writes.append(("entry", args))
        return "weft-new"

    async def fake_write_review(*args: object, **kwargs: object) -> str:
        writes.append(("review", kwargs))
        return "weft-review"

    result = await upsert_entry(
        entry,
        pool="pool",
        project_id="project",
        dry_run=False,
        find_existing_memories_fn=lambda *_args, **_kwargs: [
            {"id": "weft-old", "content": "FILE_HASH: def456"}
        ],
        write_entry_to_weft_fn=fake_write_entry,
        write_review_record_fn=fake_write_review,
    )

    assert result == {
        "action": "updated",
        "new_id": "weft-new",
        "old_ids": ["weft-old"],
        "review_id": "weft-review",
    }
    assert [kind for kind, _payload in writes] == ["entry", "review"]
    review_kwargs = writes[1][1]
    assert review_kwargs["old_ids"] == ["weft-old"]
    assert review_kwargs["new_id"] == "weft-new"
    assert review_kwargs["reason"] == "hash-changed"


def _run_main() -> int:
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                result = test_fn()
                if hasattr(result, "__await__"):
                    import asyncio

                    asyncio.run(result)
            except Exception as exc:  # pragma: no cover - manual runner
                failures += 1
                print(f"FAIL: {name}: {exc}")
            else:
                print(f"PASS: {name}")
    return failures


if __name__ == "__main__":
    sys.exit(_run_main())
