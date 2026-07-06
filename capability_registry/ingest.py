"""Dry-run and write ingestion pipeline for the Capability Registry."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fnmatch
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from capability_registry.config import RegistryConfig, RepoConfig, load_config
from capability_registry.models import CapabilityEntry
from capability_registry.scanner import scan_repo
from capability_registry.slug_classifier import classify_entries
from capability_registry.upsert import upsert_entry


StoreMemoryFn = Callable[..., Any | Awaitable[Any]]


def format_dry_run_report(entries: list[CapabilityEntry]) -> str:
    """Format proposed capability entries for human review."""
    if not entries:
        return "Capability Registry dry run: 0 entries."

    blocks = [f"Capability Registry dry run: {len(entries)} entries."]
    for entry in entries:
        symbol = entry.symbol_name or "<module>"
        content_preview = entry.content[:200]
        blocks.append(
            "\n".join(
                [
                    "",
                    f"FILE: {entry.file_path}",
                    f"SYMBOL: {symbol}",
                    f"KIND: {entry.symbol_kind or 'unknown'}",
                    f"TOPICS: {', '.join(entry.topics)}",
                    f"CONTENT_PREVIEW: {content_preview}",
                ]
            )
        )
    return "\n".join(blocks)


def scan_and_classify(repo_cfg: RepoConfig) -> list[CapabilityEntry]:
    """Scan one configured repo and return classified, useful entries."""
    entries = scan_repo(
        repo_cfg.slug,
        Path(repo_cfg.root),
        repo_cfg.include_globs,
        repo_cfg.exclude_dirs,
    )
    for entry in entries:
        manual_slugs = _manual_slugs_for_entry(repo_cfg, entry)
        if manual_slugs:
            entry.capability_slugs = manual_slugs

    classify_entries(entries)
    return [
        entry
        for entry in entries
        if entry.capability_slugs or entry.docstring
    ]


async def write_entry_to_weft(
    entry: CapabilityEntry,
    pool: Any,
    project_id: str | None,
    *,
    store_memory_fn: StoreMemoryFn | None = None,
) -> str:
    """Write one capability entry to Weft and return the memory id."""
    from weft.models import MemoryCreate, MemorySource, MemoryType
    from weft.store import store_memory

    create = MemoryCreate(
        type=MemoryType.solution,
        content=entry.content,
        topic=entry.topics,
        source=MemorySource.ingest,
        confidence=0.75,
        project_id=project_id or None,
        pinned=False,
    )
    if store_memory_fn is None:
        from weft.db.connection import acquire

        async with acquire(pool):
            memory = await store_memory(pool, create)
        return str(memory.id)

    store_fn = store_memory_fn
    memory = store_fn(pool, create)
    if hasattr(memory, "__await__"):
        memory = await memory
    return str(memory.id)


async def run_ingestion(
    config: RegistryConfig,
    pool: Any = None,
) -> dict[str, Any]:
    """Run dry-run or write ingestion for every configured repository."""
    entries: list[CapabilityEntry] = []
    for repo in config.repos:
        entries.extend(scan_and_classify(repo))

    if config.dry_run:
        print(format_dry_run_report(entries))
        return {
            "mode": "dry_run",
            "entry_count": len(entries),
            "entries": [dataclasses.asdict(entry) for entry in entries],
        }

    if pool is None:
        raise ValueError("pool required for write mode")

    results = [
        await upsert_entry(
            entry,
            pool,
            config.weft_project_id or None,
            dry_run=False,
        )
        for entry in entries
    ]
    return {
        "mode": "write",
        "entry_count": len(entries),
        "results": results,
    }


def _manual_slugs_for_entry(
    repo_cfg: RepoConfig,
    entry: CapabilityEntry,
) -> list[str] | None:
    for pattern, slugs in repo_cfg.manual_capability_slugs.items():
        if entry.file_path == pattern or fnmatch.fnmatch(entry.file_path, pattern):
            return list(slugs)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest capability entries into Weft.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Enable write mode. Dry-run is the default.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.write:
        config.dry_run = False
    if config.dry_run:
        result = await run_ingestion(config)
    else:
        from weft.config import load_config as load_weft_config
        from weft.db.connection import create_pool

        pool = await create_pool(load_weft_config())
        try:
            result = await run_ingestion(config, pool)
        finally:
            await pool.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
