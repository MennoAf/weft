"""Idempotent upsert and stale-file helpers for capability entries."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

from capability_registry.models import CapabilityEntry


MemoryRecord = dict[str, Any]
FindExistingFn = Callable[..., list[MemoryRecord] | Awaitable[list[MemoryRecord]]]
WriteEntryFn = Callable[..., str | Awaitable[str]]
WriteReviewFn = Callable[..., str | Awaitable[str]]

_FILE_HASH_RE = re.compile(r"^FILE_HASH:\s*([0-9a-fA-F]+)\s*$", re.MULTILINE)


def extract_file_hash_from_content(content: str) -> str | None:
    """Return the FILE_HASH value from a capability memory content block."""
    match = _FILE_HASH_RE.search(content)
    if match is None:
        return None
    return match.group(1)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _identity_topics(entry: CapabilityEntry) -> set[str]:
    topics = {f"repo:{entry.repo_slug}", f"file:{entry.file_path}"}
    if entry.symbol_name:
        topics.add(f"symbol:{entry.symbol_name}")
    return topics


async def find_existing_memories(
    entry: CapabilityEntry,
    pool: Any,
    project_id: str | None,
) -> list[MemoryRecord]:
    """Find active capability memories for the same repo/file/symbol identity."""
    from weft.models import MemoryStatus
    from weft.store import list_memories

    candidates = await list_memories(
        pool,
        status=MemoryStatus.active,
        topic=f"file:{entry.file_path}",
        project_id=project_id,
        exact_scope=True,
        limit=1000,
    )
    required_topics = _identity_topics(entry)
    matches = []
    for memory in candidates:
        memory_topics = set(memory.topic)
        if required_topics.issubset(memory_topics):
            matches.append(memory.to_dict())
    return matches


def _most_recent_memory(memories: list[MemoryRecord]) -> MemoryRecord:
    return sorted(
        memories,
        key=lambda memory: str(memory.get("updated_at") or memory.get("created_at") or ""),
        reverse=True,
    )[0]


async def upsert_entry(
    entry: CapabilityEntry,
    pool: Any,
    project_id: str | None,
    *,
    dry_run: bool = True,
    find_existing_memories_fn: FindExistingFn = find_existing_memories,
    write_entry_to_weft_fn: WriteEntryFn | None = None,
    write_review_record_fn: WriteReviewFn | None = None,
) -> dict[str, Any]:
    """Decide whether to create, skip, or update a capability memory."""
    existing = await _maybe_await(find_existing_memories_fn(entry, pool, project_id))
    existing_ids = [str(memory.get("id")) for memory in existing if memory.get("id")]
    review_reason: str | None = None

    if not existing:
        action = "create"
    else:
        latest = _most_recent_memory(existing)
        prior_hash = extract_file_hash_from_content(str(latest.get("content", "")))
        if prior_hash == entry.file_hash:
            action = "skip"
        else:
            action = "update"
            review_reason = "missing-file-hash" if prior_hash is None else "hash-changed"

    if dry_run:
        return {
            "action": action,
            "entry": dataclasses.asdict(entry),
            "existing_ids": existing_ids,
            "review_reason": review_reason,
        }

    if action == "skip":
        return {"action": "skipped", "existing_ids": existing_ids}

    if write_entry_to_weft_fn is None:
        from capability_registry.ingest import write_entry_to_weft

        write_entry_to_weft_fn = write_entry_to_weft

    new_id = await _maybe_await(write_entry_to_weft_fn(entry, pool, project_id))
    if action == "create":
        return {"action": "created", "id": new_id}

    if write_review_record_fn is None:
        write_review_record_fn = write_review_record_to_weft

    review_id = await _maybe_await(
        write_review_record_fn(
            entry,
            pool,
            project_id,
            old_ids=existing_ids,
            new_id=new_id,
            reason=review_reason or "hash-changed",
        )
    )
    return {
        "action": "updated",
        "new_id": new_id,
        "old_ids": existing_ids,
        "review_id": review_id,
    }


async def write_review_record_to_weft(
    entry: CapabilityEntry,
    pool: Any,
    project_id: str | None,
    *,
    old_ids: list[str],
    new_id: str,
    reason: str,
) -> str:
    """Write a review issue for a changed capability artifact."""
    from weft.db.connection import acquire
    from weft.models import MemoryCreate, MemorySource, MemoryType
    from weft.store import store_memory

    content = _review_record_content(
        entry,
        old_ids=old_ids,
        new_id=new_id,
        reason=reason,
    )
    create = MemoryCreate(
        type=MemoryType.issue,
        content=content,
        topic=_review_topics(entry),
        source=MemorySource.ingest,
        confidence=0.8,
        project_id=project_id or None,
        pinned=False,
    )
    async with acquire(pool):
        memory = await store_memory(pool, create)
    return memory.id


def _review_topics(entry: CapabilityEntry) -> list[str]:
    topics = [
        "capability-review",
        "capability-stale",
        f"repo:{entry.repo_slug}",
        f"file:{entry.file_path}",
    ]
    if entry.symbol_name:
        topics.append(f"symbol:{entry.symbol_name}")
    return topics


def _review_record_content(
    entry: CapabilityEntry,
    *,
    old_ids: list[str],
    new_id: str,
    reason: str,
) -> str:
    symbol = entry.symbol_name or "(module)"
    return "\n".join(
        [
            "CAPABILITY_REVIEW: stale-artifact",
            f"REASON: {reason}",
            f"REPO: {entry.repo_slug}",
            f"FILE: {entry.file_path}",
            f"SYMBOL: {symbol}",
            f"OLD_MEMORY_IDS: {', '.join(old_ids) if old_ids else '(none)'}",
            f"NEW_MEMORY_ID: {new_id}",
            f"NEW_FILE_HASH: {entry.file_hash}",
            "NEXT_ACTION: inspect the old and new capability memories; archive or revise stale records after review.",
        ]
    )


def check_stale_files(entries: list[CapabilityEntry]) -> list[dict[str, Any]]:
    """Return changed or missing file records for capability entries."""
    stale: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry.file_path)
        if not path.exists():
            stale.append(
                {
                    "status": "missing",
                    "repo_slug": entry.repo_slug,
                    "file_path": entry.file_path,
                }
            )
            continue

        actual_hash = _sha256_file(path)
        if actual_hash != entry.file_hash:
            stale.append(
                {
                    "status": "changed",
                    "repo_slug": entry.repo_slug,
                    "file_path": entry.file_path,
                    "expected_hash": entry.file_hash,
                    "actual_hash": actual_hash,
                }
            )
    return stale


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
