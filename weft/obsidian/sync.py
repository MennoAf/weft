"""Obsidian vault sync — scans, parses, and stores vault notes as Weft memories."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from weft.embeddings.base import EmbeddingProvider
from weft.models import MemoryCreate, MemorySource, MemoryType
from weft.store import store_memory

from .config import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_SPLIT_THRESHOLD,
    FolderMapping,
    resolve_folder_mapping,
)
from .hash_store import HashStore
from .parser import parse_note, split_by_headings

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    files_found: int = 0
    files_synced: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    memories_created: int = 0
    memories_archived: int = 0


def discover_vault_files(
    vault_path: Path,
    excluded_dirs: set[str] | None = None,
) -> list[str]:
    """Walk the vault and return relative paths to all .md files."""
    if excluded_dirs is None:
        excluded_dirs = DEFAULT_EXCLUDED_DIRS

    # Normalize excluded dirs to lowercase for case-insensitive matching
    excluded_lower = {d.lower() for d in excluded_dirs}

    files = []
    for p in vault_path.rglob("*.md"):
        rel = p.relative_to(vault_path)
        if any(part.lower() in excluded_lower for part in rel.parts):
            continue
        files.append(str(rel))

    return sorted(files)


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def sync_vault(
    vault_path: Path,
    pool: asyncpg.Pool,
    embedding_provider: EmbeddingProvider | None = None,
    *,
    hash_store: HashStore | None = None,
    excluded_dirs: set[str] | None = None,
    folder_map: dict[str, FolderMapping] | None = None,
    split_threshold: int = DEFAULT_SPLIT_THRESHOLD,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    batch_size: int = 50,
) -> SyncResult:
    """Sync an Obsidian vault into Weft memories."""
    vault_path = vault_path.resolve()

    if hash_store is None:
        hash_store = HashStore()
    hash_store.set_vault_path(str(vault_path))

    result = SyncResult()

    md_files = discover_vault_files(vault_path, excluded_dirs)
    result.files_found = len(md_files)

    current_files = set(md_files)

    for i in range(0, len(md_files), batch_size):
        batch = md_files[i : i + batch_size]
        for rel_path in batch:
            try:
                synced = await _sync_file(
                    vault_path,
                    rel_path,
                    pool,
                    embedding_provider,
                    hash_store=hash_store,
                    folder_map=folder_map,
                    split_threshold=split_threshold,
                    max_file_size=max_file_size,
                    result=result,
                )
                if synced:
                    result.files_synced += 1
                else:
                    result.files_skipped += 1
            except Exception as exc:
                logger.warning("Error syncing %s: %s", rel_path, exc)
                result.files_errored += 1

    # Clean up memories for deleted files
    stale = hash_store.all_paths - current_files
    for rel_path in stale:
        old_ids = hash_store.get_memory_ids(rel_path)
        for mid in old_ids:
            await _archive_memory(pool, mid)
            result.memories_archived += 1
        hash_store.remove(rel_path)

    hash_store.save()
    return result


async def _sync_file(
    vault_path: Path,
    rel_path: str,
    pool: asyncpg.Pool,
    embedding_provider: EmbeddingProvider | None,
    *,
    hash_store: HashStore,
    folder_map: dict[str, FolderMapping] | None,
    split_threshold: int,
    max_file_size: int,
    result: SyncResult,
) -> bool:
    """Sync a single file. Returns True if ingested, False if skipped."""
    full_path = vault_path / rel_path

    try:
        size = full_path.stat().st_size
    except FileNotFoundError:
        logger.warning("File disappeared: %s", rel_path)
        return False

    if size > max_file_size:
        logger.warning("Skipping oversized file (%d bytes): %s", size, rel_path)
        return False

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping binary file: %s", rel_path)
        return False
    except FileNotFoundError:
        logger.warning("File disappeared: %s", rel_path)
        return False

    content_hash = compute_hash(content)
    if hash_store.get_hash(rel_path) == content_hash:
        return False

    # Archive old memories for this file
    old_ids = hash_store.get_memory_ids(rel_path)
    for mid in old_ids:
        await _archive_memory(pool, mid)
        result.memories_archived += 1

    parsed = parse_note(content)
    mapping = resolve_folder_mapping(rel_path, folder_map)

    # Frontmatter overrides
    memory_type = mapping.memory_type
    if "type" in parsed.frontmatter:
        try:
            memory_type = MemoryType(parsed.frontmatter["type"])
        except ValueError:
            pass

    confidence = mapping.confidence
    if "confidence" in parsed.frontmatter:
        try:
            confidence = float(parsed.frontmatter["confidence"])
        except (ValueError, TypeError):
            pass

    # Build topics: source marker + file path + folder topics + tags
    topics = ["obsidian", f"file:{rel_path}"] + mapping.topics
    if parsed.tags:
        topics.extend(parsed.tags)
    topics = list(dict.fromkeys(topics))  # dedupe preserving order

    title = (
        parsed.title
        or parsed.frontmatter.get("name")
        or Path(rel_path).stem
    )

    sections = split_by_headings(parsed.body, split_threshold)

    memory_ids: list[str] = []
    for section in sections:
        section_content = _build_content(
            title, section, parsed, mapping
        )

        embedding = None
        if embedding_provider:
            try:
                embedding = await embedding_provider.embed(section_content)
            except Exception as exc:
                logger.warning("Failed to embed %s: %s", rel_path, exc)

        create = MemoryCreate(
            type=memory_type,
            content=section_content,
            topic=topics,
            source=MemorySource.ingest,
            confidence=confidence,
        )
        memory = await store_memory(pool, create, embedding=embedding)
        memory_ids.append(memory.id)
        result.memories_created += 1

    hash_store.update(rel_path, content_hash, memory_ids)
    return True


def _build_content(title, section, parsed, mapping):
    """Build the memory content string from a note section."""
    if section.total > 1 and section.heading:
        text = f"# {title} -- {section.heading}\n\n{section.content}"
    else:
        text = f"# {title}\n\n{section.content}"

    meta_parts: list[str] = []
    if parsed.date:
        meta_parts.append(f"Date: {parsed.date}")
    if parsed.wikilinks:
        linked = [link["target"] for link in parsed.wikilinks]
        meta_parts.append(f"Links: {', '.join(linked)}")

    fm = parsed.frontmatter

    # People-specific fields
    if mapping.memory_type == MemoryType.user_model:
        for key in ("company", "role", "email", "how_met"):
            if fm.get(key):
                meta_parts.append(f"{key.replace('_', ' ').title()}: {fm[key]}")

    # Recipe-specific fields
    if "recipes" in mapping.topics:
        for key in ("cuisine", "prep_time", "cook_time", "servings", "source"):
            if fm.get(key):
                meta_parts.append(
                    f"{key.replace('_', ' ').title()}: {fm[key]}"
                )
        if fm.get("lissy_approved") is True:
            meta_parts.append("Lissy approved: yes")
        elif fm.get("lissy_approved") is False:
            meta_parts.append("Lissy approved: no")

    # Media-specific fields
    if "media" in mapping.topics:
        for key in ("author", "rating", "date_finished"):
            if fm.get(key):
                meta_parts.append(
                    f"{key.replace('_', ' ').title()}: {fm[key]}"
                )
        if fm.get("type"):
            meta_parts.append(f"Media type: {fm['type']}")

    if meta_parts:
        text += "\n\n" + "\n".join(meta_parts)

    return text


async def _archive_memory(pool: asyncpg.Pool, memory_id: str):
    """Archive a memory by ID."""
    try:
        await pool.execute(
            "UPDATE memories SET status = 'archived', updated_at = NOW() WHERE id = $1",
            memory_id,
        )
    except Exception as exc:
        logger.warning("Failed to archive memory %s: %s", memory_id, exc)
