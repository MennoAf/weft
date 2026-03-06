"""Tests for Obsidian vault sync — uses tmp_path fixtures and mocked store."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.obsidian.config import (
    DEFAULT_EXCLUDED_DIRS,
    FolderMapping,
    resolve_folder_mapping,
)
from weft.obsidian.hash_store import HashStore
from weft.obsidian.sync import (
    SyncResult,
    compute_hash,
    discover_vault_files,
    sync_vault,
)
from weft.models import MemoryType


# ---- Config tests ----


class TestResolveMapping:
    def test_inbox(self):
        m = resolve_folder_mapping("inbox/quick-note.md")
        assert m.memory_type == MemoryType.fact
        assert "inbox" in m.topics
        assert m.confidence == 0.5

    def test_people(self):
        m = resolve_folder_mapping("people/alice.md")
        assert m.memory_type == MemoryType.user_model

    def test_nested_wktw(self):
        m = resolve_folder_mapping("wktw/clients/acme.md")
        assert "wktw" in m.topics
        assert "clients" in m.topics

    def test_wktw_root(self):
        m = resolve_folder_mapping("wktw/notes.md")
        assert "wktw" in m.topics

    def test_journal_daily(self):
        m = resolve_folder_mapping("journal/daily/2026-03-06.md")
        assert "journal" in m.topics

    def test_recipes(self):
        m = resolve_folder_mapping("recipes/carbonara.md")
        assert "recipes" in m.topics

    def test_media(self):
        m = resolve_folder_mapping("media/dune.md")
        assert "media" in m.topics

    def test_writing_blog(self):
        m = resolve_folder_mapping("writing/blog/my-post.md")
        assert "writing" in m.topics
        assert "blog" in m.topics

    def test_writing_ideas(self):
        m = resolve_folder_mapping("writing/ideas/concept.md")
        assert "creative" in m.topics

    def test_tools(self):
        m = resolve_folder_mapping("tools/weft.md")
        assert "tools" in m.topics

    def test_finances_income(self):
        m = resolve_folder_mapping("wktw/finances/income/jan-2026.md")
        assert "finances" in m.topics
        assert "income" in m.topics

    def test_finances_expenses(self):
        m = resolve_folder_mapping("wktw/finances/expenses/api-credits.md")
        assert "finances" in m.topics
        assert "expenses" in m.topics

    def test_unknown_folder_defaults(self):
        m = resolve_folder_mapping("random/stuff.md")
        assert m.memory_type == MemoryType.fact
        assert "notes" in m.topics

    def test_root_file_defaults(self):
        m = resolve_folder_mapping("readme.md")
        assert m.memory_type == MemoryType.fact


# ---- HashStore tests ----


class TestHashStore:
    def test_empty_store(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        assert hs.file_count == 0
        assert hs.get_hash("test.md") is None
        assert hs.get_memory_ids("test.md") == []

    def test_update_and_retrieve(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        hs.update("notes/test.md", "abc123", ["weft-001"])
        assert hs.get_hash("notes/test.md") == "abc123"
        assert hs.get_memory_ids("notes/test.md") == ["weft-001"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "hashes.json"
        hs = HashStore(path)
        hs.update("test.md", "abc", ["weft-1"])
        hs.save()

        hs2 = HashStore(path)
        assert hs2.get_hash("test.md") == "abc"

    def test_remove(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        hs.update("test.md", "abc", ["weft-1"])
        hs.remove("test.md")
        assert hs.get_hash("test.md") is None

    def test_vault_path_change_clears(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        hs.set_vault_path("/vault/one")
        hs.update("test.md", "abc", ["weft-1"])
        hs.set_vault_path("/vault/two")
        assert hs.file_count == 0

    def test_same_vault_path_preserves(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        hs.set_vault_path("/vault/one")
        hs.update("test.md", "abc", ["weft-1"])
        hs.set_vault_path("/vault/one")
        assert hs.file_count == 1

    def test_corrupted_json_recovers(self, tmp_path):
        path = tmp_path / "hashes.json"
        path.write_text("{invalid json", encoding="utf-8")
        hs = HashStore(path)
        assert hs.file_count == 0

    def test_all_paths(self, tmp_path):
        hs = HashStore(tmp_path / "hashes.json")
        hs.update("a.md", "h1", ["m1"])
        hs.update("b.md", "h2", ["m2"])
        assert hs.all_paths == {"a.md", "b.md"}


# ---- Discover tests ----


class TestDiscoverVaultFiles:
    def test_finds_md_files(self, tmp_path):
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "test.md").write_text("hello")
        (tmp_path / "inbox").mkdir()
        (tmp_path / "inbox" / "quick.md").write_text("note")

        files = discover_vault_files(tmp_path)
        assert "notes/test.md" in files
        assert "inbox/quick.md" in files

    def test_excludes_obsidian_dir(self, tmp_path):
        (tmp_path / ".obsidian").mkdir()
        (tmp_path / ".obsidian" / "config.md").write_text("config")
        files = discover_vault_files(tmp_path)
        assert not any(".obsidian" in f for f in files)

    def test_excludes_assets(self, tmp_path):
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "note.md").write_text("hidden")
        files = discover_vault_files(tmp_path)
        assert files == []

    def test_excludes_trash(self, tmp_path):
        (tmp_path / ".trash").mkdir()
        (tmp_path / ".trash" / "deleted.md").write_text("gone")
        files = discover_vault_files(tmp_path)
        assert files == []

    def test_ignores_non_md(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.json").write_text("{}")
        files = discover_vault_files(tmp_path)
        assert files == []

    def test_custom_excludes(self, tmp_path):
        (tmp_path / "private").mkdir()
        (tmp_path / "private" / "secret.md").write_text("shh")
        files = discover_vault_files(tmp_path, excluded_dirs={"private"})
        assert files == []


# ---- Sync tests (mocked DB) ----


def _make_vault(tmp_path: Path) -> Path:
    """Create a minimal test vault."""
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "inbox").mkdir()
    (vault / "inbox" / "quick.md").write_text(
        "---\ncreated: 2026-03-06\n---\n\nQuick thought"
    )

    (vault / "people").mkdir()
    (vault / "people" / "alice.md").write_text(
        "---\nname: Alice Smith\ncompany: Acme\ntags: [friend]\n---\n\nMet at PyCon."
    )

    (vault / "recipes").mkdir()
    (vault / "recipes" / "pasta.md").write_text(
        "---\ntitle: Pasta Carbonara\ncuisine: Italian\nlissy_approved: true\ntags: [pasta]\n---\n\n## Ingredients\n- Pasta\n- Eggs"
    )

    return vault


def _mock_pool():
    """Create a mock asyncpg pool."""
    pool = AsyncMock()
    pool.execute = AsyncMock()
    return pool


def _mock_memory(memory_id="weft-test1"):
    """Create a mock Memory object."""
    m = MagicMock()
    m.id = memory_id
    return m


class TestSyncVault:
    @pytest.mark.asyncio
    async def test_syncs_new_files(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        counter = {"n": 0}

        async def fake_store(pool, create, embedding=None):
            counter["n"] += 1
            m = MagicMock()
            m.id = f"weft-{counter['n']}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=fake_store):
            result = await sync_vault(vault, pool, hash_store=hs)

        assert result.files_found == 3
        assert result.files_synced == 3
        assert result.memories_created == 3

    @pytest.mark.asyncio
    async def test_skips_unchanged_files(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        counter = {"n": 0}

        async def fake_store(pool, create, embedding=None):
            counter["n"] += 1
            m = MagicMock()
            m.id = f"weft-{counter['n']}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=fake_store):
            await sync_vault(vault, pool, hash_store=hs)
            counter["n"] = 0
            result = await sync_vault(vault, pool, hash_store=hs)

        assert result.files_synced == 0
        assert result.files_skipped == 3

    @pytest.mark.asyncio
    async def test_resyncs_changed_file(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        counter = {"n": 0}

        async def fake_store(pool, create, embedding=None):
            counter["n"] += 1
            m = MagicMock()
            m.id = f"weft-{counter['n']}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=fake_store):
            await sync_vault(vault, pool, hash_store=hs)

            # Modify a file
            (vault / "inbox" / "quick.md").write_text("---\n---\n\nUpdated content")
            counter["n"] = 0
            result = await sync_vault(vault, pool, hash_store=hs)

        assert result.files_synced == 1
        assert result.files_skipped == 2
        assert result.memories_archived == 1  # old memory archived

    @pytest.mark.asyncio
    async def test_archives_deleted_files(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        counter = {"n": 0}

        async def fake_store(pool, create, embedding=None):
            counter["n"] += 1
            m = MagicMock()
            m.id = f"weft-{counter['n']}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=fake_store):
            await sync_vault(vault, pool, hash_store=hs)

            # Delete a file
            (vault / "inbox" / "quick.md").unlink()
            counter["n"] = 0
            result = await sync_vault(vault, pool, hash_store=hs)

        assert result.files_found == 2
        assert result.memories_archived == 1

    @pytest.mark.asyncio
    async def test_memory_content_includes_title(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        stored_creates = []

        async def capture_store(pool, create, embedding=None):
            stored_creates.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored_creates)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture_store):
            await sync_vault(vault, pool, hash_store=hs)

        # Find the pasta recipe
        pasta = next(c for c in stored_creates if "Pasta Carbonara" in c.content)
        assert "Lissy approved: yes" in pasta.content
        assert "Cuisine: Italian" in pasta.content
        assert "recipes" in pasta.topic

    @pytest.mark.asyncio
    async def test_people_mapped_to_user_model(self, tmp_path):
        vault = _make_vault(tmp_path)
        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        stored_creates = []

        async def capture_store(pool, create, embedding=None):
            stored_creates.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored_creates)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture_store):
            await sync_vault(vault, pool, hash_store=hs)

        alice = next(c for c in stored_creates if "Alice" in c.content)
        assert alice.type == MemoryType.user_model
        assert "contacts" in alice.topic
        assert "Company: Acme" in alice.content

    @pytest.mark.asyncio
    async def test_tool_note_content(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "tools").mkdir()
        (vault / "tools" / "weft.md").write_text(
            "---\nname: Weft\ndescription: Persistent memory\n"
            "created_by_me: true\nis_public: true\ncategory: ai-tooling\n"
            "development_status: active\n---\n\n## What Is The Core Function\nMemory storage."
        )

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")
        stored = []

        async def capture(pool, create, embedding=None):
            stored.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture):
            await sync_vault(vault, pool, hash_store=hs)

        tool = stored[0]
        assert "tools" in tool.topic
        assert "Description: Persistent memory" in tool.content
        assert "Created by me: yes" in tool.content
        assert "Public: yes" in tool.content
        assert "Category: ai-tooling" in tool.content

    @pytest.mark.asyncio
    async def test_finance_income_content(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "wktw" / "finances" / "income").mkdir(parents=True)
        (vault / "wktw" / "finances" / "income" / "jan-payment.md").write_text(
            "---\nclient: Birdy Grey\namount: 2500\n"
            "date_received: 2026-01-15\ninvoice_id: INV-001\n"
            "category: income\n---\n\n## Notes\nFirst payment."
        )

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")
        stored = []

        async def capture(pool, create, embedding=None):
            stored.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture):
            await sync_vault(vault, pool, hash_store=hs)

        inc = stored[0]
        assert "finances" in inc.topic
        assert "income" in inc.topic
        assert "Client: Birdy Grey" in inc.content
        assert "Amount: 2500" in inc.content
        assert "Date Received: 2026-01-15" in inc.content
        assert "Invoice Id: INV-001" in inc.content

    @pytest.mark.asyncio
    async def test_finance_expense_content(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "wktw" / "finances" / "expenses").mkdir(parents=True)
        (vault / "wktw" / "finances" / "expenses" / "api-credits.md").write_text(
            "---\nvendor: Anthropic\namount: 150.00\n"
            "date_paid: 2026-02-01\ncategory: api-credits\n"
            "recurring: monthly\n---\n\n## Notes\nClaude API usage."
        )

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")
        stored = []

        async def capture(pool, create, embedding=None):
            stored.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture):
            await sync_vault(vault, pool, hash_store=hs)

        exp = stored[0]
        assert "finances" in exp.topic
        assert "expenses" in exp.topic
        assert "Vendor: Anthropic" in exp.content
        assert "Amount: 150.0" in exp.content
        assert "Category: api-credits" in exp.content
        assert "Recurring: monthly" in exp.content

    @pytest.mark.asyncio
    async def test_frontmatter_type_override(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "notes").mkdir()
        (vault / "notes" / "decision.md").write_text(
            "---\ntype: decision\n---\n\nWe decided to use PostgreSQL."
        )

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")
        stored = []

        async def capture(pool, create, embedding=None):
            stored.append(create)
            m = MagicMock()
            m.id = f"weft-{len(stored)}"
            return m

        with patch("weft.obsidian.sync.store_memory", side_effect=capture):
            await sync_vault(vault, pool, hash_store=hs)

        assert stored[0].type == MemoryType.decision

    @pytest.mark.asyncio
    async def test_binary_file_skipped(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        # Write a file with invalid UTF-8
        (vault / "binary.md").write_bytes(b"\x80\x81\x82\x83")

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        with patch("weft.obsidian.sync.store_memory") as mock_store:
            result = await sync_vault(vault, pool, hash_store=hs)

        assert result.files_found == 1
        assert result.files_synced == 0
        mock_store.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_file_skipped(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "huge.md").write_text("x" * 100)

        pool = _mock_pool()
        hs = HashStore(tmp_path / "hashes.json")

        with patch("weft.obsidian.sync.store_memory") as mock_store:
            result = await sync_vault(vault, pool, hash_store=hs, max_file_size=50)

        assert result.files_found == 1
        assert result.files_synced == 0
        mock_store.assert_not_called()


class TestComputeHash:
    def test_deterministic(self):
        assert compute_hash("hello") == compute_hash("hello")

    def test_different_content(self):
        assert compute_hash("hello") != compute_hash("world")
