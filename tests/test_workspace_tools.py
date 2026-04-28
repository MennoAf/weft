"""Workspace MCP tools — create/add/remove/list + workspace-scoped weft_remember.

These tests exercise the full MCP-tool path (membership checks, RLS predicates,
cross-user reads). The two-user scenarios switch ``WEFT_USER_ID`` env + the
``current_user_id`` contextvar so ``acquire()`` issues the right
``SET LOCAL app.user_id`` per call.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from weft.auth import current_user_id
from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext


_FAKE_EMBEDDING = [0.1] * 768


class FakeEmbeddingProvider:
    provider_name = "fake"
    dimensions = 768

    async def embed(self, text: str) -> list[float]:
        return list(_FAKE_EMBEDDING)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(_FAKE_EMBEDDING) for _ in texts]


def _make_ctx(app: AppContext) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    ctx.list_roots = AsyncMock(return_value=[])
    return ctx


@contextmanager
def _as_user(monkeypatch: pytest.MonkeyPatch, user_id: str):
    """Switch both env (for ``get_user_id()``) and contextvar (for ``acquire``)."""
    monkeypatch.setenv("WEFT_USER_ID", user_id)
    tok = current_user_id.set(user_id)
    try:
        yield
    finally:
        current_user_id.reset(tok)


@pytest.fixture
def app(pool):
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=WeftConfig(),
    )


@pytest.fixture
def ctx(app):
    return _make_ctx(app)


# ---------------------------------------------------------------------------
# weft_workspace_create
# ---------------------------------------------------------------------------


class TestWorkspaceCreate:
    async def test_creates_workspace_and_owner_membership(self, ctx, monkeypatch):
        from weft.mcp.tools import weft_workspace_create, weft_workspace_list

        with _as_user(monkeypatch, "alice-uid"):
            result = await weft_workspace_create(
                ctx,
                name="Brandon Cleanroom",
                description="AIO shared brain with Brandon",
                metadata={"project": "aio_cleanroom"},
            )

            assert result["id"].startswith("ws-")
            assert result["name"] == "Brandon Cleanroom"
            assert result["description"] == "AIO shared brain with Brandon"
            assert result["created_by"] == "alice-uid"
            assert result["metadata"] == {"project": "aio_cleanroom"}

            # Owner is auto-added as a member, so list returns it.
            listed = await weft_workspace_list(ctx)
            assert listed["count"] == 1
            assert listed["workspaces"][0]["id"] == result["id"]
            assert listed["workspaces"][0]["is_owner"] is True
            members = listed["workspaces"][0]["members"]
            assert len(members) == 1
            assert members[0]["member_identity"]["user_id"] == "alice-uid"
            assert members[0]["role"] == "admin"

    async def test_default_metadata_is_empty_dict(self, ctx, monkeypatch):
        from weft.mcp.tools import weft_workspace_create

        with _as_user(monkeypatch, "alice-uid"):
            result = await weft_workspace_create(ctx, name="Solo")
            assert result["metadata"] == {}
            assert result["description"] is None


# ---------------------------------------------------------------------------
# weft_workspace_add_member / remove_member
# ---------------------------------------------------------------------------


class TestWorkspaceAddMember:
    async def test_owner_can_add_member(self, ctx, monkeypatch):
        from weft.mcp.tools import (
            weft_workspace_add_member,
            weft_workspace_create,
            weft_workspace_list,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            add_result = await weft_workspace_add_member(
                ctx, workspace_id=ws["id"], user_id="bob-uid",
            )
            assert "error" not in add_result
            assert add_result["member_identity"]["user_id"] == "bob-uid"
            assert add_result["role"] == "member"

        # Bob can now see the workspace
        with _as_user(monkeypatch, "bob-uid"):
            listed = await weft_workspace_list(ctx)
            assert listed["count"] == 1
            assert listed["workspaces"][0]["id"] == ws["id"]
            assert listed["workspaces"][0]["is_owner"] is False

    async def test_non_owner_cannot_add_member(self, ctx, monkeypatch):
        from weft.mcp.tools import (
            weft_workspace_add_member,
            weft_workspace_create,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            await weft_workspace_add_member(
                ctx, workspace_id=ws["id"], user_id="bob-uid",
            )

        # Bob is a member but not owner — cannot add charlie.
        with _as_user(monkeypatch, "bob-uid"):
            result = await weft_workspace_add_member(
                ctx, workspace_id=ws["id"], user_id="charlie-uid",
            )
            assert result.get("error") == "Permission denied"

    async def test_add_member_to_missing_workspace_returns_not_found(
        self, ctx, monkeypatch,
    ):
        from weft.mcp.tools import weft_workspace_add_member

        with _as_user(monkeypatch, "alice-uid"):
            result = await weft_workspace_add_member(
                ctx, workspace_id="ws-doesnotexist", user_id="bob-uid",
            )
            assert result.get("error") == "Not found"


class TestWorkspaceRemoveMember:
    async def test_owner_can_remove_member(self, ctx, monkeypatch):
        from weft.mcp.tools import (
            weft_workspace_add_member,
            weft_workspace_create,
            weft_workspace_list,
            weft_workspace_remove_member,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            await weft_workspace_add_member(
                ctx, workspace_id=ws["id"], user_id="bob-uid",
            )
            removed = await weft_workspace_remove_member(
                ctx, workspace_id=ws["id"], user_id="bob-uid",
            )
            assert removed["removed"] is True

        # Bob is no longer a member.
        with _as_user(monkeypatch, "bob-uid"):
            listed = await weft_workspace_list(ctx)
            assert listed["count"] == 0

    async def test_owner_cannot_remove_self(self, ctx, monkeypatch):
        from weft.mcp.tools import (
            weft_workspace_create,
            weft_workspace_remove_member,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            result = await weft_workspace_remove_member(
                ctx, workspace_id=ws["id"], user_id="alice-uid",
            )
            assert result.get("error") == "Permission denied"
            assert "owner" in result.get("detail", "").lower()


# ---------------------------------------------------------------------------
# weft_remember with workspace_id
# ---------------------------------------------------------------------------


class TestWorkspaceScopedRemember:
    async def test_member_can_write_workspace_memory(self, ctx, monkeypatch):
        from weft.mcp.tools import weft_remember, weft_workspace_create

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            mem = await weft_remember(
                ctx,
                content="shared knowledge across the team",
                type="fact",
                workspace_id=ws["id"],
                check_contradictions=False,
            )
            assert "error" not in mem
            assert mem["workspace_id"] == ws["id"]

    async def test_non_member_cannot_write_workspace_memory(self, ctx, monkeypatch):
        from weft.mcp.tools import weft_remember, weft_workspace_create

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")

        with _as_user(monkeypatch, "mallory-uid"):
            result = await weft_remember(
                ctx,
                content="should not land",
                type="fact",
                workspace_id=ws["id"],
                check_contradictions=False,
            )
            assert result.get("error") == "Invalid input"
            assert "not a member" in result.get("detail", "")

    async def test_workspace_member_can_read_others_workspace_memory(
        self, ctx, monkeypatch,
    ):
        """RLS subquery: bob (member, not author) sees alice's workspace row."""
        from weft.mcp.tools import (
            weft_recall,
            weft_remember,
            weft_workspace_add_member,
            weft_workspace_create,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            await weft_workspace_add_member(
                ctx, workspace_id=ws["id"], user_id="bob-uid",
            )
            await weft_remember(
                ctx,
                content="alice wrote a workspace fact about ferrets",
                type="fact",
                workspace_id=ws["id"],
                check_contradictions=False,
            )

        with _as_user(monkeypatch, "bob-uid"):
            results = await weft_recall(ctx, query="ferrets", limit=10)
            contents = [
                r["content"] for r in results.get("results", [])
            ]
            assert any("ferrets" in c for c in contents), (
                f"bob should see alice's workspace memory; got: {contents}"
            )

    async def test_non_member_cannot_read_workspace_memory(
        self, ctx, monkeypatch,
    ):
        """RLS: mallory (not a member) does NOT see workspace rows."""
        from weft.mcp.tools import (
            weft_recall,
            weft_remember,
            weft_workspace_create,
        )

        with _as_user(monkeypatch, "alice-uid"):
            ws = await weft_workspace_create(ctx, name="shared")
            await weft_remember(
                ctx,
                content="alice's secret about platypuses",
                type="fact",
                workspace_id=ws["id"],
                check_contradictions=False,
            )

        with _as_user(monkeypatch, "mallory-uid"):
            results = await weft_recall(ctx, query="platypuses", limit=10)
            contents = [r["content"] for r in results.get("results", [])]
            assert not any("platypuses" in c for c in contents), (
                f"mallory (non-member) must not see workspace memory; got: {contents}"
            )
