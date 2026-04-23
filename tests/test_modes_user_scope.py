"""Tests for user_id OR-NULL scoping in modes list function.

Verifies that user_id parameter correctly filters modes:
- user_id=None: returns all rows (existing behavior)
- user_id="user-a": returns user-a rows + NULL rows, excludes other users
- user_id="user-a" with only NULL rows: returns them
"""

from __future__ import annotations

import pytest

from weft.modes import list_modes, upsert_mode
from weft.models import ModeCreate, ModeWeights


# --- Helpers ---


async def _make_mode(
    pool,
    name: str,
    user_id: str | None = None,
    **kwargs
) -> str:
    """Create a mode. If user_id is provided, manually set it in DB."""
    create = ModeCreate(
        name=name,
        description=kwargs.get("description", f"Mode {name}"),
        weights=kwargs.get("weights", ModeWeights()),
        project_id=kwargs.get("project_id"),
        agent_id=kwargs.get("agent_id"),
    )
    mode = await upsert_mode(pool, create)
    if user_id is not None:
        # Manually override user_id in the database
        await pool.execute(
            "UPDATE modes SET user_id = $1 WHERE id = $2",
            user_id,
            mode.id,
        )
    return mode.id


# --- list_modes with user_id ---


async def test_list_modes_user_id_none_returns_all(pool):
    """user_id=None should return all modes (existing behavior)."""
    await _make_mode(pool, "user-a-mode", user_id="user-a")
    await _make_mode(pool, "user-b-mode", user_id="user-b")
    await _make_mode(pool, "global-mode", user_id=None)

    results = await list_modes(pool, user_id=None)
    assert len(results) == 3
    names = {m.name for m in results}
    assert "user-a-mode" in names
    assert "user-b-mode" in names
    assert "global-mode" in names


async def test_list_modes_user_id_filters_to_user_and_null(pool):
    """user_id='user-a' should return user-a rows + NULL rows, exclude others."""
    await _make_mode(pool, "user-a-mode-1", user_id="user-a")
    await _make_mode(pool, "user-a-mode-2", user_id="user-a")
    await _make_mode(pool, "user-b-mode", user_id="user-b")
    await _make_mode(pool, "global-mode", user_id=None)

    results = await list_modes(pool, user_id="user-a")
    assert len(results) == 3
    names = {m.name for m in results}
    assert "user-a-mode-1" in names
    assert "user-a-mode-2" in names
    assert "global-mode" in names
    assert "user-b-mode" not in names


async def test_list_modes_user_id_with_only_null_rows(pool):
    """user_id='user-a' should return NULL rows even if no user-a rows exist."""
    await _make_mode(pool, "global-mode-1", user_id=None)
    await _make_mode(pool, "global-mode-2", user_id=None)
    await _make_mode(pool, "user-b-mode", user_id="user-b")

    results = await list_modes(pool, user_id="user-a")
    assert len(results) == 2
    names = {m.name for m in results}
    assert "global-mode-1" in names
    assert "global-mode-2" in names
    assert "user-b-mode" not in names
