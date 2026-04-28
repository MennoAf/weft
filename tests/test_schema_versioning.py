"""Tests for the schema upgrade chain."""

from __future__ import annotations

import pytest

from weft.schema.versioning import (
    CURRENT_VERSION,
    SUPPORTED_VERSIONS,
    upgrade_to_current,
)


def test_current_version_in_supported():
    assert CURRENT_VERSION in SUPPORTED_VERSIONS


def test_v1_row_unchanged():
    row = {
        "schema_version": 1,
        "user_id": "u-123",
        "author_identity": {"kind": "local_user", "user_id": "u-123"},
        "visibility": "private",
        "provenance": {"source": "self"},
        "sharing_metadata": {},
        "workspace_id": None,
    }
    upgraded = upgrade_to_current(dict(row))
    assert upgraded == row


def test_v0_user_row_backfills_to_private():
    row = {"user_id": "u-123", "content": "hi"}
    upgraded = upgrade_to_current(row)
    assert upgraded["schema_version"] == 1
    assert upgraded["visibility"] == "private"
    assert upgraded["author_identity"] == {"kind": "local_user", "user_id": "u-123"}
    assert upgraded["provenance"] == {"source": "self"}
    assert upgraded["sharing_metadata"] == {}
    assert upgraded["workspace_id"] is None


def test_v0_global_row_backfills_to_system():
    row = {"user_id": None, "content": "seed memory"}
    upgraded = upgrade_to_current(row)
    assert upgraded["schema_version"] == 1
    assert upgraded["visibility"] == "global"
    assert upgraded["author_identity"] == {"kind": "system", "component": "seed"}


def test_upgrade_does_not_mutate_input():
    row = {"user_id": "u-1"}
    snapshot = dict(row)
    upgrade_to_current(row)
    # Input dict should not gain v1 fields.
    assert row == snapshot


def test_missing_upgrade_fn_raises():
    # A row claiming a version we don't have an upgrade fn for must fail
    # loud rather than silently passing through.
    row = {"schema_version": -1}
    with pytest.raises(ValueError, match="No upgrade path"):
        upgrade_to_current(row)
