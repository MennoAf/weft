"""RC-FL-02 contract tests for the public schema and migration manifest."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.refresh_public_schema_manifest import (
    discover_source_state,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "database-schema.json"
POLICY_PATH = ROOT / "tests" / "fixtures" / "rc_public_schema_policy.json"


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
        json.loads(POLICY_PATH.read_text(encoding="utf-8")),
    )


def test_manifest_matches_independent_literal_policy_and_current_source() -> None:
    manifest, policy = _documents()
    state = discover_source_state()

    # The policy is a tracked, literal decision record.  It must not be a
    # projection of the generated manifest or a runtime MIGRATIONS fixture.
    assert policy["schema"] == "weft.public-schema-policy.v1"
    assert isinstance(policy["public_tables"], list)
    assert policy["public_tables"]
    assert manifest["tables"]
    assert {row["name"] for row in manifest["tables"]} == {
        row["name"] for row in policy["public_tables"]
    }
    policy_by_name = {row["name"]: row for row in policy["public_tables"]}
    for table in manifest["tables"]:
        expected = policy_by_name[table["name"]]
        for field in ("category", "purpose", "ownership_rls", "export", "runtime_contract"):
            assert table[field] == expected[field]

    assert manifest["migration_head"] == state["migration_head"]
    assert manifest["migrations"] == state["migrations"]
    assert manifest["pending_files"] == [
        {
            "path": item["path"],
            "status": "dormant",
            "disposition": next(
                decision["disposition"]
                for decision in policy["pending_files"]
                if decision["path"] == item["path"]
            ),
        }
        for item in state["pending_files"]
    ]
    validate_manifest(manifest, policy, state=state)


def test_manifest_rejects_duplicate_missing_and_stale_migrations() -> None:
    manifest, policy = _documents()
    state = discover_source_state()

    duplicate = copy.deepcopy(manifest)
    duplicate["migrations"].append(copy.deepcopy(duplicate["migrations"][0]))
    with pytest.raises(ValueError, match="duplicate migration entries"):
        validate_manifest(duplicate, policy, state=state)

    missing = copy.deepcopy(manifest)
    missing["migrations"].pop()
    with pytest.raises(ValueError, match="migration entries differ"):
        validate_manifest(missing, policy, state=state)

    stale = copy.deepcopy(manifest)
    stale["migrations"][0]["description"] = "stale description"
    with pytest.raises(ValueError, match="migration entries differ"):
        validate_manifest(stale, policy, state=state)

    wrong_head = copy.deepcopy(manifest)
    wrong_head["migration_head"] -= 1
    with pytest.raises(ValueError, match="stale migration_head"):
        validate_manifest(wrong_head, policy, state=state)


def test_manifest_rejects_missing_or_misclassified_pending_draft() -> None:
    manifest, policy = _documents()
    state = discover_source_state()

    missing = copy.deepcopy(manifest)
    missing["pending_files"] = []
    with pytest.raises(ValueError, match="pending-file disposition differs"):
        validate_manifest(missing, policy, state=state)

    misclassified = copy.deepcopy(manifest)
    misclassified["pending_files"][0]["status"] = "applied"
    with pytest.raises(ValueError, match="pending-file disposition"):
        validate_manifest(misclassified, policy, state=state)


def test_manifest_rejects_missing_or_extra_public_tables() -> None:
    manifest, policy = _documents()
    state = discover_source_state()

    missing = copy.deepcopy(manifest)
    missing["tables"].pop()
    with pytest.raises(ValueError, match="public table inventory differs"):
        validate_manifest(missing, policy, state=state)

    extra = copy.deepcopy(manifest)
    extra["tables"].append(copy.deepcopy(extra["tables"][0]))
    extra["tables"][-1]["name"] = "not_a_real_table"
    with pytest.raises(ValueError, match="public table inventory differs"):
        validate_manifest(extra, policy, state=state)


@pytest.mark.asyncio
async def test_manifest_tables_match_real_migrated_postgres(pool) -> None:
    """Compare the declaration to the current source migrated in testcontainers."""
    manifest, policy = _documents()
    state = discover_source_state()
    validate_manifest(manifest, policy, state=state)

    rows = await pool.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    actual = {row["table_name"] for row in rows}
    declared = {row["name"] for row in manifest["tables"]}
    assert actual == declared

    ledger = await pool.fetch(
        "SELECT version, description FROM public.schema_migrations ORDER BY version"
    )
    assert [(row["version"], row["description"]) for row in ledger] == [
        (row["version"], row["description"]) for row in manifest["migrations"]
    ]
