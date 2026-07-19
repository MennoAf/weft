"""Public MCP manifest and preliminary inventory gates."""

from __future__ import annotations

import json

import pytest

from weft.tool_inventory import generate_artifacts, validate_public_tool_manifest


pytestmark = pytest.mark.asyncio


async def test_public_tool_removal_requires_approved_deprecation_record():
    with pytest.raises(ValueError, match="weft_removed"):
        validate_public_tool_manifest(
            ["weft_kept"],
            ["weft_kept", "weft_removed"],
            [],
        )

    with pytest.raises(ValueError, match="missing fields"):
        validate_public_tool_manifest(
            ["weft_kept"],
            ["weft_kept", "weft_removed"],
            [{"tool_name": "weft_removed", "status": "approved"}],
        )

    result = validate_public_tool_manifest(
        ["weft_kept"],
        ["weft_kept", "weft_removed"],
        [{
            "tool_name": "weft_removed",
            "status": "approved",
            "decision_date": "2026-07-18",
            "owner_intent": "Remove after compatibility window",
            "replacement_path": "weft_kept",
            "rollback_plan": "Restore the compatibility alias",
            "telemetry_evidence": {
                "since": "2026-06-19",
                "through": "2026-07-18",
                "recorder_version": "2",
                "valid_days": 30,
                "gap_days": 0,
                "failure_total": 0,
                "deprecation_eligible": True,
                "tool_call_count": 0,
            },
        }],
    )
    assert result["approved_removed"] == ["weft_removed"]


async def test_public_tool_removal_rejects_unproven_telemetry():
    record = {
        "tool_name": "weft_removed",
        "status": "approved",
        "decision_date": "2026-07-18",
        "owner_intent": "Remove after compatibility window",
        "replacement_path": "weft_kept",
        "rollback_plan": "Restore the compatibility alias",
        "telemetry_evidence": {
            "since": "2026-06-19",
            "through": "2026-07-18",
            "recorder_version": "2",
            "valid_days": 29,
            "gap_days": 1,
            "failure_total": 0,
            "deprecation_eligible": False,
            "tool_call_count": 0,
        },
    }
    with pytest.raises(ValueError, match="does not prove"):
        validate_public_tool_manifest(
            ["weft_kept"], ["weft_kept", "weft_removed"], [record]
        )


async def test_generated_preliminary_inventory_is_complete(tmp_path):
    deprecations = tmp_path / "deprecations.json"
    deprecations.write_text('{"records": []}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.json"

    await generate_artifacts(
        tmp_path,
        manifest_path=manifest,
        inventory_path=inventory,
        deprecations_path=deprecations,
    )

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    inventory_data = json.loads(inventory.read_text(encoding="utf-8"))
    assert manifest_data["tools"]
    assert len(inventory_data["tools"]) == len(manifest_data["tools"])
    assert inventory_data["status"] == "preliminary"

    required = {
        "mcp_name",
        "lifecycle",
        "external_call_count",
        "observation_window",
        "internal_callers",
        "hooks_or_scheduler_references",
        "tests",
        "docs_or_public_contracts",
        "migrations_or_tables",
        "recovery_or_admin_purpose",
        "owner_intent",
        "replacement_path",
        "rollback_plan",
    }
    assert all(required <= set(row) for row in inventory_data["tools"])
    assert all(
        row["observation_window"]["status"] == "PENDING-30-VALID-DAYS"
        for row in inventory_data["tools"]
    )


async def test_checked_in_manifest_has_no_unapproved_removals():
    from pathlib import Path

    from weft.mcp import mcp

    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
        (root / "inventory/public-tool-manifest.json").read_text(encoding="utf-8")
    )
    deprecations = json.loads(
        (root / "inventory/approved-deprecations.json").read_text(encoding="utf-8")
    )
    current = [tool.name for tool in await mcp.list_tools()]
    validate_public_tool_manifest(
        current,
        baseline["tools"],
        deprecations["records"],
    )
