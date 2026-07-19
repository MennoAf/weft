"""Generate public-tool manifests and preliminary lifecycle inventory.

The inventory is evidence for later review, not an automatic deletion engine.
Unknown ownership or replacement facts stay explicitly unconfirmed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA_VERSION = 1
INVENTORY_SCHEMA_VERSION = 1
APPROVED_DEPRECATION_STATUS = "approved"


@dataclass(frozen=True, slots=True)
class ToolInventoryRow:
    mcp_name: str
    lifecycle: str
    external_call_count: int | None
    observation_window: dict[str, Any]
    internal_callers: list[str]
    hooks_or_scheduler_references: list[str]
    tests: list[str]
    docs_or_public_contracts: list[str]
    migrations_or_tables: list[str]
    recovery_or_admin_purpose: str
    owner_intent: str
    replacement_path: str
    rollback_plan: str


def classify_lifecycle(name: str, tags: Iterable[str]) -> str:
    """Return a conservative lifecycle class from explicit public metadata."""
    tag_set = set(tags)
    if "deprecated" in tag_set:
        return "deprecated-observe"
    if any(token in name for token in ("token_", "fsck", "quarantine", "consolidate")):
        return "admin-or-recovery"
    if any(token in name for token in ("turn_", "episode_", "trigger_", "tracker_")):
        return "supporting-or-experimental"
    return "default-user-facing"


def validate_public_tool_manifest(
    current_names: Iterable[str],
    baseline_names: Iterable[str],
    deprecation_records: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    """Reject public-tool removals lacking a complete approved record."""
    current = set(current_names)
    baseline = set(baseline_names)
    records = list(deprecation_records)
    record_names = [record.get("tool_name") for record in records]
    duplicates = sorted({name for name in record_names if record_names.count(name) > 1})
    if duplicates:
        raise ValueError("duplicate deprecation records: " + ", ".join(duplicates))

    required_approval_fields = {
        "tool_name",
        "status",
        "decision_date",
        "owner_intent",
        "replacement_path",
        "rollback_plan",
    }
    approved: set[str] = set()
    for record in records:
        if record.get("status") != APPROVED_DEPRECATION_STATUS:
            continue
        missing = sorted(
            field
            for field in required_approval_fields
            if not isinstance(record.get(field), str) or not record[field].strip()
        )
        if missing:
            raise ValueError(
                f"approved deprecation record for {record.get('tool_name')!r} "
                f"missing fields: {', '.join(missing)}"
            )
        try:
            decision_date = datetime.fromisoformat(record["decision_date"]).date()
        except ValueError as exc:
            raise ValueError(
                f"approved deprecation record for {record['tool_name']!r} "
                "has invalid decision_date"
            ) from exc

        evidence = record.get("telemetry_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(
                f"approved deprecation record for {record['tool_name']!r} "
                "requires structured telemetry_evidence"
            )
        required_evidence = {
            "since",
            "through",
            "recorder_version",
            "valid_days",
            "gap_days",
            "failure_total",
            "deprecation_eligible",
            "tool_call_count",
        }
        if required_evidence - set(evidence):
            raise ValueError(
                f"approved deprecation record for {record['tool_name']!r} "
                "has incomplete telemetry_evidence"
            )
        try:
            since = datetime.fromisoformat(str(evidence["since"])).date()
            through = datetime.fromisoformat(str(evidence["through"])).date()
        except ValueError as exc:
            raise ValueError(
                f"approved deprecation record for {record['tool_name']!r} "
                "has invalid telemetry date range"
            ) from exc
        evidence_is_valid = (
            isinstance(evidence["recorder_version"], str)
            and bool(evidence["recorder_version"].strip())
            and isinstance(evidence["valid_days"], int)
            and evidence["valid_days"] >= 30
            and evidence["gap_days"] == 0
            and evidence["failure_total"] == 0
            and evidence["deprecation_eligible"] is True
            and evidence["tool_call_count"] == 0
            and (through - since).days + 1 >= 30
            and through <= decision_date
        )
        if not evidence_is_valid:
            raise ValueError(
                f"approved deprecation record for {record['tool_name']!r} "
                "does not prove 30 valid zero-use telemetry days"
            )
        approved.add(record["tool_name"])
    removed = sorted(baseline - current)
    unapproved = sorted(set(removed) - approved)
    if unapproved:
        raise ValueError(
            "public MCP tools removed without approved deprecation records: "
            + ", ".join(unapproved)
        )
    return {
        "added": sorted(current - baseline),
        "removed": removed,
        "approved_removed": sorted(set(removed) & approved),
    }


def _references(root: Path, needle: str, *, directories: tuple[str, ...]) -> list[str]:
    refs: list[str] = []
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if needle in text:
                refs.append(str(path.relative_to(root)))
    return refs


async def generate_artifacts(
    root: Path,
    *,
    manifest_path: Path,
    inventory_path: Path,
    deprecations_path: Path,
) -> None:
    """Write a current public manifest and honest preliminary inventory."""
    from weft.mcp import mcp

    tools = sorted(await mcp.list_tools(), key=lambda tool: tool.name)
    deprecations = (
        json.loads(deprecations_path.read_text(encoding="utf-8"))
        if deprecations_path.exists()
        else {"records": []}
    )
    records = deprecations.get("records", [])
    approved_by_name = {
        record["tool_name"]: record
        for record in records
        if record.get("status") == APPROVED_DEPRECATION_STATUS
    }
    generated_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "tools": [tool.name for tool in tools],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for tool in tools:
        approved = approved_by_name.get(tool.name, {})
        code_refs = _references(root, tool.name, directories=("weft",))
        test_refs = _references(root, tool.name, directories=("tests", "benchmarks"))
        doc_refs = _references(root, tool.name, directories=("docs",))
        row = ToolInventoryRow(
            mcp_name=tool.name,
            lifecycle=classify_lifecycle(tool.name, tool.tags),
            external_call_count=None,
            observation_window={
                "status": "PENDING-30-VALID-DAYS",
                "valid_days_required": 30,
                "source": "weft_tool_usage_daily + weft_tool_usage_coverage",
            },
            internal_callers=[ref for ref in code_refs if ref != "weft/mcp/tools.py"],
            hooks_or_scheduler_references=[
                ref for ref in code_refs if "scheduler" in ref or "hook" in ref
            ],
            tests=test_refs,
            docs_or_public_contracts=doc_refs,
            migrations_or_tables=[
                ref for ref in code_refs if "migration" in ref or "/db/" in ref
            ],
            recovery_or_admin_purpose=(
                approved.get("purpose") or "UNCONFIRMED"
            ),
            owner_intent=approved.get("owner_intent") or "UNCONFIRMED",
            replacement_path=approved.get("replacement_path") or "UNCONFIRMED",
            rollback_plan=approved.get("rollback_plan") or "UNCONFIRMED",
        )
        rows.append(asdict(row))

    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "preliminary",
        "warning": (
            "Zero means not observed. No deprecation recommendation is valid "
            "until telemetry reports 30 valid coverage days."
        ),
        "tools": rows,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
