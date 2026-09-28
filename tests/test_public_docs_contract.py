"""Contract checks for the GitHub-readable database/schema guide."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "database-schema.md"
README = ROOT / "README.md"
MANIFEST = ROOT / "docs" / "database-schema.json"

REQUIRED_HEADINGS = (
    "## Table groups",
    "## RLS and runtime roles",
    "## Owner-managed migrations",
    "## pgvector width and codec ordering",
    "## Export exclusions",
    "## Dormant-file status",
)

# These patterns are intentionally conservative: public docs may name the
# configuration variables, but must not contain values that can be used to
# connect or authenticate.
FORBIDDEN_PUBLIC_PATTERNS = (
    re.compile(r"(?:postgres(?:ql)?|redis)://[^\s)\]`]+", re.IGNORECASE),
    re.compile(r"(?:postgres(?:ql)?|redis)://[^\s:@]+:[^\s@]+@", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])Users[\\/][^\s`)]*", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])home[\\/][^\s`)]*", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/](?!/)[^\s`)]*", re.IGNORECASE),
)
FORBIDDEN_SECRET_MARKERS = (
    "sk-",
    "sbp_",
    "api_key=",
    "access_token=",
    "password=",
    "secret=",
    "token=",
)


def _public_text() -> str:
    return GUIDE.read_text(encoding="utf-8") + "\n" + README.read_text(encoding="utf-8")


def test_readme_links_canonical_schema_guide() -> None:
    readme = README.read_text(encoding="utf-8")
    assert re.search(r"\]\(docs/database-schema\.md\)", readme)


def test_guide_has_exact_required_headings() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert all(heading in guide.splitlines() for heading in REQUIRED_HEADINGS)


def test_guide_documents_every_manifest_table_and_count() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tables = manifest["tables"]
    guide = GUIDE.read_text(encoding="utf-8")

    assert len(tables) == 52
    assert guide.count("## ") >= len(REQUIRED_HEADINGS)
    for table in tables:
        assert f"`{table['name']}`" in guide, table["name"]
        assert table["purpose"] in guide, table["name"]

    exported = [table for table in tables if table["export"]["included"]]
    assert len(exported) == 11
    assert "52 public tables" in guide
    assert "11 exported" in guide
    assert "41 excluded" in guide


def test_guide_carries_security_and_runtime_policy_markers() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert f"v{manifest['migration_head']}" in guide
    assert "restricted application runtime role" in guide
    assert "SUPERUSER" in guide
    assert "BYPASSRLS" in guide
    assert "table ownership" in guide
    assert "768" in guide
    assert "codec" in guide.lower()
    assert "pending_v51_episode_turns_fts.py" in guide
    assert "not discovered" in guide
    assert "append-only" in guide
    assert "vNN_*.py" in guide


def test_public_docs_reject_credentials_dsns_and_private_paths() -> None:
    text = _public_text()
    for pattern in FORBIDDEN_PUBLIC_PATTERNS:
        assert not pattern.search(text), pattern.pattern
    lowered = text.lower()
    for marker in FORBIDDEN_SECRET_MARKERS:
        assert marker not in lowered, marker


def test_applied_migration_history_is_not_part_of_this_change() -> None:
    """The RC docs task must not edit migration history, before or after commit."""
    for diff_args in (
        ("git", "diff", "--name-only", "HEAD", "--", "weft/db/migrations"),
        ("git", "diff", "--cached", "--name-only", "--", "weft/db/migrations"),
    ):
        changed = subprocess.run(
            list(diff_args),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert changed == []
