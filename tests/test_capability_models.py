"""Tests for Capability Registry data models."""

from __future__ import annotations

from capability_registry.models import CapabilityEntry


def test_topics_include_repo_file_symbol_and_capabilities() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/escalation.py",
        symbol_name="LazyEscalationPolicy",
        symbol_kind="class",
        capability_slugs=["bot-block-hardening", "crawler-escalation"],
    )

    assert entry.topics == [
        "repo:muttr",
        "file:crawl/escalation.py",
        "symbol:LazyEscalationPolicy",
        "capability:bot-block-hardening",
        "capability:crawler-escalation",
    ]


def test_topics_omit_symbol_when_entry_is_file_level() -> None:
    entry = CapabilityEntry(repo_slug="weft", file_path="scripts/audit.py")

    assert entry.topics == ["repo:weft", "file:scripts/audit.py"]


def test_content_formats_unclassified_file_level_entry() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/escalation.py",
        file_hash="abc123",
    )

    assert entry.content == "\n".join(
        [
            "CAPABILITY: (unclassified)",
            "REPO: muttr",
            "FILE: crawl/escalation.py",
            "FILE_HASH: abc123",
        ]
    )


def test_content_includes_optional_fields_in_contract_order() -> None:
    entry = CapabilityEntry(
        repo_slug="muttr",
        file_path="crawl/escalation.py",
        symbol_name="LazyEscalationPolicy",
        symbol_kind="class",
        docstring="Lazy escalation for bot-block handling.",
        imports=["requests", "time"],
        reuse_notes="Use when cheap retries should precede expensive escalation.",
        file_hash="def456",
        capability_slugs=["bot-block-hardening", "crawler-escalation"],
    )

    assert entry.content == "\n".join(
        [
            "CAPABILITY: bot-block-hardening, crawler-escalation",
            "REPO: muttr",
            "FILE: crawl/escalation.py",
            "SYMBOL: LazyEscalationPolicy (class)",
            "DOCSTRING: Lazy escalation for bot-block handling.",
            "IMPORTS: requests, time",
            "REUSE_NOTES: Use when cheap retries should precede expensive escalation.",
            "FILE_HASH: def456",
        ]
    )


def test_content_truncates_docstring_and_imports() -> None:
    entry = CapabilityEntry(
        repo_slug="repo",
        file_path="module.py",
        docstring="x" * 501,
        imports=[f"import_{index}" for index in range(25)],
    )

    lines = entry.content.splitlines()
    assert f"DOCSTRING: {'x' * 500}" in lines
    assert "IMPORTS: " + ", ".join(f"import_{index}" for index in range(20)) in lines
    assert "import_20" not in entry.content


def test_default_lists_are_not_shared() -> None:
    first = CapabilityEntry(repo_slug="one", file_path="a.py")
    second = CapabilityEntry(repo_slug="two", file_path="b.py")

    first.imports.append("requests")
    first.capability_slugs.append("crawler")

    assert second.imports == []
    assert second.capability_slugs == []
