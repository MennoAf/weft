"""Tests for Capability Registry lookup helpers."""

from __future__ import annotations

import sys

import pytest

from capability_registry.lookup import (
    format_lookup_results,
    lookup_capabilities,
    parse_capability_entry_from_content,
    query_to_topics,
)


def test_query_to_topics_natural_language_bot_block() -> None:
    topics = query_to_topics("bot blocked crawler")

    assert "capability:bot-block-hardening" in topics


def test_query_to_topics_explicit_tag() -> None:
    assert query_to_topics("capability:bot-block-hardening") == [
        "capability:bot-block-hardening"
    ]


def test_query_to_topics_repo_tag() -> None:
    assert query_to_topics("repo:muttr") == ["repo:muttr"]


def test_query_to_topics_unknown_query() -> None:
    topics = query_to_topics("some unknown thing")

    assert isinstance(topics, list)
    assert len(topics) >= 1


def test_query_to_topics_empty_query() -> None:
    assert query_to_topics("") == []
    assert query_to_topics("   ") == []


def test_parse_content_all_fields() -> None:
    content = "\n".join(
        [
            "CAPABILITY: bot-block-hardening",
            "REPO: muttr",
            "FILE: crawl/escalation.py",
            "SYMBOL: LazyEscalationPolicy (class)",
            "DOCSTRING: Lazy escalation for bot blocks.",
            "IMPORTS: requests, time",
            "REUSE_NOTES: Reuse by subclassing.",
            "FILE_HASH: abc123",
        ]
    )

    parsed = parse_capability_entry_from_content(content)

    assert parsed["capability"] == "bot-block-hardening"
    assert parsed["repo"] == "muttr"
    assert parsed["file"] == "crawl/escalation.py"
    assert parsed["symbol"] == "LazyEscalationPolicy (class)"
    assert parsed["docstring"] == "Lazy escalation for bot blocks."
    assert parsed["imports"] == "requests, time"
    assert parsed["reuse_notes"] == "Reuse by subclassing."
    assert parsed["file_hash"] == "abc123"


def test_parse_content_missing_optional_fields() -> None:
    content = "\n".join(
        [
            "CAPABILITY: bot-block-hardening",
            "REPO: muttr",
            "FILE: crawl/escalation.py",
            "FILE_HASH: def456",
        ]
    )

    parsed = parse_capability_entry_from_content(content)

    assert parsed["repo"] == "muttr"
    assert parsed["file_hash"] == "def456"
    assert parsed["symbol"] in (None, "")


def test_parse_content_malformed_lines_do_not_crash() -> None:
    content = "not a field line\nREPO: muttr\n:::\nUNKNOWN_FIELD: ignored\n"

    parsed = parse_capability_entry_from_content(content)

    assert parsed["repo"] == "muttr"
    assert parsed["file"] is None


def test_format_lookup_results_nonempty() -> None:
    results = [
        {
            "memory_id": "x",
            "topics": ["repo:muttr"],
            "parsed": {
                "repo": "muttr",
                "file": "crawl/esc.py",
                "symbol": "LazyEscalationPolicy",
                "capability": "bot-block-hardening",
                "reuse_notes": "Reuse by subclassing.",
                "docstring": "",
            },
        }
    ]

    formatted = format_lookup_results(results)

    assert "muttr" in formatted
    assert "LazyEscalationPolicy" in formatted
    assert "Reuse by subclassing." in formatted


def test_format_lookup_results_module_fallback_and_no_notes() -> None:
    results = [
        {
            "memory_id": "x",
            "topics": ["repo:muttr"],
            "parsed": {
                "repo": "muttr",
                "file": "crawl/esc.py",
                "symbol": None,
                "capability": "crawler",
                "reuse_notes": None,
                "docstring": None,
            },
        }
    ]

    formatted = format_lookup_results(results)

    assert "(module)" in formatted
    assert "(no notes)" in formatted


def test_format_lookup_results_empty() -> None:
    formatted = format_lookup_results([])

    assert isinstance(formatted, str)


@pytest.mark.asyncio
async def test_lookup_capabilities_parses_and_dedupes() -> None:
    memory = {
        "id": "weft-1",
        "content": "CAPABILITY: bot-block-hardening\nREPO: muttr\nFILE: crawl/esc.py\nFILE_HASH: abc123",
        "topic": ["capability:bot-block-hardening", "repo:muttr"],
    }
    calls: list[str] = []

    async def fake_list_memories(pool: object, topic: str, project_id: object, limit: int) -> list[dict]:
        calls.append(topic)
        return [memory]

    results = await lookup_capabilities(
        "bot blocked crawler",
        pool="pool",
        project_id="project",
        list_memories_fn=fake_list_memories,
    )

    assert len(calls) >= 2  # bot-block-hardening + crawler topics both queried
    assert len(results) == 1  # same memory id deduplicated
    assert results[0]["memory_id"] == "weft-1"
    assert results[0]["topics"] == ["capability:bot-block-hardening", "repo:muttr"]
    assert results[0]["parsed"]["repo"] == "muttr"
    assert results[0]["parsed"]["file_hash"] == "abc123"


@pytest.mark.asyncio
async def test_lookup_capabilities_respects_limit() -> None:
    def make_memory(index: int) -> dict:
        return {
            "id": f"weft-{index}",
            "content": f"REPO: muttr\nFILE: file{index}.py\nFILE_HASH: h{index}",
            "topic": ["capability:crawler"],
        }

    async def fake_list_memories(pool: object, topic: str, project_id: object, limit: int) -> list[dict]:
        return [make_memory(index) for index in range(5)]

    results = await lookup_capabilities(
        "capability:crawler",
        pool="pool",
        project_id="project",
        limit=3,
        list_memories_fn=fake_list_memories,
    )

    assert len(results) == 3
    assert [result["memory_id"] for result in results] == ["weft-0", "weft-1", "weft-2"]


@pytest.mark.asyncio
async def test_lookup_capabilities_no_matches_returns_empty() -> None:
    async def fake_list_memories(pool: object, topic: str, project_id: object, limit: int) -> list[dict]:
        return []

    results = await lookup_capabilities(
        "some unknown thing",
        pool="pool",
        project_id="project",
        list_memories_fn=fake_list_memories,
    )

    assert results == []


@pytest.mark.asyncio
async def test_lookup_capabilities_empty_query_skips_recall() -> None:
    async def exploding_list_memories(*args: object, **kwargs: object) -> list[dict]:
        raise AssertionError("recall should not be called for an empty query")

    results = await lookup_capabilities(
        "   ",
        pool="pool",
        project_id="project",
        list_memories_fn=exploding_list_memories,
    )

    assert results == []


def _run_main() -> int:
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                result = test_fn()
                if hasattr(result, "__await__"):
                    import asyncio

                    asyncio.run(result)
            except Exception as exc:  # pragma: no cover - manual runner
                failures += 1
                print(f"FAIL: {name}: {exc}")
            else:
                print(f"PASS: {name}")
    return failures


if __name__ == "__main__":
    sys.exit(_run_main())
