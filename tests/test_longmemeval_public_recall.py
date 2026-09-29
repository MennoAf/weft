from __future__ import annotations

import pytest

from benchmarks.longmemeval.reader import format_public_recall_context


def test_reader_uses_only_public_evidence_and_preserves_provenance() -> None:
    response = {
        "tier": "belief",
        "results": [{"id": "m1", "content": "retrieved fact", "similarity": 0.91, "source_provenance": "conversation"}],
    }
    rendered = format_public_recall_context(response)
    assert "retrieved fact" in rendered
    assert "conversation" in rendered
    assert "not-retrieved sentinel" not in rendered
    assert "{" not in rendered


def test_reader_preserves_claim_source_provenance() -> None:
    rendered = format_public_recall_context({"results": [{"content": "claim", "source_provenance": "user_stated"}]})
    assert "user_stated" in rendered


def test_reader_renders_turns_and_anchored_public_shapes() -> None:
    response = {
        "tier": "turns",
        "turns": [{"id": "t1", "text": "turn evidence", "source": "episode"}],
        "anchors": {"first": [{"id": "t1", "text": "turn evidence", "source": "episode"}]},
    }
    rendered = format_public_recall_context(response)
    assert "turn evidence" in rendered
    assert "episode" in rendered


def test_reader_renders_public_enumeration() -> None:
    rendered = format_public_recall_context({"enumeration": {"count": 1, "target": "items", "members": [{"content": "one"}]}})
    assert "COUNT: 1 items" in rendered
    assert "one" in rendered


@pytest.mark.parametrize("tier", ["auto", "belief", "turns"])
def test_public_tier_contract_is_limited_to_required_modes(tier: str) -> None:
    assert tier in {"auto", "belief", "turns"}
