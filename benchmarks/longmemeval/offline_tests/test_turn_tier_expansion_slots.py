"""Provider-free contract tests for turn_tier_expansion_slots wiring.

Covers the round-9 harness prep:
  (a) a profile with expansion pins ``retrieval.turn_tier_expansion_slots``
      in the manifest, the faithful S36 selection refuses invalid pins, and
      the adapter runner's drift refusal reconciles manifest vs invocation;
  (b) the default profile omits the field entirely (byte-identical manifest
      semantics) and the router defaults the expansion to 0;
  (c) the router forwards the pinned value to ``recall_turns``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from benchmarks.longmemeval import router as bench_router
from benchmarks.longmemeval.faithful_s36 import (
    FaithfulRunError,
    _dataset_hash,
    _select_instances,
)
from benchmarks.longmemeval.full_s_profile import (
    FULL_S_CASE_COUNT,
    build_full_s_manifest,
    normalize_turn_tier_expansion_slots,
)


def _synthetic_source() -> list[dict]:
    return [
        {
            "question_id": f"q{i:03d}",
            "haystack_session_ids": [f"s{i}a"],
            "haystack_dates": ["2024/01/01"],
            "haystack_sessions": [[{"role": "user", "content": "seed"}]],
        }
        for i in range(FULL_S_CASE_COUNT)
    ]


def _build_manifest(tmp_path: pytest.Path, *, expansion: int = 0) -> dict:
    source = tmp_path / "source.json"
    normalized = tmp_path / "normalized.json"
    hashes = [tmp_path / "hash_one.py", tmp_path / "hash_two.py"]
    source.write_text(json.dumps(_synthetic_source()), encoding="utf-8")
    for h in hashes:
        h.write_text("# pinned source\n", encoding="utf-8")
    return build_full_s_manifest(
        source, normalized, source_hash_paths=[str(h) for h in hashes],
        turn_tier_expansion_slots=expansion,
    )


def test_profile_with_expansion_pins_manifest_field(tmp_path: pytest.Path) -> None:
    manifest = _build_manifest(tmp_path, expansion=8)
    assert manifest["retrieval"]["turn_tier_expansion_slots"] == 8


def test_default_profile_omits_field_byte_identical_semantics(
    tmp_path: pytest.Path,
) -> None:
    manifest = _build_manifest(tmp_path)
    assert "turn_tier_expansion_slots" not in manifest["retrieval"]
    assert manifest["retrieval"] == {
        "tier": "turns", "top_k": 10, "recall_k": 10, "label_blind": True,
    }


@pytest.mark.parametrize("bad", [-1, True, "3", None, 1.5])
def test_normalize_turn_tier_expansion_slots_refuses_invalid(bad) -> None:
    with pytest.raises(Exception) as excinfo:
        normalize_turn_tier_expansion_slots(bad)
    assert "turn_tier_expansion_slots" in str(excinfo.value)


def test_normalize_turn_tier_expansion_slots_accepts_zero_and_positive() -> None:
    assert normalize_turn_tier_expansion_slots(0) == 0
    assert normalize_turn_tier_expansion_slots(8) == 8


def test_s36_selection_refuses_invalid_expansion_pin(
    tmp_path: pytest.Path,
) -> None:
    manifest = _build_manifest(tmp_path, expansion=8)
    manifest["retrieval"]["turn_tier_expansion_slots"] = -1
    manifest_path = tmp_path / "full_s_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("[]", encoding="utf-8")
    # The expansion-pin refusal fires before any dataset access.
    with pytest.raises(FaithfulRunError, match="turn_tier_expansion_slots"):
        _select_instances(dataset_path, manifest_path)


def test_s36_selection_accepts_valid_expansion_pin(tmp_path: pytest.Path) -> None:
    manifest = _build_manifest(tmp_path, expansion=8)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(_synthetic_source()), encoding="utf-8")
    manifest["dataset"] = {"sha256": _dataset_hash(dataset_path)}
    manifest_path = tmp_path / "full_s_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        # The dataset records lack full Instance fields, so something later
        # may still raise — what matters is that the valid pin PASSES the
        # expansion drift refusal instead of failing on it.
        _select_instances(dataset_path, manifest_path)
    assert "turn_tier_expansion_slots" not in str(excinfo.value)


def test_adapter_resolver_refuses_manifest_invocation_drift() -> None:
    resolve = bench_router._resolve_turn_tier_expansion_slots
    assert resolve(None, 5) == 5
    assert resolve({}, 0) == 0
    assert resolve({"turn_tier_expansion_slots": 8}, 8) == 8
    with pytest.raises(ValueError, match="drift"):
        resolve({"turn_tier_expansion_slots": 8}, 5)
    with pytest.raises(ValueError, match="non-negative int"):
        resolve({"turn_tier_expansion_slots": -1}, 5)


class _Pool:
    def acquire(self):
        return _async_context(_Conn())


class _Conn:
    def transaction(self):
        return _async_context(self)


class _Embedder:
    async def embed(self, query):
        return [1.0]


class _AsyncContext:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc):
        return False


def _async_context(value):
    return _AsyncContext(value)


class _StubTurn:
    role = "user"
    occurred_at = datetime(2026, 1, 5, tzinfo=timezone.utc)
    created_at = datetime(2026, 1, 5, tzinfo=timezone.utc)
    content = "session sibling content"
    token_count = 30

    def __init__(self, turn_id: str):
        self.id = turn_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expansion_arg", "expected"),
    [(8, 8), (None, 0), (0, 0)],
)
async def test_router_forwards_pinned_expansion_slots_to_recall_turns(
    monkeypatch, expansion_arg, expected,
) -> None:
    captured: dict = {}

    async def fake_set_user_context_value(conn, user_id):
        return None

    async def fake_recall_turns(*args, **kwargs):
        captured["expansion_slots"] = kwargs.get("expansion_slots")
        return [_StubTurn("et-base")]

    monkeypatch.setattr(
        "benchmarks.longmemeval.router.set_user_context_value",
        fake_set_user_context_value,
    )
    monkeypatch.setattr(
        "benchmarks.longmemeval.router.recall_turns", fake_recall_turns,
    )

    kwargs = {}
    if expansion_arg is not None:
        kwargs["turn_tier_expansion_slots"] = expansion_arg
    await bench_router.retrieve(
        _Pool(), _Embedder(),
        question="How many visits to the shop?",
        question_type="single-session-user",
        project_id="lme_q1", tier="turns",
        **kwargs,
    )
    assert captured["expansion_slots"] == expected
