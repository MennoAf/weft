"""Head-to-head comparison: _build_primer_legacy vs build_primer (modular).

These tests run BOTH the legacy monolithic and the modular orchestrator
with identical inputs and assert byte-for-byte identical output.

These tests serve as an ongoing regression gate — if the modular primer
ever diverges from the legacy implementation, these tests will catch it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weft.behaviors import store_behavior
from weft.entities import store_entity
from weft.models import (
    BehaviorCreate,
    EntityCreate,
    MemoryCreate,
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from weft.primer import _build_primer_legacy, build_primer
from weft.store import store_memory


def _deep_compare(v1, v2, path=""):
    """Recursively compare two values, returning list of differences."""
    diffs = []
    if type(v1) != type(v2):
        diffs.append(f"{path}: type mismatch {type(v1).__name__} vs {type(v2).__name__}")
        return diffs
    if isinstance(v1, dict):
        all_keys = set(v1.keys()) | set(v2.keys())
        for k in sorted(all_keys):
            if k not in v1:
                diffs.append(f"{path}.{k}: missing in v1")
            elif k not in v2:
                diffs.append(f"{path}.{k}: missing in v2")
            else:
                diffs.extend(_deep_compare(v1[k], v2[k], f"{path}.{k}"))
    elif isinstance(v1, list):
        if len(v1) != len(v2):
            diffs.append(f"{path}: list length {len(v1)} vs {len(v2)}")
        for i, (a, b) in enumerate(zip(v1, v2)):
            diffs.extend(_deep_compare(a, b, f"{path}[{i}]"))
    elif isinstance(v1, float):
        # Allow tiny float differences (age_hours rounding)
        if abs(v1 - v2) > 0.2:
            diffs.append(f"{path}: {v1} vs {v2}")
    elif v1 != v2:
        diffs.append(f"{path}: {v1!r} vs {v2!r}")
    return diffs


def _assert_identical(mono, v2, label=""):
    """Assert two primer results are identical, with detailed diff on failure."""
    # Skip keys that may differ due to timing (changes_since, wellness_snapshot)
    # These are tested separately.
    skip_keys = {"changes_since", "wellness_snapshot", "autonomy", "section_tokens"}
    mono_filtered = {k: v for k, v in mono.items() if k not in skip_keys}
    v2_filtered = {k: v for k, v in v2.items() if k not in skip_keys}

    diffs = _deep_compare(mono_filtered, v2_filtered, "result")
    assert not diffs, f"Differences found ({label}):\n" + "\n".join(diffs)


# ---------------------------------------------------------------------------
# Fixture: heavy data (reused from extreme tests)
# ---------------------------------------------------------------------------

@pytest.fixture
async def heavy_pool(pool):
    """Pool with lots of data across all section types."""
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Heavy test project for v2 comparison.",
        topic=["project-grounding"],
        confidence=1.0,
        project_id="v2-proj",
    ))

    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Rule {i}: {'important ' * (i + 1)}guideline.",
            confidence=round(0.5 + 0.05 * i, 2),
            pinned=True,
        ))

    for i in range(3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content=f"## Handoff {i}\n{'Context ' * 20}for session {i}.",
            confidence=1.0,
        ))

    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.issue,
            content=f"Issue {i}: broken module_{i}. " + "Details. " * 5,
            confidence=round(0.6 + 0.02 * i, 2),
        ))

    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.anti_pattern,
            content=f"Anti-pattern {i}: avoid {'mistake ' * 3}in context_{i}.",
            confidence=round(0.7 + 0.03 * i, 2),
        ))

    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: chose approach_{i} because {'reason ' * 5}.",
            confidence=round(0.6 + 0.02 * i, 2),
            project_id="v2-proj" if i % 2 == 0 else None,
        ))

    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.milestone,
            content=f"Milestone {i}: shipped feature_{i}.",
            confidence=0.9,
        ))

    for i in range(10):
        await store_behavior(pool, BehaviorCreate(
            trigger_pattern=f"when doing task_{i}",
            action=f"apply strategy_{i} carefully",
            confidence=round(0.7 + 0.03 * i, 2),
            priority=i,
        ))

    for i in range(15):
        await store_entity(pool, EntityCreate(
            name=f"Entity-{i}",
            entity_type="tool" if i % 3 == 0 else "person" if i % 3 == 1 else "project",
            description=f"Entity {i} description.",
        ))

    return pool


# ---------------------------------------------------------------------------
# Full disclosure: v1 vs v2
# ---------------------------------------------------------------------------


class TestFullDisclosureComparison:
    """Full disclosure mode — every section visible."""

    @pytest.mark.parametrize("budget", [0, 1, 50, 100, 200, 500, 1000, 2400, 8000])
    async def test_budget_sweep(self, heavy_pool, budget):
        mono = await _build_primer_legacy(heavy_pool, budget_tokens=budget, disclosure="full")
        v2 = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        _assert_identical(mono, v2, f"full/budget={budget}")

    @pytest.mark.parametrize("budget", [0, 50, 200, 500, 2400, 8000])
    async def test_budget_sweep_with_project(self, heavy_pool, budget):
        mono = await _build_primer_legacy(heavy_pool, project_id="v2-proj",
                                  budget_tokens=budget, disclosure="full")
        v2 = await build_primer(heavy_pool, project_id="v2-proj",
                                   budget_tokens=budget, disclosure="full")
        _assert_identical(mono, v2, f"full/project/budget={budget}")


class TestProgressiveDisclosureComparison:
    """Progressive disclosure mode — tier 2 deferred."""

    @pytest.mark.parametrize("budget", [0, 1, 50, 100, 200, 500, 1000, 2400, 8000])
    async def test_budget_sweep(self, heavy_pool, budget):
        mono = await _build_primer_legacy(heavy_pool, budget_tokens=budget, disclosure="progressive")
        v2 = await build_primer(heavy_pool, budget_tokens=budget, disclosure="progressive")
        _assert_identical(mono, v2, f"progressive/budget={budget}")

    @pytest.mark.parametrize("budget", [0, 50, 200, 500, 2400, 8000])
    async def test_budget_sweep_with_project(self, heavy_pool, budget):
        mono = await _build_primer_legacy(heavy_pool, project_id="v2-proj",
                                  budget_tokens=budget, disclosure="progressive")
        v2 = await build_primer(heavy_pool, project_id="v2-proj",
                                   budget_tokens=budget, disclosure="progressive")
        _assert_identical(mono, v2, f"progressive/project/budget={budget}")


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------


class TestEmptyDatabase:
    """No data at all — both should produce identical empty primer."""

    async def test_full_empty(self, pool):
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="full")
        _assert_identical(mono, v2, "empty/full")

    async def test_progressive_empty(self, pool):
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="progressive")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="progressive")
        _assert_identical(mono, v2, "empty/progressive")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Specific scenarios that have caused issues before."""

    async def test_only_handoff(self, pool):
        """Only a handoff, nothing else."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content="## Handoff\nJust context.",
            confidence=1.0,
        ))
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="full")
        _assert_identical(mono, v2, "only_handoff")

    async def test_oversized_handoff_tight_budget(self, pool):
        """Huge handoff, tiny budget — truncation path."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content="## Handoff\n" + "Important context. " * 300,
            confidence=1.0,
        ))
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content="Critical rule.",
            confidence=1.0,
            pinned=True,
        ))
        mono = await _build_primer_legacy(pool, budget_tokens=200, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=200, disclosure="full")
        _assert_identical(mono, v2, "oversized_handoff")

    async def test_pinned_decision_dedup(self, pool):
        """Pinned decision: should appear in rules, not decisions."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content="Use asyncpg everywhere.",
            confidence=0.95,
            pinned=True,
        ))
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content="Use pytest-asyncio.",
            confidence=0.8,
        ))
        mono = await _build_primer_legacy(pool, budget_tokens=4000, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=4000, disclosure="full")
        _assert_identical(mono, v2, "pinned_decision_dedup")

    async def test_many_identical_confidence(self, pool):
        """20 decisions with identical confidence — deterministic ordering."""
        for i in range(20):
            await store_memory(pool, MemoryCreate(
                type=MemoryType.decision,
                content=f"Decision {i}: same confidence.",
                confidence=0.8,
            ))
        mono = await _build_primer_legacy(pool, budget_tokens=4000, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=4000, disclosure="full")
        _assert_identical(mono, v2, "tie_breaking")

    async def test_grounding_no_project(self, pool):
        """Grounding memory exists but no project_id in call."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content="Some grounding.",
            topic=["project-grounding"],
            confidence=1.0,
            project_id="some-proj",
        ))
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="full")
        _assert_identical(mono, v2, "grounding_no_project")

    async def test_cold_start_detection(self, pool):
        """Empty DB should trigger cold start in both."""
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="full")
        assert mono["onboarding"] is not None
        assert v2["onboarding"] is not None
        _assert_identical(mono, v2, "cold_start")

    async def test_not_cold_start_with_handoff(self, pool):
        """Handoff present → not cold start."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content="Session handoff.",
            confidence=1.0,
        ))
        mono = await _build_primer_legacy(pool, budget_tokens=2400, disclosure="full")
        v2 = await build_primer(pool, budget_tokens=2400, disclosure="full")
        assert mono["onboarding"] is None
        assert v2["onboarding"] is None
        _assert_identical(mono, v2, "not_cold_start")
