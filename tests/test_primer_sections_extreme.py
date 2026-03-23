"""Extreme-condition equivalence tests for primer section builders.

These tests stress the section builders under conditions where bugs are most
likely to surface: tight budgets, heavy data loads, boundary token counts,
and competing sections.  Every test runs both the monolithic build_primer
and the section builders, asserting identical output.

These tests exist because there have been prior memory loss incidents.
The primer is the critical path for session context — it must not regress.
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
from weft.primer import build_primer
from weft.primer_sections.anti_patterns import build_anti_patterns_section
from weft.primer_sections.behaviors import build_behaviors_section
from weft.primer_sections.context import PrimerContext
from weft.primer_sections.decisions import build_decisions_section
from weft.primer_sections.entities import build_entities_section
from weft.primer_sections.grounding import build_grounding_section
from weft.primer_sections.handoff import build_handoff_section
from weft.primer_sections.issues import build_issues_section
from weft.primer_sections.recent_work import build_recent_work_section
from weft.primer_sections.rules import build_rules_section
from weft.store import store_memory


def _make_ctx(pool, *, project_id=None, budget_tokens=2400):
    return PrimerContext(
        user_id="test-user", project_id=project_id, agent_id=None,
        pool=pool, budget_tokens=budget_tokens, query=None, query_vec=None,
        disclosure="full", mode=None,
    )


async def _run_all_sections(ctx):
    """Run all budget-packed sections in the correct order, returning results."""
    results = {}
    results["grounding"] = await build_grounding_section(ctx)
    results["rules"] = await build_rules_section(ctx)
    results["behaviors"] = await build_behaviors_section(ctx)
    results["handoff"] = await build_handoff_section(ctx)
    results["recent_work"] = await build_recent_work_section(ctx)
    results["issues"] = await build_issues_section(ctx)
    results["anti_patterns"] = await build_anti_patterns_section(ctx)
    results["decisions"] = await build_decisions_section(ctx)
    results["entities"] = await build_entities_section(ctx)
    return results


def _assert_equivalence(mono, ctx, results):
    """Assert section builders match monolithic primer on all dimensions."""
    # Token totals
    assert ctx.used_tokens == mono["total_tokens"], (
        f"used_tokens mismatch: sections={ctx.used_tokens}, mono={mono['total_tokens']}"
    )
    assert ctx.section_tokens == mono["section_tokens"], (
        f"section_tokens mismatch:\n  sections={ctx.section_tokens}\n  mono={mono['section_tokens']}"
    )

    # Budget invariant
    assert ctx.used_tokens <= ctx.budget_tokens, "sections exceeded budget"
    assert mono["total_tokens"] <= mono["budget_tokens"], "mono exceeded budget"

    # Excluded count
    assert ctx.excluded == mono["excluded"], (
        f"excluded mismatch: sections={ctx.excluded}, mono={mono['excluded']}"
    )

    # Section content — item IDs must match in order
    _check_ids(results["rules"].items, mono["rules"], "rules")
    _check_ids(results["handoff"].items, mono["handoff"], "handoff")
    _check_ids(results["issues"].items, mono["issues"]["items"], "issues")
    _check_ids(results["anti_patterns"].items, mono["anti_patterns"], "anti_patterns")
    _check_ids(results["decisions"].items, mono["decisions"], "decisions")

    # Grounding
    mono_grounding = mono["grounding"]
    sect_grounding = (
        results["grounding"].items[0]["grounding_line"]
        if results["grounding"].items else None
    )
    assert sect_grounding == mono_grounding, (
        f"grounding mismatch: sections={sect_grounding!r}, mono={mono_grounding!r}"
    )

    # Entities (order may vary — compare as sets)
    mono_ent_names = {e["name"] for e in mono["entities"]}
    sect_ent_names = {e["name"] for e in results["entities"].items}
    assert sect_ent_names == mono_ent_names, (
        f"entity mismatch: sections={sect_ent_names}, mono={mono_ent_names}"
    )

    # Behaviors (compare IDs as sets since fetch order may differ)
    mono_beh_ids = {b["id"] for b in mono["behaviors"]}
    sect_beh_ids = {b["id"] for b in results["behaviors"].items}
    assert sect_beh_ids == mono_beh_ids, (
        f"behavior mismatch: sections={sect_beh_ids}, mono={mono_beh_ids}"
    )

    # Recent work
    mono_rw_ids = {m["id"] for m in mono["recent_work"]}
    sect_rw_ids = {m["id"] for m in results["recent_work"].items}
    assert sect_rw_ids == mono_rw_ids, (
        f"recent_work mismatch: sections={sect_rw_ids}, mono={mono_rw_ids}"
    )


def _check_ids(section_items, mono_items, name):
    """Assert item IDs match in order."""
    sect_ids = [i["id"] for i in section_items]
    mono_ids = [i["id"] for i in mono_items]
    assert sect_ids == mono_ids, (
        f"{name} ID mismatch:\n  sections={sect_ids}\n  mono={mono_ids}"
    )


# ---------------------------------------------------------------------------
# Fixture: heavily populate the database
# ---------------------------------------------------------------------------

@pytest.fixture
async def heavy_pool(pool):
    """Pool with lots of data — every section type has many items."""
    # Grounding
    await store_memory(pool, MemoryCreate(
        type=MemoryType.fact,
        content="Heavy test project — stress testing primer budget packing.",
        topic=["project-grounding"],
        confidence=1.0,
        project_id="heavy-proj",
    ))

    # 10 pinned rules (various confidence levels)
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content=f"Rule {i}: {'important ' * (i + 1)}guideline for the project.",
            confidence=round(0.5 + 0.05 * i, 2),
            pinned=True,
        ))

    # 3 handoffs (only newest should appear)
    for i in range(3):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content=f"## Handoff {i}\n{'Context ' * 20}for session {i}.",
            confidence=1.0,
        ))

    # 20 issues
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.issue,
            content=f"Issue {i}: something is broken in module_{i}. " + "Details. " * 5,
            confidence=round(0.6 + 0.02 * i, 2),
        ))

    # 10 anti-patterns
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.anti_pattern,
            content=f"Anti-pattern {i}: never do {'bad thing ' * 3}in context_{i}.",
            confidence=round(0.7 + 0.03 * i, 2),
        ))

    # 20 decisions (half project-scoped, half global)
    for i in range(20):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content=f"Decision {i}: we chose approach_{i} because {'reason ' * 5}.",
            confidence=round(0.6 + 0.02 * i, 2),
            project_id="heavy-proj" if i % 2 == 0 else None,
        ))

    # 10 milestones (recent work)
    for i in range(10):
        await store_memory(pool, MemoryCreate(
            type=MemoryType.milestone,
            content=f"Milestone {i}: completed feature_{i} with {'test ' * 3}coverage.",
            confidence=0.9,
        ))

    # 10 behaviors
    for i in range(10):
        await store_behavior(pool, BehaviorCreate(
            trigger_pattern=f"when doing task_{i}",
            action=f"apply strategy_{i} with {'careful ' * 2}attention",
            confidence=round(0.7 + 0.03 * i, 2),
            priority=i,
        ))

    # 15 entities
    for i in range(15):
        await store_entity(pool, EntityCreate(
            name=f"Entity-{i}",
            entity_type="tool" if i % 3 == 0 else "person" if i % 3 == 1 else "project",
            description=f"Description for entity {i} with {'some ' * 3}detail.",
        ))

    return pool


# ---------------------------------------------------------------------------
# Extreme budget tests
# ---------------------------------------------------------------------------


class TestTinyBudget:
    """Budget = 50 tokens. Barely fits grounding, everything else excluded."""

    async def test_equivalence_50_tokens(self, heavy_pool):
        budget = 50
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

    async def test_equivalence_50_tokens_with_project(self, heavy_pool):
        budget = 50
        mono = await build_primer(heavy_pool, project_id="heavy-proj",
                                  budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, project_id="heavy-proj", budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


class TestSmallBudget:
    """Budget = 200 tokens. Forces hard prioritization."""

    async def test_equivalence_200_tokens(self, heavy_pool):
        budget = 200
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

    async def test_equivalence_200_tokens_with_project(self, heavy_pool):
        budget = 200
        mono = await build_primer(heavy_pool, project_id="heavy-proj",
                                  budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, project_id="heavy-proj", budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


class TestMediumBudget:
    """Budget = 500 tokens. Some sections fit, others don't."""

    async def test_equivalence_500_tokens(self, heavy_pool):
        budget = 500
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


class TestGenerousBudget:
    """Budget = 8000 tokens. Everything should fit — test max item caps."""

    async def test_equivalence_8000_tokens(self, heavy_pool):
        budget = 8000
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

    async def test_equivalence_8000_tokens_with_project(self, heavy_pool):
        budget = 8000
        mono = await build_primer(heavy_pool, project_id="heavy-proj",
                                  budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, project_id="heavy-proj", budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


class TestBoundaryBudgets:
    """Budgets at exact section cap boundaries."""

    @pytest.mark.parametrize("budget", [100, 150, 250, 800, 1000, 1500])
    async def test_equivalence_at_boundary(self, heavy_pool, budget):
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


# ---------------------------------------------------------------------------
# Oversized handoff under tight budget
# ---------------------------------------------------------------------------


class TestOversizedHandoff:
    """Handoff larger than its cap — tests truncation path."""

    async def test_oversized_handoff_truncation(self, pool):
        """Store a 2000-token handoff with 500-token budget."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content="## Session Handoff\n" + "Important context. " * 200,
            confidence=1.0,
        ))
        # Also add a rule so the handoff has to compete
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content="Critical rule: always validate inputs.",
            confidence=1.0,
            pinned=True,
        ))

        budget = 500
        mono = await build_primer(pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

        # Handoff should be truncated, not dropped
        if mono["handoff"]:
            assert len(mono["handoff"][0]["content"]) < len("Important context. " * 200)

    async def test_very_tight_budget_with_oversized_handoff(self, pool):
        """Budget = 100 tokens, handoff is huge. Rules eat budget first."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.handoff,
            content="## Handoff\n" + "word " * 500,
            confidence=1.0,
        ))
        await store_memory(pool, MemoryCreate(
            type=MemoryType.fact,
            content="Rule one.",
            confidence=1.0,
            pinned=True,
        ))

        budget = 100
        mono = await build_primer(pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


# ---------------------------------------------------------------------------
# Duplicate prevention
# ---------------------------------------------------------------------------


class TestDuplicatePrevention:
    """Pinned memory that could appear in multiple sections."""

    async def test_pinned_decision_appears_once(self, pool):
        """A pinned decision should appear in rules, not also in decisions."""
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content="We decided to use asyncpg for all database access.",
            confidence=0.95,
            pinned=True,
        ))
        # Also add a non-pinned decision
        await store_memory(pool, MemoryCreate(
            type=MemoryType.decision,
            content="Use pytest-asyncio for async test fixtures.",
            confidence=0.8,
        ))

        budget = 4000
        mono = await build_primer(pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

        # Verify no ID appears in both rules and decisions
        rule_ids = {r["id"] for r in mono["rules"]}
        decision_ids = {d["id"] for d in mono["decisions"]}
        assert rule_ids.isdisjoint(decision_ids), "Pinned decision appeared in both sections"


# ---------------------------------------------------------------------------
# Zero budget
# ---------------------------------------------------------------------------


class TestZeroBudget:
    """Budget = 0. Nothing should be included."""

    async def test_zero_budget(self, heavy_pool):
        budget = 0
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

        assert ctx.used_tokens == 0
        assert mono["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Budget = 1 (pathological)
# ---------------------------------------------------------------------------


class TestBudgetOne:
    """Budget = 1 token. Nothing fits but nothing should crash."""

    async def test_budget_one(self, heavy_pool):
        budget = 1
        mono = await build_primer(heavy_pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


# ---------------------------------------------------------------------------
# Many items of same type with identical scores
# ---------------------------------------------------------------------------


class TestTieBreaking:
    """Items with identical scores — tie-breaking must be deterministic."""

    async def test_identical_confidence_decisions(self, pool):
        """20 decisions all with confidence=0.8 — order must match."""
        for i in range(20):
            await store_memory(pool, MemoryCreate(
                type=MemoryType.decision,
                content=f"Decision {i}: identical confidence choice.",
                confidence=0.8,
            ))

        budget = 4000
        mono = await build_primer(pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)

    async def test_identical_confidence_issues(self, pool):
        """20 issues all with same confidence — order must match."""
        for i in range(20):
            await store_memory(pool, MemoryCreate(
                type=MemoryType.issue,
                content=f"Issue {i}: same severity bug.",
                confidence=0.7,
            ))

        budget = 4000
        mono = await build_primer(pool, budget_tokens=budget, disclosure="full")
        ctx = _make_ctx(pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)
        _assert_equivalence(mono, ctx, results)


# ---------------------------------------------------------------------------
# Progressive disclosure mode under pressure
# ---------------------------------------------------------------------------


class TestProgressiveDisclosureEquivalence:
    """Progressive disclosure with heavy data."""

    async def test_progressive_heavy(self, heavy_pool):
        """Progressive mode with lots of data — tier 1 tokens must match."""
        budget = 2400
        mono = await build_primer(heavy_pool, budget_tokens=budget,
                                  disclosure="progressive")

        # For progressive, we compare tier 1 sections only
        # (tier 2 are deferred in mono, but our builders still compute them)
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)

        # Tier 1 section tokens must match
        for section in ["grounding", "rules", "handoff", "issues", "anti_patterns"]:
            assert ctx.section_tokens.get(section, 0) == mono["section_tokens"].get(section, 0), (
                f"tier 1 section '{section}' token mismatch: "
                f"sections={ctx.section_tokens.get(section, 0)}, "
                f"mono={mono['section_tokens'].get(section, 0)}"
            )

    async def test_progressive_tiny(self, heavy_pool):
        """Progressive mode with tiny budget."""
        budget = 100
        mono = await build_primer(heavy_pool, budget_tokens=budget,
                                  disclosure="progressive")
        ctx = _make_ctx(heavy_pool, budget_tokens=budget)
        results = await _run_all_sections(ctx)

        for section in ["grounding", "rules", "handoff", "issues", "anti_patterns"]:
            assert ctx.section_tokens.get(section, 0) == mono["section_tokens"].get(section, 0)
