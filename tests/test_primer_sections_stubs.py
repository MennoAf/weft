"""Structural and contractual tests for the primer_sections stub package.

These tests verify that the stub directory, dataclasses, and section builder
functions have the correct shape BEFORE any logic is implemented.  They are
the acceptance criteria for the design task (loom-f1b6f106).
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. PrimerContext instantiation
# ---------------------------------------------------------------------------


class TestPrimerContext:
    """Verify PrimerContext dataclass can be created and validates inputs."""

    def test_instantiate_with_required_fields(self):
        from weft.primer_sections.context import PrimerContext

        pool = MagicMock()
        ctx = PrimerContext(
            user_id="user-1",
            project_id="proj-1",
            agent_id=None,
            pool=pool,
            budget_tokens=2400,
            query=None,
            query_vec=None,
            disclosure="progressive",
            mode=None,
        )
        assert ctx.user_id == "user-1"
        assert ctx.project_id == "proj-1"
        assert ctx.pool is pool
        assert ctx.budget_tokens == 2400

    def test_query_without_query_vec_raises(self):
        from weft.primer_sections.context import PrimerContext

        pool = MagicMock()
        with pytest.raises(ValueError, match="query_vec"):
            PrimerContext(
                user_id="user-1",
                project_id=None,
                agent_id=None,
                pool=pool,
                budget_tokens=2400,
                query="something",
                query_vec=None,  # query set but no vector
                disclosure="progressive",
                mode=None,
            )

    def test_query_vec_without_query_is_ok(self):
        """query_vec without query is valid (pre-computed vector)."""
        from weft.primer_sections.context import PrimerContext

        pool = MagicMock()
        ctx = PrimerContext(
            user_id="user-1",
            project_id=None,
            agent_id=None,
            pool=pool,
            budget_tokens=2400,
            query=None,
            query_vec=[0.1, 0.2],
            disclosure="progressive",
            mode=None,
        )
        assert ctx.query_vec is not None

    def test_biased_property(self):
        from weft.primer_sections.context import PrimerContext

        pool = MagicMock()
        ctx_unbiased = PrimerContext(
            user_id="u", project_id=None, agent_id=None, pool=pool,
            budget_tokens=2400, query=None, query_vec=None,
            disclosure="full", mode=None,
        )
        assert ctx_unbiased.biased is False

        ctx_biased = PrimerContext(
            user_id="u", project_id=None, agent_id=None, pool=pool,
            budget_tokens=2400, query=None, query_vec=[0.1],
            disclosure="full", mode=None,
        )
        assert ctx_biased.biased is True


# ---------------------------------------------------------------------------
# 2. SectionResult shape
# ---------------------------------------------------------------------------


class TestSectionResult:
    """Verify SectionResult has the expected fields."""

    def test_instantiate(self):
        from weft.primer_sections.context import SectionResult

        r = SectionResult(items=[], tokens_used=0, skipped=False, skip_reason=None)
        assert r.tokens_used == 0
        assert r.skipped is False

    def test_skipped_result(self):
        from weft.primer_sections.context import SectionResult

        r = SectionResult(items=[], tokens_used=0, skipped=True, skip_reason="no project_id")
        assert r.skipped is True
        assert r.skip_reason == "no project_id"


# ---------------------------------------------------------------------------
# 3. SECTION_BUDGETS completeness
# ---------------------------------------------------------------------------


class TestSectionBudgets:
    """Verify SECTION_BUDGETS covers all expected sections."""

    EXPECTED_SECTIONS = {
        "grounding", "rules", "behaviors", "handoff", "recent_work",
        "issues", "anti_patterns", "decisions", "entities",
    }

    def test_all_sections_have_budgets(self):
        from weft.primer_sections.context import SECTION_BUDGETS

        assert self.EXPECTED_SECTIONS <= set(SECTION_BUDGETS.keys())

    def test_budgets_are_positive_ints(self):
        from weft.primer_sections.context import SECTION_BUDGETS

        for name, budget in SECTION_BUDGETS.items():
            assert isinstance(budget, int), f"{name} budget is not int"
            assert budget > 0, f"{name} budget is not positive"


# ---------------------------------------------------------------------------
# 4. Section builder stub signatures
# ---------------------------------------------------------------------------

# Map of module name → expected async function name
_SECTION_BUILDERS = {
    "weft.primer_sections.grounding": "build_grounding_section",
    "weft.primer_sections.rules": "build_rules_section",
    "weft.primer_sections.behaviors": "build_behaviors_section",
    "weft.primer_sections.handoff": "build_handoff_section",
    "weft.primer_sections.recent_work": "build_recent_work_section",
    "weft.primer_sections.issues": "build_issues_section",
    "weft.primer_sections.anti_patterns": "build_anti_patterns_section",
    "weft.primer_sections.decisions": "build_decisions_section",
    "weft.primer_sections.entities": "build_entities_section",
    "weft.primer_sections.changes_since": "build_changes_since_section",
    "weft.primer_sections.wellness": "build_wellness_section",
    "weft.primer_sections.onboarding": "build_onboarding_section",
    "weft.primer_sections.disclosure": "apply_progressive_disclosure",
}


class TestSectionBuilderSignatures:
    """Every section builder must be async and accept PrimerContext → SectionResult."""

    @pytest.mark.parametrize("module_path,func_name", _SECTION_BUILDERS.items())
    def test_function_exists_and_is_async(self, module_path, func_name):
        import importlib

        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        assert inspect.iscoroutinefunction(func), f"{func_name} is not async"

    @pytest.mark.parametrize("module_path,func_name", _SECTION_BUILDERS.items())
    def test_function_has_ctx_parameter(self, module_path, func_name):
        import importlib

        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        # First positional parameter should be ctx
        assert len(params) >= 1, f"{func_name} has no parameters"
        assert params[0].name == "ctx", f"{func_name} first param is not 'ctx'"


# ---------------------------------------------------------------------------
# 5. Smoke test: stubs raise NotImplementedError
# ---------------------------------------------------------------------------

# Sections that have been implemented (no longer stubs).
# Update this set as sections are implemented.
_IMPLEMENTED_SECTIONS = set(_SECTION_BUILDERS.keys())  # All sections now implemented

_STUB_ONLY_BUILDERS = {
    k: v for k, v in _SECTION_BUILDERS.items()
    if k not in _IMPLEMENTED_SECTIONS
}


class TestSectionBuilderStubsRaise:
    """Each unimplemented stub must raise NotImplementedError when called."""

    @pytest.mark.parametrize("module_path,func_name", _STUB_ONLY_BUILDERS.items())
    @pytest.mark.asyncio
    async def test_stub_raises_not_implemented(self, module_path, func_name):
        import importlib
        from weft.primer_sections.context import PrimerContext

        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)

        pool = MagicMock()
        ctx = PrimerContext(
            user_id="test", project_id=None, agent_id=None, pool=pool,
            budget_tokens=2400, query=None, query_vec=None,
            disclosure="full", mode=None,
        )
        with pytest.raises(NotImplementedError):
            await func(ctx)


# ---------------------------------------------------------------------------
# 6. __init__.py re-exports
# ---------------------------------------------------------------------------


class TestInitReexports:
    """The package __init__ must re-export PrimerContext, SectionResult, and all builders."""

    def test_primer_context_importable(self):
        from weft.primer_sections import PrimerContext  # noqa: F401

    def test_section_result_importable(self):
        from weft.primer_sections import SectionResult  # noqa: F401

    @pytest.mark.parametrize("func_name", _SECTION_BUILDERS.values())
    def test_builder_importable_from_package(self, func_name):
        import weft.primer_sections as pkg

        assert hasattr(pkg, func_name), f"{func_name} not re-exported from __init__.py"
