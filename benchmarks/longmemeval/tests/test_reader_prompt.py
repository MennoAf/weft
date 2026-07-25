"""Prompt-contract tests for evidence-grounded Reader behavior."""

from benchmarks.longmemeval.reader import _system_prompt_for


def test_base_prompt_requires_exact_evidence_and_scoped_aggregation() -> None:
    prompt = _system_prompt_for("multi-session")
    assert "exact evidence" in prompt
    assert "matching items first" in prompt
    assert "nearby distractors" in prompt
    assert "matching evidence is insufficient" in prompt


def test_temporal_prompt_requires_date_anchored_selection() -> None:
    prompt = _system_prompt_for("temporal-reasoning")
    assert "dates" in prompt
    assert "exact evidence" in prompt
    assert "current" in prompt


def test_preference_prompt_resolves_conflicting_states_by_date() -> None:
    prompt = _system_prompt_for("single-session-preference")
    assert "current from previous" in prompt
    assert "dates" in prompt
    assert "do not merge or average" in prompt


def test_preference_prompt_prioritizes_constraints_and_keeps_audit_internal() -> None:
    prompt = _system_prompt_for("single-session-preference")
    assert "hard constraints and avoidances first" in prompt
    assert "Never recommend something that" in prompt
    assert "Direct evidence outranks a weak analogy" in prompt
    assert "Keep this audit internal" in prompt
    assert "output must remain answer-only" in prompt


def test_preference_abs_prompt_preserves_specific_evidence_abstention() -> None:
    prompt = _system_prompt_for("single-session-preference_abs")
    assert "do not bridge a material evidence gap" in prompt
    assert "SPECIFIC detail" in prompt
    assert "respond with exactly: I don't know" in prompt
    assert "Plausibility is not evidence" in prompt


def test_abstention_prompt_rejects_related_but_wrong_evidence() -> None:
    prompt = _system_prompt_for("single-session-user_abs")
    assert "SPECIFIC detail" in prompt
    assert "related-but-different" in prompt
    assert "Plausibility is not evidence" in prompt
