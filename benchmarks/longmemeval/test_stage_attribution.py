"""Hermetic tests for benchmark-local LongMemEval stage attribution.

This module lives beside the benchmark package, not under ``tests/``: the
nested benchmark conftest imports shared fixtures that start Postgres/Redis
containers. These tests use only the pure evaluator and in-memory literals.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from benchmarks.longmemeval.dataset import Session, Turn
from benchmarks.longmemeval.stage_attribution import (
    INPUT_SCHEMA,
    REPORT_SCHEMA,
    evaluate_stage_attribution,
)


_SYNTHETIC_SOURCE = "My commute is 45 minutes each way."
_FACT = "45 minutes each way"


def _input() -> dict[str, object]:
    """Return manually annotated synthetic evidence, never a dataset gold answer."""
    source_session = Session(
        session_id="synthetic-session-1",
        date="2024-01-01",
        turns=(Turn(role="user", content=_SYNTHETIC_SOURCE),),
        has_answer=True,
    )
    source_turn = source_session.turns[0]
    return {
        "schema": INPUT_SCHEMA,
        "target": {
            "fact": _FACT,
            "source_turn": {
                "session_id": source_session.session_id,
                "turn_index": 0,
                "role": source_turn.role,
                "content": source_turn.content,
            },
            "source_basis": "manual_source_review",
        },
        "stages": {
            "writer_selection": {
                "captured": True,
                "contents": ["Commute: 45 minutes each way, as of 2024-01."],
            },
            "persisted_memory_readback": {
                "captured": True,
                "verified_after_save": True,
                "contents": ["Commute: 45 minutes each way, as of 2024-01."],
            },
            "retrieved_memory_evidence": {
                "captured": True,
                "contents": ["Commute is 45 minutes each way."],
            },
            "final_answer": {"captured": True, "text": "45 minutes each way"},
            "judge": {"captured": True, "label": True},
        },
    }


def test_complete_source_grounded_chain_reports_coverage_without_raw_text() -> None:
    report = evaluate_stage_attribution(_input()).to_dict()

    assert report["schema"] == REPORT_SCHEMA
    assert report["attribution"] == "exact_evidence_chain_judge_accepted"
    stages = report["stages"]
    assert stages["writer_selection"] == {
        "captured": True,
        "status": "present",
        "item_count": 1,
        "matching_item_indexes": [0],
        "verified_after_save": None,
    }
    assert stages["persisted_memory_contents"] == {
        "captured": True,
        "status": "present",
        "item_count": 1,
        "matching_item_indexes": [0],
        "verified_after_save": True,
    }
    assert stages["retrieved_memory_evidence"]["status"] == "present"
    assert stages["final_answer"]["status"] == "present"
    serialized = json.dumps(report, sort_keys=True)
    assert _FACT not in serialized
    assert _SYNTHETIC_SOURCE not in serialized
    assert "Commute:" not in serialized


def test_writer_selection_omission_is_distinct_from_persistence_gap() -> None:
    data = _input()
    data["stages"]["writer_selection"]["contents"] = []

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "writer_selection_miss"
    assert report["stages"]["writer_selection"]["captured"] is True
    assert report["stages"]["writer_selection"]["status"] == "missing"
    # Later-stage evidence cannot erase the first observed writer omission.
    assert report["stages"]["persisted_memory_contents"]["status"] == "present"


def test_writer_output_followed_by_missing_verified_readback_is_persistence_gap() -> None:
    data = _input()
    data["stages"]["persisted_memory_readback"]["contents"] = ["A different saved memory."]

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "persistence_gap"
    assert report["stages"]["writer_selection"]["status"] == "present"
    assert report["stages"]["persisted_memory_contents"] == {
        "captured": True,
        "status": "missing",
        "item_count": 1,
        "matching_item_indexes": [],
        "verified_after_save": True,
    }


def test_exact_fact_requires_complete_value_unit_and_qualifier() -> None:
    data = _input()
    data["stages"]["writer_selection"]["contents"] = ["Commute is 45 minutes."]

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "writer_selection_miss"
    assert report["stages"]["writer_selection"]["status"] == "missing"
    # Literal matching deliberately does not infer equivalent-looking evidence.
    assert report["stages"]["persisted_memory_contents"]["status"] == "present"


def test_retrieval_miss_is_distinct_from_writer_or_persistence_gap() -> None:
    data = _input()
    data["stages"]["retrieved_memory_evidence"]["contents"] = ["A different memory."]

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "retrieval_miss"
    assert report["stages"]["writer_selection"]["status"] == "present"
    assert report["stages"]["persisted_memory_contents"]["status"] == "present"
    assert report["stages"]["retrieved_memory_evidence"]["status"] == "missing"
    assert report["stages"]["final_answer"]["status"] == "present"


def test_reader_miss_is_distinct_from_retrieval_miss() -> None:
    data = _input()
    data["stages"]["final_answer"]["text"] = "I don't know."

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "reader_miss"
    assert report["stages"]["retrieved_memory_evidence"]["status"] == "present"
    assert report["stages"]["final_answer"]["status"] == "missing"


def test_judge_disagreement_is_distinct_from_answer_coverage() -> None:
    data = _input()
    data["stages"]["judge"]["label"] = False

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "judge_rejected_exactly_covered_answer"
    assert report["stages"]["final_answer"]["status"] == "present"
    assert report["stages"]["judge"]["status"] == "rejected"


def test_missing_capture_is_not_reported_as_negative_observation() -> None:
    data = _input()
    data["stages"]["writer_selection"] = {"captured": False, "contents": None}

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "missing_writer_selection_capture"
    assert report["stages"]["writer_selection"] == {
        "captured": False,
        "status": "not_captured",
        "item_count": None,
        "matching_item_indexes": [],
        "verified_after_save": None,
    }

    data = _input()
    data["stages"]["persisted_memory_readback"] = {
        "captured": False,
        "verified_after_save": False,
        "contents": None,
    }
    report = evaluate_stage_attribution(data).to_dict()
    assert report["attribution"] == "missing_persisted_readback"
    assert report["stages"]["persisted_memory_contents"] == {
        "captured": False,
        "status": "not_captured",
        "item_count": None,
        "matching_item_indexes": [],
        "verified_after_save": False,
    }

    data = _input()
    data["stages"]["persisted_memory_readback"]["verified_after_save"] = False
    report = evaluate_stage_attribution(data).to_dict()
    assert report["attribution"] == "missing_persisted_readback_verification"
    assert report["stages"]["persisted_memory_contents"]["status"] == "unverified"

    data = _input()
    data["stages"]["retrieved_memory_evidence"] = {"captured": False, "contents": None}
    report = evaluate_stage_attribution(data).to_dict()
    assert report["attribution"] == "missing_retrieval_evidence_capture"
    assert report["stages"]["retrieved_memory_evidence"]["status"] == "not_captured"


def test_source_annotation_must_be_explicit_and_point_to_haystack_turn() -> None:
    data = _input()
    data["target"]["source_basis"] = "gold_answer"
    with pytest.raises(ValueError, match="source_basis"):
        evaluate_stage_attribution(data)

    data = _input()
    data["target"]["source_turn"]["role"] = "assistant"
    with pytest.raises(ValueError, match="role must be 'user'"):
        evaluate_stage_attribution(data)

    data = _input()
    data["target"]["source_turn"]["content"] = "No commute fact appears in this source turn."
    with pytest.raises(ValueError, match="source_turn.content"):
        evaluate_stage_attribution(data)


def test_strict_contract_rejects_unknown_fields_and_unbounded_captures() -> None:
    data = _input()
    data["answer"] = "45 minutes each way"  # Gold/reference-like extra field is rejected.
    with pytest.raises(ValueError, match="unknown keys"):
        evaluate_stage_attribution(data)

    data = _input()
    data["stages"]["writer_selection"]["contents"] = ["x"] * 33
    with pytest.raises(ValueError, match="exceeds 32 items"):
        evaluate_stage_attribution(data)


def test_uncaptured_answer_and_judge_are_not_negative_observations() -> None:
    data = _input()
    data["stages"]["final_answer"] = {"captured": False, "text": None}

    report = evaluate_stage_attribution(data).to_dict()

    assert report["attribution"] == "missing_final_answer_capture"
    assert report["stages"]["final_answer"] == {
        "captured": False,
        "status": "not_captured",
        "item_count": None,
        "matching_item_indexes": [],
        "verified_after_save": None,
    }

    data = _input()
    data["stages"]["judge"] = {"captured": False, "label": None}
    report = evaluate_stage_attribution(data).to_dict()
    assert report["attribution"] == "missing_judge_label"
    assert report["stages"]["judge"]["captured"] is False
    assert report["stages"]["judge"]["status"] == "not_captured"


def test_evaluator_is_deterministic_and_does_not_mutate_input() -> None:
    data = _input()
    before = deepcopy(data)

    first = evaluate_stage_attribution(data).to_dict()
    second = evaluate_stage_attribution(data).to_dict()

    assert first == second
    assert data == before
    assert json.loads(json.dumps(first, sort_keys=True)) == first
