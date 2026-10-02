from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmarks.longmemeval.dataset import Instance, Session, Turn
from benchmarks.longmemeval.ingest import expected_turn_count
from benchmarks.longmemeval.task_shape import derive_task_shape, routing_class_for
from weft.text_generation import GenerationResponse


def _instance() -> Instance:
    return Instance(
        question_id="q1",
        question_type="single-session-user",
        question="What color was the bicycle?",
        answer="blue",
        question_date="2024-01-01",
        sessions=(
            Session(
                "s1",
                "2023/12/01",
                (Turn("user", "The bicycle was blue."),),
                has_answer=True,
            ),
            Session(
                "s2",
                "2023/12/02",
                (Turn("assistant", "Unrelated."),),
                has_answer=False,
            ),
        ),
    )


def test_unshimmed_adapter_import_and_expected_turn_count() -> None:
    from benchmarks.longmemeval import adapter

    instance = replace(
        _instance(),
        sessions=(
            Session(
                "s1",
                "2023/12/01",
                (Turn("user", "u"), Turn("unknown", "skip"), Turn("tool", "t")),
            ),
        ),
    )
    assert adapter.expected_turn_count(instance) == 2
    assert expected_turn_count(instance) == 2


def test_task_shape_is_invariant_under_gold_metadata_mutation() -> None:
    original = _instance()
    mutated = replace(
        original,
        question_type="temporal-reasoning_abs",
        answer="a different answer",
        sessions=tuple(replace(s, has_answer=not s.has_answer) for s in original.sessions),
    )
    assert derive_task_shape(original.question, original.sessions) == derive_task_shape(
        mutated.question, mutated.sessions
    )
    assert routing_class_for(original.question, original.sessions) == routing_class_for(
        mutated.question, mutated.sessions
    )


@pytest.mark.asyncio
async def test_router_uses_shape_and_preserves_candidate_order(monkeypatch) -> None:
    import benchmarks.longmemeval.router as router

    calls: list[dict] = []
    candidate_ids = ("c1", "c2", "c3")

    async def fake_turns(*args, **kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(memory=SimpleNamespace(id=value)) for value in candidate_ids]

    async def fail_belief(*args, **kwargs):
        raise AssertionError("unexpected belief fallback")

    monkeypatch.setattr(router, "_retrieve_turns", fake_turns)
    monkeypatch.setattr(router, "_retrieve_belief", fail_belief)
    original = _instance()
    mutated = replace(original, question_type="multi-session_abs", answer="changed")
    shape_a = derive_task_shape(original.question, original.sessions)
    shape_b = derive_task_shape(mutated.question, mutated.sessions)
    results_a = await router.retrieve(None, None, question=original.question, project_id="p", task_shape=shape_a, tier="turns")
    results_b = await router.retrieve(None, None, question=mutated.question, project_id="p", task_shape=shape_b, tier="turns")
    assert calls[0]["policy"] == calls[1]["policy"]
    assert [result.memory.id for result in results_a] == list(candidate_ids)
    assert [result.memory.id for result in results_a] == [result.memory.id for result in results_b]


class SpyProvider:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(text="stable answer", model=request.model, input_tokens=1, output_tokens=2)


@pytest.mark.asyncio
async def test_reader_injected_provider_is_label_blind(monkeypatch) -> None:
    import benchmarks.longmemeval.reader as reader_module

    provider = SpyProvider()
    monkeypatch.setattr(reader_module, "AsyncAnthropic", lambda: (_ for _ in ()).throw(AssertionError("SDK constructed")))
    reader = reader_module.Reader(provider=provider, model="fake-model", abstention_model="other-model")
    first = await reader.read_answer(
        question="What color was the bicycle?", question_date="2024-01-01",
        question_type="single-session-user", task_shape="single-session", memories=[],
    )
    second = await reader.read_answer(
        question="What color was the bicycle?", question_date="2024-01-01",
        question_type="temporal-reasoning_abs", task_shape="single-session", memories=[],
    )
    assert first == second
    assert [request.model for request in provider.requests] == ["fake-model", "fake-model"]
    assert provider.requests[0].system == provider.requests[1].system


def _raw_instance() -> dict:
    return {
        "question_id": "q1", "question_type": "single-session-user", "question": "q",
        "answer": "a", "question_date": "2024-01-01", "haystack_session_ids": ["s1"],
        "haystack_dates": ["2024/01/01"], "haystack_sessions": [[{"role": "user", "content": "x"}]],
    }


@pytest.mark.parametrize("malformed", ["", 0, False, {}, "s1"])
def test_dataset_rejects_malformed_present_answer_ids(malformed) -> None:
    value = _raw_instance()
    value["answer_session_ids"] = malformed
    with pytest.raises(ValueError):
        Instance.from_dict(value)


def test_dataset_rejects_duplicate_unknown_and_non_string_ids() -> None:
    for ids, message in [(["s1", "s1"], "duplicate"), (["unknown"], "unknown"), ([1], "strings")]:
        value = _raw_instance()
        value["answer_session_ids"] = ids
        with pytest.raises(ValueError, match=message):
            Instance.from_dict(value)


def test_missing_answer_ids_default_to_no_evidence() -> None:
    instance = Instance.from_dict(_raw_instance())
    assert all(not session.has_answer for session in instance.sessions)


def test_summary_counts_missing_records_in_full_denominator() -> None:
    from benchmarks.longmemeval.adapter import summarize_recall_records

    summary = summarize_recall_records(
        [{"question_id": "q1", "question_type": "single", "recall_at_k_hit": True}],
        requested_question_ids=["q1", "q2"],
    )
    assert summary["n_questions"] == 2
    assert summary["n_hits"] == 1
    assert summary["recall_at_k"] == 0.5
