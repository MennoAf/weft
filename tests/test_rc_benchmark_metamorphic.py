"""RC-FL-17 deterministic, provider-free benchmark metamorphic checks.

The runtime is run twice with identical permitted inputs while only evaluation
(gold) metadata changes. A trace may differ only in post-hoc scoring data;
shape, routing, retrieval, Reader requests, and output contracts must not.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from benchmarks.longmemeval.adapter import summarize_recall_records
from benchmarks.longmemeval.dataset import Instance, Session, Turn
from benchmarks.longmemeval.reader import Reader
from benchmarks.longmemeval.router import retrieve
from benchmarks.longmemeval.task_shape import derive_task_shape
from weft.models import Memory, MemoryRecall, MemorySource, MemoryStatus, MemoryType
from weft.text_generation import GenerationResponse


class SpyProvider:
    """Deterministic generation provider; it never constructs an SDK/client."""

    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return GenerationResponse(
            text="stable answer",
            model=request.model,
            input_tokens=3,
            output_tokens=2,
        )


def _instance(**gold: object) -> Instance:
    data = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What color was the bicycle?",
        "answer": "blue",
        "question_date": "2024-01-01",
        "sessions": (
            Session("s1", "2023/01/01", (Turn("user", "The bicycle was blue."),)),
            Session("s2", "2023/01/02", (Turn("user", "An unrelated note."),)),
        ),
    }
    data.update(gold)
    return Instance(**data)


@pytest.mark.asyncio
async def test_gold_mutation_keeps_full_runtime_trace_identical(monkeypatch):
    """Changing every gold field cannot influence the label-blind runtime."""
    import benchmarks.longmemeval.router as router

    candidate_ids = ("c1", "c2", "c3")
    route_calls: list[dict] = []

    async def fake_turns(*args, **kwargs):
        route_calls.append(
            {
                "question": kwargs["question"],
                "question_type": kwargs["question_type"],
                "top_k": kwargs["policy"].top_k,
                "tier": "turns",
            }
        )
        return [
            MemoryRecall(
                memory=Memory(
                    id=cid,
                    type=MemoryType.fact,
                    content=f"candidate content {cid}",
                    source=MemorySource.conversation,
                    project_id="p",
                    status=MemoryStatus.active,
                    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    accessed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                ),
                similarity=0.9 - (index * 0.1),
            )
            for index, cid in enumerate(candidate_ids)
        ]

    async def forbidden_belief(*args, **kwargs):
        raise AssertionError("belief fallback is not part of this deterministic trace")

    monkeypatch.setattr(router, "_retrieve_turns", fake_turns)
    monkeypatch.setattr(router, "_retrieve_belief", forbidden_belief)

    original = _instance()
    mutated = replace(
        original,
        question_type="temporal-reasoning_abs",
        answer="a different gold answer",
        sessions=tuple(
            replace(session, has_answer=not session.has_answer)
            for session in original.sessions
        ),
    )

    async def run(instance: Instance) -> dict:
        shape = derive_task_shape(instance.question, instance.sessions)
        provider = SpyProvider()
        reader = Reader(provider=provider, model="fake-reader")
        retrieved = await retrieve(
            None,
            None,
            question=instance.question,
            project_id="p",
            task_shape=shape,
            tier="turns",
        )
        # Gold question_type is deliberately absent from the runtime boundary.
        # Reader behavior is selected by the non-gold shape only.
        response = await reader.read_answer(
            question=instance.question,
            question_date=instance.question_date,
            task_shape=shape,
            memories=retrieved,
        )
        request = provider.requests[0]
        expected_memory_text = "\n\n".join(
            f"[{index}] (relevance={0.9 - ((index - 1) * 0.1):.2f}) "
            f"candidate content {cid}"
            for index, cid in enumerate(candidate_ids, start=1)
        )
        assert request.messages[0]["content"].endswith(
            f"Memories:\n{expected_memory_text}"
        )
        return {
            "task_shape": shape,
            "route": {
                "routing_class": shape.routing_class,
                "router_question_type": route_calls[-1]["question_type"],
                "top_k": route_calls[-1]["top_k"],
                "tier": route_calls[-1]["tier"],
            },
            "candidate_ids": tuple(item.memory.id for item in retrieved),
            "prompt": {"system": request.system, "messages": request.messages},
            "provider_model_sequence": (type(provider).__name__, request.model),
            "output_contract": {
                "hypothesis": response.hypothesis,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "cached_input_tokens": response.cached_input_tokens,
                "output_tokens": response.output_tokens,
            },
        }

    trace_a = await run(original)
    trace_b = await run(mutated)
    assert trace_a == trace_b
    assert trace_a["candidate_ids"] == candidate_ids
    assert [call["question_type"] for call in route_calls] == [
        "single-session-user",
        "single-session-user",
    ]


def test_label_blind_shape_rejects_gold_metadata_at_runtime_boundary():
    """A caller cannot smuggle gold labels into the shape seam."""
    with pytest.raises(ValueError, match="gold metadata"):
        derive_task_shape(
            "What color was the bicycle?",
            metadata={"question_type": "multi-session"},
        )


def test_existing_recall_summary_preserves_missing_requested_denominator():
    """The recall sidecar also keeps missing questions as denominator misses."""
    summary = summarize_recall_records(
        [{"question_id": "q1", "question_type": "single", "recall_at_k_hit": True}],
        requested_question_ids=["q1", "q2"],
    )
    assert summary["n_questions"] == 2
    assert summary["n_hits"] == 1
    assert summary["recall_at_k"] == 0.5
