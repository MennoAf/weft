"""Regression test: rank_memories must tiebreak deterministically on memory id.

Found by property-based discovery:
``rank_memories`` used a bare stable sort with no tiebreaker, so tied
scores came out in input order — memory-level rankings were
input-order dependent. ``rank_turns`` in the same module already
tiebreaks on ``turn.id`` "for reproducible A/B comparisons"; this makes
``rank_memories`` consistent with that pattern.

This is a deliberate behavior change: tie order becomes id-ordered
instead of input-ordered.
"""

from __future__ import annotations

from datetime import datetime, timezone

from weft.models import Memory, MemoryRecall, MemoryType
from weft.relevance import rank_memories

NOW = datetime(2026, 2, 25, 12, 0, 0, tzinfo=timezone.utc)


def _tied_memory(memory_id: str) -> MemoryRecall:
    """Two memories rank identically only when every factor matches.

    accessed_at/created_at/updated_at are pinned so construction-time
    wall-clock jitter cannot perturb the recency factor and break the tie.
    """
    return MemoryRecall(
        memory=Memory(
            id=memory_id,
            type=MemoryType.fact,
            topic=[],
            content=f"tied memory {memory_id}",
            confidence=0.8,
            usefulness_score=0.6,
            access_count=0,
            last_boosted_at=None,
            created_at=NOW,
            updated_at=NOW,
            accessed_at=NOW,
        ),
        similarity=0.5,
    )


def test_tied_scores_rank_in_id_order_regardless_of_input_order():
    """Swapped input lists yield identical output lists."""
    mem_aaa = _tied_memory("mem-aaa")
    mem_bbb = _tied_memory("mem-bbb")

    forward = rank_memories([mem_aaa, mem_bbb], now=NOW)
    swapped = rank_memories([mem_bbb, mem_aaa], now=NOW)

    expected = ["mem-aaa", "mem-bbb"]
    assert [s.memory.id for s in forward] == expected
    assert [s.memory.id for s in swapped] == expected
    # Output identity, not just id sequence: same order in both runs.
    assert [s.memory.id for s in forward] == [s.memory.id for s in swapped]


def test_tiebreak_does_not_disturb_non_tied_ordering():
    """Higher-scoring memories still outrank the tied pair."""
    low_a = _tied_memory("mem-aaa")
    low_b = _tied_memory("mem-bbb")
    high = MemoryRecall(
        memory=Memory(
            id="mem-zzz",
            type=MemoryType.fact,
            topic=[],
            content="clearly better",
            confidence=0.9,
            usefulness_score=0.9,
            access_count=0,
            last_boosted_at=None,
            created_at=NOW,
            updated_at=NOW,
            accessed_at=NOW,
        ),
        similarity=0.95,
    )

    result = rank_memories([low_a, low_b, high], now=NOW)
    assert [s.memory.id for s in result] == ["mem-zzz", "mem-aaa", "mem-bbb"]
