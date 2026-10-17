"""Regression tests: non-finite inputs must not poison the relevance pipeline.

Found by property-based discovery:

1. ``MemoryRecall.similarity`` had no finiteness constraint, so a NaN or
   inf similarity constructed cleanly and ``score_memory`` multiplied it
   through, corrupting every downstream ``rank_memories`` comparison.
2. ``confidence_factor(float("nan"))`` and ``usefulness_factor(float("nan"))``
   returned 1.0 — CPython's ``max(0.0, min(1.0, nan))`` comparison ordering
   keeps 1.0 — so poison read as best-quality.

Production callers legitimately produce similarity outside [0, 1]: vector
search computes ``1 - cosine_distance`` (negative for opposing embeddings)
and facet-boosted keyword ranks exceed 1.0 (``min(1.0, rank) * 1.15``).
The model therefore rejects only non-finite values.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from weft.models import Memory, MemoryRecall, MemoryType
from weft.relevance import confidence_factor, usefulness_factor


def _memory(**overrides) -> Memory:
    fields: dict = {
        "type": MemoryType.fact,
        "content": "non-finite guard regression",
        "confidence": 0.8,
        "usefulness_score": 0.6,
    }
    fields.update(overrides)
    return Memory(**fields)


class TestMemoryRecallSimilarityRejectsNonFinite:
    def test_nan_similarity_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MemoryRecall(memory=_memory(), similarity=float("nan"))

    def test_inf_similarity_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MemoryRecall(memory=_memory(), similarity=float("inf"))

    def test_negative_inf_similarity_raises_validation_error(self):
        with pytest.raises(ValidationError):
            MemoryRecall(memory=_memory(), similarity=float("-inf"))

    def test_finite_similarity_outside_unit_range_still_accepted(self):
        """Pin the deliberate scope: bound only NaN/inf, not [0, 1]."""
        recall = MemoryRecall(memory=_memory(), similarity=-0.5)
        assert math.isclose(recall.similarity, -0.5)

        recall = MemoryRecall(memory=_memory(), similarity=1.15)
        assert math.isclose(recall.similarity, 1.15)


class TestFactorFunctionsTreatNonFiniteAsFloor:
    def test_confidence_factor_nan_returns_floor(self):
        assert confidence_factor(float("nan")) == 0.0

    def test_confidence_factor_inf_returns_floor(self):
        assert confidence_factor(float("inf")) == 0.0
        assert confidence_factor(float("-inf")) == 0.0

    def test_confidence_factor_finite_unchanged(self):
        assert confidence_factor(0.8) == 0.8
        assert confidence_factor(1.5) == 1.0
        assert confidence_factor(-0.5) == 0.0

    def test_usefulness_factor_nan_returns_default_floor(self):
        assert usefulness_factor(float("nan")) == 0.5

    def test_usefulness_factor_inf_returns_default_floor(self):
        assert usefulness_factor(float("inf")) == 0.5
        assert usefulness_factor(float("-inf")) == 0.5

    def test_usefulness_factor_nan_honors_custom_floor(self):
        assert usefulness_factor(float("nan"), floor=0.25) == 0.25

    def test_usefulness_factor_finite_unchanged(self):
        assert usefulness_factor(0.0) == 0.5
        assert usefulness_factor(1.0) == 1.0
        assert usefulness_factor(0.5) == 0.75
