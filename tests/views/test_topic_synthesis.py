"""Tests for weft.views.topic_synthesis.

All tests mock the Anthropic provider — no live API calls are made.
Covers the four done_when assertions:
  V4: provenance map cites ONLY memory ids present in the input set, ≥1 id cited.
  V5: when projected cost > MAX_SYNTH_COST_PER_CALL_USD the function returns None.
  named constant: MAX_SYNTH_COST_PER_CALL_USD is a module-level named constant.
  non-empty content: on a non-empty input set the returned content is non-empty.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.models import Memory, MemoryType
from weft.views.topic_synthesis import (
    MAX_SYNTH_COST_PER_CALL_USD,
    SYNTHESIZER_VERSION,
    _projected_cost,
    synthesize_digest,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_memory(
    *,
    mem_id: str,
    content: str = "Some memory content.",
    topic: list[str] | None = None,
) -> Memory:
    return Memory(
        id=mem_id,
        type=MemoryType.fact,
        content=content,
        topic=topic or ["test"],
    )


def _mock_response(content: str, provenance: dict) -> MagicMock:
    """Build a mock Anthropic response that returns the given synthesis payload."""
    payload = json.dumps({"content": content, "provenance": provenance})
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=payload)]
    # Minimal usage object for cost calculation.
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    return mock_response


# ---------------------------------------------------------------------------
# Module-level named constant (done_when assertion #3)
# ---------------------------------------------------------------------------


class TestNamedConstant:
    def test_max_synth_cost_is_module_level_named_constant(self):
        """MAX_SYNTH_COST_PER_CALL_USD must be a module-level named constant = 0.01."""
        import weft.views.topic_synthesis as mod

        assert hasattr(mod, "MAX_SYNTH_COST_PER_CALL_USD"), (
            "MAX_SYNTH_COST_PER_CALL_USD must be defined at module level"
        )
        assert mod.MAX_SYNTH_COST_PER_CALL_USD == pytest.approx(0.01), (
            f"Expected 0.01, got {mod.MAX_SYNTH_COST_PER_CALL_USD}"
        )

    def test_max_synth_cost_value(self):
        """The imported constant value is 0.01."""
        assert MAX_SYNTH_COST_PER_CALL_USD == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# V4: Provenance — keys must be a SUBSET of input ids, ≥1 cited
# ---------------------------------------------------------------------------


class TestProvenanceV4:
    @pytest.mark.asyncio
    async def test_provenance_cites_only_input_ids(self):
        """Provenance map keys must be a subset of the input memory ids (V4)."""
        mem_a = _make_memory(mem_id="weft-aaa111", content="Weft ships Tier-1 gather.")
        mem_b = _make_memory(mem_id="weft-bbb222", content="Topic digest cache added.")
        input_ids = {"weft-aaa111", "weft-bbb222"}

        # Model returns provenance citing both real ids AND a hallucinated one.
        mock_resp = _mock_response(
            content="Weft shipped Tier-1 gather and topic digest cache.",
            provenance={
                "weft-aaa111": ["Weft shipped Tier-1 gather"],
                "weft-bbb222": ["topic digest cache"],
                "weft-HALLUCINATED999": ["this id was invented"],  # must be filtered
            },
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        # Use a small budget_tokens to stay under the cost cap (this test is about V4, not V5).
        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem_a, mem_b], budget_tokens=100)

        assert result is not None
        prov_keys = set(result["provenance"].keys())
        assert prov_keys.issubset(input_ids), (
            f"Provenance keys {prov_keys} are not a subset of input ids {input_ids}"
        )
        assert len(prov_keys) >= 1, "Provenance map must cite at least one input id"
        assert "weft-HALLUCINATED999" not in prov_keys, (
            "Hallucinated id must be filtered from provenance"
        )

    @pytest.mark.asyncio
    async def test_provenance_at_least_one_id_for_nonempty_input(self):
        """On a non-empty input with a well-formed model response, ≥1 id is cited (V4)."""
        mem = _make_memory(mem_id="weft-ccc333", content="Weft memory is working well.")

        mock_resp = _mock_response(
            content="Weft memory is working well.",
            provenance={"weft-ccc333": ["Weft memory is working well"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        # Use a small budget_tokens to stay under the cost cap (this test is about V4, not V5).
        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result is not None
        assert len(result["provenance"]) >= 1, "Must cite at least one memory id (V4)"
        assert "weft-ccc333" in result["provenance"]

    @pytest.mark.asyncio
    async def test_provenance_all_hallucinated_ids_are_dropped(self):
        """If the model cites ONLY hallucinated ids, provenance is empty (filtered)."""
        mem = _make_memory(mem_id="weft-real0001", content="Real memory content.")

        mock_resp = _mock_response(
            content="Some content from the model.",
            provenance={
                "weft-fake0001": ["span 1"],
                "weft-fake0002": ["span 2"],
            },
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        # Use a small budget_tokens to stay under the cost cap (this test is about V4, not V5).
        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result is not None
        # All hallucinated — provenance must be empty after filtering.
        assert "weft-fake0001" not in result["provenance"]
        assert "weft-fake0002" not in result["provenance"]

    @pytest.mark.asyncio
    async def test_provenance_subset_with_multiple_memories(self):
        """With five memories, only the cited subset appears in provenance."""
        mems = [_make_memory(mem_id=f"weft-m{i}", content=f"Content {i}") for i in range(5)]
        input_ids = {m.id for m in mems}

        # Model cites only two of the five.
        mock_resp = _mock_response(
            content="Synthesis over some of the memories.",
            provenance={
                "weft-m0": ["Content 0"],
                "weft-m3": ["Content 3"],
                "weft-invented": ["not real"],  # hallucinated
            },
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        # Use a small budget_tokens to stay under the cost cap (this test is about V4, not V5).
        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest(mems, budget_tokens=100)

        assert result is not None
        prov_keys = set(result["provenance"].keys())
        assert prov_keys.issubset(input_ids)
        assert "weft-invented" not in prov_keys
        assert len(prov_keys) >= 1


# ---------------------------------------------------------------------------
# V5: Cost cap + abstention
# ---------------------------------------------------------------------------


class TestCostCapV5:
    def test_projected_cost_helper(self):
        """_projected_cost correctly computes input + output cost at Haiku rates."""
        # 1M input + 1M output = $1.00 + $5.00 = $6.00
        cost = _projected_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(6.0, abs=0.001)

    def test_projected_cost_zero_tokens(self):
        assert _projected_cost(0, 0) == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_abstains_when_projected_cost_exceeds_cap(self):
        """synthesize_digest returns None without an API call when projected cost > cap (V5)."""
        # Create enough memories that the token estimate pushes projected cost over 0.01.
        # At Haiku rates: cap = $0.01, budget_tokens = 2000.
        # Output cost alone = 2000 * (5.00/1_000_000) = $0.01 — exactly the cap.
        # We need to exceed it: budget_tokens=2001 with enough input tokens to push over.
        # Simpler: use a very large budget_tokens value that guarantees the breach.
        large_budget = 10_000  # output cost = 0.05 >> 0.01 cap
        mems = [_make_memory(mem_id="weft-cost001", content="Content for cost test.")]

        mock_client = AsyncMock()

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest(mems, budget_tokens=large_budget)

        assert result is None, (
            "synthesize_digest must return None when projected cost exceeds cap (V5)"
        )
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_abstain_when_cost_is_within_cap(self):
        """When projected cost is within the cap, the call proceeds (V5 inverse)."""
        # Tiny budget → low projected cost.
        tiny_budget = 10  # 10 output tokens * $5/1M = $0.00005 — well under cap
        mem = _make_memory(mem_id="weft-cheap001", content="A short memory.")

        mock_resp = _mock_response(
            content="Short synthesis.",
            provenance={"weft-cheap001": ["Short synthesis"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=tiny_budget)

        # Should not abstain — call must have been made.
        mock_client.messages.create.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_max_synth_cost_constant_gates_abstention(self):
        """The abstention threshold is exactly MAX_SYNTH_COST_PER_CALL_USD (V5).

        Verifies that modifying the constant is the single lever — the runtime
        check references it, not an inline literal.
        """
        import weft.views.topic_synthesis as mod

        original = mod.MAX_SYNTH_COST_PER_CALL_USD
        try:
            # Lower the cap to near-zero — even a tiny budget should now abstain.
            mod.MAX_SYNTH_COST_PER_CALL_USD = 0.000001
            mem = _make_memory(mem_id="weft-gate001", content="Gate test memory.")
            mock_client = AsyncMock()

            with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
                result = await synthesize_digest([mem], budget_tokens=50)

            assert result is None, "Should abstain when cap is near-zero"
            mock_client.messages.create.assert_not_called()
        finally:
            mod.MAX_SYNTH_COST_PER_CALL_USD = original


# ---------------------------------------------------------------------------
# Non-empty content (done_when assertion #4)
# ---------------------------------------------------------------------------


class TestNonEmptyContent:
    @pytest.mark.asyncio
    async def test_content_is_nonempty_for_nonempty_input(self):
        """On a non-empty input set, the returned content must be non-empty."""
        mem = _make_memory(
            mem_id="weft-ne001",
            content="Weft has a Tier-1 topic gather that returns complete memory sets.",
        )

        mock_resp = _mock_response(
            content="Weft's Tier-1 gather returns the complete active memory set for a topic.",
            provenance={"weft-ne001": ["Tier-1 gather returns the complete active memory set"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        # Use a small budget_tokens to stay under the cost cap (this test is about content, not V5).
        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result is not None
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0, "content must be non-empty for a non-empty input"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_input(self):
        """Empty memory list must return None immediately (no API call)."""
        mock_client = AsyncMock()

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([])

        assert result is None
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_includes_cost_field(self):
        """The result dict must include a 'cost' field (non-negative float)."""
        mem = _make_memory(mem_id="weft-cost-field001", content="Memory with cost field.")

        mock_resp = _mock_response(
            content="Synthesis with cost.",
            provenance={"weft-cost-field001": ["synthesis"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result is not None
        assert "cost" in result
        assert isinstance(result["cost"], float)
        assert result["cost"] >= 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_returns_none(self):
        """An API error during the synthesis call returns None gracefully."""
        mem = _make_memory(mem_id="weft-err001", content="Error test memory.")
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """If the model returns malformed JSON, synthesize_digest returns None."""
        mem = _make_memory(mem_id="weft-json001", content="JSON test memory.")

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="{not valid json!!!")]
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result is None

    @pytest.mark.asyncio
    async def test_fenced_json_is_handled(self):
        """Model output wrapped in markdown fences is stripped and parsed correctly."""
        mem = _make_memory(mem_id="weft-fence001", content="Fenced JSON test.")

        payload = json.dumps({
            "content": "Synthesis from fenced output.",
            "provenance": {"weft-fence001": ["Synthesis from fenced output"]},
        })
        fenced = f"```json\n{payload}\n```"

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=fenced)]
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result is not None
        assert result["content"] == "Synthesis from fenced output."
        assert "weft-fence001" in result["provenance"]
