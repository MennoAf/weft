"""Tests for weft.views.topic_synthesis.

All tests mock the Anthropic provider — no live API calls are made.
Covers the contract assertions, refined by the cost-policy decision
(Weft weft-d58f7350, loom-8e41000e):
  V4: provenance map cites ONLY memory ids present in the input set, >=1 id cited.
  V5: when *projected* cost > MAX_SYNTH_COST_PER_CALL_USD the call abstains.
  named constant: MAX_SYNTH_COST_PER_CALL_USD is a module-level named constant.
  non-empty content: on a non-empty input set the returned content is non-empty.
  projection fix: the projection uses EXPECTED_OUTPUT_TOKENS, not budget_tokens,
    so defaults (budget_tokens=2000) no longer force abstention (the no-op bug).
  structured result: synthesize_digest returns a SynthesisResult whose status +
    projected_cost_usd + memory_count let the caller record fire/abstain telemetry.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.models import Memory, MemoryType
from weft.views.topic_synthesis import (
    EXPECTED_OUTPUT_TOKENS,
    MAX_SYNTH_COST_PER_CALL_USD,
    SYNTHESIZER_VERSION,
    SynthesisResult,
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
# Module-level named constant
# ---------------------------------------------------------------------------


class TestNamedConstant:
    def test_max_synth_cost_is_module_level_named_constant(self):
        """MAX_SYNTH_COST_PER_CALL_USD must be a module-level named constant = 0.10."""
        import weft.views.topic_synthesis as mod

        assert hasattr(mod, "MAX_SYNTH_COST_PER_CALL_USD"), (
            "MAX_SYNTH_COST_PER_CALL_USD must be defined at module level"
        )
        assert mod.MAX_SYNTH_COST_PER_CALL_USD == pytest.approx(0.10), (
            f"Expected 0.10, got {mod.MAX_SYNTH_COST_PER_CALL_USD}"
        )

    def test_max_synth_cost_value(self):
        """The imported constant value is 0.10 (raised from 0.01 per weft-d58f7350)."""
        assert MAX_SYNTH_COST_PER_CALL_USD == pytest.approx(0.10)

    def test_expected_output_tokens_constant(self):
        """EXPECTED_OUTPUT_TOKENS is a module-level constant used in the projection."""
        assert isinstance(EXPECTED_OUTPUT_TOKENS, int)
        assert 0 < EXPECTED_OUTPUT_TOKENS < 2000, (
            "Expected output must be a positive fraction of a typical budget"
        )


# ---------------------------------------------------------------------------
# V4: Provenance — keys must be a SUBSET of input ids, >=1 cited
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

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem_a, mem_b], budget_tokens=100)

        assert result.status == "synthesized"
        prov_keys = set(result.provenance.keys())
        assert prov_keys.issubset(input_ids), (
            f"Provenance keys {prov_keys} are not a subset of input ids {input_ids}"
        )
        assert len(prov_keys) >= 1, "Provenance map must cite at least one input id"
        assert "weft-HALLUCINATED999" not in prov_keys, (
            "Hallucinated id must be filtered from provenance"
        )

    @pytest.mark.asyncio
    async def test_provenance_at_least_one_id_for_nonempty_input(self):
        """On a non-empty input with a well-formed model response, >=1 id is cited (V4)."""
        mem = _make_memory(mem_id="weft-ccc333", content="Weft memory is working well.")

        mock_resp = _mock_response(
            content="Weft memory is working well.",
            provenance={"weft-ccc333": ["Weft memory is working well"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result.status == "synthesized"
        assert len(result.provenance) >= 1, "Must cite at least one memory id (V4)"
        assert "weft-ccc333" in result.provenance

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

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result.status == "synthesized"
        # All hallucinated — provenance must be empty after filtering.
        assert "weft-fake0001" not in result.provenance
        assert "weft-fake0002" not in result.provenance

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

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest(mems, budget_tokens=100)

        assert result.status == "synthesized"
        prov_keys = set(result.provenance.keys())
        assert prov_keys.issubset(input_ids)
        assert "weft-invented" not in prov_keys
        assert len(prov_keys) >= 1


# ---------------------------------------------------------------------------
# V5: Cost cap + abstention (projection fix)
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
    async def test_defaults_do_not_abstain(self):
        """REGRESSION (the no-op bug): at the default budget_tokens=2000 and the
        real $0.10 cap, a small-input synthesis must PROCEED, not abstain.

        Before the projection fix, the worst-case output term (2000 * $5/1M =
        $0.010) consumed the entire old $0.01 cap before any input was counted,
        so synthesize_digest always returned None at defaults. This pins that
        the feature is no longer a no-op at default parameters.
        """
        mem = _make_memory(mem_id="weft-default001", content="A normal-sized memory.")

        mock_resp = _mock_response(
            content="Synthesis at default budget.",
            provenance={"weft-default001": ["Synthesis at default budget"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem])  # default budget_tokens=2000

        assert result.status == "synthesized", (
            "Default parameters must NOT abstain (the no-op bug must stay fixed)"
        )
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_budget_does_not_force_abstention(self):
        """A large budget_tokens must NOT drive abstention — the projection
        ignores it (output is already bounded by max_tokens). budget_tokens is
        still passed through as the API max_tokens.
        """
        mem = _make_memory(mem_id="weft-bigbudget001", content="Short memory, big budget.")

        mock_resp = _mock_response(
            content="Synthesis with a large output budget.",
            provenance={"weft-bigbudget001": ["Synthesis with a large output budget"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100_000)

        assert result.status == "synthesized", (
            "A large budget_tokens must not force abstention after the projection fix"
        )
        # budget_tokens flows through as the API hard cap.
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["max_tokens"] == 100_000

    @pytest.mark.asyncio
    async def test_abstains_when_projected_cost_exceeds_cap(self):
        """When projected cost > cap, abstain WITHOUT an API call, and surface
        the projected cost + memory_count for telemetry (V5)."""
        import weft.views.topic_synthesis as mod

        mems = [_make_memory(mem_id="weft-cost001", content="Content for cost test.")]
        mock_client = AsyncMock()

        original = mod.MAX_SYNTH_COST_PER_CALL_USD
        try:
            # A near-zero cap: even the expected-output + tiny-input projection breaches it.
            mod.MAX_SYNTH_COST_PER_CALL_USD = 0.000001
            with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
                result = await synthesize_digest(mems, budget_tokens=2000)
        finally:
            mod.MAX_SYNTH_COST_PER_CALL_USD = original

        assert result.status == "abstained", (
            "synthesize_digest must abstain when projected cost exceeds cap (V5)"
        )
        mock_client.messages.create.assert_not_called()
        # Telemetry fields populated on abstain.
        assert result.memory_count == 1
        assert result.projected_cost_usd > 0.000001
        assert result.content is None

    @pytest.mark.asyncio
    async def test_does_not_abstain_when_cost_is_within_cap(self):
        """When projected cost is within the cap, the call proceeds (V5 inverse)."""
        mem = _make_memory(mem_id="weft-cheap001", content="A short memory.")

        mock_resp = _mock_response(
            content="Short synthesis.",
            provenance={"weft-cheap001": ["Short synthesis"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=10)

        mock_client.messages.create.assert_called_once()
        assert result.status == "synthesized"

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

            assert result.status == "abstained", "Should abstain when cap is near-zero"
            mock_client.messages.create.assert_not_called()
        finally:
            mod.MAX_SYNTH_COST_PER_CALL_USD = original


# ---------------------------------------------------------------------------
# Structured result + non-empty content
# ---------------------------------------------------------------------------


class TestResultShape:
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

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=100)

        assert result.status == "synthesized"
        assert isinstance(result.content, str)
        assert len(result.content) > 0, "content must be non-empty for a non-empty input"

    @pytest.mark.asyncio
    async def test_returns_empty_status_for_empty_input(self):
        """Empty memory list returns an 'empty' result immediately (no API call)."""
        mock_client = AsyncMock()

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([])

        assert result.status == "empty"
        assert result.memory_count == 0
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_result_includes_actual_cost_on_synthesis(self):
        """A synthesized result carries the actual incurred cost (non-negative float)."""
        mem = _make_memory(mem_id="weft-cost-field001", content="Memory with cost field.")

        mock_resp = _mock_response(
            content="Synthesis with cost.",
            provenance={"weft-cost-field001": ["synthesis"]},
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result.status == "synthesized"
        assert isinstance(result.cost_usd, float)
        assert result.cost_usd >= 0.0
        assert result.synthesized is True

    def test_synthesis_result_is_dataclass(self):
        """SynthesisResult carries the telemetry-relevant fields."""
        r = SynthesisResult(status="abstained", memory_count=431, projected_cost_usd=0.12)
        assert r.status == "abstained"
        assert r.memory_count == 431
        assert r.projected_cost_usd == pytest.approx(0.12)
        assert r.synthesized is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_returns_error_status(self):
        """An API error during the synthesis call yields an 'error' result."""
        mem = _make_memory(mem_id="weft-err001", content="Error test memory.")
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result.status == "error"
        assert result.content is None

    @pytest.mark.asyncio
    async def test_malformed_json_returns_error_status(self):
        """If the model returns malformed JSON, the result status is 'error'."""
        mem = _make_memory(mem_id="weft-json001", content="JSON test memory.")

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="{not valid json!!!")]
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        with patch("weft.views.topic_synthesis._get_client", return_value=mock_client):
            result = await synthesize_digest([mem], budget_tokens=50)

        assert result.status == "error"

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

        assert result.status == "synthesized"
        assert result.content == "Synthesis from fenced output."
        assert "weft-fence001" in result.provenance


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


def test_synthesizer_version_exported():
    """SYNTHESIZER_VERSION remains a module-level string (provenance tag)."""
    assert isinstance(SYNTHESIZER_VERSION, str)
    assert SYNTHESIZER_VERSION
