"""Tests for provider-neutral text-generation primitives."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.text_generation import (
    AnthropicTextGenerationProvider,
    GenerationRequest,
    OpenAITextGenerationProvider,
    model_for_role,
    provider_for_role,
)


@pytest.mark.asyncio
async def test_anthropic_adapter_normalizes_text_usage_and_stop_reason() -> None:
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", text="internal"), SimpleNamespace(type="text", text="answer")],
        model="claude-haiku-4-5-20251001-resolved",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
    )
    provider = AnthropicTextGenerationProvider(client)
    request = GenerationRequest(
        model="claude-haiku-4-5-20251001",
        system="classify",
        messages=({"role": "user", "content": "hello"},),
        max_tokens=8,
    )

    result = await provider.generate(request)

    assert result.text == "answer"
    assert result.model == "claude-haiku-4-5-20251001-resolved"
    assert result.stop_reason == "end_turn"
    assert (result.input_tokens, result.output_tokens) == (12, 4)
    client.messages.create.assert_awaited_once_with(
        model=request.model,
        max_tokens=8,
        messages=[{"role": "user", "content": "hello"}],
        system="classify",
    )


@pytest.mark.asyncio
async def test_openai_adapter_maps_responses_request_and_normalizes_output() -> None:
    responses = SimpleNamespace(create=AsyncMock())
    client = SimpleNamespace(responses=responses)
    responses.create.return_value = SimpleNamespace(
        output_text="generated answer",
        model="gpt-5.6-luna",
        status="completed",
        usage=SimpleNamespace(input_tokens=21, output_tokens=7),
    )
    provider = OpenAITextGenerationProvider(client)
    schema = {
        "type": "json_schema",
        "name": "result",
        "strict": True,
        "schema": {"type": "array", "items": {"type": "string"}},
    }

    result = await provider.generate(
        GenerationRequest(
            model="gpt-5.6-luna",
            system="Return JSON only.",
            messages=({"role": "user", "content": "classify this"},),
            max_tokens=64,
            response_format=schema,
        )
    )

    assert result.text == "generated answer"
    assert result.model == "gpt-5.6-luna"
    assert result.stop_reason == "completed"
    assert (result.input_tokens, result.output_tokens) == (21, 7)
    responses.create.assert_awaited_once_with(
        model="gpt-5.6-luna",
        input=[
            {"role": "developer", "content": "Return JSON only."},
            {"role": "user", "content": "classify this"},
        ],
        max_output_tokens=64,
        store=False,
        text={"format": schema},
    )


@pytest.mark.asyncio
async def test_openai_adapter_extracts_raw_output_and_incomplete_reason() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[SimpleNamespace(type="output_text", text="ignore")],
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="part one"),
                    SimpleNamespace(type="output_text", text="part two"),
                ],
            ),
        ],
        model="gpt-5.6-luna",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))

    result = await OpenAITextGenerationProvider(client).generate(
        GenerationRequest(model="gpt-5.6-luna")
    )

    assert result.text == "part onepart two"
    assert result.stop_reason == "max_output_tokens"
    assert (result.input_tokens, result.output_tokens) == (3, 2)


def test_model_for_role_uses_configured_role_model() -> None:
    config = SimpleNamespace(
        text_generation=SimpleNamespace(
            provider="anthropic", models={"ingest_classifier": "luna-small"}
        )
    )

    assert model_for_role("ingest_classifier", "haiku", config) == "luna-small"
    assert model_for_role("unknown", "haiku", config) == "haiku"


def test_model_for_role_uses_provider_level_model() -> None:
    config = SimpleNamespace(
        text_generation=SimpleNamespace(provider="openai", model="my-model", models={})
    )

    assert model_for_role("ingest_classifier", "haiku", config) == "my-model"


def test_model_for_role_prefers_role_model_over_provider_model() -> None:
    config = SimpleNamespace(
        text_generation=SimpleNamespace(
            provider="openai", model="my-model", models={"ingest_classifier": "role-model"}
        )
    )

    assert model_for_role("ingest_classifier", "haiku", config) == "role-model"
    assert model_for_role("codebase_summary", "haiku", config) == "my-model"


def test_model_for_role_requires_model_for_non_anthropic_provider() -> None:
    config = SimpleNamespace(
        text_generation=SimpleNamespace(provider="openai", model="", models={})
    )

    with pytest.raises(ValueError, match="WEFT_TEXT_MODEL"):
        model_for_role("ingest_classifier", "haiku", config)


def test_preflight_openai_reports_package_and_credential(monkeypatch) -> None:
    import importlib.util

    import weft.text_generation as tg

    config = SimpleNamespace(
        text_generation=SimpleNamespace(provider="openai"), api_key=None
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "openai" else None
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = tg.text_generation_preflight(config)
    assert result["ready"] is False
    assert result["package_available"] is True
    assert result["credential_present"] is False
    assert result["credential_variable"] == "OPENAI_API_KEY"
    assert "OPENAI_API_KEY" in result["reason"]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = tg.text_generation_preflight(config)
    assert result["ready"] is True
    assert result["reason"] is None


def test_preflight_anthropic_missing_package_and_credential(monkeypatch) -> None:
    import importlib.util

    import weft.text_generation as tg

    config = SimpleNamespace(
        text_generation=SimpleNamespace(provider="anthropic"), api_key=None
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = tg.text_generation_preflight(config)
    assert result["ready"] is False
    assert result["package_available"] is False
    assert result["credential_present"] is False
    assert "weft-memory[anthropic]" in result["reason"]
    assert "ANTHROPIC_API_KEY" in result["reason"]


def test_preflight_unknown_provider_is_not_ready() -> None:
    import weft.text_generation as tg

    config = SimpleNamespace(text_generation=SimpleNamespace(provider="luna"))
    result = tg.text_generation_preflight(config)
    assert result["ready"] is False
    assert result["reason"] is not None


def test_provider_for_role_reuses_injected_openai_client() -> None:
    config = SimpleNamespace(text_generation=SimpleNamespace(provider="openai", models={}))
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))

    provider = provider_for_role("codebase_summary", client, config)

    assert isinstance(provider, OpenAITextGenerationProvider)
    assert provider.client is client


def test_provider_for_role_rejects_unregistered_provider() -> None:
    config = SimpleNamespace(text_generation=SimpleNamespace(provider="luna", models={}))

    with pytest.raises(ValueError, match="unavailable"):
        provider_for_role("ingest_classifier", object(), config)
