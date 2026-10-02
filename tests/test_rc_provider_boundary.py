"""Focused R2-P provider forwarding and classifier-boundary tests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from weft.ingest_pipeline import IngestItem, classify, process
from weft.text_generation import GenerationResponse


CLASSIFIED_JSON = (
    '[{"type":"general_note","content":"parsed note",'
    '"confidence":0.9,"entities":[],"dates":[]}]'
)


class FakeProvider:
    def __init__(self, text: str = CLASSIFIED_JSON, error: BaseException | None = None):
        self.text = text
        self.error = error
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return GenerationResponse(text=self.text, model=request.model)


class FakeAnthropicClient:
    def __init__(
        self,
        text: str = CLASSIFIED_JSON,
        wait_for: asyncio.Event | None = None,
        error: BaseException | None = None,
    ):
        self.text = text
        self.wait_for = wait_for
        self.error = error
        self.calls = []
        self.close_calls = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.wait_for is not None:
            await self.wait_for.wait()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            model=kwargs["model"],
            stop_reason="end_turn",
        )

    async def aclose(self):
        self.close_calls += 1


@pytest.fixture
def anthropic_config():
    return SimpleNamespace(
        text_generation=SimpleNamespace(
            provider="anthropic",
            models={"ingest_classifier": "configured-classifier"},
        )
    )


@pytest.mark.asyncio
async def test_process_reaches_real_classify_and_routes_with_injected_provider(
    monkeypatch, anthropic_config
):
    provider = FakeProvider()
    stored = []

    @asynccontextmanager
    async def fake_acquire(_pool):
        yield SimpleNamespace()

    async def fake_store_memory(_pool, create, embedding=None):
        stored.append((create, embedding))
        return SimpleNamespace(id="memory-1")

    monkeypatch.setattr("weft.ingest_pipeline.acquire", fake_acquire)
    monkeypatch.setattr("weft.store.store_memory", fake_store_memory)
    monkeypatch.setattr("weft.config.load_config", lambda: anthropic_config)

    result = await process(
        IngestItem(text="a real note that should route", source="test"),
        pool=object(),
        generation_provider=provider,
    )

    assert [intent.content for intent in result.intents] == ["parsed note"]
    assert result.memories_created == 1
    assert stored[0][0].content == "parsed note"
    assert provider.requests[0].model == "configured-classifier"


@pytest.mark.asyncio
async def test_classify_fallback_uses_internal_client_and_closes_once(
    monkeypatch, anthropic_config
):
    client = FakeAnthropicClient()
    monkeypatch.setattr("weft.config.load_config", lambda: anthropic_config)
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: client)

    result = await classify("fallback uses one managed client")

    assert [intent.content for intent in result] == ["parsed note"]
    assert len(client.calls) == 1
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_unknown_provider_is_visible_before_sdk_construction(monkeypatch):
    config = SimpleNamespace(
        text_generation=SimpleNamespace(provider="unregistered", models={})
    )
    constructor_called = False

    def fail_constructor():
        nonlocal constructor_called
        constructor_called = True
        raise AssertionError("SDK constructor must not run")

    monkeypatch.setattr("weft.config.load_config", lambda: config)
    monkeypatch.setattr("anthropic.AsyncAnthropic", fail_constructor)

    with pytest.raises(ValueError, match="unavailable"):
        await classify("invalid provider must be visible")
    assert constructor_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        FakeProvider(error=RuntimeError("transport down")),
        FakeProvider(text="{not valid json"),
        FakeProvider(text=""),
    ],
)
async def test_transport_malformed_and_empty_responses_abstain(provider):
    assert await classify("provider failure must abstain", generation_provider=provider) == []


@pytest.mark.asyncio
async def test_internal_generation_error_abstains_and_closes_once(monkeypatch, anthropic_config):
    client = FakeAnthropicClient(error=RuntimeError("transport down"))
    monkeypatch.setattr("weft.config.load_config", lambda: anthropic_config)
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: client)

    assert await classify("internal provider error must abstain") == []
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_internal_client(monkeypatch, anthropic_config):
    client = FakeAnthropicClient(wait_for=asyncio.Event())
    monkeypatch.setattr("weft.config.load_config", lambda: anthropic_config)
    monkeypatch.setattr("anthropic.AsyncAnthropic", lambda: client)

    task = asyncio.create_task(classify("cancel while provider is generating"))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.close_calls == 1
