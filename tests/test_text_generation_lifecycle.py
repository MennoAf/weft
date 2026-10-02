"""Focused lifecycle tests for managed text-generation provider ownership."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from weft.text_generation import (
    AnthropicTextGenerationProvider,
    GenerationRequest,
    OpenAITextGenerationProvider,
    managed_provider_for_role,
    provider_for_role,
)


ROLE = "ingest_classifier"
REQUEST = GenerationRequest(model="test-model", messages=({"role": "user", "content": "hi"},))


def _config(provider: str) -> SimpleNamespace:
    return SimpleNamespace(text_generation=SimpleNamespace(provider=provider, models={}))


class _Closeable:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.close_calls = 0
        self.error = error
        self.started: asyncio.Event | None = None
        self.release: asyncio.Event | None = None

    async def aclose(self) -> None:
        self.close_calls += 1

    async def wait_for_release(self) -> None:
        if self.started is not None and self.release is not None:
            self.started.set()
            await self.release.wait()


class _SyncCloseOnly:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _AwaitableCloseOnly:
    def __init__(self) -> None:
        self.close_calls = 0
        self.finished = False

    def close(self):
        self.close_calls += 1
        return self._finish_close()

    async def _finish_close(self) -> None:
        self.finished = True


class _OpenAIClient(_Closeable):
    def __init__(self, *, error: BaseException | None = None) -> None:
        super().__init__(error=error)
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **_: object) -> SimpleNamespace:
        await self.wait_for_release()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            output_text="answer", model="test-model", status="completed", usage=None
        )


class _AnthropicClient(_Closeable):
    def __init__(self, *, error: BaseException | None = None) -> None:
        super().__init__(error=error)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **_: object) -> SimpleNamespace:
        await self.wait_for_release()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="answer")],
            model="test-model",
            stop_reason="end_turn",
            usage=None,
        )


class _SdkModules(dict[str, list[_Closeable]]):
    def __init__(self) -> None:
        super().__init__({"anthropic": [], "openai": []})
        self.constructed = {
            "anthropic": asyncio.Event(),
            "openai": asyncio.Event(),
        }


@pytest.fixture
def sdk_modules(monkeypatch: pytest.MonkeyPatch) -> _SdkModules:
    monkeypatch.setenv("OPENAI_API_KEY", "isolated-test-key")
    constructed = _SdkModules()

    class FakeAsyncAnthropic(_AnthropicClient):
        def __init__(self, **_: object) -> None:
            super().__init__()
            constructed["anthropic"].append(self)
            constructed.constructed["anthropic"].set()

    class FakeAsyncOpenAI(_OpenAIClient):
        def __init__(self, **_: object) -> None:
            super().__init__()
            constructed["openai"].append(self)
            constructed.constructed["openai"].set()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    return constructed


@pytest.mark.asyncio
async def test_unknown_provider_fails_before_either_sdk_constructor(
    sdk_modules: dict[str, list[_Closeable]],
) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        async with managed_provider_for_role(ROLE, config=_config("unknown")):
            raise AssertionError("managed scope should not enter")

    assert sdk_modules == {"anthropic": [], "openai": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
async def test_owned_provider_closes_once_after_success(
    provider_name: str,
    sdk_modules: dict[str, list[_Closeable]],
) -> None:
    async with managed_provider_for_role(ROLE, config=_config(provider_name)) as provider:
        result = await provider.generate(REQUEST)
        assert result.text == "answer"

    client = sdk_modules[provider_name][0]
    assert client.close_calls == 1
    await provider.aclose()  # type: ignore[attr-defined]
    assert client.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
async def test_owned_provider_closes_once_after_generation_error(
    provider_name: str,
    sdk_modules: dict[str, list[_Closeable]],
) -> None:
    with pytest.raises(RuntimeError, match="generation failed"):
        async with managed_provider_for_role(ROLE, config=_config(provider_name)) as provider:
            client = sdk_modules[provider_name][0]
            client.error = RuntimeError("generation failed")
            await provider.generate(REQUEST)

    assert client.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
async def test_owned_provider_closes_after_real_task_cancellation(
    provider_name: str,
    sdk_modules: _SdkModules,
) -> None:
    ready_for_cancel = asyncio.Event()

    async def operate() -> None:
        async with managed_provider_for_role(ROLE, config=_config(provider_name)) as provider:
            await sdk_modules.constructed[provider_name].wait()
            client = sdk_modules[provider_name][0]
            client.started = asyncio.Event()
            client.release = asyncio.Event()
            generation = asyncio.create_task(provider.generate(REQUEST))
            try:
                await client.started.wait()
                ready_for_cancel.set()
                await generation
            finally:
                if not generation.done():
                    generation.cancel()
                await asyncio.gather(generation, return_exceptions=True)

    task = asyncio.create_task(operate())
    try:
        await ready_for_cancel.wait()
        client = sdk_modules[provider_name][0]
        # The operation task owns the managed scope; cancel it while generate() is blocked.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.close_calls == 1
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
async def test_borrowed_provider_never_closes_after_real_task_cancellation(
    provider_name: str,
) -> None:
    client = _AnthropicClient() if provider_name == "anthropic" else _OpenAIClient()
    started = asyncio.Event()
    release = asyncio.Event()
    client.started = started
    client.release = release

    async def operate() -> None:
        async with managed_provider_for_role(ROLE, config=_config(provider_name), client=client) as provider:
            await provider.generate(REQUEST)

    task = asyncio.create_task(operate())
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.close_calls == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_anthropic_owned_constructor_receives_explicit_key(
    sdk_modules: dict[str, list[_Closeable]],
) -> None:
    async with managed_provider_for_role(
        ROLE, config=_config("anthropic"), anthropic_api_key="isolated-key"
    ):
        pass

    assert len(sdk_modules["anthropic"]) == 1
    assert sdk_modules["anthropic"][0].close_calls == 1


@pytest.mark.asyncio
async def test_config_selection_is_per_call_without_provider_or_client_cache(
    sdk_modules: dict[str, list[_Closeable]],
) -> None:
    anthropic_client = _AnthropicClient()
    openai_client = _OpenAIClient()

    async with managed_provider_for_role(ROLE, config=_config("anthropic"), client=anthropic_client) as first:
        assert isinstance(first, AnthropicTextGenerationProvider)
    async with managed_provider_for_role(ROLE, config=_config("openai"), client=openai_client) as second:
        assert isinstance(second, OpenAITextGenerationProvider)

    assert first is not second
    assert first.client is anthropic_client
    assert second.client is openai_client
    assert anthropic_client.close_calls == 0
    assert openai_client.close_calls == 0
    assert sdk_modules == {"anthropic": [], "openai": []}


def test_provider_for_role_preserves_injected_client_compatibility() -> None:
    anthropic_client = _AnthropicClient()
    openai_client = _OpenAIClient()

    anthropic = provider_for_role(ROLE, anthropic_client, _config("anthropic"))
    openai = provider_for_role(ROLE, openai_client, _config("openai"))

    assert isinstance(anthropic, AnthropicTextGenerationProvider)
    assert isinstance(openai, OpenAITextGenerationProvider)
    assert anthropic.client is anthropic_client
    assert openai.client is openai_client
    assert anthropic.owns_client is False
    assert openai.owns_client is False


@pytest.mark.asyncio
@pytest.mark.parametrize("close_only_factory", [_SyncCloseOnly, _AwaitableCloseOnly])
async def test_anthropic_close_only_cleanup_is_awaitable_and_idempotent(close_only_factory) -> None:
    client = close_only_factory()
    provider = AnthropicTextGenerationProvider(client, owns_client=True)

    await provider.aclose()
    if isinstance(client, _AwaitableCloseOnly):
        assert client.finished is True
    await provider.aclose()

    assert client.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("close_only_factory", [_SyncCloseOnly, _AwaitableCloseOnly])
async def test_openai_close_only_cleanup_is_awaitable_and_idempotent(close_only_factory) -> None:
    client = close_only_factory()
    provider = OpenAITextGenerationProvider(client, owns_client=True)

    await provider.aclose()
    if isinstance(client, _AwaitableCloseOnly):
        assert client.finished is True
    await provider.aclose()

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_adapter_aclose_is_idempotent_for_owned_async_aclose_clients() -> None:
    anthropic_client = _AnthropicClient()
    openai_client = _OpenAIClient()
    anthropic = AnthropicTextGenerationProvider(anthropic_client, owns_client=True)
    openai = OpenAITextGenerationProvider(openai_client, owns_client=True)

    await anthropic.aclose()
    await anthropic.aclose()
    await openai.aclose()
    await openai.aclose()

    assert anthropic_client.close_calls == 1
    assert openai_client.close_calls == 1
