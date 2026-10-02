"""Focused RC-R2-C tests for the public provider-aware ingest command.

The command, run_ingest, summary functions, and provider adapters remain real.
Only external boundaries (config, SDK, pool/storage/embedding) are replaced.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from click.testing import CliRunner

from weft.cli import cli


class _FakePool:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _FakeEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeResponses:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.requests: list[dict] = []
        self.error = error

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        model = kwargs["model"]
        return SimpleNamespace(
            output_text=f"synthetic {model} response",
            model=model,
            status="completed",
            usage=SimpleNamespace(input_tokens=2, output_tokens=3),
        )


class _FakeSDK:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.responses = _FakeResponses(error=error)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _SDKModule:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.instances: list[_FakeSDK] = []
        self.error = error

    def AsyncOpenAI(self, **kwargs):
        sdk = _FakeSDK(error=self.error)
        sdk.constructor_kwargs = kwargs
        self.instances.append(sdk)
        return sdk


def _config(provider: str, *, models: dict[str, str] | None = None):
    return SimpleNamespace(
        database=SimpleNamespace(url="postgresql://test-only"),
        embedding=SimpleNamespace(provider="fake", model="fake", dimensions=3),
        text_generation=SimpleNamespace(provider=provider, models=models or {}),
        api_key=None,
    )


def _git_repo(tmp_path: Path) -> Path:
    source = tmp_path / "src.py"
    source.write_text("\n".join(f"# line {index}" for index in range(12)))
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=RC Test", "-c", "user.email=rc@example.invalid", "commit", "-m", "fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def _patch_cli_boundaries(
    monkeypatch,
    config,
    pool: _FakePool,
    stored: list[dict],
    load_calls: list[object] | None = None,
):
    def load_config_spy():
        if load_calls is not None:
            load_calls.append(config)
        return config

    monkeypatch.setattr("weft.cli.load_config", load_config_spy)
    monkeypatch.setattr("asyncpg.create_pool", lambda *args, **kwargs: _pool(pool))
    monkeypatch.setattr("weft.embeddings.get_provider", lambda *args, **kwargs: _FakeEmbedding())

    async def _store(*args, **kwargs):
        stored.append(kwargs)

    monkeypatch.setattr("weft.ingest.upsert_by_topic", _store)


async def _pool(pool: _FakePool) -> _FakePool:
    return pool


def test_real_click_run_ingest_and_both_role_models(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path)
    pool = _FakePool()
    stored: list[dict] = []
    config = _config(
        "openai",
        models={
            "codebase_summary": "summary-model",
            "codebase_architecture": "architecture-model",
        },
    )
    sdk_module = _SDKModule()
    load_calls: list[object] = []
    _patch_cli_boundaries(monkeypatch, config, pool, stored, load_calls)
    monkeypatch.setitem(sys.modules, "openai", sdk_module)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("WEFT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-openai-key")

    result = CliRunner().invoke(
        cli,
        ["ingest", str(repo), "--project-id", "rc-project", "--depth", "full"],
    )

    assert result.exit_code == 0, result.output
    assert len(load_calls) == 1
    assert load_calls[0] is config
    assert "Ingest complete" in result.output
    assert "test-only-openai-key" not in result.output
    assert sdk_module.instances and len(sdk_module.instances) == 1
    sdk = sdk_module.instances[0]
    assert sdk.close_calls == 1
    assert pool.close_calls == 1
    assert [request["model"] for request in sdk.responses.requests] == [
        "summary-model",
        "architecture-model",
    ]
    assert len(stored) == 2
    assert {entry["content"] for entry in stored} == {
        "synthetic summary-model response",
        "synthetic architecture-model response",
    }


def test_alternate_provider_does_not_construct_anthropic_or_require_anthropic_keys(
    monkeypatch, tmp_path
):
    repo = _git_repo(tmp_path)
    pool = _FakePool()
    stored: list[dict] = []
    config = _config("openai")
    sdk_module = _SDKModule()
    anthropic_module = ModuleType("anthropic")

    def fail_anthropic(**kwargs):
        raise AssertionError("Anthropic must not be constructed for OpenAI ingest")

    anthropic_module.AsyncAnthropic = fail_anthropic
    _patch_cli_boundaries(monkeypatch, config, pool, stored)
    monkeypatch.setitem(sys.modules, "openai", sdk_module)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("WEFT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "alternate-only-key")

    result = CliRunner().invoke(
        cli, ["ingest", str(repo), "--project-id", "rc-project", "--depth", "architecture"]
    )

    assert result.exit_code == 0, result.output
    assert len(sdk_module.instances) == 1
    assert pool.close_calls == 1


def test_default_anthropic_missing_key_is_visible_without_pool_or_sdk(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path)
    config = _config("anthropic")
    config.api_key = None
    pool_called = False
    sdk_called = False

    def fail_pool(*args, **kwargs):
        nonlocal pool_called
        pool_called = True
        raise AssertionError("pool must not be allocated before missing-key validation")

    def fail_anthropic(**kwargs):
        nonlocal sdk_called
        sdk_called = True
        raise AssertionError("SDK must not be constructed without an Anthropic key")

    monkeypatch.setattr("weft.cli.load_config", lambda: config)
    monkeypatch.setattr("asyncpg.create_pool", fail_pool)
    anthropic_module = ModuleType("anthropic")
    anthropic_module.AsyncAnthropic = fail_anthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_module)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("WEFT_API_KEY", raising=False)

    result = CliRunner().invoke(cli, ["ingest", str(repo), "--project-id", "rc-project"])

    assert result.exit_code == 1
    assert "No API key found" in result.output
    assert pool_called is False
    assert sdk_called is False


def test_unknown_provider_rejects_before_either_sdk_constructor(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path)
    pool = _FakePool()
    config = _config("not-registered")
    _patch_cli_boundaries(monkeypatch, config, pool, [])
    constructors = []

    class _SentinelModule:
        def __getattr__(self, name):
            constructors.append(name)
            raise AssertionError(f"unexpected SDK access: {name}")

    monkeypatch.setitem(sys.modules, "anthropic", _SentinelModule())
    monkeypatch.setitem(sys.modules, "openai", _SentinelModule())

    result = CliRunner().invoke(cli, ["ingest", str(repo), "--project-id", "rc-project"])

    assert result.exit_code == 1
    assert "unavailable" in result.output
    assert constructors == []
    assert pool.close_calls == 1


def test_cli_error_closes_owned_provider_and_pool_without_response_leak(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path)
    pool = _FakePool()
    config = _config("openai")
    sdk_module = _SDKModule(error=RuntimeError("provider transport failed"))
    _patch_cli_boundaries(monkeypatch, config, pool, [])
    monkeypatch.setitem(sys.modules, "openai", sdk_module)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key-not-output")

    result = CliRunner().invoke(
        cli, ["ingest", str(repo), "--project-id", "rc-project", "--depth", "full"]
    )

    assert result.exit_code != 0
    assert "secret-key-not-output" not in result.output
    assert sdk_module.instances[0].close_calls == 1
    assert pool.close_calls == 1


def test_real_click_ingest_cancellation_closes_provider_and_pool_once(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path)
    pool = _FakePool()
    config = _config("openai")
    sdk_module = _SDKModule()
    _patch_cli_boundaries(monkeypatch, config, pool, [])
    monkeypatch.setitem(sys.modules, "openai", sdk_module)
    monkeypatch.setenv("OPENAI_API_KEY", "cancel-only-key")

    owning_task: asyncio.Task | None = None
    started = asyncio.Event()
    cancel_scheduled = asyncio.Event()

    class _SuspendedResponses(_FakeResponses):
        async def create(self, **kwargs):
            self.requests.append(kwargs)
            started.set()
            assert owning_task is not None
            asyncio.get_running_loop().call_soon(owning_task.cancel)
            cancel_scheduled.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled generation must not return")

    def make_sdk(**kwargs):
        sdk = _FakeSDK()
        sdk.responses = _SuspendedResponses()
        sdk_module.instances.append(sdk)
        return sdk

    sdk_module.AsyncOpenAI = make_sdk

    original_run = asyncio.run

    def run_with_task(coro, *args, **kwargs):
        nonlocal owning_task

        async def run_coro():
            nonlocal owning_task
            owning_task = asyncio.create_task(coro)
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                await asyncio.wait_for(cancel_scheduled.wait(), timeout=2)
                await owning_task
            finally:
                if owning_task is not None and not owning_task.done():
                    owning_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await owning_task

        return original_run(run_coro(), *args, **kwargs)

    monkeypatch.setattr("weft.cli.asyncio.run", run_with_task)
    with pytest.raises(asyncio.CancelledError):
        CliRunner().invoke(
            cli,
            ["ingest", str(repo), "--project-id", "rc-project", "--depth", "full"],
        )

    assert sdk_module.instances[0].close_calls == 1
    assert pool.close_calls == 1
