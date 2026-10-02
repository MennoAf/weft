"""RC-FL-05: configured text generation reaches MCP ingestion."""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from weft.cache import NullCache
from weft.config import WeftConfig
from weft.mcp.server import AppContext
from weft.text_generation import GenerationResponse


CLASSIFIED_JSON = (
    '[{"type":"general_note","content":"parsed through MCP",'
    '"confidence":0.9,"entities":[],"dates":[]}]'
)


class FakeGenerationProvider:
    provider_name = "fake"

    def __init__(self, text: str = CLASSIFIED_JSON, error: BaseException | None = None):
        self.text = text
        self.error = error
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return GenerationResponse(text=self.text, model=request.model)

    async def aclose(self):
        pass


class FakeEmbeddingProvider:
    provider_name = "fake"


@pytest.fixture
def app(pool):
    config = WeftConfig()
    config.text_generation.provider = "fake"
    config.text_generation.models["ingest_classifier"] = "rc-classifier-model"
    return AppContext(
        pool=pool,
        cache=NullCache(),
        embedding=FakeEmbeddingProvider(),
        config=config,
    )


@pytest.fixture
def ctx(app):
    context = MagicMock()
    context.request_context.lifespan_context = app
    context.list_roots.return_value = []
    return context


@pytest.fixture
def public_boundaries(monkeypatch):
    """Keep the public tool -> process/classify path while isolating DB writes."""
    from weft.ingest_pipeline import IngestResult

    captured = {}

    @asynccontextmanager
    async def fake_acquire(_pool):
        yield SimpleNamespace()

    async def fake_route(intents, _pool, _embedding, *, project_id, source):
        captured["project_id"] = project_id
        captured["source"] = source
        captured["intents"] = intents
        return IngestResult(intents=list(intents))

    monkeypatch.setattr("weft.mcp.tools.acquire", fake_acquire)
    monkeypatch.setattr("weft.ingest_pipeline.route", fake_route)
    return captured


@pytest.mark.asyncio
async def test_public_ingest_forwards_configured_provider_role_and_model(
    ctx, app, public_boundaries, monkeypatch
):
    provider = FakeGenerationProvider()
    app.generation_provider = provider
    selected = {}
    from weft.ingest_pipeline import model_for_role as real_model_for_role

    def record_model_role(role, default, config=None):
        selected["role"] = role
        selected["provider"] = config.text_generation.provider
        return real_model_for_role(role, default, config)

    monkeypatch.setattr("weft.ingest_pipeline.model_for_role", record_model_role)

    from weft.mcp.tools import weft_ingest

    result = await weft_ingest(
        ctx,
        "a configured public ingest note",
        source="rc-test",
        project_id="explicit-project",
    )

    assert result == {
        "memories_created": 0,
        "entities_created": 0,
        "entities_linked": 0,
        "alerts_created": 0,
        "intents": 1,
        "errors": [],
    }
    assert selected == {"role": "ingest_classifier", "provider": "fake"}
    assert provider.requests[0].model == "rc-classifier-model"
    assert public_boundaries["project_id"] == "explicit-project"
    assert public_boundaries["source"] == "rc-test"


@pytest.mark.asyncio
async def test_invalid_provider_is_visible_without_constructing_provider(
    ctx, app, public_boundaries, monkeypatch
):
    app.config.text_generation.provider = "not-registered"
    constructed = []

    def fail_constructor(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr("anthropic.AsyncAnthropic", fail_constructor)

    from weft.mcp.tools import weft_ingest

    result = await weft_ingest(ctx, "an invalid provider note")

    assert result["error"] == "Invalid input"
    assert "unavailable" in result["detail"]
    assert constructed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text,metadata", [("hi", {}), ("👍", {}), ("bot note", {"is_bot": True})])
async def test_skip_preserves_response_shape_without_provider_selection(
    ctx, public_boundaries, monkeypatch, text, metadata
):
    selected = []

    @asynccontextmanager
    async def fail_if_selected(*args, **kwargs):
        selected.append((args, kwargs))
        raise AssertionError("skipped input must not select a provider")
        yield  # pragma: no cover

    monkeypatch.setattr("weft.ingest_pipeline.managed_provider_for_role", fail_if_selected)

    from weft.mcp.tools import weft_ingest

    result = await weft_ingest(ctx, text, metadata=metadata)

    assert result == {
        "memories_created": 0,
        "entities_created": 0,
        "entities_linked": 0,
        "alerts_created": 0,
        "intents": 0,
        "errors": [],
    }
    assert selected == []


@pytest.mark.asyncio
async def test_abstention_preserves_response_shape(
    ctx, public_boundaries, monkeypatch
):
    provider = FakeGenerationProvider(text="")

    @asynccontextmanager
    async def fake_managed_provider(role, *, config=None, **_kwargs):
        yield provider

    monkeypatch.setattr("weft.ingest_pipeline.managed_provider_for_role", fake_managed_provider)

    from weft.mcp.tools import weft_ingest

    result = await weft_ingest(ctx, "provider abstains from this note")

    assert result == {
        "memories_created": 0,
        "entities_created": 0,
        "entities_linked": 0,
        "alerts_created": 0,
        "intents": 0,
        "errors": [],
    }
    assert len(provider.requests) == 1
