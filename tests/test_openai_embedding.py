"""Tests for OpenAI embedding provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weft.embeddings.openai import OpenAIEmbeddingProvider


@pytest.fixture(autouse=True)
def _set_api_key():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
        yield


def _fake_embedding(dims: int = 768) -> list[float]:
    return [0.1] * dims


class TestOpenAIProviderInit:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEmbeddingProvider()

    def test_default_dimensions(self):
        provider = OpenAIEmbeddingProvider()
        assert provider.dimensions == 768

    def test_custom_dimensions(self):
        provider = OpenAIEmbeddingProvider(dimensions=1536)
        assert provider.dimensions == 1536

    def test_provider_name(self):
        provider = OpenAIEmbeddingProvider()
        assert provider.provider_name == "openai"

    def test_default_model(self):
        provider = OpenAIEmbeddingProvider()
        assert provider._model_name == "text-embedding-3-small"

    def test_custom_model(self):
        provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-large")
        assert provider._model_name == "text-embedding-3-large"


class TestOpenAIEmbed:
    @pytest.fixture
    def provider(self):
        return OpenAIEmbeddingProvider()

    async def test_embed_single(self, provider):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=_fake_embedding())]
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed("test text")

        assert len(result) == 768
        provider._client.embeddings.create.assert_awaited_once_with(
            input="test text",
            model="text-embedding-3-small",
            dimensions=768,
        )

    async def test_embed_batch(self, provider):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=_fake_embedding()),
            MagicMock(embedding=_fake_embedding()),
        ]
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await provider.embed_batch(["text one", "text two"])

        assert len(result) == 2
        assert all(len(v) == 768 for v in result)
        provider._client.embeddings.create.assert_awaited_once_with(
            input=["text one", "text two"],
            model="text-embedding-3-small",
            dimensions=768,
        )

    async def test_embed_batch_empty(self, provider):
        result = await provider.embed_batch([])
        assert result == []

    async def test_embed_passes_custom_dimensions(self):
        provider = OpenAIEmbeddingProvider(dimensions=1536)
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=_fake_embedding(1536))]
        provider._client.embeddings.create = AsyncMock(return_value=mock_response)

        await provider.embed("test")

        provider._client.embeddings.create.assert_awaited_once_with(
            input="test",
            model="text-embedding-3-small",
            dimensions=1536,
        )


class TestGetProvider:
    def test_registry_returns_openai(self):
        from weft.embeddings import get_provider

        provider = get_provider("openai")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimensions == 768

    def test_registry_with_custom_dims(self):
        from weft.embeddings import get_provider

        provider = get_provider("openai", dimensions=512)
        assert provider.dimensions == 512
