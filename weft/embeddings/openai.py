"""OpenAI embedding provider — text-embedding-3-small with Matryoshka truncation."""

from __future__ import annotations

import logging
import os

import openai

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 768  # Matryoshka truncation from native 1536


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-small provider with configurable output dimensions.

    Uses Matryoshka truncation to reduce from native 1536 dims to 768 by default,
    balancing quality and storage cost.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
    ):
        self._model_name = model_name
        self._dimensions = dimensions
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI embeddings. "
                "Set it in ~/.weft/.env or your environment."
            )
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "openai"

    async def embed(self, text: str) -> list[float]:
        """Embed a single text using the OpenAI API."""
        response = await self._client.embeddings.create(
            input=text,
            model=self._model_name,
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model_name,
            dimensions=self._dimensions,
        )
        # API returns embeddings in same order as input
        return [item.embedding for item in response.data]
