"""FastEmbed provider — local ONNX-based embeddings, no API key required."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSIONS = 384


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> TextEmbedding:
    """Lazily load the model on first use. Cached for reuse."""
    logger.info("Loading FastEmbed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


class FastEmbedProvider:
    """Local embedding provider using FastEmbed (ONNX runtime)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._dimensions = DEFAULT_DIMENSIONS

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "fastembed"

    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Runs model in thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._embed_sync, text)
        return result

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Runs model in thread pool to avoid blocking."""
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._embed_batch_sync, texts)
        return result

    def _embed_sync(self, text: str) -> list[float]:
        model = _get_model(self._model_name)
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self._model_name)
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]
