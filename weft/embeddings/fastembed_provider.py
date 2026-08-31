"""FastEmbed provider — local ONNX-based embeddings, no API key required."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Native output width of BAAI/bge-small-en-v1.5; configured vectors may be padded.
NATIVE_DIMENSIONS = 384
DEFAULT_DIMENSIONS = 768


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> TextEmbedding:
    """Lazily load the model on first use. Cached for reuse."""
    logger.info("Loading FastEmbed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


class FastEmbedProvider:
    """Local embedding provider using FastEmbed (ONNX runtime).

    If ``dimensions`` exceeds the model's native output (384), vectors are
    zero-padded so they fit the existing 768-wide pgvector columns without a
    schema change.
    Cosine similarity is unaffected because the extra zeros contribute nothing
    to dot-product or magnitude.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, dimensions: int = DEFAULT_DIMENSIONS, **_kwargs):
        self._model_name = model_name
        self._native_dimensions = NATIVE_DIMENSIONS
        self._dimensions = max(dimensions, NATIVE_DIMENSIONS)

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

    def _pad(self, vec: list[float]) -> list[float]:
        """Zero-pad vector if target dimensions exceed native model output."""
        if self._dimensions > len(vec):
            return vec + [0.0] * (self._dimensions - len(vec))
        return vec

    def _embed_sync(self, text: str) -> list[float]:
        model = _get_model(self._model_name)
        embeddings = list(model.embed([text]))
        return self._pad(embeddings[0].tolist())

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self._model_name)
        embeddings = list(model.embed(texts))
        return [self._pad(e.tolist()) for e in embeddings]
