"""Embedding provider protocol — all providers implement this interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All providers must implement embed() and embed_batch().
    The dimensions property reports the output vector size.
    """

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the output vectors."""
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable provider name for config/logging."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Convert a single text to an embedding vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts to embedding vectors."""
        ...
