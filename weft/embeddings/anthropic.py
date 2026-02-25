"""Anthropic embedding provider stub — no API available yet."""

from __future__ import annotations


class AnthropicEmbeddingProvider:
    """Anthropic embedding provider. Anthropic has not released an embedding API yet."""

    @property
    def dimensions(self) -> int:
        return 0

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic has not released an embedding API yet. "
            "This stub will be implemented when an API becomes available."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic embedding API not yet available.")
