"""OpenAI embedding provider stub — text-embedding-3-small (1536d)."""

from __future__ import annotations


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-small provider. Not yet implemented."""

    @property
    def dimensions(self) -> int:
        return 1536

    @property
    def provider_name(self) -> str:
        return "openai"

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "OpenAI embedding provider not yet implemented. "
            "Requires openai SDK and OPENAI_API_KEY env var. "
            "Will use text-embedding-3-small model (1536 dimensions)."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAI embedding provider not yet implemented.")
