"""Google embedding provider stub — text-embedding-004 (768d)."""

from __future__ import annotations


class GoogleEmbeddingProvider:
    """Google text-embedding-004 provider. Not yet implemented."""

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def provider_name(self) -> str:
        return "google"

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Google embedding provider not yet implemented. "
            "Requires google-genai SDK and GOOGLE_API_KEY env var. "
            "Will use text-embedding-004 model (768 dimensions)."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Google embedding provider not yet implemented.")
