"""Embedding providers — pluggable interface for text-to-vector conversion."""

from __future__ import annotations

from weft.embeddings.base import EmbeddingProvider

_PROVIDERS: dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    """Register an embedding provider class by name."""
    _PROVIDERS[name] = cls


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Instantiate an embedding provider by name."""
    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys())) or "(none)"
        raise ValueError(f"Unknown embedding provider: {name!r}. Available: {available}")
    return _PROVIDERS[name](**kwargs)


def _register_builtins() -> None:
    """Register all built-in providers."""
    from weft.embeddings.anthropic import AnthropicEmbeddingProvider
    from weft.embeddings.fastembed_provider import FastEmbedProvider
    from weft.embeddings.google import GoogleEmbeddingProvider
    from weft.embeddings.openai import OpenAIEmbeddingProvider

    register_provider("fastembed", FastEmbedProvider)
    register_provider("google", GoogleEmbeddingProvider)
    register_provider("openai", OpenAIEmbeddingProvider)
    register_provider("anthropic", AnthropicEmbeddingProvider)


_register_builtins()

__all__ = ["EmbeddingProvider", "get_provider", "register_provider"]
