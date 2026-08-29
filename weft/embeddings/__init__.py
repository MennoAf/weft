"""Embedding providers — pluggable interface for text-to-vector conversion."""

from __future__ import annotations

from weft.embeddings.base import EmbeddingProvider

_PROVIDERS: dict[str, type] = {}
_PROVIDER_MODULES = {
    "anthropic": ("weft.embeddings.anthropic", "AnthropicEmbeddingProvider"),
    "fastembed": ("weft.embeddings.fastembed_provider", "FastEmbedProvider"),
    "google": ("weft.embeddings.google", "GoogleEmbeddingProvider"),
    "openai": ("weft.embeddings.openai", "OpenAIEmbeddingProvider"),
}


def register_provider(name: str, cls: type) -> None:
    """Register an embedding provider class by name."""
    _PROVIDERS[name] = cls


def _load_builtin(name: str) -> type | None:
    """Load one provider implementation lazily, returning None if unavailable."""
    module_info = _PROVIDER_MODULES.get(name)
    if module_info is None:
        return None
    module_name, class_name = module_info
    try:
        import importlib
        module = importlib.import_module(module_name)
        provider_cls = getattr(module, class_name)
    except (ImportError, AttributeError):
        return None
    register_provider(name, provider_cls)
    return provider_cls


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Instantiate an embedding provider, loading optional dependencies on demand."""
    provider_cls = _PROVIDERS.get(name) or _load_builtin(name)
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_MODULES)) or "(none)"
        raise ValueError(
            f"Embedding provider {name!r} is unavailable or unknown. "
            f"Install the matching optional dependency; available providers: {available}"
        )
    return provider_cls(**kwargs)

__all__ = ["EmbeddingProvider", "get_provider", "register_provider"]
