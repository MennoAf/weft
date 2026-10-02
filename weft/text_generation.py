"""Provider-neutral text generation primitives.

The application owns request shaping, role/model selection, and response
normalization. Provider adapters translate only the normalized request to a
provider SDK and return a normalized response. This keeps future providers
(such as an eventual Luna adapter) out of feature-specific parsing code.
"""
from __future__ import annotations

import inspect
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    """A provider-independent text-generation request."""

    model: str
    system: str | None = None
    messages: Sequence[dict[str, str]] = field(default_factory=tuple)
    max_tokens: int = 512
    timeout: float | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResponse:
    """Normalized provider output used by feature parsers."""

    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class TextGenerationProvider(Protocol):
    """Protocol implemented by every text-generation provider adapter."""

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for *request* and normalize the response."""
        ...


def _response_output_text(response: Any) -> str:
    """Extract text from raw Responses output for SDK-compatible fakes."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            value = getattr(content, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _openai_stop_reason(response: Any) -> str | None:
    """Map Responses completion status to the normalized stop reason."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        return getattr(details, "reason", None) or "incomplete"
    if status == "completed":
        return "completed"
    return status if isinstance(status, str) else None


class OpenAITextGenerationProvider:
    """OpenAI Responses API adapter.

    The OpenAI SDK is an optional dependency and is imported lazily so the
    default Anthropic installation does not require it. ``client`` is
    injectable for tests; production construction reads ``OPENAI_API_KEY`` and
    uses the SDK's bounded timeout/retry controls. The SDK owns retries for
    eligible transient failures (429/503), avoiding a second retry loop here.
    """

    provider_name = "openai"
    default_timeout = 30.0
    default_max_retries = 2

    def __init__(
        self,
        client: Any | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        owns_client: bool | None = None,
    ) -> None:
        self.owns_client = client is None if owns_client is None else owns_client
        self._closed = False
        if client is not None:
            self.client = client
            return
        try:
            import openai
        except ImportError as exc:
            raise ValueError(
                "The OpenAI text-generation adapter requires the 'openai' extra. "
                "Install it with: pip install weft-memory[openai]"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI text generation. "
                "Set it in ~/.weft/.env or your environment."
            )
        if timeout is None:
            timeout = float(
                os.environ.get("WEFT_OPENAI_TEXT_TIMEOUT", self.default_timeout)
            )
        if max_retries is None:
            max_retries = int(
                os.environ.get(
                    "WEFT_OPENAI_TEXT_RETRIES", self.default_max_retries
                )
            )
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def aclose(self) -> None:
        """Close an internally owned SDK client at most once."""
        if not self.owns_client or self._closed:
            return
        self._closed = True
        close = getattr(self.client, "aclose", None)
        if not callable(close):
            close = getattr(self.client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Translate a normalized request to ``responses.create``."""
        input_items: list[dict[str, str]] = []
        if request.system is not None:
            input_items.append({"role": "developer", "content": request.system})
        input_items.extend(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": input_items,
            "max_output_tokens": request.max_tokens,
            "store": False,
        }
        client = self.client
        if request.timeout is not None and hasattr(client, "with_options"):
            # ``timeout`` is an SDK request option, not a Responses API body
            # field. Client-level timeout remains the default otherwise.
            client = client.with_options(timeout=request.timeout)
        if request.response_format is not None:
            kwargs["text"] = {"format": request.response_format}
        response = await client.responses.create(**kwargs)
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            text = _response_output_text(response)
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            text=text,
            model=getattr(response, "model", request.model),
            stop_reason=_openai_stop_reason(response),
            input_tokens=int(
                getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
            ),
            output_tokens=int(
                getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
                or 0
            ),
        )


class AnthropicTextGenerationProvider:
    """Anthropic Messages adapter.

    ``client`` is injectable so feature tests can use an SDK-shaped fake and
    provider tests can use a completely fake provider without network calls.
    """

    provider_name = "anthropic"

    def __init__(self, client: Any, *, owns_client: bool = False) -> None:
        self.client = client
        self.owns_client = owns_client
        self._closed = False

    async def aclose(self) -> None:
        """Close an internally owned SDK client at most once."""
        if not self.owns_client or self._closed:
            return
        self._closed = True
        close = getattr(self.client, "aclose", None)
        if not callable(close):
            close = getattr(self.client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Translate a normalized request to ``messages.create``."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": list(request.messages),
        }
        if request.system is not None:
            kwargs["system"] = request.system
        response = await self.client.messages.create(**kwargs)
        blocks = getattr(response, "content", None) or []
        text = ""
        fallback_text = ""
        for block in blocks:
            block_text = getattr(block, "text", None)
            if not isinstance(block_text, str):
                continue
            if getattr(block, "type", None) == "text":
                text = block_text
                break
            if not fallback_text:
                fallback_text = block_text
        else:
            text = fallback_text
        usage = getattr(response, "usage", None)
        return GenerationResponse(
            text=text,
            model=getattr(response, "model", request.model),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def model_for_role(role: str, default: str, config: Any | None = None) -> str:
    """Resolve a configured model for a logical role.

    Explicit ``text_generation.models`` configuration wins. Environment
    overrides are applied by ``load_config`` and therefore remain available to
    deployments without exposing provider-specific logic here.
    """
    if config is None:
        from weft.config import load_config

        config = load_config()
    models = getattr(getattr(config, "text_generation", None), "models", {})
    return models.get(role, default)


def provider_for_role(
    role: str,
    client: Any | None = None,
    config: Any | None = None,
) -> TextGenerationProvider:
    """Build the configured provider adapter for a logical role.

    The legacy ``client`` argument is retained for Anthropic callers. OpenAI
    constructs its own optional SDK client when no injected client is supplied.
    Unknown providers fail explicitly rather than silently using another API.
    """
    if config is None:
        from weft.config import load_config

        config = load_config()
    provider = getattr(getattr(config, "text_generation", None), "provider", "anthropic")
    if provider == "anthropic":
        if client is None:
            raise ValueError(
                f"Anthropic client is required for text-generation role {role!r}"
            )
        return AnthropicTextGenerationProvider(client)
    if provider == "openai":
        # Existing call sites pass an Anthropic client in ``client``. Reuse an
        # injected object only when it is demonstrably OpenAI-shaped; otherwise
        # construct the configured OpenAI client from OPENAI_API_KEY.
        openai_client = client if hasattr(client, "responses") else None
        return OpenAITextGenerationProvider(openai_client)
    raise ValueError(
        f"Text-generation provider {provider!r} is unavailable for role {role!r}; "
        "install/register an adapter before selecting it"
    )


@asynccontextmanager
async def managed_provider_for_role(
    role: str,
    *,
    config: Any | None = None,
    client: Any | None = None,
    anthropic_api_key: str | None = None,
) -> AsyncIterator[TextGenerationProvider]:
    """Lazily select one provider and manage an internally owned SDK client.

    Provider selection is validated before optional SDK imports or constructors.
    Injected clients are borrowed; clients constructed on context entry are
    owned and closed by the scope on every exit path.
    """
    if config is None:
        from weft.config import load_config

        config = load_config()
    provider_name = getattr(
        getattr(config, "text_generation", None), "provider", "anthropic"
    )
    if provider_name not in {"anthropic", "openai"}:
        raise ValueError(
            f"Text-generation provider {provider_name!r} is unavailable for role {role!r}; "
            "install/register an adapter before selecting it"
        )

    if client is not None:
        if provider_name == "anthropic":
            provider = AnthropicTextGenerationProvider(client)
        else:
            provider = OpenAITextGenerationProvider(client=client)
    elif provider_name == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ValueError(
                "The Anthropic text-generation adapter requires the 'anthropic' extra. "
                "Install it with: pip install weft-memory[anthropic]"
            ) from exc
        sdk_client = (
            AsyncAnthropic(api_key=anthropic_api_key)
            if anthropic_api_key is not None
            else AsyncAnthropic()
        )
        provider = AnthropicTextGenerationProvider(sdk_client, owns_client=True)
    else:
        provider = OpenAITextGenerationProvider(owns_client=True)

    try:
        yield provider
    finally:
        await provider.aclose()  # type: ignore[attr-defined]


__all__ = [
    "AnthropicTextGenerationProvider",
    "OpenAITextGenerationProvider",
    "GenerationRequest",
    "GenerationResponse",
    "TextGenerationProvider",
    "managed_provider_for_role",
    "model_for_role",
    "provider_for_role",
]
