"""LLM-specific facade over the generic `aether.registry`.

Owns three things and nothing else:
  1. The `kind` constant ("llm_provider") so callers don't pass it everywhere.
  2. Protocol validation — registered classes must define `async complete()`.
  3. Built-in provider registrations (lazy, so SDK imports stay deferred).

The actual storage lives in `aether.registry.REGISTRY`. Tests, factories,
and other consumers can read it directly there.
"""
import inspect
from aether.registry import register, register_lazy

LLM_PROVIDER_KIND = "llm_provider"


def register_provider(name: str, **metadata):
    """Class decorator for third-party LLM providers.

    `**metadata` is forwarded to the registry as-is. Conventional keys
    consumed by `Aether()` when auto-detecting from env:
        api_key_env: str | None — env var holding the API key
        model_env:   str | None — env var overriding default_model

    Other keys are ignored by the framework but available to anyone
    inspecting the spec.

        @register_provider("ollama", model_env="OLLAMA_MODEL")
        class OllamaProvider:
            async def complete(self, request): ...
    """
    def decorator(cls):
        _validate_provider_class(cls)
        return register(LLM_PROVIDER_KIND, name, **metadata)(cls)
    return decorator


def _validate_provider_class(cls: type) -> None:
    """Fail at registration time, not at first call."""
    if not isinstance(cls, type):
        raise TypeError(
            f"register_provider expects a class, got {type(cls).__name__}"
        )
    complete = getattr(cls, "complete", None)
    if complete is None or not inspect.iscoroutinefunction(complete):
        raise TypeError(
            f"{cls.__name__} must define `async def complete(self, request)` "
            "to satisfy the LLMProvider protocol."
        )


# --- Built-in providers (lazy registration) ------------------------------

def _load_openai():
    from aether.extensions.llm.openai import OpenAIProvider
    return OpenAIProvider

def _load_gemini():
    from aether.extensions.llm.gemini import GeminiProvider
    return GeminiProvider

def _load_fake():
    from aether.extensions.llm.fake import FakeProvider
    return FakeProvider

register_lazy(
    LLM_PROVIDER_KIND, "openai", _load_openai,
    api_key_env="OPENAI_API_KEY", model_env="OPENAI_MODEL",
)
register_lazy(
    LLM_PROVIDER_KIND, "gemini", _load_gemini,
    api_key_env="GEMINI_API_KEY", model_env="GEMINI_MODEL",
)
register_lazy(
    LLM_PROVIDER_KIND, "fake", _load_fake,
    api_key_env=None, model_env=None,
)
