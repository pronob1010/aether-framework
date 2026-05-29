import pytest
from aether import Aether, register, register_provider
from aether.llm.contracts import LLMRequest, LLMResponse
from aether.providers.llm.factory import make_provider
from aether.providers.llm.registry import LLM_PROVIDER_KIND
from aether.providers.llm.builder import (
    ProviderConfig,
    RetryConfig,
    build_provider,
)
from aether.providers.llm.retrying import RetryingProvider
from aether.registry import REGISTRY, get, list_kind


@pytest.fixture
def cleanup_registry():
    """Snapshot the nested registry, restore after — keeps tests isolated."""
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- Built-ins ----------------------------------------------------------

def test_builtins_are_registered_under_llm_provider_kind():
    names = list_kind(LLM_PROVIDER_KIND)
    assert "openai" in names
    assert "gemini" in names
    assert "fake" in names


# --- LLM-specific decorator --------------------------------------------

def test_register_provider_adds_to_llm_kind_with_metadata(cleanup_registry):
    @register_provider("my_llm", api_key_env="MY_LLM_KEY", model_env="MY_LLM_MODEL")
    class MyProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(text="ok", model="m", input_tokens=0, output_tokens=0)

    spec = get(LLM_PROVIDER_KIND, "my_llm")
    assert spec.metadata == {"api_key_env": "MY_LLM_KEY", "model_env": "MY_LLM_MODEL"}


def test_register_provider_accepts_arbitrary_metadata(cleanup_registry):
    """**metadata is open-ended — extra keys ride through to the spec."""
    @register_provider("my_llm", api_key_env="K", supports_streaming=True, max_context=8192)
    class MyProvider:
        async def complete(self, request): ...

    spec = get(LLM_PROVIDER_KIND, "my_llm")
    assert spec.metadata["supports_streaming"] is True
    assert spec.metadata["max_context"] == 8192


def test_registered_provider_works_via_factory(cleanup_registry):
    @register_provider("my_llm")
    class MyProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def complete(self, request): ...

    provider = make_provider("my_llm", default_model="custom")
    assert isinstance(provider, MyProvider)


def test_registered_provider_composes_with_decorators(cleanup_registry):
    @register_provider("my_llm")
    class MyProvider:
        async def complete(self, request): ...

    provider = build_provider(ProviderConfig(name="my_llm", retry=RetryConfig()))
    assert isinstance(provider, RetryingProvider)
    assert isinstance(provider.inner_provider, MyProvider)


@pytest.mark.asyncio
async def test_registered_provider_works_via_facade(cleanup_registry, monkeypatch):
    @register_provider("my_llm", api_key_env="MY_LLM_KEY")
    class MyProvider:
        def __init__(self, api_key: str):
            self.api_key = api_key

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                text=f"hello from {self.api_key}",
                model="m",
                input_tokens=0,
                output_tokens=0,
            )

    monkeypatch.setenv("LLM_PROVIDER", "my_llm")
    monkeypatch.setenv("MY_LLM_KEY", "secret-key")
    client = Aether.from_env(with_retry=False, with_circuit_breaker=False)
    answer = await client.ask("ping")
    assert answer == "hello from secret-key"


# --- Validation --------------------------------------------------------

def test_register_provider_rejects_class_without_complete(cleanup_registry):
    with pytest.raises(TypeError, match="complete"):
        @register_provider("broken")
        class Broken:
            pass


def test_register_provider_rejects_class_with_sync_complete(cleanup_registry):
    with pytest.raises(TypeError, match="complete"):
        @register_provider("broken")
        class Broken:
            def complete(self, request):
                pass


def test_register_provider_rejects_non_class(cleanup_registry):
    with pytest.raises(TypeError, match="expects a class"):
        register_provider("broken")("not a class")


def test_unknown_provider_error_lists_known_names(cleanup_registry):
    @register_provider("zebra")
    class Z:
        async def complete(self, request): ...

    with pytest.raises(ValueError, match="zebra"):
        make_provider("zibra")


# --- Generic registry (proving it's not LLM-coupled) -------------------

def test_generic_register_works_for_arbitrary_kind(cleanup_registry):
    """The registry doesn't care about kind — vector stores, DBs, anything."""
    @register("vector_store", "pinecone", dimension=1536, metric="cosine")
    class PineconeStore:
        def __init__(self, **kwargs): pass

    spec = get("vector_store", "pinecone")
    assert spec.metadata == {"dimension": 1536, "metric": "cosine"}

    cls = spec._factory()
    assert cls is PineconeStore
    assert "pinecone" in list_kind("vector_store")


def test_kinds_are_namespaced(cleanup_registry):
    """Same name under different kinds doesn't collide."""
    @register("vector_store", "fake")
    class FakeVectorStore: pass

    # "fake" is also a registered llm_provider (built-in) — no conflict
    assert get("vector_store", "fake")._factory() is FakeVectorStore
    assert get(LLM_PROVIDER_KIND, "fake")._factory().__name__ == "FakeProvider"


def test_get_raises_for_unknown_kind(cleanup_registry):
    with pytest.raises(KeyError, match="unknown kind"):
        get("nonexistent_kind", "anything")


def test_get_raises_for_unknown_name(cleanup_registry):
    with pytest.raises(KeyError, match="unknown llm_provider"):
        get(LLM_PROVIDER_KIND, "no_such_provider")
