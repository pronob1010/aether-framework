from aether.llm.contracts import LLMProvider
from aether.registry import REGISTRY, list_kind
# Importing LLM_PROVIDER_KIND also loads the registry module, which
# registers the built-in providers at the bottom of that file.
from aether.providers.llm.registry import LLM_PROVIDER_KIND


def make_provider(name: str, **kwargs) -> LLMProvider:
    specs = REGISTRY[LLM_PROVIDER_KIND]
    if name not in specs:
        raise ValueError(
            f"unknown provider: {name!r}. "
            f"Known providers: {list_kind(LLM_PROVIDER_KIND)}."
        )
    cls = specs[name]._factory()
    return cls(**kwargs)
