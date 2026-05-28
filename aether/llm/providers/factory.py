from aether.llm.contracts import LLMProvider

def make_provider(name: str, **kwargs) -> LLMProvider:
    match name:
        case "openai":
            from aether.llm.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(**kwargs)
        case "gemini":
            from aether.llm.providers.gemini_provider import GeminiProvider
            return GeminiProvider(**kwargs)
        case "fake":
            from aether.llm.providers.fake_provider import FakeProvider
            return FakeProvider(**kwargs)
        case _:
            raise ValueError(
                f"unknown provider: {name!r}. "
                f"Known providers: 'openai', 'gemini', 'fake'."
            )
