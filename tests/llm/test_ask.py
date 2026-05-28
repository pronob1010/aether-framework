import pytest
from aether.llm.ask import ask
from aether.llm.providers.fake_provider import FakeProvider

@pytest.mark.asyncio
async def test_ask_returns_provider_text():
    provider = FakeProvider(canned_response="Hello, world!")
    response = await ask("What is the meaning of life?", provider)
    assert response == "Hello, world!"

@pytest.mark.asyncio
async def test_ask_passes_question_to_provider():
    provider = FakeProvider()
    await ask("What is the meaning of life?", provider)
    assert len(provider.calls) == 1
    assert provider.calls[0].prompt == "What is the meaning of life?"

@pytest.mark.asyncio
async def test_ask_leaves_model_unset_by_default():
    provider = FakeProvider()
    await ask("What is the meaning of life?", provider)
    assert provider.calls[0].model is None