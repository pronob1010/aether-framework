import pytest
from aether import Aether
from aether.extensions.llm.builder import ProviderConfig
from aether.extensions.llm.fake import FakeProvider


@pytest.mark.asyncio
async def test_aether_ask_delegates_to_provider():
    fake = FakeProvider(canned_response="42")
    client = Aether(fake)
    answer = await client.ask("What is the meaning of life?")
    assert answer == "42"


@pytest.mark.asyncio
async def test_aether_complete_returns_full_response():
    fake = FakeProvider(canned_response="42")
    client = Aether(fake)
    response = await client.complete("What is the meaning of life?")
    assert response.text == "42"
    assert response.model == "fake-model"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_aether_complete_forwards_model_and_temperature():
    fake = FakeProvider()
    client = Aether(fake)
    await client.complete("hi", model="gpt-4o", temperature=0.0)
    sent = fake.calls[0]
    assert sent.model == "gpt-4o"
    assert sent.temperature == 0.0


@pytest.mark.asyncio
async def test_aether_ask_uses_complete_internally():
    """ask() is a thin wrapper — same request goes through."""
    fake = FakeProvider(canned_response="hello")
    client = Aether(fake)
    answer = await client.ask("ping")
    assert answer == "hello"
    assert fake.calls[0].messages[0].content == "ping"


def test_aether_from_config_builds_provider():
    client = Aether.from_config(ProviderConfig(name="fake"))
    assert isinstance(client._provider, FakeProvider)


def test_aether_from_env_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        Aether.from_env()


def test_aether_from_env_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Aether.from_env()


def test_aether_from_env_fake_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    client = Aether.from_env(
        with_retry=False,
        with_circuit_breaker=False,
        with_cost_tracking=False,
    )
    assert isinstance(client._provider, FakeProvider)
