import pytest
from aether import Aether
from aether.llm.providers.builder import ProviderConfig
from aether.llm.providers.fake_provider import FakeProvider


@pytest.mark.asyncio
async def test_aether_ask_delegates_to_provider():
    fake = FakeProvider(canned_response="42")
    client = Aether(fake)
    answer = await client.ask("What is the meaning of life?")
    assert answer == "42"
    assert fake.calls[0].prompt == "What is the meaning of life?"


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
    client = Aether.from_env(with_retry=False, with_circuit_breaker=False)
    assert isinstance(client._provider, FakeProvider)
