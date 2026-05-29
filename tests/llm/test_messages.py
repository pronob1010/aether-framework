"""Phase A foundation tests: multi-turn conversations via Message lists.

Tool calling builds on top of this; these tests pin the contract so the
later tool-loop tests have a stable base.
"""
import pytest
from aether import Aether, Message
from aether.llm.contracts import LLMRequest
from aether.extensions.llm.fake import FakeProvider


@pytest.mark.asyncio
async def test_string_prompt_is_wrapped_into_single_user_message():
    fake = FakeProvider()
    client = Aether(fake)
    await client.ask("hello")
    sent = fake.calls[0]
    assert len(sent.messages) == 1
    assert sent.messages[0].role == "user"
    assert sent.messages[0].content == "hello"


@pytest.mark.asyncio
async def test_message_list_passes_through_unchanged():
    fake = FakeProvider()
    client = Aether(fake)
    convo = [
        Message(role="system", content="You are terse."),
        Message(role="user",   content="hi"),
        Message(role="assistant", content="hi"),
        Message(role="user",   content="and again"),
    ]
    await client.complete(convo)
    sent = fake.calls[0]
    assert len(sent.messages) == 4
    assert [m.role for m in sent.messages] == ["system", "user", "assistant", "user"]
    assert sent.messages[-1].content == "and again"


@pytest.mark.asyncio
async def test_fake_provider_input_tokens_count_last_user_message():
    """FakeProvider's token estimate uses the most recent user turn — sanity check."""
    fake = FakeProvider(canned_response="ok")
    convo = [
        Message(role="user", content="first"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="three words here"),
    ]
    response = await fake.complete(LLMRequest(messages=convo))
    assert response.input_tokens == 3   # "three words here"


@pytest.mark.asyncio
async def test_stream_accepts_message_list():
    fake = FakeProvider(canned_response="a b")
    client = Aether(fake)
    convo = [
        Message(role="system", content="be brief"),
        Message(role="user", content="say it"),
    ]
    chunks = [c async for c in client.stream(convo)]
    assert len(chunks) == 2
    # The provider received the full conversation.
    assert fake.calls[0].messages[0].role == "system"
