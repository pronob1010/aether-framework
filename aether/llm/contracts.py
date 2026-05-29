from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable
from pydantic import BaseModel


class ToolCall(BaseModel):
    """One tool invocation the LLM wants the framework to execute.

    `id` is provider-assigned; we echo it back when delivering the tool
    result so the LLM can match call → result.
    """
    id: str
    name: str
    arguments: dict[str, Any] = {}


class Message(BaseModel):
    """One turn in a conversation.

    Roles:
      - 'system'    instructions / persona
      - 'user'      user input
      - 'assistant' LLM output (may include tool_calls)
      - 'tool'      result from executing a tool the LLM asked for
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []         # only on assistant turns
    tool_call_id: str | None = None         # only on tool turns


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    tools: list[str] | None = None          # populated in Phase C


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCall] = []         # populated in Phase C


class LLMStreamChunk(BaseModel):
    """One delta in a streaming response.

    `text` is the new tokens since the previous chunk (a delta, not cumulative).
    Provider metadata (`model`, `finish_reason`, token counts) is populated
    when the underlying SDK emits it — typically only on the final chunk.
    """
    text: str = ""
    model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[ToolCall] = []         # populated in Phase C


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        ...
