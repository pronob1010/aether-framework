from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from aether.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
)


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Aether's Message list to OpenAI's chat-completions format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.role == "tool" and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        # tool_calls (assistant role) wiring happens in Phase C.
        out.append(d)
    return out


class OpenAIProvider:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        completion = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=_to_openai_messages(request.messages),
            temperature=request.temperature,
        )
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model=completion.model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        stream = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=_to_openai_messages(request.messages),
            temperature=request.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            # Final usage-only chunk has no choices.
            if not chunk.choices:
                if chunk.usage:
                    yield LLMStreamChunk(
                        text="",
                        model=chunk.model,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                continue
            choice = chunk.choices[0]
            yield LLMStreamChunk(
                text=choice.delta.content or "",
                model=chunk.model,
                finish_reason=choice.finish_reason,
            )
