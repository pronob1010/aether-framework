from typing import AsyncIterator
from google import genai
from google.genai import types
from aether.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
)


# Gemini's role names differ from OpenAI's.
_ROLE_MAP = {"user": "user", "assistant": "model"}


def _split_system_and_conversation(
    messages: list[Message],
) -> tuple[str | None, list[Message]]:
    """Gemini takes system instructions on the config, not inline in the conversation."""
    system_parts: list[str] = []
    conversation: list[Message] = []
    for msg in messages:
        if msg.role == "system" and msg.content:
            system_parts.append(msg.content)
        else:
            conversation.append(msg)
    return ("\n".join(system_parts) if system_parts else None, conversation)


def _to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for msg in messages:
        if msg.role in _ROLE_MAP and msg.content is not None:
            contents.append(types.Content(
                role=_ROLE_MAP[msg.role],
                parts=[types.Part.from_text(text=msg.content)],
            ))
        # Tool turns: deferred to Phase C.
    return contents


class GeminiProvider:
    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        system, conversation = _split_system_and_conversation(request.messages)
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            system_instruction=system,
        )
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=_to_gemini_contents(conversation),
            config=config,
        )
        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            model=response.model_version or model,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        model = request.model or self.default_model
        system, conversation = _split_system_and_conversation(request.messages)
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            system_instruction=system,
        )
        stream = await self.client.aio.models.generate_content_stream(
            model=model,
            contents=_to_gemini_contents(conversation),
            config=config,
        )
        async for chunk in stream:
            usage = chunk.usage_metadata
            finish = None
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish = str(chunk.candidates[0].finish_reason)
            yield LLMStreamChunk(
                text=chunk.text or "",
                model=chunk.model_version or model,
                finish_reason=finish,
                input_tokens=usage.prompt_token_count if usage else None,
                output_tokens=usage.candidates_token_count if usage else None,
            )
