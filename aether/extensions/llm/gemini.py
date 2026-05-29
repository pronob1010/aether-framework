from typing import Any, AsyncIterator
from google import genai
from google.genai import types
from aether.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
)
from aether.tools import get_tool


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
        if msg.role == "assistant" and msg.tool_calls:
            # Model emitted a function call in a prior turn.
            parts = [
                types.Part.from_function_call(name=tc.name, args=tc.arguments)
                for tc in msg.tool_calls
            ]
            if msg.content:
                parts.insert(0, types.Part.from_text(text=msg.content))
            contents.append(types.Content(role="model", parts=parts))
        elif msg.role == "tool" and msg.tool_call_id is not None:
            # Result of a function the framework executed.
            # Gemini expects function_response wrapped in a user-role Content.
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=msg.tool_call_id,
                    response={"result": msg.content or ""},
                )],
            ))
        elif msg.role in _ROLE_MAP and msg.content is not None:
            contents.append(types.Content(
                role=_ROLE_MAP[msg.role],
                parts=[types.Part.from_text(text=msg.content)],
            ))
    return contents


def _tools_config(tool_names: list[str] | None) -> list[types.Tool] | None:
    if not tool_names:
        return None
    declarations: list[types.FunctionDeclaration] = []
    for name in tool_names:
        spec = get_tool(name)
        declarations.append(types.FunctionDeclaration(
            name=spec.schema["name"],
            description=spec.schema.get("description", ""),
            parameters=spec.schema["parameters"],
        ))
    return [types.Tool(function_declarations=declarations)]


def _parse_function_calls(response: Any) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        parts = getattr(cand.content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                # Gemini doesn't supply call IDs — synthesize one using the name.
                # The framework only needs IDs for matching call → result, and
                # since Gemini takes function_responses by NAME, that's enough.
                parsed.append(ToolCall(
                    id=fc.name,
                    name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))
    return parsed


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
            tools=_tools_config(request.tools),
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
            tool_calls=_parse_function_calls(response),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream with tool-call support.

        Gemini emits function calls atomically (not chunked across deltas),
        so a tool call appears on a single chunk's candidate as a
        `function_call` Part — no per-chunk accumulation needed.
        """
        model = request.model or self.default_model
        system, conversation = _split_system_and_conversation(request.messages)
        config = types.GenerateContentConfig(
            temperature=request.temperature,
            system_instruction=system,
            tools=_tools_config(request.tools),
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

            tool_calls = _parse_function_calls(chunk)
            if tool_calls:
                # Emit a control-flow chunk so the facade can dispatch.
                yield LLMStreamChunk(
                    text="",
                    model=chunk.model_version or model,
                    finish_reason=finish or "tool_calls",
                    tool_calls=tool_calls,
                    input_tokens=usage.prompt_token_count if usage else None,
                    output_tokens=usage.candidates_token_count if usage else None,
                )
                continue

            yield LLMStreamChunk(
                text=chunk.text or "",
                model=chunk.model_version or model,
                finish_reason=finish,
                input_tokens=usage.prompt_token_count if usage else None,
                output_tokens=usage.candidates_token_count if usage else None,
            )
