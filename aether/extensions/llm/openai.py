import json
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
from aether.llm.contracts import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
)
from aether.tools import get_tool


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate Aether's Message list to OpenAI's chat-completions format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.role == "assistant" and msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.role == "tool" and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        out.append(d)
    return out


def _tools_payload(tool_names: list[str] | None) -> list[dict[str, Any]] | None:
    if not tool_names:
        return None
    return [
        {"type": "function", "function": get_tool(name).schema}
        for name in tool_names
    ]


def _parse_tool_calls(msg: Any) -> list[ToolCall]:
    raw = getattr(msg, "tool_calls", None) or []
    parsed: list[ToolCall] = []
    for tc in raw:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        parsed.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return parsed


def _build_tool_calls(buffers: dict[int, dict[str, str]]) -> list[ToolCall]:
    """Turn the per-index accumulators into parsed ToolCall objects."""
    out: list[ToolCall] = []
    for idx in sorted(buffers):
        buf = buffers[idx]
        try:
            args = json.loads(buf["args"]) if buf["args"] else {}
        except json.JSONDecodeError:
            args = {}
        out.append(ToolCall(
            id=buf["id"] or f"tool_{idx}",
            name=buf["name"],
            arguments=args,
        ))
    return out


class OpenAIProvider:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": _to_openai_messages(request.messages),
            "temperature": request.temperature,
        }
        if tools := _tools_payload(request.tools):
            kwargs["tools"] = tools
        completion = await self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        return LLMResponse(
            text=message.content or "",
            model=completion.model,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            tool_calls=_parse_tool_calls(message),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Stream with tool-call support.

        Text deltas pass through as they arrive. Tool-call deltas (name and
        arguments arrive incrementally across chunks) are accumulated per
        `index` and emitted as a single consolidated chunk when the LLM
        signals `finish_reason="tool_calls"`.
        """
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": _to_openai_messages(request.messages),
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools := _tools_payload(request.tools):
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        # Per-index accumulators: each tool_call's name/args arrive across
        # multiple chunks. Index is OpenAI's parallel-call slot.
        buffers: dict[int, dict[str, str]] = {}

        async for chunk in stream:
            # Final usage-only chunk (no choices).
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
            delta = choice.delta

            # Accumulate any tool-call deltas in this chunk.
            for tc_delta in (getattr(delta, "tool_calls", None) or []):
                idx = tc_delta.index
                buf = buffers.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc_delta.id:
                    buf["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        buf["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        buf["args"] += tc_delta.function.arguments

            # Text delta — pass through.
            if delta.content:
                yield LLMStreamChunk(
                    text=delta.content,
                    model=chunk.model,
                    finish_reason=choice.finish_reason,
                )

            # End-of-stream with tool calls — emit accumulated set.
            if choice.finish_reason == "tool_calls" and buffers:
                yield LLMStreamChunk(
                    text="",
                    model=chunk.model,
                    finish_reason="tool_calls",
                    tool_calls=_build_tool_calls(buffers),
                )
                buffers = {}
