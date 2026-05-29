from typing import AsyncIterator
from aether.llm.contracts import LLMRequest, LLMResponse, LLMStreamChunk

class FakeProvider:
    def __init__(self, canned_response: str = "This is a fake response."):
        self.canned_response = canned_response
        self.calls: list[LLMRequest] = []

    def _last_user_text(self, request: LLMRequest) -> str:
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.canned_response,
            model='fake-model',
            input_tokens=len(self._last_user_text(request).split()),
            output_tokens=len(self.canned_response.split()),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        self.calls.append(request)
        words = self.canned_response.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            delta = word if i == 0 else f" {word}"
            yield LLMStreamChunk(
                text=delta,
                model='fake-model' if is_last else None,
                finish_reason='stop' if is_last else None,
                input_tokens=len(self._last_user_text(request).split()) if is_last else None,
                output_tokens=len(words) if is_last else None,
            )
