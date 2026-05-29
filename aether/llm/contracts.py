from typing import Protocol, runtime_checkable
from pydantic import BaseModel

class LLMRequest(BaseModel):
    prompt: str
    model: str | None = None
    temperature: float = 0.7

class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int

@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...
