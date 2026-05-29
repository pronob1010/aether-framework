from openai import AsyncOpenAI
from aether.llm.contracts import LLMRequest, LLMResponse

class OpenAIProvider:
    def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = await self.client.chat.completions.create(
            model=request.model or self.default_model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
        )
        return LLMResponse(
            text=model.choices[0].message.content or "",
            model=model.model,
            input_tokens=model.usage.prompt_tokens,
            output_tokens=model.usage.completion_tokens,
        )
    