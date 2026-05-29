from google import genai
from google.genai import types
from aether.llm.contracts import LLMRequest, LLMResponse

class GeminiProvider:
    def __init__(self, api_key: str, default_model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=request.prompt,
            config=types.GenerateContentConfig(temperature=request.temperature),
        )
        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            model=response.model_version or model,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )
