from aether.llm.contracts import LLMRequest, LLMProvider, LLMResponse

async def ask(question: str, provider: LLMProvider) -> str:
    response = await provider.complete(LLMRequest(prompt=question))
    return response.text