from aether.llm.contracts import LLMRequest, LLMProvider, Message

async def ask(question: str, provider: LLMProvider) -> str:
    response = await provider.complete(LLMRequest(
        messages=[Message(role="user", content=question)],
    ))
    return response.text
