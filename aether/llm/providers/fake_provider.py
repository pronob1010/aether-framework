from aether.llm.contracts import LLMRequest, LLMResponse

class FakeProvider:
    def __init__(self, canned_response: str = "This is a fake response."):
        self.canned_response = canned_response
        self.calls: list[LLMRequest] = []
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.canned_response,
            model='fake-model',
            input_tokens=len(request.prompt.split()),
            output_tokens=len(self.canned_response.split()),
        )