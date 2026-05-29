# Aether

A small Python framework for building AI-native applications, organized
around classical software-engineering patterns: Strategy, Adapter,
Decorator, Factory, Builder, Facade, and a generic Plugin Registry.

**Status:** `pre-1.0, not yet on PyPI.`

## Why

Most LLM frameworks ship the *features* you need (streaming, retries,
tool calling) wired together in ways you can't easily pull apart. Aether
ships the *primitives* — small typed objects you compose — plus a
sensible facade for the 90% case.

Three things you can do that most frameworks make hard:

1. Swap the LLM provider with one env var, no code changes.
2. Inspect cumulative cost (`client.usage.total_cost_usd`) without
   threading a tracker through your app.
3. Write a Python function, decorate it with `@register_tool`, and
   it becomes available to any provider — schema generated for you.

## Quick start

```python
import asyncio
from aether import Aether

async def main():
    client = Aether()                 # reads LLM_PROVIDER, OPENAI_API_KEY, ...
    answer = await client.ask("What is the meaning of life?")
    print(answer)
    print(f"Spent ${client.usage.total_cost_usd:.6f}")

asyncio.run(main())
```

```bash
export LLM_PROVIDER=openai       # or gemini, fake
export OPENAI_API_KEY=sk-...
python try_it.py
```

`Aether()` builds a fully resilient client by default: retry →
circuit breaker → cost tracking, in the correct nesting order.
Opt out via `Aether(with_retry=False, ...)`. For an explicit config,
pass `Aether(config=ProviderConfig(...))`. For a pre-built provider
(testing, custom wrapping), pass `Aether(some_provider)` positionally.

## Features

### Streaming

```python
async for delta in client.stream_text("Tell me a story"):
    print(delta, end="", flush=True)
```

Rich-chunk variant if you need metadata:

```python
async for chunk in client.stream("..."):
    chunk.text             # delta
    chunk.finish_reason    # only on final chunk
    chunk.output_tokens    # only on final chunk
```

Both Retry and CircuitBreaker decorators handle streaming. Retry only
applies to the handshake (before the first chunk yields); errors
mid-stream propagate so the caller never sees duplicate output.

### Tool calling

```python
from aether import Aether, register_tool

@register_tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

client = Aether()
answer = await client.ask("What is 17 + 25?", tools=["add"])
# Aether runs the LLM↔tool loop and returns the final assistant text.
```

The framework auto-generates the JSON Schema from your function's
signature and docstring. Async tools work too. Tool errors become
content (the LLM sees the failure as a tool result) rather than
exceptions that abort the conversation.

Three reference tools ship under `aether.extensions.tools`:

```python
import aether.extensions.tools  # registers get_current_time, http_get, read_file
```

### Cost tracking

On by default. Reports tokens and (where pricing is known) dollar cost:

```python
client = Aether()
await client.ask("hi")
await client.ask("how are you")

client.usage.total_requests          # 2
client.usage.total_input_tokens      # 7
client.usage.total_output_tokens     # 23
client.usage.total_cost_usd          # 0.00002
client.usage.by_model                # {"gpt-4o-mini": TokenUsage(...)}
```

Cost tracking sits **outermost** in the decorator stack, so it only
counts what actually billed — not retried attempts.

### Multi-turn conversations

`prompt` is a string for the single-turn case, or a `list[Message]`
for multi-turn:

```python
from aether import Aether, Message

convo = [
    Message(role="system",    content="You are terse."),
    Message(role="user",      content="hi"),
    Message(role="assistant", content="hi"),
    Message(role="user",      content="say more"),
]
answer = await client.ask(convo)
```

### Resilience

Retry (with exponential backoff) and Circuit Breaker both ship as
decorators. The builder composes them in the load-bearing order:
retry inside circuit breaker, so one retry-exhausted call counts as
**one** breaker failure, not N.

### Observability hooks

Subscribe sync or async callbacks to 10 lifecycle events. Use them
for logging, tracing, metrics, debugging — no need to thread loggers
through your code.

```python
from aether import Aether
from aether.events import REQUEST_COMPLETE, TOOL_ERROR

client = Aether()

@client.on(REQUEST_COMPLETE)
def log_latency(event):
    print(f"{event.request.model}: {event.duration_seconds:.2f}s")

@client.on(TOOL_ERROR)
async def alert(event):
    await send_alert(f"Tool {event.call.name} failed: {event.error}")

await client.ask("hi")
```

The 10 events:

| Event | Fires when |
|---|---|
| `request.start` / `request.complete` / `request.error` | Around each provider.complete() call (including each tool-loop iteration) |
| `stream.start` / `stream.chunk` / `stream.complete` / `stream.error` | Around streaming responses, one `stream.chunk` per delta |
| `tool.start` / `tool.complete` / `tool.error` | Around each tool dispatch in the tool loop |

Subscriber exceptions are caught and logged — observability never
breaks the request path. Share an `EventBus` across multiple clients
by passing `events=bus` to each `Aether()`.

## Architecture

```
aether/
├── client.py                ← Aether facade (the front door)
├── registry.py              ← generic plugin registry (any kind)
├── config.py                ← runtime config (env-driven, call-time)
├── events.py                ← EventBus + 10 lifecycle event types
├── llm/                     ← user-facing LLM API
│   ├── contracts.py         ← LLMProvider Protocol + Message, ToolCall, ...
│   └── ask.py               ← thin convenience
├── tools/                   ← user-facing tool API
│   ├── registry.py          ← @register_tool decorator
│   ├── schema.py            ← signature → JSON Schema
│   └── (dispatch_tool, get_tool, list_tools exposed via __init__)
└── extensions/              ← all plugin implementations
    ├── llm/
    │   ├── openai.py, gemini.py, fake.py   ← adapters
    │   ├── retrying.py, circuit_breaker.py, cost_tracking.py  ← decorators
    │   ├── registry.py      ← LLM-provider registration helper
    │   ├── factory.py       ← name → instance
    │   └── builder.py       ← config → composed stack
    └── tools/
        ├── time.py          ← get_current_time
        ├── http.py          ← http_get
        └── file.py          ← read_file
```

Each layer only knows the one below it. The **generic registry** at
the top level (`aether/registry.py`) is the single source of truth
for what's pluggable — providers, tools, and any future "kind" (vector
stores, databases, ...) live in nested dicts keyed by `kind`.

## Extending

### Register a new LLM provider

```python
from aether import register_provider
from aether.llm.contracts import LLMRequest, LLMResponse

@register_provider("ollama", api_key_env="OLLAMA_API_KEY", model_env="OLLAMA_MODEL")
class OllamaProvider:
    def __init__(self, api_key: str | None = None, default_model: str = "llama3"):
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...
```

Now `LLM_PROVIDER=ollama` works with `Aether()` — no
framework code changes. Retry, CircuitBreaker, and CostTracking
wrap it automatically.

### Register a new tool

```python
from aether import register_tool

@register_tool(description="Look up a customer by ID")
def get_customer(customer_id: str) -> dict:
    """
    Args:
        customer_id: Internal customer ID (UUID).
    """
    return {"id": customer_id, "name": "..."}
```

Tool is now available as `tools=["get_customer"]` in any
`Aether.complete()` call. JSON Schema is generated from the
signature + docstring; the LLM sees the `customer_id` description
verbatim.

### Register any other kind of plugin

The same registry mechanism underlies both providers and tools.
For future subsystems (vector stores, databases, ...) the pattern
is `register(kind, name, **metadata)`:

```python
from aether import register

@register("vector_store", "pinecone", dimension=1536)
class PineconeStore:
    ...
```

## Configuration

All runtime defaults live in `aether/config.py` and read from env
vars **at call time** (not import time), so tests and live config
reloads work cleanly.

| Env var | Default | Affects |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Which provider `Aether()` builds |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` | — | Per-provider API keys |
| `OPENAI_MODEL` / `GEMINI_MODEL` | (provider default) | Override default model |
| `AETHER_DEFAULT_TEMPERATURE` | `0.7` | Default sampling temp |
| `AETHER_MAX_TOOL_ITERATIONS` | `10` | Tool-loop cap before giving up |
| `AETHER_HTTP_TOOL_TIMEOUT` | `10.0` | Default timeout (seconds) for `http_get` |
| `AETHER_HTTP_TOOL_MAX_BYTES` | `100000` | Max body size before `http_get` truncates |
| `AETHER_FILE_TOOL_MAX_BYTES` | `200000` | Max bytes before `read_file` truncates |

Precedence: **per-call kwarg > env var > in-code fallback.** Invalid
env values silently fall back to the default rather than crashing.

## Testing

```bash
.venv/bin/python -m pytest tests/
```

The `FakeProvider` lets you write end-to-end tests with no API calls:

```python
from aether import Aether
from aether.extensions.llm.fake import FakeProvider

async def test_my_agent_logic():
    fake = FakeProvider(canned_response="hello")
    client = Aether(fake)
    assert await client.ask("hi") == "hello"
```

For scripted multi-turn flows (including tool calls):

```python
from aether.llm.contracts import LLMResponse, ToolCall

fake = FakeProvider(responses=[
    LLMResponse(text="", model="...", input_tokens=1, output_tokens=1,
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]),
    LLMResponse(text="The answer is 5.", model="...", input_tokens=1, output_tokens=1),
])
```

## Design principles

1. **Explicit over magical** — no hidden globals; what you import is what runs.
2. **Composable over monolithic** — decorators, providers, tools all opt-in.
3. **Observable by default** — cost tracking on by default; usage exposed via `client.usage`.
4. **Failure-aware** — retry + circuit breaker compose correctly.
5. **Cost & latency aware** — tokens + dollar cost tracked per model.
6. **Typed contracts** — Pydantic models for requests/responses; Protocol for providers.
7. **Async-first** — `complete()` and `stream()` are both async; sync tools wrapped transparently.
8. **Testable without API calls** — `FakeProvider` plays both single-turn and scripted multi-turn flows.

## Status & roadmap

Shipped: provider abstraction, resilience (retry + circuit breaker),
cost tracking, streaming, multi-turn messages, tool calling, reference
tools, env-driven config, observability hooks (Observer pattern).

On deck (not yet built):
- Caching decorator
- Streaming + tool calls together (the punted piece)
- Anthropic provider
- `pyproject.toml` and PyPI release

## License

Not yet specified.
