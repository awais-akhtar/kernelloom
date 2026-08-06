# LangChain integration

KernelLoom provides local chat and embedding adapters for LangChain. They let an
application use a resident GGUF model through the standard chat-model and
embedding interfaces.

## Supported LangChain surfaces

1. **Async local calls**: native `ainvoke` and `astream` use KernelLoom's async
   bridge rather than blocking the application event loop.
2. **Tool schemas**: `bind_tools` converts LangChain functions, tools,
   Pydantic models, and JSON schemas into llama.cpp tool definitions.
3. **Structured output**: `with_structured_output` supports validated Pydantic
   results and dictionary schemas through the normal LangChain contract.
4. **Local embeddings**: `KernelLoomEmbeddings` batches documents through a
   dedicated local sequence-embedding GGUF model.
5. **Response metadata**: responses expose standard token usage plus
   backend, device, model ID, finish reason, and measured latency metadata.

## Install

```bash
pip install "kernelloom[langchain,llama]"
```

## Chat model

```python
from kernelloom import KernelLoomModel, ModelConfig
from kernelloom.langchain import KernelLoomChatModel

runtime = KernelLoomModel(ModelConfig(
    "./models/model.gguf",
    model_id="local-chat",
    system_prompt="Be accurate and concise.",
))
chat = KernelLoomChatModel(runtime)

try:
    response = chat.invoke("Explain continuous batching.")
    print(response.content)
    print(response.usage_metadata)
    print(response.response_metadata["latency_ms"])
finally:
    runtime.close()
```

The adapter participates in LangChain caching, callbacks, tracing, retry,
fallback, batch, configurable, and runnable composition APIs inherited from
`BaseChatModel`.

## Prompt pipelines

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about {subject}."),
    ("human", "{question}"),
])

chain = prompt | chat
response = chain.invoke({
    "subject": "local inference",
    "question": "When should I use a quantized model?",
})
print(response.content)
```

## Native async and streaming

```python
import asyncio

async def main():
    response = await chat.ainvoke("Explain paged attention.")
    print(response.content)

    async for chunk in chat.astream("Summarize that in one sentence."):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

Synchronous and asynchronous callbacks receive each generated fragment. Calls
to one resident model are serialized to protect its native context.

## Tool calling and agents

Use an instruct GGUF model with a tool-capable chat template. Tool support is a
property of the model and template; an incompatible model may not produce a
reliable tool call.

```python
from langchain_core.tools import tool

@tool
def lookup_order(order_id: str) -> str:
    """Look up the current state of an order."""
    return f"Order {order_id} is ready."

model_with_tools = chat.bind_tools([lookup_order])
message = model_with_tools.invoke("Where is order A-104?")

for call in message.tool_calls:
    print(call["name"], call["args"], call["id"])
```

Force any tool or one named tool when the model/template supports it:

```python
required = chat.bind_tools([lookup_order], tool_choice="any")
specific = chat.bind_tools([lookup_order], tool_choice="lookup_order")
```

`disable_streaming="tool_calling"` is enabled by default. Ordinary responses
still stream, while tool requests use a complete response so their JSON
arguments are not lost across partial chunks.

## Structured output

```python
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    title: str
    priority: str = Field(description="low, medium, or high")

extractor = chat.with_structured_output(Ticket)
ticket = extractor.invoke("The checkout page is down for every customer.")
print(ticket.title, ticket.priority)
```

To retain the raw local-model message when validation fails:

```python
result = chat.with_structured_output(Ticket, include_raw=True).invoke(
    "Turn this report into a ticket."
)
print(result["raw"])
print(result["parsed"])
print(result["parsing_error"])
```

## Completely local RAG

Use a dedicated GGUF embedding model that provides sequence-level pooling:

```python
from kernelloom.langchain import KernelLoomEmbeddings

embeddings = KernelLoomEmbeddings(
    "./models/nomic-embed-text-v1.5.Q8_0.gguf",
    model_id="local-embeddings",
    threads=8,
)

try:
    documents = embeddings.embed_documents([
        "KernelLoom keeps models resident.",
        "Paged KV caches reduce allocation churn.",
    ])
    query = embeddings.embed_query("How are local models kept warm?")
    print(len(documents), len(query))
finally:
    embeddings.close()
```

`aembed_documents` and `aembed_query` are available for async ingestion and
retrieval. The adapter works with LangChain vector stores because it implements
the standard `Embeddings` interface.

## Messages and tool results

Human, AI, system, and tool messages are preserved. Tool-call IDs and assistant
tool-call history are passed back to llama.cpp, allowing an agent loop to join a
tool result to the request that created it. Text content blocks are flattened;
KernelLoom's current high-level LangChain adapter is text-only.

## Token counting and metadata

For GGUF models, LangChain token-budget helpers use the model's own tokenizer:

```python
count = chat.get_num_tokens("Count this with the local tokenizer.")
```

Every complete `AIMessage` includes:

- `usage_metadata` with input, output, and total tokens when the backend reports it;
- `response_metadata` with model ID, backend, device, latency, finish reason, and raw usage;
- `tool_calls` in LangChain's standard normalized format.

## Per-call generation settings

```python
response = chat.invoke(
    "Give one deterministic answer.",
    max_new_tokens=120,
    temperature=0,
    top_p=0.9,
    repetition_penalty=1.05,
    stop_strings=["END"],
)
```

## Lifecycle and production use

The chat adapter accepts a resident `KernelLoomModel` and does not silently load
a second copy. Close it during application shutdown. The adapter itself can be
used as a context manager:

```python
runtime = KernelLoomModel("./models/model.gguf")

with KernelLoomChatModel(runtime) as chat:
    print(chat.invoke("Hello").content)
```

Use separate chat and embedding models. Only create multiple instances when RAM
and the backend can support them. Benchmark the exact models and workload rather
than assuming thread or batch settings transfer between machines.
