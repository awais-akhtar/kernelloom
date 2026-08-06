# LangChain integration

KernelLoom provides `KernelLoomChatModel`, a `BaseChatModel` implementation for
using a resident local model in LangChain runnables, prompt pipelines and
applications.

## Install

Install LangChain support with the runtime required by the model:

```bash
pip install "kernelloom[langchain,llama]"
```

For OpenVINO GenAI:

```bash
pip install "kernelloom[langchain,genai]"
```

## Basic invocation

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
finally:
    runtime.close()
```

## Use message objects

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = chat.invoke([
    SystemMessage(content="You teach Python to experienced engineers."),
    HumanMessage(content="Explain the descriptor protocol."),
])
print(response.content)
```

LangChain human, AI, system and tool message types are mapped to their matching
chat roles. Message content is converted to text before it reaches the backend.

## Prompt pipeline

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

## Streaming

```python
for chunk in chat.stream("Compare prefill and decode workloads."):
    print(chunk.content, end="", flush=True)
```

LangChain callbacks receive each fragment through `on_llm_new_token`.

## Per-call generation settings

Generation arguments pass through to `KernelLoomModel`:

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

The available values are `max_new_tokens`, `temperature`, `top_p`, `top_k`,
`repetition_penalty` and `stop_strings`.

## Lifecycle

`KernelLoomChatModel` does not take ownership of the underlying runtime. Keep
the `KernelLoomModel` alive for as long as the chain is used and close it during
application shutdown:

```python
runtime = KernelLoomModel("./models/model.gguf")
chat = KernelLoomChatModel(runtime)

try:
    # Build and run chains here.
    ...
finally:
    runtime.close()
```

One `KernelLoomModel` serializes access to its backend. Create separate model
instances only when the backend and available memory can support them.
