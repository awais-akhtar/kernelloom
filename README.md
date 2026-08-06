# KernelLoom

[![Tests](https://github.com/awais-akhtar/kernelloom/actions/workflows/ci.yml/badge.svg)](https://github.com/awais-akhtar/kernelloom/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kernelloom.svg)](https://pypi.org/project/kernelloom/)
[![Python](https://img.shields.io/pypi/pyversions/kernelloom.svg)](https://pypi.org/project/kernelloom/)
[![License](https://img.shields.io/github/license/awais-akhtar/kernelloom.svg)](https://github.com/awais-akhtar/kernelloom/blob/main/LICENSE)

KernelLoom is a local AI inference engine built to make models run smoothly on
your own hardware. It keeps weights resident, gives GGUF and OpenVINO GenAI one
Python API, serves a familiar HTTP protocol, integrates with LangChain, and
includes hardware-aware inspection and planning tools. It is not a cloud-model
gateway: model execution stays on the machine running KernelLoom.

KernelLoom is local by default. It does not download models, enable telemetry,
or expose a network service unless you start one.

## Highlights

- Run GGUF chat models with `llama-cpp-python` on CPU or with optional GPU layers.
- Run exported OpenVINO GenAI models on supported CPU, GPU and NPU devices.
- Invoke a model with a string, chat messages, streaming Python, LangChain or HTTP.
- Build local LangChain agents with tool calling and validated structured output.
- Create local embeddings for RAG through LangChain, Python, CLI or `/v1/embeddings`.
- Run a complete plug-and-play RAG pipeline with file loading, chunking, MMR,
  metadata filters, citations, namespaces, and memory, SQLite, or custom vector stores.
- Keep several named models resident behind one local service.
- Use native async invoke and streaming APIs without blocking an event loop.
- Tune mmap, mlock, micro-batches, batch threads, KQV offload and flash attention.
- Start reproducible multi-model servers from one JSON configuration file.
- Diagnose runtimes, inspect hardware, benchmark generation and monitor metrics.
- Use OpenAI-compatible chat completion, text completion and streaming routes.
- Configure and test models from the built-in browser console.
- Inspect GGUF, SafeTensors, ONNX and OpenVINO IR without loading full weights.
- Build separate prefill and decode plans under memory, quality and power limits.
- Benchmark and calibrate supported OpenVINO devices with recorded evidence.
- Manage paged KV-cache metadata and deadline-aware inference queues.
- Persist plans, calibrations, model roles and audit events in local SQLite.

## Requirements

- Python 3.11 or newer
- A local model file or exported model directory
- A runtime extra matching the model you want to execute

Model inspection and analytical planning have no mandatory third-party
dependencies.

## Installation

Install only the features you need:

```bash
pip install kernelloom
pip install "kernelloom[llama]"       # GGUF execution
pip install "kernelloom[openvino]"    # OpenVINO model execution
pip install "kernelloom[genai]"       # OpenVINO GenAI text generation
pip install "kernelloom[onnx]"        # richer ONNX inspection
pip install "kernelloom[server]"      # HTTP API and browser console
pip install "kernelloom[langchain]"   # LangChain adapter
pip install "kernelloom[all]"         # all main runtime integrations
```

## Quick start

### Run a GGUF model

```python
from kernelloom import KernelLoomModel, ModelConfig

config = ModelConfig(
    model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
    model_id="qwen-local",
    device="CPU",
    context_length=4096,
    threads=8,
    batch_threads=8,
    batch_size=512,
    micro_batch_size=128,
    use_mmap=True,
)

with KernelLoomModel(config) as model:
    print(model.invoke("Explain prefix caching in two paragraphs."))
```

When `threads=0`, KernelLoom leaves one logical CPU available for the rest of
the system. Set `gpu_layers` to a positive value only when your llama.cpp build
supports the intended accelerator.

### Chat and stream

```python
from kernelloom import KernelLoomModel, ModelConfig

model = KernelLoomModel(ModelConfig(
    "./models/model.gguf",
    system_prompt="You are a concise technical assistant.",
))

messages = [
    {"role": "user", "content": "Why does quantization help CPU inference?"},
]

try:
    result = model.chat(messages, max_new_tokens=180, temperature=0.2)
    print(result.text)
    print(result.backend, result.device, result.latency_ms)

    for text in model.stream(messages, max_new_tokens=180):
        print(text, end="", flush=True)
finally:
    model.close()
```

Async applications use the same resident model:

```python
import asyncio
from kernelloom import KernelLoomModel

async def main():
    model = KernelLoomModel("./models/model.gguf")
    try:
        await model.aload()
        print(await model.ainvoke("Explain local inference."))
        async for fragment in model.astream("Give me a faster summary."):
            print(fragment, end="", flush=True)
    finally:
        model.close()

asyncio.run(main())
```

### Run an OpenVINO GenAI model

Pass the exported model directory, not an individual XML file:

```python
from kernelloom import KernelLoomModel, ModelConfig

config = ModelConfig(
    model_path="./models/phi-4-mini-openvino",
    model_id="phi-local",
    backend="openvino",
    device="CPU",
)

with KernelLoomModel(config) as model:
    print(model.invoke("Write a short release note."))
```

If OpenVINO is installed in a separate environment, point KernelLoom to that
interpreter before starting Python:

```powershell
$env:KERNELLOOM_ACCELERATOR_PYTHON = "D:\runtimes\openvino\Scripts\python.exe"
```

The native worker communicates through inherited stdin/stdout pipes. It does
not open a separate port.

## Command line

Generate one response:

```bash
kernelloom run ./models/model.gguf "Write a haiku about compilers."
kernelloom run ./models/model.gguf "Explain NUMA." --threads 8 --context-length 8192
kernelloom chat ./models/model.gguf
kernelloom benchmark ./models/model.gguf "Explain KV caches" --runs 5
kernelloom embed ./models/embedding-model.gguf "document text"
kernelloom inspect ./models/model.gguf
kernelloom hardware
kernelloom doctor
```

Start the browser console and API:

```bash
kernelloom serve
kernelloom serve --host 127.0.0.1 --port 11435
kernelloom serve --model-path ./models/model.gguf --model-id local
kernelloom serve --config kernelloom.json
```

A configuration file can preload several named local models:

```json
{
  "server": {"host": "127.0.0.1", "port": 11435, "max_models": 2},
  "models": [
    {
      "model_path": "./models/chat.gguf",
      "model_id": "chat",
      "threads": 8,
      "micro_batch_size": 128
    }
  ]
}
```

Open [http://127.0.0.1:11435](http://127.0.0.1:11435) to configure a model and
test responses.

## OpenAI-compatible API

Install the server extra and start KernelLoom:

```bash
pip install "kernelloom[server,llama]"
kernelloom serve
```

Load a model:

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path":"./models/model.gguf","model_id":"local","device":"CPU"}'
```

Send a chat request:

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"local",
    "messages":[{"role":"user","content":"Hello from KernelLoom"}],
    "max_tokens":128,
    "temperature":0.2
  }'
```

Use the OpenAI Python client by changing its base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "What is a KV cache?"}],
)
print(response.choices[0].message.content)
```

Set `KERNELLOOM_API_KEY` before starting the server to require a bearer token.
The `/health`, `/ready` and `/metrics` routes remain unauthenticated for local
process checks and monitoring.

## Plug-and-play RAG

Use a chat model, an embedding model, and either the built-in persistent SQLite
store or an in-memory store:

```python
from kernelloom import KernelLoomModel, ModelConfig, RAGConfig, RAGPipeline

chat = KernelLoomModel("./models/chat.gguf")
embeddings = KernelLoomModel(ModelConfig("./models/embed.gguf", embedding=True))
rag = RAGPipeline.local(
    chat,
    embeddings,
    database="./data/knowledge.db",
    config=RAGConfig(namespace="docs", retrieval="mmr", top_k=5, fetch_k=15),
)

try:
    rag.ingest("./docs", metadata={"audience": "developers"})
    result = rag.ask(
        "How do I start the API?",
        filters={"audience": "developers"},
    )
    print(result.answer)
    print(result.to_dict()["sources"])
finally:
    rag.close()
```

The pipeline accepts inline text, `Document` objects, text/Markdown, JSON,
JSONL, CSV, directories, custom loaders, custom splitters, LangChain-compatible
embedders, and custom vector databases. See the [complete RAG guide](docs/RAG.md)
for async use, multi-tenant namespaces, retrieval tuning, and a custom database
adapter example.

## LangChain

```python
from kernelloom import KernelLoomModel, ModelConfig
from kernelloom.langchain import KernelLoomChatModel

runtime = KernelLoomModel(ModelConfig("./models/model.gguf"))
llm = KernelLoomChatModel(runtime)

response = llm.invoke("Give me three names for a migration tool.")
print(response.content)

for chunk in llm.stream("Explain model quantization."):
    print(chunk.content, end="", flush=True)
```

The adapter also supports native async calls, LangChain tools and Pydantic
structured output. Use a GGUF instruct model whose chat template supports tool
calling:

```python
from pydantic import BaseModel, Field

class SearchCode(BaseModel):
    query: str = Field(description="Code search query")

tool_model = llm.bind_tools([SearchCode])
response = tool_model.invoke("Find the database connection code")
print(response.tool_calls)

structured_model = llm.with_structured_output(SearchCode)
print(structured_model.invoke("Create a search for authentication middleware"))
```

For completely local RAG, use a dedicated sequence-embedding GGUF model:

```python
from kernelloom.langchain import KernelLoomEmbeddings

embeddings = KernelLoomEmbeddings("./models/nomic-embed-text.gguf")
try:
    query_vector = embeddings.embed_query("How does prefix caching work?")
    document_vectors = embeddings.embed_documents(["document one", "document two"])
finally:
    embeddings.close()
```

Close the underlying `KernelLoomModel` when the application shuts down.

## Hardware inspection and execution planning

```python
from kernelloom import AdaptiveExecutionEngine

engine = AdaptiveExecutionEngine("./engine-data")
try:
    hardware = engine.hardware(refresh=True)
    model = engine.inspect_model("./models/model.gguf")
    plan = engine.compile_model(
        "./models/model.gguf",
        prompt_tokens=512,
        context_tokens=4096,
        memory_budget_gb=12,
        quality_loss_limit=0.08,
        power_mode="balanced",
        backend_compile=False,
    )
    print(hardware["profile"]["devices"])
    print(model["source_format"], plan["status"])
finally:
    engine.close()
```

Result states have strict meanings:

- `planned` means an analytical placement was created.
- `compiled` means the selected vendor backend accepted the source model.
- `verified` means output passed an explicit numerical comparison.

Hardware detection by itself is not reported as verified model execution.

## Supported inputs

| Input | Inspect | Plan | Execute |
| --- | --- | --- | --- |
| GGUF v2/v3 | Yes | Yes | Yes, through llama.cpp |
| OpenVINO GenAI directory | Yes | Yes | Yes, through OpenVINO GenAI |
| OpenVINO IR (`.xml`) | Yes | Yes | Generic tensor inference |
| ONNX (`.onnx`) | Yes | Yes | Generic tensor inference through OpenVINO |
| SafeTensors | Yes | Yes | Convert to an executable format first |

Backend support also depends on the model operators, installed runtime, device
driver and available memory.

## Documentation

- [Getting started and model configuration](https://github.com/awais-akhtar/kernelloom/blob/main/docs/GETTING_STARTED.md)
- [HTTP API and browser console](https://github.com/awais-akhtar/kernelloom/blob/main/docs/API_SERVER.md)
- [LangChain integration](https://github.com/awais-akhtar/kernelloom/blob/main/docs/LANGCHAIN.md)
- [Compiler and runtime API](https://github.com/awais-akhtar/kernelloom/blob/main/docs/ENGINE_API.md)
- [Architecture](https://github.com/awais-akhtar/kernelloom/blob/main/docs/ENGINE.md)
- [Deployment, security and publishing](https://github.com/awais-akhtar/kernelloom/blob/main/docs/DEPLOYMENT.md)

## Development

```bash
git clone https://github.com/awais-akhtar/kernelloom.git
cd kernelloom
python -m venv .venv
python -m pip install -e ".[dev,langchain,server]"
python -m pytest
python -m build
python -m twine check dist/*
```

Every push to `main` runs the test matrix and publishes a unique PyPI
post-release such as `0.3.0.post12`. See the deployment guide before enabling
that workflow on a fork.

## Project status

KernelLoom is currently alpha software. Validate model compatibility and output
quality for your workload before relying on it in production.

## License

KernelLoom is available under the [MIT License](https://github.com/awais-akhtar/kernelloom/blob/main/LICENSE).
