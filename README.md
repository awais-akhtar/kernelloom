# KernelLoom

[![Tests](https://github.com/awais-akhtar/kernelloom/actions/workflows/ci.yml/badge.svg)](https://github.com/awais-akhtar/kernelloom/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/kernelloom.svg)](https://pypi.org/project/kernelloom/)
[![Python](https://img.shields.io/pypi/pyversions/kernelloom.svg)](https://pypi.org/project/kernelloom/)
[![License](https://img.shields.io/github/license/awais-akhtar/kernelloom.svg)](https://github.com/awais-akhtar/kernelloom/blob/main/LICENSE)

KernelLoom runs GGUF and OpenVINO GenAI models on the machine where you start
it. It provides a Python API, a small HTTP service, and local RAG components.
KernelLoom is alpha software; verify the exact model, runtime, and hardware
combination before production use.

The recommended surfaces are the model API, the server, and the RAG pipeline.
The lower-level compiler, scheduling, and hardware-planning modules are useful
for experiments and host applications, but they are not a promise that every
model format can be generated on every device.

## What is included

- GGUF execution through llama.cpp, with CPU tuning, model warmup, and bounded
  embedding/token caches.
- OpenVINO GenAI execution through an isolated local worker for exported model
  directories.
- A named-model HTTP service with OpenAI-style chat, completion, embedding, and
  streaming endpoints.
- A browser control page for model configuration, warm/cache controls, hardware
  inspection, CPU plans, and RAG collections.
- Document loading, chunking, local SQLite or optional FAISS retrieval, plus
  adapters for custom embedders and vector stores.
- A LangChain chat and embedding adapter for local GGUF models.

## Install

Choose the pieces you use rather than installing every optional dependency:

```bash
pip install kernelloom
pip install "kernelloom[llama]"              # GGUF execution
pip install "kernelloom[server]"             # HTTP API and browser page
pip install "kernelloom[genai]"              # OpenVINO GenAI text generation
pip install "kernelloom[openvino]"           # generic OpenVINO / ONNX tooling
pip install "kernelloom[langchain]"          # LangChain adapter
pip install "kernelloom[fastembed]"          # local ONNX embedding models
pip install "kernelloom[faiss]"              # local native vector search
pip install "kernelloom[rag]"                # FastEmbed and FAISS together
pip install "kernelloom[all]"                # all of the optional integrations above
```

`[all]` is convenient but large. Install it only when the runtime dependencies
are appropriate for the target machine.

## Quick start

```python
from kernelloom import KernelLoomModel, ModelConfig

config = ModelConfig(
    model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
    model_id="local-chat",
    cpu_profile="latency",
    reserve_cores=1,
    auto_batch_size=True,
    warmup=True,
)

with KernelLoomModel(config) as model:
    print(model.invoke("Explain KV caches in two short paragraphs."))
```

The CPU profile is a reproducible starting point, not a benchmark result.
Measure the model, quantization, context length, and representative prompts on
the deployment hardware. See the [CPU-first guide](https://github.com/awais-akhtar/kernelloom/blob/main/docs/CPU_FIRST.md).

## Local server and browser control

```bash
pip install "kernelloom[server,llama]"
kernelloom serve --host 127.0.0.1 --port 11435
```

Open `http://127.0.0.1:11435/`. The page uses the same API as applications and
does not save settings by itself. It can load and update resident models, show
cache and warmup state, inspect local hardware, apply a CPU plan, stream a chat
response, and create a RAG collection from already-loaded models.

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path":"./models/model.gguf","model_id":"local","device":"CPU"}'
```

The server has no authentication until `KERNELLOOM_API_KEY` is set. Keep it on
loopback for development. If it is reachable by other machines, use a strong
key, TLS, firewall rules, and a process account restricted to the model and
knowledge paths you intend to expose.

## Local RAG

Use separate chat and embedding models. The built-in SQLite store is persistent
and simple; the optional FAISS store is an in-memory exact-search store with
native vector math. For large collections or approximate nearest-neighbor
search, supply a database adapter that fits the workload.

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
    answer = rag.ask("How do I start the API?", filters={"audience": "developers"})
    print(answer.answer)
    print(answer.to_dict()["sources"])
finally:
    rag.close()
    chat.close()
    embeddings.close()
```

`RAGAnswer.sources` is the retrieval trace. A generated answer can mention
source labels in its prompt, but KernelLoom does not verify or enforce citations
in model output. Read the [RAG guide](https://github.com/awais-akhtar/kernelloom/blob/main/docs/RAG.md) for persistence, filters, FAISS, custom stores, and server routes.

## Local boundary

Core GGUF and OpenVINO execution use local files and local native runtimes.
Optional integrations have their own behavior: FastEmbed can download a chosen
embedding model on first use unless it is already cached, and custom embedders
or vector stores may send text elsewhere. Review each integration before using
it in an air-gapped or sensitive environment.

## Command line

```bash
kernelloom run ./models/model.gguf "Write a haiku about compilers."
kernelloom chat ./models/model.gguf
kernelloom benchmark ./models/model.gguf "Explain KV caches" --runs 5
kernelloom embed ./models/embedding-model.gguf "document text"
kernelloom warm ./models/model.gguf --cpu-profile latency
kernelloom cpu-plan --profile throughput
kernelloom inspect ./models/model.gguf
kernelloom hardware
kernelloom doctor
```

## Documentation

- [Getting started and model configuration](https://github.com/awais-akhtar/kernelloom/blob/main/docs/GETTING_STARTED.md)
- [CPU-first runtime](https://github.com/awais-akhtar/kernelloom/blob/main/docs/CPU_FIRST.md)
- [HTTP API and browser control](https://github.com/awais-akhtar/kernelloom/blob/main/docs/API_SERVER.md)
- [RAG pipeline](https://github.com/awais-akhtar/kernelloom/blob/main/docs/RAG.md)
- [LangChain integration](https://github.com/awais-akhtar/kernelloom/blob/main/docs/LANGCHAIN.md)
- [Compiler and runtime API](https://github.com/awais-akhtar/kernelloom/blob/main/docs/ENGINE_API.md)
- [Architecture](https://github.com/awais-akhtar/kernelloom/blob/main/docs/ENGINE.md)
- [Deployment, security, and publishing](https://github.com/awais-akhtar/kernelloom/blob/main/docs/DEPLOYMENT.md)

## Development and releases

```bash
git clone https://github.com/awais-akhtar/kernelloom.git
cd kernelloom
python -m venv .venv
python -m pip install -e ".[dev,langchain,server]"
python -m pytest
python -m build
python -m twine check dist/*
```

Every PyPI release uses an explicit `major.minor.patch` version. Update the
version to the next release (for example, `0.4.0` to `0.4.1`), commit it, and
push it to `main`. The publish workflow checks PyPI first and stops if that
version already exists, so the next release must be bumped normally instead of
reusing the same release number. A successful upload still depends on GitHub
environment approval, the PyPI token, package validation, and PyPI availability.

KernelLoom is available under the [MIT License](https://github.com/awais-akhtar/kernelloom/blob/main/LICENSE).
