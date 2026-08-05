# KernelLoom

KernelLoom runs local language models behind a small Python API or an
OpenAI-compatible HTTP service. It supports GGUF models through llama.cpp and
OpenVINO GenAI model directories through an isolated native worker.

The project also includes the lower-level pieces needed to inspect model
containers, profile CPU, GPU and NPU hardware, build execution plans, schedule
inference work and manage paged KV-cache metadata.

## Install

Choose the runtime that matches your model:

```bash
pip install "kernelloom[llama]"       # GGUF models
pip install "kernelloom[openvino]"    # OpenVINO models
pip install "kernelloom[server]"      # HTTP API and browser console
pip install "kernelloom[langchain]"   # LangChain adapter
```

The inspection and planning API has no required third-party dependencies.

## Run a model from Python

```python
from kernelloom import KernelLoomModel, ModelConfig

config = ModelConfig(
    model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
    model_id="qwen-local",
    device="CPU",
    context_length=4096,
    threads=8,
)

with KernelLoomModel(config) as model:
    answer = model.invoke("Explain prefix caching in two paragraphs.")
    print(answer)
```

`KernelLoomModel` accepts a plain prompt or chat messages:

```python
answer = model.chat([
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "Why does quantization help CPU inference?"},
])
print(answer.text)
```

For OpenVINO GenAI, pass the exported model directory and select a device:

```python
model = KernelLoomModel(ModelConfig(
    model_path="./models/phi-4-mini-openvino",
    backend="openvino",
    device="CPU",
))
```

Set `KERNELLOOM_ACCELERATOR_PYTHON` when OpenVINO is installed in a separate
environment. The worker communicates over inherited stdin/stdout pipes and
does not open its own network port.

## LangChain

```python
from kernelloom import KernelLoomModel, ModelConfig
from kernelloom.langchain import KernelLoomChatModel

runtime = KernelLoomModel(ModelConfig("./models/model.gguf"))
llm = KernelLoomChatModel(runtime)

response = llm.invoke("Give me three names for a database migration tool.")
print(response.content)
```

The adapter implements LangChain's chat generation and streaming methods, so
it can be used in chains, agents and runnable pipelines.

## API and browser console

Start the local service:

```bash
kernelloom serve
```

Open `http://127.0.0.1:11435` to load a model and test responses. The default
bind address is loopback-only. If the service will be reachable by other
machines, set `KERNELLOOM_API_KEY` and send it as a bearer token.

Models can also be loaded over the API:

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path":"./models/model.gguf","model_id":"local","device":"CPU"}'
```

Then call the OpenAI-compatible chat endpoint:

```python
import requests

response = requests.post(
    "http://127.0.0.1:11435/v1/chat/completions",
    json={
        "model": "local",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128,
    },
    timeout=120,
)
print(response.json()["choices"][0]["message"]["content"])
```

Any client that can be pointed at a custom OpenAI base URL can use the same
endpoint. Streaming responses use standard server-sent events.

## Inspection and execution planning

```python
from kernelloom import AdaptiveExecutionEngine

engine = AdaptiveExecutionEngine("./engine-data")
try:
    model = engine.inspect_model("./models/model.gguf")
    plan = engine.compile_model(
        "./models/model.gguf",
        context_tokens=4096,
        memory_budget_gb=12,
        backend_compile=False,
    )
    print(model["source_format"], plan["status"])
finally:
    engine.close()
```

Supported inspection formats are GGUF v2/v3, SafeTensors, ONNX and OpenVINO
IR. A planned placement is reported as `planned`; KernelLoom only reports
`compiled` or `verified` when the corresponding backend operation completed.

## Command line

```bash
kernelloom run ./models/model.gguf "Write a short release note."
kernelloom serve --host 127.0.0.1 --port 11435
```

## Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest
python -m build
python -m twine check dist/*
```

KernelLoom is licensed under the MIT License.
