# HTTP API and browser console

KernelLoom can keep named models resident behind a local FastAPI service. The
service provides a browser console, model lifecycle routes, OpenAI-compatible
chat completions, text completions and chat streaming.

## Install and start

Install the server plus a model runtime:

```bash
pip install "kernelloom[server,llama]"
```

Start on the loopback interface:

```bash
kernelloom serve
```

The defaults are:

- Host: `127.0.0.1`
- Port: `11435`
- Browser console: `http://127.0.0.1:11435/`
- API base URL: `http://127.0.0.1:11435/v1`
- Interactive API schema: `http://127.0.0.1:11435/docs`

Choose another port when required:

```bash
kernelloom serve --host 127.0.0.1 --port 8080
```

For a reproducible multi-model service, create `kernelloom.json`:

```json
{
  "server": {"host": "127.0.0.1", "port": 11435, "max_models": 2},
  "models": [
    {"model_path": "./models/chat.gguf", "model_id": "chat", "threads": 8},
    {"model_path": "./models/code.gguf", "model_id": "code", "threads": 8}
  ]
}
```

Relative model paths are resolved beside the configuration file. Start it with
`kernelloom serve --config kernelloom.json`.

## Preload a model

Load a model during server startup:

```bash
kernelloom serve \
  --model-path ./models/model.gguf \
  --model-id assistant \
  --backend auto \
  --device CPU
```

If loading fails, startup fails instead of accepting requests without the
requested model.

## Browser console

Open the service root in a browser. The console provides fields for:

- model path;
- model ID;
- backend;
- device;
- context length;
- optional API key;
- prompt, response limit and temperature.

The console operates on the same API routes documented below. Loaded models
remain resident until replaced, unloaded, or the server stops.

## Authentication

The service has no authentication when `KERNELLOOM_API_KEY` is unset. This is
appropriate only for a loopback-only development service.

Set a key before binding to a network interface:

```powershell
$env:KERNELLOOM_API_KEY = "replace-with-a-long-random-value"
kernelloom serve --host 0.0.0.0
```

Send the key as a bearer token:

```bash
curl http://127.0.0.1:11435/v1/models \
  -H "Authorization: Bearer replace-with-a-long-random-value"
```

All `/v1` routes require the token when configured. `/health`, `/` and the API
schema remain available. Put TLS and stronger access controls in front of the
service before using it across an untrusted network.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health and number of loaded models. |
| `GET` | `/ready` | Readiness, resident model IDs and configured capacity. |
| `GET` | `/metrics` | Dependency-free Prometheus text metrics. |
| `GET` | `/v1/models` | List resident models. |
| `POST` | `/v1/models/load` | Load or replace a named model. |
| `DELETE` | `/v1/models/{model_id}` | Unload a resident model. |
| `POST` | `/v1/chat/completions` | Chat completion, with optional SSE streaming. |
| `POST` | `/v1/completions` | Plain text completion. |

## Load a model

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path":"./models/model.gguf",
    "model_id":"assistant",
    "backend":"auto",
    "device":"CPU",
    "context_length":4096,
    "batch_size":512,
    "micro_batch_size":128,
    "threads":8,
    "batch_threads":8,
    "gpu_layers":0,
    "use_mmap":true,
    "flash_attention":false,
    "system_prompt":"Answer clearly and briefly."
  }'
```

The body accepts the fields defined by `ModelConfig`. Loading another model
with the same `model_id` replaces and closes the previous instance.

An OpenVINO example:

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path":"./models/model-openvino",
    "model_id":"openvino-chat",
    "backend":"openvino",
    "device":"GPU"
  }'
```

## List and unload models

```bash
curl http://127.0.0.1:11435/v1/models
curl -X DELETE http://127.0.0.1:11435/v1/models/assistant
```

## Chat completion

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"assistant",
    "messages":[
      {"role":"system","content":"You explain software architecture."},
      {"role":"user","content":"What is an isolated worker?"}
    ],
    "max_tokens":256,
    "temperature":0.2,
    "top_p":0.9,
    "top_k":50,
    "repetition_penalty":1.05,
    "stop":["END"]
  }'
```

The response follows the familiar `chat.completion` shape. KernelLoom also
adds a `kernelloom` object containing the backend, device and measured request
latency.

## Chat streaming

Set `stream` to `true`:

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"assistant",
    "messages":[{"role":"user","content":"Explain paged KV caches."}],
    "stream":true,
    "max_tokens":256
  }'
```

The response uses server-sent events. Each event contains a
`chat.completion.chunk`, followed by `data: [DONE]`.

## Text completion

```bash
curl http://127.0.0.1:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"assistant",
    "prompt":"Complete this sentence: local inference is",
    "max_tokens":80,
    "temperature":0.3
  }'
```

Text completion currently returns a complete response rather than an event
stream. Use chat completions when streaming is required.

## Python requests client

```python
import requests

base_url = "http://127.0.0.1:11435"
headers = {"Authorization": "Bearer replace-with-a-long-random-value"}

response = requests.post(
    f"{base_url}/v1/chat/completions",
    headers=headers,
    json={
        "model": "assistant",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 128,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

## OpenAI Python client

Install the OpenAI client separately:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="replace-with-a-long-random-value",
)

response = client.chat.completions.create(
    model="assistant",
    messages=[{"role": "user", "content": "Describe tensor parallelism."}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

If authentication is disabled, the client still requires a non-empty local
placeholder for its `api_key` argument.

## Application factory

Embed the FastAPI application in another Python process:

```python
from kernelloom import ModelConfig
from kernelloom.server import create_app

app = create_app(initial_model=ModelConfig(
    "./models/model.gguf",
    model_id="assistant",
))
```

Run it with any ASGI server that supports FastAPI lifespan events. The lifespan
handler loads the initial model and closes all resident models on shutdown.

## Operational notes

- A model ID must be loaded before it can answer completion requests.
- One model instance serializes its backend calls for predictable native access.
- Loading multiple models increases memory use; the service does not overcommit deliberately.
- The resident-model limit defaults to four. Set `--max-models`, the JSON `max_models`, or `KERNELLOOM_MAX_MODELS`.
- `/metrics` reports request, active, completed, failed and total generation-time counters.
- Model paths are local filesystem paths on the server machine.
- Do not expose model-loading routes to untrusted users.
- HTTP status `400` indicates model configuration or loading failure.
- HTTP status `404` indicates an unknown model ID.
- HTTP status `401` indicates an invalid or missing configured bearer token.
