# HTTP API and browser control

KernelLoom keeps named models resident in a FastAPI process. The service offers
an OpenAI-style subset for chat, text completion, embeddings, and chat
streaming. It also exposes local control routes for models, hardware, CPU plans,
runtime configuration, and RAG collections. It is not a full OpenAI API
implementation.

## Start the service

Install the server and one model runtime:

```bash
pip install "kernelloom[server,llama]"
kernelloom serve --host 127.0.0.1 --port 11435
```

Default addresses:

- Browser control: `http://127.0.0.1:11435/`
- API base: `http://127.0.0.1:11435/v1`
- Interactive schema: `http://127.0.0.1:11435/docs`
- Metrics: `http://127.0.0.1:11435/metrics`

For reproducible startup, create `kernelloom.json`:

```json
{
  "server": {"host": "127.0.0.1", "port": 11435, "max_models": 2},
  "models": [
    {
      "model_path": "./models/chat.gguf",
      "model_id": "chat",
      "cpu_profile": "latency",
      "warmup": true
    },
    {
      "model_path": "./models/embed.gguf",
      "model_id": "embed",
      "embedding": true
    }
  ]
}
```

```bash
kernelloom serve --config kernelloom.json
```

Relative paths are resolved beside the JSON file. If a configured model cannot
load, startup fails before the server accepts requests.

## Browser control

The page at `/` is a dependency-free client for the same `/v1` routes. It can:

- edit every `ModelConfig` setting, including CPU threads, memory mapping,
  GPU layers, cache limits, generation defaults, OpenVINO device JSON, and
  scheduler JSON;
- load or replace a model, reuse an identical resident configuration, warm it,
  clear caches, or unload it;
- inspect local hardware and copy a CPU plan into the model form;
- stream a chat response from a selected model; and
- create memory, SQLite, or optional FAISS RAG collections from loaded chat and
  embedding models.

It can also download the resident-model configuration as `kernelloom.json`.
The page does not browse the server filesystem and does not save settings by
itself. Paths are supplied as text and resolved on the server machine.

## Authentication and network exposure

Set `KERNELLOOM_API_KEY` to require a bearer token on every `/v1` route:

```powershell
$env:KERNELLOOM_API_KEY = "replace-with-a-long-random-value"
kernelloom serve --host 0.0.0.0
```

```bash
curl http://127.0.0.1:11435/v1/models \
  -H "Authorization: Bearer replace-with-a-long-random-value"
```

`/`, `/assets/console.css`, `/assets/console.js`, `/health`, `/ready`, and
`/metrics` remain unauthenticated so a local browser and process checks can
reach them. Put TLS, firewall rules, rate limits, and stronger identity controls
in front of a network-facing service. Do not let untrusted users call model-load
or RAG-ingestion routes: both accept server-local paths.

## Route map

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Lightweight process health and model count. |
| `GET` | `/ready` | Resident model IDs and configured capacity. |
| `GET` | `/metrics` | Prometheus-style request and resident-model metrics. |
| `GET` | `/v1/models` | List resident models. |
| `GET` | `/v1/models/{id}` | Model status plus reusable configuration. |
| `POST` | `/v1/models/load` | Load, reuse, or replace a named model. |
| `DELETE` | `/v1/models/{id}` | Unload a model not used by a RAG collection. |
| `POST` | `/v1/models/{id}/warm` | Run bounded local warmup. |
| `POST` | `/v1/models/{id}/cache/clear` | Clear caches without unloading. |
| `GET` | `/v1/hardware` | Local CPU, GPU, NPU, and runtime profile. |
| `GET` | `/v1/cpu-plan` | CPU profile recommendation. |
| `GET` | `/v1/runtime/config` | Export resident model settings. |
| `POST` | `/v1/runtime/config/validate` | Validate a runtime configuration object. |
| `POST` | `/v1/chat/completions` | OpenAI-style chat completion with optional SSE streaming. |
| `POST` | `/v1/completions` | Plain text completion. |
| `POST` | `/v1/embeddings` | Embeddings from a model configured with `embedding=true`. |
| `GET`, `POST` | `/v1/rag/collections` | List or create managed RAG collections. |
| `GET`, `DELETE` | `/v1/rag/collections/{id}` | Inspect or remove a collection. |
| `POST` | `/v1/rag/collections/{id}/ingest` | Load, split, embed, and upsert local content. |
| `POST` | `/v1/rag/collections/{id}/retrieve` | Return retrieved chunks and scores. |
| `POST` | `/v1/rag/collections/{id}/query` | Retrieve context and run the chat model. |
| `POST` | `/v1/rag/collections/{id}/warm` | Warm model, embedder, store, and optional queries. |
| `DELETE` | `/v1/rag/collections/{id}/namespaces/{namespace}` | Delete one collection namespace. |

## Model lifecycle

Load a GGUF model:

```bash
curl http://127.0.0.1:11435/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path":"./models/chat.gguf",
    "model_id":"chat",
    "backend":"auto",
    "cpu_profile":"latency",
    "reserve_cores":1,
    "auto_batch_size":true,
    "warmup":true
  }'
```

For llama.cpp, `gpu_layers` determines GPU offload when the installed build
supports it. The `device` field selects an OpenVINO target for OpenVINO models.
`device="AUTO"` follows the discovered OpenVINO device preference order; it is
not a benchmark or compatibility guarantee.

An identical load request reuses the current native context. A changed request
replaces a named model only after the replacement loads successfully. A model
referenced by a managed RAG collection cannot be replaced or unloaded until the
collection is removed. This avoids changing an embedding model beneath an index.

```bash
curl -X POST http://127.0.0.1:11435/v1/models/chat/warm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Warm the local assistant.","iterations":1,"max_tokens":1}'

curl -X POST http://127.0.0.1:11435/v1/models/chat/cache/clear
```

## Hardware and CPU plans

```bash
curl "http://127.0.0.1:11435/v1/cpu-plan?profile=throughput&reserve_cores=1"
curl "http://127.0.0.1:11435/v1/hardware?refresh=true"
```

The CPU plan contains thread and batch starting values. It uses process-visible
cores, including a Linux affinity mask when available, but it does not measure
token throughput. The hardware route reports devices exposed by the local system
and installed runtimes; test the target model before treating a listed device as
an execution guarantee.

## Chat, completion, and streaming

```bash
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"chat",
    "messages":[{"role":"user","content":"Explain paged KV caches."}],
    "max_tokens":128,
    "temperature":0.2
  }'
```

The implemented message fields are `role`, `content`, `name`, `tool_call_id`,
and `tool_calls`. Generation settings include `max_tokens`, `temperature`,
`top_p`, `top_k`, `repetition_penalty`, and `stop`. Backend support for tools
and response formats depends on the model and runtime; the HTTP layer does not
add a complete OpenAI tools or responses API.

Set `stream` to `true` for server-sent events:

```bash
curl -N http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"chat",
    "messages":[{"role":"user","content":"Give three cache tips."}],
    "max_tokens":128,
    "stream":true
  }'
```

The service emits `chat.completion.chunk` data events and ends with `[DONE]`.
Text completion is non-streaming:

```bash
curl http://127.0.0.1:11435/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"chat","prompt":"Complete: local inference is","max_tokens":64}'
```

## Embeddings

Load an embedding GGUF model with `embedding=true`, then request vectors:

```bash
curl http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"embed","input":["first document","second document"]}'
```

The endpoint uses the model's local embedding path and returns vectors in
OpenAI-style `data` entries. It rejects a model not configured for embeddings.

## Managed RAG collections

Create a collection from two resident models:

```bash
curl -X POST http://127.0.0.1:11435/v1/rag/collections \
  -H "Content-Type: application/json" \
  -d '{
    "id":"manuals",
    "model":"chat",
    "embedding_model":"embed",
    "database":"./data/manuals.db",
    "config":{
      "namespace":"product",
      "chunk_size":900,
      "chunk_overlap":120,
      "retrieval":"mmr",
      "top_k":4,
      "fetch_k":12
    }
  }'
```

`database` accepts `memory`, `faiss`, or a SQLite path. `faiss` creates an
in-memory exact-search index and requires `pip install "kernelloom[faiss]"`.
SQLite is persistent but scores stored vectors in Python, so it is a practical
local store rather than a high-scale ANN database.

Ingest a server-local directory, supported text file, or inline text:

```bash
curl -X POST http://127.0.0.1:11435/v1/rag/collections/manuals/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sources":"./knowledge/manuals",
    "metadata":{"product":"kernelloom"},
    "namespace":"product",
    "batch_size":32
  }'
```

Retrieve without generation:

```bash
curl -X POST http://127.0.0.1:11435/v1/rag/collections/manuals/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I configure CPU threads?","filters":{"product":"kernelloom"}}'
```

Ask a question with a retrieval trace:

```bash
curl -X POST http://127.0.0.1:11435/v1/rag/collections/manuals/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"How do I configure CPU threads?",
    "generation":{"max_new_tokens":180,"temperature":0.2}
  }'
```

The response contains the answer and `sources` with document IDs, scores, and
metadata. Source labels in generated text are not verified citations. See
[RAG.md](RAG.md) for store characteristics and custom database adapters.

## OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
response = client.chat.completions.create(
    model="chat",
    messages=[{"role": "user", "content": "What is a KV cache?"}],
)
print(response.choices[0].message.content)
```

When server authentication is enabled, pass the configured bearer token as
`api_key`.

## Runtime configuration export and validation

`GET /v1/runtime/config` returns the resident-model portion of a portable
`kernelloom.json` file. It does not know the host and port that launched the
process, so it uses the standard loopback defaults. Adjust those values before
using it for another deployment.

```bash
curl -X POST http://127.0.0.1:11435/v1/runtime/config/validate \
  -H "Content-Type: application/json" \
  --data-binary @kernelloom.json
```

## Errors and monitoring

- `400` means an invalid configuration, missing optional dependency, source
  load issue, or runtime failure.
- `401` means the bearer token is missing or wrong.
- `404` means a named model or RAG collection is not resident.
- `409` means a RAG collection is using the model, so it cannot be replaced or
  unloaded yet.
- `422` means an API payload has the wrong basic shape.

`/metrics` reports request count, active requests, completed requests, failed
requests, total generation time, resident model count, and active RAG collection
count. It does not publish prompt text or cached embeddings.
