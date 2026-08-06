# KernelLoom documentation

Use the README for installation and a short end-to-end example. These guides
cover individual integration and runtime surfaces in more detail.

| Guide | Contents |
| --- | --- |
| [Getting started](GETTING_STARTED.md) | Model formats, installation, configuration, generation, streaming, CPU tuning and OpenVINO isolation. |
| [HTTP API](API_SERVER.md) | Browser console, authentication, model lifecycle, chat, completions, SSE streaming and client examples. |
| [LangChain](LANGCHAIN.md) | Chat model adapter, messages, prompt pipelines, streaming and lifecycle. |
| [RAG pipeline](RAG.md) | End-to-end ingestion, local persistence, retrieval, citations, async usage, and custom database adapters. |
| [CPU-first runtime](CPU_FIRST.md) | CPU execution profiles, warm models, bounded caches, hardware selection, and measurement. |
| [Engine API](ENGINE_API.md) | Inspection, planning, direct inference, benchmarks, calibration, KV cache, scheduler and orchestration. |
| [Architecture](ENGINE.md) | Frontends, compiler, hardware layer, native worker and persistence boundaries. |
| [Deployment](DEPLOYMENT.md) | Service security, environment variables, data, health checks, builds and PyPI automation. |

## Recommended path

1. Start with [Getting started](GETTING_STARTED.md).
2. Read [CPU-first runtime](CPU_FIRST.md), then choose [HTTP API](API_SERVER.md), [LangChain](LANGCHAIN.md), or the [RAG pipeline](RAG.md) for application integration.
3. Read [Engine API](ENGINE_API.md) when using hardware planning or native worker controls.
4. Review [Deployment](DEPLOYMENT.md) before exposing a service or enabling PyPI publishing.

KernelLoom is alpha software. Treat planning estimates as estimates and verify
the exact model and hardware combination used by your application.
