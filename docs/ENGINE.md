# KernelLoom engine architecture

KernelLoom separates model understanding, planning and execution into
small modules that applications can use independently.

For callable examples, see the [compiler and runtime API guide](ENGINE_API.md).

## Model frontends

`frontends.py` reads bounded metadata from GGUF, SafeTensors, ONNX and OpenVINO
IR files. Weight payloads are not loaded merely to inspect a container. Parsed
models are represented by the structures in `ir.py`.

## Compiler and lowering

`compiler.py` builds independent prefill and decode plans. It considers memory
capacity, estimated compute, memory traffic, transfers, synchronization and a
caller-supplied quality-loss budget. `codegen.py` emits bounded backend
candidates; final executable compilation remains the responsibility of a real
vendor runtime.

## Hardware and backends

`hardware.py` profiles local CPU, GPU and NPU capabilities and records which
installed runtimes expose each device. `backends.py` validates backend support.
Detection is not reported as successful model execution.

## Stateful runtime

`runtime.py` combines inspection, planning, optional native execution and local
SQLite persistence. `memory.py` provides paged KV-cache metadata and
`scheduler.py` manages compatible, prioritised inference requests.

The optional native worker in `worker.py` is launched by `device_runtime.py`.
It communicates over inherited JSONL pipes and opens no listening socket.

The higher-level `kernelloom.model` module selects llama.cpp for GGUF files and
the isolated OpenVINO GenAI path for exported model directories. The
`kernelloom.server` module keeps named instances resident behind HTTP routes.

## Persistence boundary

`storage.py` is deliberately small. Applications may pass its `EngineStore` to
`AdaptiveExecutionEngine`, or pass a directory and let the engine create the
store. The runtime uses ordinary SQLite sessions plus an audit-event method,
which also makes the boundary straightforward to adapt to another host store.

## Result meanings

- `planned`: an analytical placement was produced.
- `compiled`: the selected backend accepted the model without an outstanding
  planned conversion.
- `verified`: model output passed an explicit numerical comparison.

Synthetic hardware probes and full-model verification are separate evidence.
The package does not treat one as proof of the other.

## Public layers

| Layer | Primary entry point | Intended use |
| --- | --- | --- |
| Model API | `KernelLoomModel` | Prompt, chat and streaming generation. |
| Framework adapter | `KernelLoomChatModel` | LangChain runnables and chat pipelines. |
| HTTP service | `create_app()` or `kernelloom serve` | Browser testing and OpenAI-compatible clients. |
| Engine API | `AdaptiveExecutionEngine` | Inspection, planning, direct execution and evidence. |
| Components | Compiler, frontend, cache and scheduler classes | Custom engine embedding and research. |
