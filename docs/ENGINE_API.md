# Compiler and runtime API

`AdaptiveExecutionEngine` exposes KernelLoom's lower-level model inspection,
hardware profiling, planning, native inference, calibration, scheduling and
persistence features.

## Create and close an engine

```python
from kernelloom import AdaptiveExecutionEngine

engine = AdaptiveExecutionEngine("./engine-data")
try:
    print(engine.readiness())
finally:
    engine.close()
```

The argument may be a directory or an `EngineStore`. A directory creates a
`kernelloom.sqlite3` database plus runtime cache directories beneath it.

Use `accelerator_python` when native dependencies live in another environment:

```python
engine = AdaptiveExecutionEngine(
    "./engine-data",
    accelerator_python="D:/runtimes/openvino/Scripts/python.exe",
)
```

## Status and hardware

```python
status = engine.status(project="demo", refresh_hardware=False)
readiness = engine.readiness(project="demo")
hardware = engine.hardware(refresh=True)
```

- `readiness()` reads bounded local state without forcing a Windows driver probe.
- `hardware()` profiles available CPU, GPU and NPU devices and installed runtimes.
- `status()` combines inventory, backends, scheduler, KV cache and runtime state.

Profiles report discovery and estimates. Device presence does not prove a
particular model is executable on that device.

## Inspect a model

```python
model = engine.inspect_model(
    "./models/model.gguf",
    project="demo",
    include_tensors=False,
)

print(model["source_format"])
print(model["parameter_count"])
print(model["weight_bytes"])
print(model["architecture"])
```

Supported containers are GGUF v2/v3, SafeTensors, ONNX and OpenVINO IR. Header
and graph inspection is bounded; inspecting a container does not load the full
weight payload merely to report metadata.

For incomplete metadata, optional hints can improve analytical planning:

```python
model = engine.inspect_model(
    "./models/model.onnx",
    parameter_count_hint=3_000_000_000,
    quantization_bits_hint=8,
)
```

Hints are inputs, not measured evidence.

## Build an execution plan

```python
plan = engine.compile_model(
    "./models/model.onnx",
    project="demo",
    prompt_tokens=512,
    context_tokens=4096,
    memory_budget_gb=12,
    quality_loss_limit=0.08,
    power_mode="balanced",
    max_device_transitions=4,
    backend_compile=True,
)
```

The compiler creates separate prefill and decode plans. It considers device
memory, estimated compute and memory traffic, transfer costs, precision support,
quality-loss limits and the maximum number of device transitions.

Available power modes are:

- `performance` for the largest practical batches and performance bias;
- `balanced` for the default trade-off;
- `efficiency` for reduced power and batch pressure.

Set `backend_compile=False` to produce an analytical plan without invoking a
vendor compiler.

```python
for saved in engine.plans(project="demo", limit=10):
    print(saved["id"], saved["status"])
```

## Result semantics

KernelLoom keeps planning evidence separate from execution evidence:

- `planned`: an analytical placement exists.
- `compiled`: the vendor backend accepted the source model on the selected target.
- `verified`: execution output passed an explicit numerical comparison.

A planned precision conversion that has not been materialized does not become
`compiled` merely because the unmodified source model compiled.

## OpenVINO GenAI text generation

Use the high-level `KernelLoomModel` for most applications. The direct methods
are useful when controlling the worker explicitly:

```python
loaded = engine.load_direct_llm(
    "./models/model-openvino",
    device_id="cpu:0",
    project="demo",
    model_id="resident-chat",
    scheduler={
        "enabled": True,
        "enable_prefix_caching": True,
        "dynamic_split_fuse": True,
        "max_num_seqs": 4,
    },
)

result = engine.generate_direct(
    "resident-chat",
    project="demo",
    messages=[{"role": "user", "content": "Hello"}],
    generation={
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 0.9,
    },
)

print(result["text"])
engine.unload_direct_model("resident-chat", project="demo")
```

`direct_status(start=False)` reports whether the isolated worker is available
or running. Passing `start=True` starts it and returns live status.

## Generic ONNX and OpenVINO inference

Load an ONNX file or OpenVINO XML IR:

```python
loaded = engine.load_direct_model(
    "./models/encoder.onnx",
    device_id="cpu:0",
    project="demo",
    model_id="encoder",
)
```

Inspect `loaded["inputs"]` before supplying values. For a safe structural smoke
test, ask the worker to generate bounded automatic input tensors:

```python
result = engine.infer_direct(
    "encoder",
    inputs="auto",
    output_mode="summary",
    project="demo",
)
print(result["outputs"])
```

Application input values are keyed by model input name or index:

```python
result = engine.infer_direct(
    "encoder",
    inputs={"input_ids": [[1, 2, 3, 4]]},
    output_mode="inline",
)
```

Output modes are:

- `summary`: shape, dtype, element count, digest and simple numeric statistics;
- `inline`: includes values only when the result remains under the safety limit;
- `base64`: includes the contiguous output bytes encoded as base64.

## Benchmark a resident model

```python
benchmark = engine.benchmark_direct_model(
    "encoder",
    project="demo",
    iterations=20,
    warmup=2,
    inputs="auto",
)
```

This measures the loaded generic model and supplied input shape. It is separate
from the synthetic device benchmark described below.

## Calibrate a model across devices

```python
calibration = engine.calibrate_direct_model(
    "./models/model.onnx",
    project="demo",
    devices=["cpu:0", "gpu:0"],
    iterations=10,
    absolute_tolerance=0.03,
    keep_resident=True,
    model_id="calibrated-model",
)

print(calibration["winner_profile_device"])
print(engine.direct_calibrations(project="demo"))
```

Calibration runs the same model on requested OpenVINO targets, compares output
against a reference within the supplied tolerance, and persists the evidence.
It currently accepts ONNX and OpenVINO IR inputs.

## Benchmark and autotune devices

```python
device_result = engine.benchmark_device(
    "cpu:0",
    project="demo",
    iterations=20,
    dimension=256,
)

tuning = engine.autotune_device(
    "cpu:0",
    project="demo",
    phase="decode",
    iterations=12,
)
```

The device benchmark is a synthetic numerical matrix workload. It is not a
full-model throughput claim. Retrieve saved results with
`engine.benchmarks(project="demo")`.

## Paged KV-cache metadata

KernelLoom's in-process cache tracks token-block ownership, prefix sharing,
copy-on-write behavior and deterministic eviction metadata:

```python
session = engine.create_kv_session(
    "chat-1",
    prefix_tokens=[1, 2, 3, 4],
    priority=80,
)

engine.append_kv("chat-1", [5, 6, 7])
print(engine.status()["kv_cache"])
engine.release_kv("chat-1")
```

This component manages metadata. A backend must connect it to physical model KV
tensors before it represents native cache allocation.

## Deadline-aware scheduling

Queue prefill or decode work:

```python
engine.queue_request(
    request_id="request-1",
    model_id="assistant",
    phase="decode",
    remaining_tokens=128,
    priority=80,
    deadline_seconds=2.0,
    session_id="chat-1",
)

batch = engine.next_batch(power_mode="balanced")
```

Priorities are mapped to the engine's background, normal, interactive and
critical levels. The scheduler reserves capacity for interactive work and
groups compatible requests into bounded microbatches.

## Model roles and orchestration

The runtime includes role configuration for chat, code, reasoning, research,
supervisor, draft, vision and voice workloads. Execution is delegated to a
gateway supplied by the host application.

The gateway contract provides:

- `specs()` returning provider dictionaries;
- `discover_models()` returning available model dictionaries;
- `complete(messages, provider_id=..., model=..., allow_external=...)` returning
  an object with `provider`, `model`, `content` and `metadata` attributes.

Bind and configure it:

```python
engine.bind_gateway(gateway)

engine.upsert_model_role(
    "code",
    project="demo",
    provider_id="local",
    model_id="coder",
    system_prompt="Return complete, testable code.",
)

result = engine.orchestrate_models(
    "Build a parser with unit tests.",
    project="demo",
    mode="supervised-code",
    max_rounds=2,
    allow_external=False,
)
print(result["approved"], result["final_output"])
```

`supervised-code` uses reasoning, code and supervisor roles. `sequence` runs up
to eight supplied roles in order:

```python
result = engine.orchestrate_models(
    "Evaluate this design.",
    mode="sequence",
    roles=["reasoning", "research", "supervisor"],
)
```

Review prior runs with `engine.orchestration_runs(project="demo")`. External
providers require both a configured gateway and `allow_external=True`.

## Projects, persistence and audit

Most methods accept a `project` string. Projects share one SQLite database but
keep model records, plans, benchmarks, calibrations, roles and orchestration
runs logically separated.

Runtime actions add structured audit events through `EngineStore`. Filesystem
paths, plan data and evidence remain local unless the host application sends
them elsewhere.
