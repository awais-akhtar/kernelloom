# CPU-first local inference

KernelLoom is designed to make useful local AI possible on machines without a
discrete GPU. The goal is not to promise a universal token rate—model size,
quantization, RAM bandwidth, context length, and workload all matter—but to
provide a safe, measurable starting point and keep repeated work off the CPU.

## The execution algorithm

`plan_cpu_execution()` starts from the CPU cores available to the process. On
Linux it respects process affinity/cgroup limits; elsewhere it uses the logical
core count exposed by the operating system. It then applies one of four modes:

| Profile | Core policy | Best for |
| --- | --- | --- |
| `latency` | Reserve configured system cores; smaller prefill batch | Interactive chat and coding |
| `balanced` / `auto` | Reserve system cores; medium batch | General local assistant use |
| `throughput` | Use all process-visible cores; larger batch | Offline ingestion and batch work |
| `efficient` | Use roughly half of available work cores | Laptops and background tasks |

```python
from kernelloom import plan_cpu_execution

for profile in ("latency", "balanced", "throughput", "efficient"):
    print(plan_cpu_execution(profile).to_dict())
```

The plan is an explainable starting point, not synthetic marketing. Benchmark
the actual model and prompt distribution before committing capacity.

## A fast CPU chat configuration

```python
from kernelloom import KernelLoomModel, ModelConfig

config = ModelConfig(
    "./models/qwen2.5-3b-instruct-q4_k_m.gguf",
    model_id="cpu-chat",
    device="CPU",
    cpu_profile="latency",
    reserve_cores=1,
    auto_batch_size=True,
    context_length=4096,
    warmup=True,
    warmup_tokens=1,
)

with KernelLoomModel(config) as model:
    print(model.invoke("Write a concise release note."))
    print(model.info()["cache"])
```

Choose a quantization that comfortably fits RAM. A smaller model that stays in
RAM will usually beat a larger one that causes paging. Keep context length close
to the real requirement; KV cache memory grows with it.

`auto_batch_size=True` lets the selected profile choose a safe starting batch
size. Leave it off if you are supplying `batch_size` and `micro_batch_size`
explicitly. Explicit `threads` and `batch_threads` always override the profile.

## Warm models and bounded caches

`load()` keeps a model resident. `warmup()` performs actual local work before
you accept users, reducing cold page faults and tokenizer/native initialization
on the first real request:

```python
from kernelloom import KernelLoomModel, ModelConfig

model = KernelLoomModel(ModelConfig("./models/model.gguf"))
try:
    model.load()
    print(model.warmup(prompt="Warm the local assistant.", iterations=1))
    print(model.cache_info())
finally:
    model.close()
```

Embedding and token caches are exact-match LRU caches. Embedding values are
stored as compact float32 arrays and bounded by both entry count and byte budget
(`embedding_cache_size` and `embedding_cache_max_bytes`), so repeated RAG
queries avoid native embedding work without a silent memory leak.

## Direct local hardware selection

KernelLoom's OpenVINO worker owns native device handles through local inherited
pipes—no TCP service, telemetry, or remote accelerator is involved. Use
`device="AUTO"` to prefer a verified local GPU, then NPU, then CPU; use an
explicit target when you need deterministic placement.

```python
from kernelloom import KernelLoomModel, ModelConfig

model = KernelLoomModel(ModelConfig(
    "./models/qwen-openvino",
    backend="openvino",
    device="AUTO",
    cpu_profile="throughput",
    warmup=True,
))
```

OpenVINO GenAI supports exported models, not arbitrary model folders. Common
compatible families include Llama, Qwen, Phi, Mistral, and Gemma when exported
for the installed OpenVINO GenAI version. Use `kernelloom hardware` to inspect
what the local runtime actually exposes.

For OpenVINO CPU targets, KernelLoom passes a profile-derived thread budget and
only lets the worker apply CPU-specific settings when that installed OpenVINO
device advertises them. Changing hardware settings changes the resident-model
signature, preventing a stale native configuration from being silently reused.

## Measure before scaling

```bash
kernelloom benchmark ./models/model-q4_k_m.gguf "Explain prompt caching" --runs 5
```

Measure cold and warm behavior separately. Keep the model loaded, call
`warmup()`, then run representative short and long prompts. Record RAM use,
latency, and throughput for each target machine class rather than extrapolating
from a single laptop.

## What the current runtime guarantees

- GGUF models run through llama.cpp on CPU or supported GPU offload builds.
- Exported OpenVINO GenAI models run through an isolated local CPU/GPU/NPU worker.
- Repeated identical server model loads reuse the resident model.
- Compiled OpenVINO cache directories persist locally for faster repeat loads.
- CPU thread and cache policies are bounded and inspectable through `model.info()`.

Native generation calls on one model are intentionally serialized for context
safety. Scale concurrent users with additional resident instances only when RAM
allows, and benchmark each workload; KernelLoom does not claim transparent
continuous batching where the installed backend cannot provide it.
