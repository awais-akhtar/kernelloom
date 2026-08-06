# Getting started

This guide covers installation, model configuration, local generation and
backend selection. See the separate API and LangChain guides when embedding
KernelLoom in a service or framework.

## Choose a model format

KernelLoom has two high-level text-generation paths:

- A `.gguf` file runs through `llama-cpp-python`.
- An OpenVINO GenAI export directory runs through an isolated OpenVINO worker.

ONNX and individual OpenVINO IR files are available through the lower-level
generic tensor inference API. SafeTensors can be inspected and planned, but
must be converted before execution.

KernelLoom does not download a model. Obtain it from a source you trust and
keep its license alongside your application.

## Install a runtime

Create a virtual environment and install the relevant extra:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "kernelloom[llama]"
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "kernelloom[llama]"
```

Replace `llama` with `genai` for OpenVINO GenAI, or install several extras:

```bash
python -m pip install "kernelloom[llama,server,langchain]"
```

## Model configuration

`ModelConfig` controls loading and default generation behavior:

```python
from kernelloom import ModelConfig

config = ModelConfig(
    model_path="./models/model.gguf",
    model_id="assistant",
    backend="auto",
    device="CPU",
    data_dir="./runtime-data",
    context_length=4096,
    batch_size=512,
    micro_batch_size=0,
    threads=0,
    batch_threads=0,
    gpu_layers=0,
    use_mmap=True,
    use_mlock=False,
    offload_kqv=True,
    flash_attention=False,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.05,
    system_prompt="You are a concise assistant.",
)
```

| Setting | Meaning |
| --- | --- |
| `model_path` | GGUF file or OpenVINO GenAI directory. Relative paths are resolved immediately. |
| `model_id` | Name used by the Python result and HTTP API. |
| `backend` | `auto`, `llama-cpp` or `openvino`. Auto selects llama.cpp for `.gguf`. |
| `device` | OpenVINO target such as `CPU`, `GPU` or `NPU`. |
| `data_dir` | SQLite state and compiled-cache directory; defaults to `~/.kernelloom`. |
| `context_length` | llama.cpp context window. Minimum accepted value is 128. |
| `batch_size` | llama.cpp prompt-processing batch size. |
| `micro_batch_size` | Physical llama.cpp batch size; zero selects up to 128. Lower it to reduce peak memory. |
| `threads` | llama.cpp CPU threads. Zero selects logical CPUs minus one. |
| `batch_threads` | Prompt-processing threads; zero follows `threads`. |
| `gpu_layers` | Number of llama.cpp layers to offload; zero keeps execution on CPU. |
| `use_mmap`, `use_mlock` | Map weights from disk or request that mapped pages stay in RAM. |
| `offload_kqv` | Offload K/Q/V operations when the llama.cpp build supports it. |
| `flash_attention` | Request llama.cpp flash attention. Backend and model support are required. |
| `numa` | Enable llama.cpp NUMA placement on suitable multi-socket systems. |
| `chat_format` | Explicit llama.cpp chat template override. Empty uses model metadata. |
| `max_new_tokens` | Default response limit. |
| `temperature` | Sampling temperature. A value of zero requests deterministic generation. |
| `top_p`, `top_k` | Sampling filters. |
| `repetition_penalty` | Default repetition penalty. |
| `system_prompt` | Added when the supplied messages do not already contain a system message. |
| `device_config` | OpenVINO device properties accepted by the worker allow-list. |
| `scheduler` | OpenVINO GenAI continuous-batching scheduler settings. |

Per-call generation values override the configuration defaults.

## Load and invoke

Calling `load()` is optional because the first generation call loads the model.
Explicit loading is useful when startup failures should happen before serving
traffic.

```python
from kernelloom import KernelLoomModel, ModelConfig

model = KernelLoomModel(ModelConfig("./models/model.gguf", model_id="local"))
model.load()

try:
    text = model.invoke("Summarize the purpose of a tokenizer.")
    print(text)
    print(model.info())
finally:
    model.close()
```

The context-manager form loads and closes automatically:

```python
with KernelLoomModel("./models/model.gguf") as model:
    print(model.invoke("Hello"))
```

Keyword arguments supplied with a path are forwarded to `ModelConfig`:

```python
model = KernelLoomModel(
    "./models/model.gguf",
    context_length=8192,
    threads=8,
    system_prompt="Answer with short paragraphs.",
)
```

## Chat messages

Messages use ordinary role/content dictionaries:

```python
result = model.chat(
    [
        {"role": "system", "content": "You explain systems programming."},
        {"role": "user", "content": "What is memory mapping?"},
    ],
    max_new_tokens=200,
    temperature=0.2,
)

print(result.text)
print(result.model_id)
print(result.backend)
print(result.device)
print(result.latency_ms)
print(result.metadata)
```

`GenerationResult.to_dict()` returns the same fields as a serializable
dictionary.

## Streaming

```python
for fragment in model.stream(
    "Explain speculative decoding.",
    max_new_tokens=300,
    temperature=0.3,
):
    print(fragment, end="", flush=True)
```

The iterator yields text fragments. Fragment boundaries are backend-specific
and should not be treated as token boundaries.

## Async applications

Model loading and native inference are blocking operations. KernelLoom provides
async wrappers that move them away from the application event loop:

```python
import asyncio
from kernelloom import KernelLoomModel

async def main():
    model = KernelLoomModel("./models/model.gguf")
    try:
        await model.aload()
        result = await model.agenerate("Explain quantization.")
        print(result.text)
        async for fragment in model.astream("Summarize that."):
            print(fragment, end="", flush=True)
    finally:
        model.close()

asyncio.run(main())
```

Calls to the same model instance are serialized because most native model
contexts are not safe for concurrent mutation. Load separate instances or
named server models when independent concurrency is required and memory allows.

## CPU configuration

Start with a quantized GGUF model that fits comfortably in available RAM:

```python
config = ModelConfig(
    "./models/model-q4_k_m.gguf",
    device="CPU",
    threads=0,
    batch_size=512,
    context_length=4096,
    gpu_layers=0,
)
```

Practical tuning order:

1. Choose a quantization that leaves memory for the context and the operating system.
2. Leave `threads=0` initially, then benchmark nearby values for your CPU.
3. Reduce `context_length` if memory pressure causes swapping.
4. Adjust `batch_size` for prompt-processing throughput after generation is stable.
5. Lower `micro_batch_size` if prompt processing spikes memory.
6. Try `use_mlock=True` only when the process is allowed to lock enough RAM.
7. Keep `gpu_layers=0` for a CPU-only runtime.

Measure the result with `kernelloom benchmark MODEL PROMPT --runs 5`. The
reported characters per second is deliberately labelled as such; it is not a
model-independent tokens-per-second estimate.

KernelLoom does not claim a predicted token rate as measured performance. Test
the exact model, prompt distribution and machine used in production.

## OpenVINO isolation

An OpenVINO runtime may be installed in the application environment, but a
separate environment avoids native dependency conflicts:

```powershell
py -3.12 -m venv D:\runtimes\kernelloom-openvino
D:\runtimes\kernelloom-openvino\Scripts\python.exe -m pip install "kernelloom[genai]"
$env:KERNELLOOM_ACCELERATOR_PYTHON = "D:\runtimes\kernelloom-openvino\Scripts\python.exe"
```

Then select a device:

```python
config = ModelConfig(
    "./models/model-openvino",
    backend="openvino",
    device="GPU",
    scheduler={
        "enabled": True,
        "enable_prefix_caching": True,
        "dynamic_split_fuse": True,
        "max_num_seqs": 4,
    },
)
```

Available devices depend on the installed OpenVINO build and system drivers.

## Errors and cleanup

- `FileNotFoundError` indicates the configured path does not exist.
- `RuntimeError` commonly indicates a missing optional runtime or a backend load failure.
- `ValueError` indicates invalid configuration or an empty prompt.

Always close a model or use it as a context manager. Closing releases the
llama.cpp object or asks the native worker to unload the resident OpenVINO model.
