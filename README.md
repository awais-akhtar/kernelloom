# OpenAgent Engine

OpenAgent Engine is a local Python library for inspecting model files, profiling
available hardware, creating adaptive execution plans, managing paged KV-cache
metadata, scheduling inference work and running supported models through an
isolated hardware worker.

The package is extracted from OpenAgent so it can be embedded in another Python
application without installing the full OpenAgent product.

## Capabilities

- Inspect GGUF v2/v3, SafeTensors, ONNX and OpenVINO IR model containers locally.
- Build graph, tensor and kernel-level intermediate representations.
- Create separate prefill and decode plans under memory and quality constraints.
- Profile CPU, GPU and NPU capabilities without enabling telemetry.
- Maintain paged KV-cache metadata with prefix sharing and deterministic eviction.
- Schedule deadline-aware, priority-aware inference microbatches.
- Validate optional OpenVINO targets before marking execution as verified.
- Keep supported OpenVINO and OpenVINO GenAI models resident in an isolated worker.
- Persist plans, calibration evidence, role profiles and audit events in local SQLite.

Analytical plans are reported as plans. The engine does not claim that a model
was compiled or verified unless the corresponding backend operation completed.

## Install

```bash
python -m pip install openagent-engine
```

The core inspector and planner have no required third-party dependencies. Add an
optional runtime only when the host application needs it:

```bash
python -m pip install "openagent-engine[onnx]"
python -m pip install "openagent-engine[openvino]"
python -m pip install "openagent-engine[genai]"
```

## Python API

```python
from openagent_engine import AdaptiveExecutionEngine

engine = AdaptiveExecutionEngine("./engine-data")
try:
    model = engine.inspect_model("./models/model.gguf")
    plan = engine.compile_model(
        "./models/model.gguf",
        context_tokens=4096,
        memory_budget_gb=12,
        backend_compile=False,
    )
    print(model["source_format"])
    print(plan["status"])
finally:
    engine.close()
```

Lower-level components are also public:

```python
from openagent_engine import AdaptiveCompiler, HardwareProfiler, ModelFrontend

model = ModelFrontend().inspect("./models/model.safetensors")
hardware = HardwareProfiler("./engine-data").profile()
plan = AdaptiveCompiler().compile(model, hardware, context_tokens=2048)
```

## Command line

```bash
openagent-engine hardware --refresh
openagent-engine inspect ./models/model.gguf
openagent-engine compile ./models/model.gguf --memory-budget-gb 12 --no-backend-compile
openagent-engine status
```

Use `--data-dir` to select where SQLite state, plans and runtime caches are kept.

## Optional isolated runtime

For native OpenVINO execution, create a separate environment containing NumPy,
OpenVINO and optionally OpenVINO GenAI. Point the package to its Python
interpreter:

```powershell
$env:OPENAGENT_ENGINE_ACCELERATOR_PYTHON = "D:\runtime\Scripts\python.exe"
```

The worker communicates through inherited stdin/stdout pipes. It does not open a
network port. `OPENAGENT_ACCELERATOR_PYTHON` remains accepted for compatibility
with existing OpenAgent installations.

## Privacy and limits

Model paths, parsed headers, plans and performance evidence remain local. The
package enables no telemetry and does not download models. Hardware support
depends on local drivers, model format, operators and backend versions. A device
being detected does not prove that a particular model can run on it.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

OpenAgent Engine is licensed under the MIT License.
