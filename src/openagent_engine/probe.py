"""Isolated OpenVINO device, compiler, and numerical-validation probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any


def devices() -> dict[str, Any]:
    import openvino as ov  # type: ignore[import-not-found]

    core = ov.Core()
    result = []
    for device in core.available_devices:
        supported = _safe_property(core, device, "SUPPORTED_PROPERTIES", [])
        supported_names = [str(item) for item in supported]
        item = {
            "id": device,
            "full_name": str(_safe_property(core, device, "FULL_DEVICE_NAME", device)),
            "capabilities": [str(value) for value in _safe_property(core, device, "OPTIMIZATION_CAPABILITIES", [])],
            "architecture": str(_safe_property(core, device, "DEVICE_ARCHITECTURE", "")),
            "type": str(_safe_property(core, device, "DEVICE_TYPE", "")),
            "supported_properties": supported_names,
        }
        for property_name, output_name in (
            ("GPU_DEVICE_TOTAL_MEM_SIZE", "memory_bytes"),
            ("NPU_DEVICE_TOTAL_MEM_SIZE", "memory_bytes"),
            ("NPU_DRIVER_VERSION", "driver_version"),
            ("GPU_DRIVER_VERSION", "driver_version"),
            ("GPU_EXECUTION_UNITS_COUNT", "execution_units"),
        ):
            if property_name in supported_names:
                value = _safe_property(core, device, property_name, None)
                if value is not None:
                    item[output_name] = _json_safe(value)
        result.append(item)
    try:
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        genai_version = getattr(ov_genai, "__version__", "installed")
    except ImportError:
        genai_version = ""
    return {
        "status": "ok",
        "openvino_version": getattr(ov, "__version__", ""),
        "openvino_genai": str(genai_version),
        "python": sys.version.split()[0],
        "devices": result,
    }


def compile_model(model_path: str, device: str, cache_dir: str = "") -> dict[str, Any]:
    import openvino as ov  # type: ignore[import-not-found]

    core = ov.Core()
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"OpenVINO or ONNX model does not exist: {path}")
    config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
    if cache_dir:
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache)
    started = time.perf_counter()
    model = core.read_model(str(path))
    read_ms = (time.perf_counter() - started) * 1000
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, device, config)
    compile_ms = (time.perf_counter() - compile_started) * 1000
    return {
        "status": "compiled",
        "model": str(path),
        "device": device,
        "read_ms": round(read_ms, 3),
        "compile_ms": round(compile_ms, 3),
        "execution_devices": [str(item) for item in _as_list(_safe_compiled_property(compiled, "EXECUTION_DEVICES", [device]))],
        "optimal_requests": _json_safe(_safe_compiled_property(compiled, "OPTIMAL_NUMBER_OF_INFER_REQUESTS", 1)),
        "inputs": [_port_record(port) for port in compiled.inputs],
        "outputs": [_port_record(port) for port in compiled.outputs],
    }


def inspect_onnx(model_path: str) -> dict[str, Any]:
    import onnx  # type: ignore[import-not-found]

    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise ValueError("A local ONNX model path is required")
    model = onnx.load(str(path), load_external_data=False)
    if len(model.graph.node) > 1_000_000 or len(model.graph.initializer) > 1_000_000:
        raise ValueError("ONNX graph exceeds the safe inspection limit")
    initializers = []
    for tensor in model.graph.initializer:
        dimensions = [int(value) for value in tensor.dims]
        elements = 1
        for dimension in dimensions:
            elements *= max(0, dimension)
        element_bytes = _onnx_element_bytes(int(tensor.data_type))
        initializers.append(
            {
                "name": str(tensor.name),
                "shape": dimensions,
                "elements": elements,
                "size_bytes": elements * element_bytes,
                "data_type": int(tensor.data_type),
            }
        )
    nodes = [
        {
            "name": str(node.name or f"{node.op_type}_{index}"),
            "op_type": str(node.op_type),
            "inputs": [str(value) for value in node.input],
            "outputs": [str(value) for value in node.output],
        }
        for index, node in enumerate(model.graph.node)
    ]
    return {
        "status": "ok",
        "path": str(path),
        "ir_version": int(model.ir_version),
        "opset": [int(item.version) for item in model.opset_import],
        "nodes": nodes,
        "initializers": initializers,
        "parameter_count": sum(int(item["elements"]) for item in initializers),
        "weight_bytes": sum(int(item["size_bytes"]) for item in initializers),
    }


def benchmark_device(
    device: str,
    iterations: int = 20,
    dimension: int = 256,
    performance_hint: str = "LATENCY",
    num_streams: str = "",
) -> dict[str, Any]:
    import numpy as np
    import openvino as ov  # type: ignore[import-not-found]
    from openvino import opset13 as ops  # type: ignore[import-not-found]

    iterations = max(3, min(int(iterations), 1000))
    dimension = max(32, min(int(dimension), 2048))
    rng = np.random.default_rng(42)
    left_data = rng.normal(0, 0.25, (1, dimension)).astype(np.float16)
    right_data = rng.normal(0, 0.25, (dimension, dimension)).astype(np.float16)
    left = ops.parameter([1, dimension], np.float16, name="left")
    right = ops.constant(right_data)
    product = ops.matmul(left, right, False, False)
    activated = ops.tanh(product)
    model = ov.Model([activated], [left], "openagent_accelerator_validation")
    core = ov.Core()
    performance_hint = str(performance_hint).strip().upper()
    if performance_hint not in {"LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
        raise ValueError("performance_hint must be LATENCY, THROUGHPUT, or CUMULATIVE_THROUGHPUT")
    config: dict[str, Any] = {"PERFORMANCE_HINT": performance_hint}
    if num_streams:
        config["NUM_STREAMS"] = str(num_streams)
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, device, config)
    compile_ms = (time.perf_counter() - compile_started) * 1000
    request = compiled.create_infer_request()
    request.infer({0: left_data})
    timings: list[float] = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = request.infer({0: left_data})[compiled.output(0)]
        timings.append((time.perf_counter() - started) * 1000)
    assert output is not None
    reference = np.tanh(left_data.astype(np.float32) @ right_data.astype(np.float32))
    max_error = float(np.max(np.abs(output.astype(np.float32) - reference)))
    relative = float(np.max(np.abs(output.astype(np.float32) - reference) / np.maximum(1e-5, np.abs(reference))))
    ordered = sorted(timings)
    return {
        "status": "verified" if max_error <= 0.02 else "failed",
        "device": device,
        "dimension": dimension,
        "iterations": iterations,
        "configuration": {"performance_hint": performance_hint, "num_streams": num_streams or "backend-default"},
        "compile_ms": round(compile_ms, 3),
        "latency_ms": {
            "min": round(min(timings), 4),
            "mean": round(statistics.fmean(timings), 4),
            "p50": round(_percentile(ordered, 0.50), 4),
            "p95": round(_percentile(ordered, 0.95), 4),
        },
        "max_absolute_error": round(max_error, 8),
        "max_relative_error": round(relative, 8),
        "tolerance": {"absolute": 0.02, "reference": "numpy-fp32"},
        "execution_devices": [str(item) for item in _as_list(_safe_compiled_property(compiled, "EXECUTION_DEVICES", [device]))],
    }


def _safe_property(core: Any, device: str, name: str, default: Any) -> Any:
    try:
        return core.get_property(device, name)
    except Exception:
        return default


def _safe_compiled_property(compiled: Any, name: str, default: Any) -> Any:
    try:
        return compiled.get_property(name)
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _port_record(port: Any) -> dict[str, Any]:
    try:
        shape = [int(value) for value in port.partial_shape.get_shape()]
    except Exception:
        shape = [str(value) for value in port.partial_shape]
    return {"name": str(port.any_name), "shape": shape, "type": str(port.element_type)}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
    return values[index]


def _onnx_element_bytes(data_type: int) -> int:
    return {
        1: 4,   # FLOAT
        2: 1,   # UINT8
        3: 1,   # INT8
        4: 2,   # UINT16
        5: 2,   # INT16
        6: 4,   # INT32
        7: 8,   # INT64
        9: 1,   # BOOL
        10: 2,  # FLOAT16
        11: 8,  # DOUBLE
        12: 4,  # UINT32
        13: 8,  # UINT64
        16: 2,  # BFLOAT16
    }.get(data_type, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("devices")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--model", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--model", required=True)
    compile_parser.add_argument("--device", required=True)
    compile_parser.add_argument("--cache-dir", default="")
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--device", required=True)
    benchmark_parser.add_argument("--iterations", type=int, default=20)
    benchmark_parser.add_argument("--dimension", type=int, default=256)
    benchmark_parser.add_argument("--performance-hint", default="LATENCY")
    benchmark_parser.add_argument("--num-streams", default="")
    args = parser.parse_args()
    os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")
    try:
        if args.action == "devices":
            result = devices()
        elif args.action == "inspect":
            result = inspect_onnx(args.model)
        elif args.action == "compile":
            result = compile_model(args.model, args.device, args.cache_dir)
        else:
            result = benchmark_device(
                args.device,
                args.iterations,
                args.dimension,
                args.performance_hint,
                args.num_streams,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "type": type(exc).__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
