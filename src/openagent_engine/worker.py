"""Persistent isolated OpenVINO execution worker.

The worker owns native runtime objects in Python 3.12 and communicates with the
host process through newline-delimited JSON on local pipes. It never
opens a listening socket and never sends telemetry.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable


MAX_INLINE_ELEMENTS = 65_536
MAX_AUTO_INPUT_ELEMENTS = 16_777_216
ALLOWED_DTYPES = {
    "bool", "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
}
ALLOWED_DEVICE_PROPERTIES = {
    "PERFORMANCE_HINT", "NUM_STREAMS", "INFERENCE_PRECISION_HINT", "CACHE_DIR",
    "EXECUTION_MODE_HINT", "PERFORMANCE_HINT_NUM_REQUESTS", "DYNAMIC_QUANTIZATION_GROUP_SIZE",
    "KV_CACHE_PRECISION", "GPU_ENABLE_SDPA_OPTIMIZATION", "GPU_ENABLE_LORA_OPERATION",
    "NPU_TURBO", "NPU_COMPILER_DYNAMIC_QUANTIZATION", "NPU_QDQ_OPTIMIZATION",
    "NPU_COMPILATION_MODE_PARAMS",
    # Applied only when the installed OpenVINO device advertises support.
    "INFERENCE_NUM_THREADS", "ENABLE_CPU_PINNING", "ENABLE_HYPER_THREADING", "SCHEDULING_CORE_TYPE",
}
CPU_TUNABLE_PROPERTIES = {
    "INFERENCE_NUM_THREADS", "ENABLE_CPU_PINNING", "ENABLE_HYPER_THREADING", "SCHEDULING_CORE_TYPE",
}


@dataclass
class GenericSession:
    id: str
    path: str
    device: str
    compiled: Any
    request: Any
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    config: dict[str, Any]
    signature: str
    loaded_at: float
    calls: int = 0
    total_infer_ms: float = 0.0


@dataclass
class LLMSession:
    id: str
    path: str
    device: str
    pipeline: Any
    config: dict[str, Any]
    signature: str
    loaded_at: float
    calls: int = 0
    total_generate_ms: float = 0.0


class HardwareWorker:
    def __init__(self) -> None:
        import openvino as ov  # type: ignore[import-not-found]

        self.ov = ov
        self.core = ov.Core()
        self.models: dict[str, GenericSession | LLMSession] = {}
        self.started_at = time.time()
        self.command_count = 0

    def execute(
        self,
        command: dict[str, Any],
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self.command_count += 1
        action = str(command.get("action", "")).strip().lower()
        if action == "ping":
            return self.status()
        if action == "status":
            return self.status()
        if action == "load":
            return self.load_generic(command)
        if action == "load_llm":
            return self.load_llm(command)
        if action == "infer":
            return self.infer(command)
        if action == "generate":
            return self.generate(command)
        if action == "generate_stream":
            return self.generate_stream(command, emit=emit)
        if action == "benchmark":
            return self.benchmark(command)
        if action == "calibrate":
            return self.calibrate(command)
        if action == "unload":
            return self.unload(str(command.get("model_id", "")))
        if action == "shutdown":
            return {"status": "stopping"}
        raise ValueError(f"Unknown worker action: {action}")

    def status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "commands": self.command_count,
            "openvino_version": getattr(self.ov, "__version__", ""),
            "devices": list(self.core.available_devices),
            "models": [self._session_status(session) for session in self.models.values()],
            "privacy_boundary": "Native models remain inside this local process; commands use inherited stdin/stdout pipes only.",
        }

    def load_generic(self, command: dict[str, Any]) -> dict[str, Any]:
        path = _model_file(command.get("path"))
        device = _device(command.get("device", "CPU"))
        model_id = _model_id(command.get("model_id"), path, device)
        config = _device_config(command.get("config"), cache_dir=command.get("cache_dir"), device=device)
        config = _filter_cpu_properties(self.core, device, config)
        signature = _session_signature(path, device, config)
        existing = self.models.get(model_id)
        if (
            isinstance(existing, GenericSession)
            and existing.path == str(path)
            and existing.device == device
            and existing.signature == signature
        ):
            result = self._session_status(existing)
            result["cache_hit"] = True
            return result
        started = time.perf_counter()
        model = self.core.read_model(str(path))
        read_ms = (time.perf_counter() - started) * 1000
        compile_started = time.perf_counter()
        compiled = self.core.compile_model(model, device, config)
        compile_ms = (time.perf_counter() - compile_started) * 1000
        session = GenericSession(
            id=model_id,
            path=str(path),
            device=device,
            compiled=compiled,
            request=compiled.create_infer_request(),
            inputs=[_port_record(port) for port in compiled.inputs],
            outputs=[_port_record(port) for port in compiled.outputs],
            config=config,
            signature=signature,
            loaded_at=time.time(),
        )
        self.models[model_id] = session
        return {
            **self._session_status(session),
            "cache_hit": False,
            "read_ms": round(read_ms, 3),
            "compile_ms": round(compile_ms, 3),
            "execution_devices": _as_list(_safe_compiled_property(compiled, "EXECUTION_DEVICES", [device])),
            "optimal_requests": _json_safe(_safe_compiled_property(compiled, "OPTIMAL_NUMBER_OF_INFER_REQUESTS", 1)),
        }

    def load_llm(self, command: dict[str, Any]) -> dict[str, Any]:
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        path = _model_directory(command.get("path"))
        device = _device(command.get("device", "GPU"))
        model_id = _model_id(command.get("model_id"), path, device)
        config = _device_config(command.get("config"), cache_dir=command.get("cache_dir"), device=device)
        config = _filter_cpu_properties(self.core, device, config)
        scheduler_config = command.get("scheduler")
        scheduler_signature = _scheduler_signature(scheduler_config)
        signature = _session_signature(path, device, config, scheduler_signature)
        existing = self.models.get(model_id)
        if (
            isinstance(existing, LLMSession)
            and existing.path == str(path)
            and existing.device == device
            and existing.signature == signature
        ):
            result = self._session_status(existing)
            result["cache_hit"] = True
            return result
        if isinstance(scheduler_config, dict) and scheduler_config.get("enabled"):
            scheduler = ov_genai.SchedulerConfig()
            for name in (
                "cache_size", "max_num_batched_tokens", "max_num_seqs", "num_kv_blocks",
                "enable_prefix_caching", "dynamic_split_fuse",
            ):
                if name in scheduler_config:
                    setattr(scheduler, name, scheduler_config[name])
            config["scheduler_config"] = scheduler
        started = time.perf_counter()
        pipeline = ov_genai.LLMPipeline(str(path), device, config)
        load_ms = (time.perf_counter() - started) * 1000
        session = LLMSession(model_id, str(path), device, pipeline, config, signature, time.time())
        self.models[model_id] = session
        return {
            **self._session_status(session),
            "cache_hit": False,
            "load_ms": round(load_ms, 3),
            "continuous_batching": "scheduler_config" in config,
            "execution_evidence": {
                "backend": "openvino-genai",
                "requested_device": device,
                "compiled_pipeline": True,
                "measured": False,
            },
        }

    def infer(self, command: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        session = self._generic(str(command.get("model_id", "")))
        values = command.get("inputs")
        if values == "auto" or values is None:
            input_map = _auto_inputs(session.compiled.inputs)
        elif isinstance(values, dict):
            input_map = {}
            for index, port in enumerate(session.compiled.inputs):
                name = str(port.any_name)
                raw = values.get(name, values.get(str(index)))
                if raw is None:
                    raise ValueError(f"Missing model input: {name}")
                input_map[name] = _decode_tensor(raw)
        else:
            raise ValueError("inputs must be an object keyed by input name/index, or 'auto'")
        started = time.perf_counter()
        outputs = session.request.infer(input_map)
        latency_ms = (time.perf_counter() - started) * 1000
        session.calls += 1
        session.total_infer_ms += latency_ms
        mode = str(command.get("output_mode", "summary"))
        records = []
        for index, port in enumerate(session.compiled.outputs):
            array = np.asarray(outputs[port])
            records.append(_encode_output(str(port.any_name), array, mode=mode))
        return {
            "status": "completed",
            "model_id": session.id,
            "device": session.device,
            "execution_devices": _as_list(_safe_compiled_property(session.compiled, "EXECUTION_DEVICES", [session.device])),
            "latency_ms": round(latency_ms, 4),
            "outputs": records,
        }

    def generate(self, command: dict[str, Any]) -> dict[str, Any]:
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        session = self._llm(str(command.get("model_id", "")))
        prompt = _generation_prompt(session, command)
        generation = _generation_config(command.get("generation"))
        config = ov_genai.GenerationConfig(**generation)
        started = time.perf_counter()
        result = session.pipeline.generate(prompt, config)
        latency_ms = (time.perf_counter() - started) * 1000
        session.calls += 1
        session.total_generate_ms += latency_ms
        text = _generated_text(result)
        return {
            "status": "completed",
            "model_id": session.id,
            "device": session.device,
            "text": text,
            "latency_ms": round(latency_ms, 3),
            "characters": len(text),
            "generation": generation,
            "execution_evidence": {
                "backend": "openvino-genai",
                "device": session.device,
                "resident_model": True,
                "measured": True,
            },
        }

    def generate_stream(
        self,
        command: dict[str, Any],
        *,
        emit: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        import openvino_genai as ov_genai  # type: ignore[import-not-found]

        if emit is None:
            raise ValueError("Streaming generation requires an event emitter")
        session = self._llm(str(command.get("model_id", "")))
        prompt = _generation_prompt(session, command)
        generation = _generation_config(command.get("generation"))
        config = ov_genai.GenerationConfig(**generation)
        pieces: list[str] = []
        first_token_ms: float | None = None
        started = time.perf_counter()

        def streamer(piece: Any) -> bool:
            nonlocal first_token_ms
            text = str(piece)
            if text:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                pieces.append(text)
                emit({"type": "token", "content": text})
            return False

        result = session.pipeline.generate(prompt, config, streamer)
        latency_ms = (time.perf_counter() - started) * 1000
        session.calls += 1
        session.total_generate_ms += latency_ms
        text = "".join(pieces) or _generated_text(result)
        return {
            "status": "completed",
            "model_id": session.id,
            "device": session.device,
            "text": text,
            "latency_ms": round(latency_ms, 3),
            "first_token_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
            "characters": len(text),
            "generation": generation,
            "streamed": bool(pieces),
            "execution_evidence": {
                "backend": "openvino-genai",
                "device": session.device,
                "resident_model": True,
                "measured": True,
            },
        }

    def benchmark(self, command: dict[str, Any]) -> dict[str, Any]:
        session = self._generic(str(command.get("model_id", "")))
        iterations = max(3, min(int(command.get("iterations", 20)), 1000))
        warmup = max(1, min(int(command.get("warmup", 2)), 20))
        values = command.get("inputs", "auto")
        for _ in range(warmup):
            self.infer({"model_id": session.id, "inputs": values, "output_mode": "summary"})
        timings = []
        last = None
        for _ in range(iterations):
            result = self.infer({"model_id": session.id, "inputs": values, "output_mode": "summary"})
            timings.append(float(result["latency_ms"]))
            last = result
        ordered = sorted(timings)
        return {
            "status": "measured",
            "model_id": session.id,
            "device": session.device,
            "execution_devices": last["execution_devices"] if last else [session.device],
            "iterations": iterations,
            "warmup": warmup,
            "latency_ms": {
                "min": round(min(timings), 4),
                "mean": round(statistics.fmean(timings), 4),
                "p50": round(_percentile(ordered, 0.50), 4),
                "p95": round(_percentile(ordered, 0.95), 4),
            },
            "inferences_per_second": round(1000.0 / max(0.000001, statistics.fmean(timings)), 4),
            "output": last["outputs"] if last else [],
        }

    def calibrate(self, command: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        path = _model_file(command.get("path"))
        requested = command.get("devices", ["CPU", "GPU", "NPU"])
        available = set(self.core.available_devices)
        devices = []
        for value in requested if isinstance(requested, list) else []:
            target = _device(value)
            root = target.split(".", 1)[0].split(":", 1)[0]
            if target in available or root in {item.split(".", 1)[0] for item in available}:
                devices.append(target)
        devices = list(dict.fromkeys(devices))
        if "CPU" not in devices and "CPU" in available:
            devices.insert(0, "CPU")
        if not devices:
            raise ValueError("No requested OpenVINO device is available")
        iterations = max(3, min(int(command.get("iterations", 10)), 100))
        tolerance = max(1e-7, min(float(command.get("absolute_tolerance", 0.03)), 1.0))
        cache_root = Path(str(command.get("cache_dir", ""))).expanduser().resolve() if command.get("cache_dir") else None
        sessions: dict[str, GenericSession] = {}
        compile_records: list[dict[str, Any]] = []
        for device in devices:
            model_id = f"calibration-{hashlib.sha256((str(path)+device).encode()).hexdigest()[:16]}"
            cache_dir = str(cache_root / _safe_name(device)) if cache_root else ""
            try:
                record = self.load_generic(
                    {
                        "path": str(path), "device": device, "model_id": model_id,
                        "cache_dir": cache_dir, "config": {"PERFORMANCE_HINT": "LATENCY"},
                    }
                )
                sessions[device] = self._generic(model_id)
                compile_records.append({**record, "valid": True})
            except Exception as exc:
                compile_records.append(
                    {
                        "status": "compile-failed", "device": device, "valid": False,
                        "error": str(exc), "type": type(exc).__name__,
                    }
                )
        if not sessions:
            raise RuntimeError("The model could not compile on any requested OpenVINO device")
        reference_device = "CPU" if "CPU" in sessions else devices[0]
        if reference_device not in sessions:
            reference_device = next(iter(sessions))
        reference_session = sessions[reference_device]
        inputs = _auto_inputs(reference_session.compiled.inputs)
        reference_outputs = reference_session.request.infer(inputs)
        reference_arrays = [np.asarray(reference_outputs[port]).astype(np.float64) for port in reference_session.compiled.outputs]
        results = []
        results.extend(
            {
                "device": record["device"], "valid": False, "status": "compile-failed",
                "error": record["error"], "comparisons": [],
            }
            for record in compile_records if not record.get("valid")
        )
        for device, session in sessions.items():
            mapped_inputs = _remap_inputs(inputs, session.compiled.inputs)
            session.request.infer(mapped_inputs)
            timings = []
            output_arrays = []
            for _ in range(iterations):
                started = time.perf_counter()
                outputs = session.request.infer(mapped_inputs)
                timings.append((time.perf_counter() - started) * 1000)
                output_arrays = [np.asarray(outputs[port]).astype(np.float64) for port in session.compiled.outputs]
            comparisons = []
            valid = len(output_arrays) == len(reference_arrays)
            for reference, output in zip(reference_arrays, output_arrays):
                if reference.shape != output.shape:
                    valid = False
                    comparisons.append({"shape_match": False})
                    continue
                difference = np.abs(reference - output)
                max_absolute = float(np.max(difference)) if difference.size else 0.0
                max_relative = float(np.max(difference / np.maximum(1e-6, np.abs(reference)))) if difference.size else 0.0
                within = bool(np.allclose(reference, output, atol=tolerance, rtol=0.02, equal_nan=False))
                valid = valid and within
                comparisons.append(
                    {
                        "shape_match": True,
                        "max_absolute_error": round(max_absolute, 9),
                        "max_relative_error": round(max_relative, 9),
                        "within_tolerance": within,
                    }
                )
            results.append(
                {
                    "device": device,
                    "valid": valid,
                    "latency_ms": {
                        "mean": round(statistics.fmean(timings), 4),
                        "p50": round(_percentile(sorted(timings), 0.50), 4),
                        "p95": round(_percentile(sorted(timings), 0.95), 4),
                    },
                    "comparisons": comparisons,
                    "execution_devices": _as_list(_safe_compiled_property(session.compiled, "EXECUTION_DEVICES", [device])),
                }
            )
        valid_results = [item for item in results if item["valid"]]
        if not valid_results:
            raise RuntimeError("Every device failed differential output validation")
        winner = min(valid_results, key=lambda item: item["latency_ms"]["mean"])
        winner_session = sessions[winner["device"]]
        keep_id = str(command.get("keep_model_id", "")).strip()
        if keep_id:
            winner_session.id = keep_id
            self.models[keep_id] = winner_session
        for device, session in sessions.items():
            if session is winner_session and keep_id:
                continue
            self.models.pop(session.id, None)
        return {
            "status": "verified",
            "path": str(path),
            "reference_device": reference_device,
            "winner": winner,
            "devices": results,
            "compile_records": compile_records,
            "absolute_tolerance": tolerance,
            "resident_model_id": keep_id,
            "scope": "Exact model, deterministic generated inputs, CPU differential reference, and warmed per-device latency.",
        }

    def unload(self, model_id: str) -> dict[str, Any]:
        existed = self.models.pop(model_id, None) is not None
        return {"status": "unloaded" if existed else "not-found", "model_id": model_id}

    def _generic(self, model_id: str) -> GenericSession:
        session = self.models.get(model_id)
        if not isinstance(session, GenericSession):
            raise KeyError(f"Generic model is not resident: {model_id}")
        return session

    def _llm(self, model_id: str) -> LLMSession:
        session = self.models.get(model_id)
        if not isinstance(session, LLMSession):
            raise KeyError(f"LLM is not resident: {model_id}")
        return session

    @staticmethod
    def _session_status(session: GenericSession | LLMSession) -> dict[str, Any]:
        if isinstance(session, GenericSession):
            average = session.total_infer_ms / session.calls if session.calls else 0.0
            return {
                "status": "resident", "kind": "generic", "model_id": session.id,
                "path": session.path, "device": session.device, "inputs": session.inputs,
                "outputs": session.outputs, "calls": session.calls,
                "average_infer_ms": round(average, 4), "loaded_at": session.loaded_at,
                "configuration_fingerprint": session.signature[:16],
            }
        average = session.total_generate_ms / session.calls if session.calls else 0.0
        return {
            "status": "resident", "kind": "llm", "model_id": session.id,
            "path": session.path, "device": session.device, "calls": session.calls,
            "average_generate_ms": round(average, 4), "loaded_at": session.loaded_at,
            "configuration_fingerprint": session.signature[:16],
        }


def _model_file(value: object) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".onnx", ".xml"}:
        raise ValueError("A local ONNX or OpenVINO XML model file is required")
    return path


def _model_directory(value: object) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("A local OpenVINO GenAI model directory is required")
    required = any((path / name).exists() for name in ("openvino_model.xml", "model.xml"))
    if not required:
        raise ValueError("The model directory does not contain OpenVINO model IR")
    return path


def _device(value: object) -> str:
    device = str(value or "CPU").strip().upper()
    if not device or len(device) > 128 or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-," for character in device):
        raise ValueError("Invalid OpenVINO device target")
    return device


def _model_id(value: object, path: Path, device: str) -> str:
    provided = str(value or "").strip()
    if provided:
        if len(provided) > 128 or any(not (character.isalnum() or character in "._-") for character in provided):
            raise ValueError("model_id contains unsupported characters")
        return provided
    return hashlib.sha256(f"{path}:{device}".encode("utf-8")).hexdigest()[:24]


def _device_config(value: object, *, cache_dir: object = "", device: str = "CPU") -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = {str(key).upper(): item for key, item in source.items() if str(key).upper() in ALLOWED_DEVICE_PROPERTIES}
    result.setdefault("PERFORMANCE_HINT", "LATENCY")
    root = str(device).split(":", 1)[0].split(".", 1)[0].upper()
    if root == "NPU":
        result.setdefault("NPU_TURBO", "YES")
        result.setdefault(
            "NPU_COMPILATION_MODE_PARAMS",
            "optimization-level=2 performance-hint-override=latency",
        )
    elif root == "GPU":
        result.setdefault("EXECUTION_MODE_HINT", "PERFORMANCE")
        result.setdefault("INFERENCE_PRECISION_HINT", "f16")
    if cache_dir:
        cache = Path(str(cache_dir)).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        result["CACHE_DIR"] = str(cache)
    return result


def _filter_cpu_properties(core: Any, device: str, config: dict[str, Any]) -> dict[str, Any]:
    """Avoid passing CPU-only knobs to OpenVINO builds that do not expose them."""

    requested = set(config).intersection(CPU_TUNABLE_PROPERTIES)
    if not requested:
        return config
    try:
        supported = {str(value).upper() for value in core.get_property(device, "SUPPORTED_PROPERTIES")}
    except Exception:
        # A conservative fallback: unknown properties can make compilation
        # fail, so omit only the optional CPU tuning hints.
        supported = set()
    return {
        key: value
        for key, value in config.items()
        if key not in CPU_TUNABLE_PROPERTIES or key in supported
    }


def _scheduler_signature(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "enabled", "cache_size", "max_num_batched_tokens", "max_num_seqs", "num_kv_blocks",
        "enable_prefix_caching", "dynamic_split_fuse",
    }
    return {str(key): item for key, item in value.items() if str(key) in allowed}


def _session_signature(path: Path, device: str, config: dict[str, Any], scheduler: dict[str, Any] | None = None) -> str:
    """Fingerprint effective native options so a changed tuning request reloads."""

    payload = {
        "path": str(path),
        "device": device,
        "config": _json_safe(config),
        "scheduler": _json_safe(scheduler or {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _generation_config(value: object) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = {
        "max_new_tokens": max(1, min(int(source.get("max_new_tokens", 256)), 4096)),
        "repetition_penalty": max(0.1, min(float(source.get("repetition_penalty", 1.05)), 4.0)),
        "apply_chat_template": False,
    }
    if bool(source.get("do_sample", False)):
        result.update(
            {
                "do_sample": True,
                "temperature": max(0.01, min(float(source.get("temperature", 0.7)), 4.0)),
                "top_p": max(0.01, min(float(source.get("top_p", 0.9)), 1.0)),
                "top_k": max(0, min(int(source.get("top_k", 50)), 1000)),
            }
        )
    stop_strings = source.get("stop_strings")
    if isinstance(stop_strings, list):
        result["stop_strings"] = {str(item) for item in stop_strings[:32] if str(item)}
    return result


def _generation_prompt(session: LLMSession, command: dict[str, Any]) -> str:
    messages = command.get("messages")
    if isinstance(messages, list):
        clean_messages = [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in messages if isinstance(item, dict)
        ]
        if not clean_messages:
            raise ValueError("messages must contain at least one role/content item")
        tokenizer = session.pipeline.get_tokenizer()
        prompt = tokenizer.apply_chat_template(clean_messages, True)
    else:
        prompt = str(command.get("prompt", ""))
    if not prompt.strip():
        raise ValueError("prompt or messages are required")
    return prompt


def _auto_inputs(ports: Any) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(42)
    result = {}
    total = 0
    for port in ports:
        shape = _concrete_shape(port.partial_shape)
        elements = int(np.prod(shape, dtype=np.int64))
        total += elements
        if total > MAX_AUTO_INPUT_ELEMENTS:
            raise ValueError("Automatic input generation exceeds the safe element limit")
        dtype = _numpy_dtype(port.element_type)
        if np.issubdtype(dtype, np.floating):
            array = rng.normal(0, 0.25, shape).astype(dtype)
        elif dtype == np.dtype("bool"):
            array = np.zeros(shape, dtype=dtype)
        else:
            array = rng.integers(0, 8, shape, dtype=dtype)
        result[str(port.any_name)] = array
    return result


def _remap_inputs(values: dict[str, Any], ports: Any) -> dict[str, Any]:
    source_values = list(values.values())
    result = {}
    for index, port in enumerate(ports):
        name = str(port.any_name)
        result[name] = values.get(name, source_values[index] if index < len(source_values) else None)
        if result[name] is None:
            raise ValueError(f"Could not map calibration input {name}")
    return result


def _decode_tensor(value: object) -> Any:
    import numpy as np

    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict):
        raise ValueError("Tensor input must be a list or encoded tensor object")
    dtype_name = str(value.get("dtype", "float32")).lower()
    if dtype_name not in ALLOWED_DTYPES:
        raise ValueError(f"Unsupported tensor dtype: {dtype_name}")
    shape = tuple(int(item) for item in value.get("shape", []))
    if not shape or any(item <= 0 or item > 1_000_000 for item in shape):
        raise ValueError("Tensor shape is invalid")
    raw = base64.b64decode(str(value.get("data_base64", "")), validate=True)
    array = np.frombuffer(raw, dtype=np.dtype(dtype_name))
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("Encoded tensor byte count does not match its shape")
    return array.reshape(shape)


def _encode_output(name: str, array: Any, *, mode: str) -> dict[str, Any]:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes()
    record = {
        "name": name,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "elements": int(contiguous.size),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "minimum": _finite_float(np.nanmin(contiguous)) if contiguous.size else None,
        "maximum": _finite_float(np.nanmax(contiguous)) if contiguous.size else None,
        "mean": _finite_float(np.nanmean(contiguous)) if contiguous.size else None,
    }
    if mode == "base64":
        record["data_base64"] = base64.b64encode(raw).decode("ascii")
    elif mode == "inline" and contiguous.size <= MAX_INLINE_ELEMENTS:
        record["data"] = contiguous.tolist()
    return record


def _port_record(port: Any) -> dict[str, Any]:
    return {
        "name": str(port.any_name),
        "shape": _shape_record(port.partial_shape),
        "type": str(port.element_type),
    }


def _shape_record(shape: Any) -> list[Any]:
    values = []
    for dimension in shape:
        values.append(int(dimension.get_length()) if dimension.is_static else {"min": int(dimension.get_min_length()), "max": int(dimension.get_max_length())})
    return values


def _concrete_shape(shape: Any) -> tuple[int, ...]:
    result = []
    for dimension in shape:
        if dimension.is_static:
            value = int(dimension.get_length())
        else:
            minimum = int(dimension.get_min_length())
            maximum = int(dimension.get_max_length())
            value = max(1, minimum)
            if maximum > 0:
                value = min(value, maximum)
        result.append(max(1, value))
    return tuple(result)


def _numpy_dtype(element_type: Any) -> Any:
    import numpy as np

    text = str(element_type).lower().replace("<type:", "").replace(">", "").replace("'", "").strip()
    if text.startswith("type:"):
        text = text.split(":", 1)[1].strip()
    aliases = {
        "bf16": "float32", "boolean": "bool", "f16": "float16",
        "f32": "float32", "f64": "float64", "i8": "int8",
        "i16": "int16", "i32": "int32", "i64": "int64",
        "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
    }
    text = aliases.get(text, text)
    try:
        return np.dtype(text)
    except TypeError:
        return np.dtype("float32")


def _generated_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    texts = getattr(result, "texts", None)
    if texts:
        return str(texts[0])
    return str(result)


def _safe_compiled_property(compiled: Any, name: str, default: Any) -> Any:
    try:
        return compiled.get_property(name)
    except Exception:
        return default


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return [_json_safe(value)]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return round(number, 9) if number == number and abs(number) != float("inf") else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))]


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="jsonl", choices=["jsonl"])
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    if args.parent_pid > 0:
        threading.Thread(
            target=_exit_when_parent_stops,
            args=(args.parent_pid,),
            name="openagent-parent-watchdog",
            daemon=True,
        ).start()
    os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")
    worker = HardwareWorker()
    for line in sys.stdin:
        request_id = ""
        command: dict[str, Any] = {}
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("Worker command must be a JSON object")
            request_id = str(command.get("request_id", ""))
            def emit(event: dict[str, Any]) -> None:
                print(
                    json.dumps(
                        {"request_id": request_id, "status": "event", "event": event},
                        sort_keys=True, default=_json_safe,
                    ),
                    flush=True,
                )

            result = worker.execute(command, emit=emit)
            response = {"request_id": request_id, "status": "ok", "result": result}
        except Exception as exc:
            response = {
                "request_id": request_id,
                "status": "error",
                "error": str(exc),
                "type": type(exc).__name__,
            }
        print(json.dumps(response, sort_keys=True, default=_json_safe), flush=True)
        if command.get("action") == "shutdown":
            break
    return 0


def _exit_when_parent_stops(parent_pid: int) -> None:
    while _process_alive(parent_pid):
        time.sleep(1.0)
    os._exit(0)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):  # type: ignore[attr-defined]
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
