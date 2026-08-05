"""Cross-platform CPU, GPU, NPU, runtime, power, and driver discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from .system_metrics import system_metrics


def accelerator_source_root(accelerator_python: str, fallback: Path) -> Path:
    """Resolve the code root without exposing the host virtualenv to the worker."""

    candidates: list[Path] = []
    configured = (
        os.environ.get("KERNELLOOM_SOURCE_ROOT", "").strip()
        or os.environ.get("OPENAGENT_ENGINE_SOURCE_ROOT", "").strip()
        or os.environ.get("OPENAGENT_SOURCE_ROOT", "").strip()
    )
    if configured:
        candidates.append(Path(configured).expanduser())
    if accelerator_python:
        try:
            executable = Path(accelerator_python).expanduser().resolve()
            candidates.append(executable.parents[2])
        except (OSError, IndexError):
            pass
    candidates.append(fallback)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if (resolved / "openagent_engine").is_dir():
            return resolved
        if (resolved / "src" / "openagent_engine").is_dir():
            return resolved / "src"
    return fallback


def accelerator_environment(source_root: Path) -> dict[str, str]:
    """Build an isolated native-worker environment with deterministic imports."""

    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


@dataclass(frozen=True)
class DeviceProfile:
    id: str
    kind: str
    name: str
    vendor: str
    memory_gb: float
    unified_memory: bool
    compute_tops: float
    memory_bandwidth_gbps: float
    precisions: tuple[str, ...]
    supported_ops: tuple[str, ...]
    backends: tuple[str, ...]
    available: bool = True
    driver_version: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    estimate_source: str = "heuristic-unbenchmarked"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["precisions"] = list(self.precisions)
        payload["supported_ops"] = list(self.supported_ops)
        payload["backends"] = list(self.backends)
        return payload


@dataclass(frozen=True)
class HardwareProfile:
    id: str
    platform: str
    total_ram_gb: float
    available_ram_gb: float
    cpu_threads: int
    devices: tuple[DeviceProfile, ...]
    runtimes: dict[str, Any]
    power: dict[str, Any]
    discovered_at: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "cpu_threads": self.cpu_threads,
            "devices": [device.to_dict() for device in self.devices],
            "runtimes": self.runtimes,
            "power": self.power,
            "discovered_at": self.discovered_at,
            "warnings": list(self.warnings),
        }


class HardwareProfiler:
    """Discover physical devices and the runtimes that can really execute on them."""

    def __init__(self, data_dir: str | Path, *, accelerator_python: str = "") -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.accelerator_python = (
            accelerator_python
            or os.environ.get("KERNELLOOM_ACCELERATOR_PYTHON", "")
            or os.environ.get("OPENAGENT_ENGINE_ACCELERATOR_PYTHON", "")
            or os.environ.get("OPENAGENT_ACCELERATOR_PYTHON", "")
        )
        package_root = Path(__file__).resolve().parents[1]
        self.project_root = accelerator_source_root(self.accelerator_python, package_root)
        self._cached: HardwareProfile | None = None
        self._cached_at = 0.0

    def profile(self, *, force: bool = False) -> HardwareProfile:
        if not force and self._cached and time.monotonic() - self._cached_at < 30:
            return self._cached
        metrics = system_metrics(self.data_dir)
        memory = metrics.get("memory", {})
        cpu_info, gpu_info, npu_info = self._os_devices()
        openvino = self._openvino_devices()
        onnx = self._onnx_runtime()
        llama = self._llama_runtime()
        runtime_map = {
            "openvino": openvino,
            "onnxruntime": onnx,
            "llama_cpp": llama,
        }
        devices: list[DeviceProfile] = []
        devices.append(self._cpu_profile(cpu_info, memory, openvino, onnx, llama))
        for index, item in enumerate(gpu_info):
            devices.append(self._gpu_profile(item, index, memory, openvino, llama))
        for index, item in enumerate(npu_info):
            devices.append(self._npu_profile(item, index, memory, openvino))
        for item in openvino.get("devices", []):
            kind = str(item.get("id", "")).split(".", 1)[0].lower()
            if kind not in {device.kind for device in devices}:
                devices.append(self._openvino_only_profile(item, memory))
        warnings: list[str] = []
        if any(device.kind == "gpu" for device in devices) and not any("openvino" in device.backends or "llama-sycl" in device.backends or "llama-vulkan" in device.backends for device in devices if device.kind == "gpu"):
            warnings.append("GPU hardware is present but no verified local inference backend exposes it yet.")
        if any(device.kind == "npu" for device in devices) and not any("openvino" in device.backends for device in devices if device.kind == "npu"):
            warnings.append("NPU hardware is present but OpenVINO does not expose it yet.")
        profile_data = {
            "platform": platform.platform(),
            "total_ram_gb": float(memory.get("total_gb", 0)),
            "available_ram_gb": float(memory.get("available_gb", 0)),
            "cpu_threads": os.cpu_count() or 1,
            "devices": [device.to_dict() for device in devices],
            "runtimes": runtime_map,
            "power": power_status(),
        }
        profile_id = hashlib.sha256(json.dumps(profile_data, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        result = HardwareProfile(
            id=profile_id,
            platform=profile_data["platform"],
            total_ram_gb=profile_data["total_ram_gb"],
            available_ram_gb=profile_data["available_ram_gb"],
            cpu_threads=profile_data["cpu_threads"],
            devices=tuple(devices),
            runtimes=runtime_map,
            power=profile_data["power"],
            discovered_at=time.time(),
            warnings=tuple(warnings),
        )
        self._cached = result
        self._cached_at = time.monotonic()
        return result

    def benchmark_openvino(
        self,
        device: str,
        *,
        iterations: int = 20,
        dimension: int = 256,
        performance_hint: str = "LATENCY",
        num_streams: str = "",
    ) -> dict[str, Any]:
        python = self.accelerator_python_path()
        if not python:
            raise RuntimeError("The isolated OpenVINO accelerator runtime is not installed.")
        arguments = [
            "benchmark", "--device", device, "--iterations", str(iterations),
            "--dimension", str(dimension), "--performance-hint", performance_hint,
        ]
        if num_streams:
            arguments.extend(["--num-streams", num_streams])
        return self._run_probe(arguments, timeout=300)

    def compile_openvino(self, model_path: str, device: str, *, cache_dir: str = "") -> dict[str, Any]:
        python = self.accelerator_python_path()
        if not python:
            raise RuntimeError("The isolated OpenVINO accelerator runtime is not installed.")
        arguments = ["compile", "--model", str(Path(model_path).expanduser().resolve()), "--device", device]
        if cache_dir:
            arguments.extend(["--cache-dir", str(Path(cache_dir).expanduser().resolve())])
        return self._run_probe(arguments, timeout=600)

    def inspect_onnx(self, model_path: str) -> dict[str, Any]:
        python = self.accelerator_python_path()
        if not python:
            raise RuntimeError("The isolated ONNX inspection runtime is not installed.")
        return self._run_probe(
            ["inspect", "--model", str(Path(model_path).expanduser().resolve())],
            timeout=120,
        )

    def accelerator_python_path(self) -> str:
        candidates = [
            self.accelerator_python,
            str(self.project_root / ".accelerator-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
        ]
        for candidate in candidates:
            if not candidate or not Path(candidate).is_file():
                continue
            resolved = str(Path(candidate).resolve())
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                probe = subprocess.run(
                    [resolved, "-c", "import sys; print(sys.executable)"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                    creationflags=creationflags,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if probe.returncode == 0 and probe.stdout.strip():
                return resolved
        return ""

    def _os_devices(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        if os.name == "nt":
            cpu = self._powershell_json(
                "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress"
            )
            gpu = self._powershell_json(
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
            )
            npu = self._powershell_json(
                "Get-PnpDevice -PresentOnly | Where-Object { $_.Class -eq 'ComputeAccelerator' -or $_.FriendlyName -match '\\bNPU\\b|Neural|AI Boost|\\bVPU\\b' } | Select-Object Class,FriendlyName,InstanceId,Status | ConvertTo-Json -Compress"
            )
            return _first_record(cpu), _records(gpu), _records(npu)
        cpu_name = platform.processor() or platform.machine()
        gpu_records: list[dict[str, Any]] = []
        lspci = shutil.which("lspci")
        if lspci:
            try:
                output = subprocess.run([lspci], capture_output=True, text=True, timeout=5, check=False).stdout
                gpu_records = [{"Name": line.strip()} for line in output.splitlines() if "VGA" in line or "3D controller" in line]
            except OSError:
                pass
        return {"Name": cpu_name, "NumberOfLogicalProcessors": os.cpu_count() or 1}, gpu_records, []

    def _powershell_json(self, command: str) -> Any:
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if not executable:
            return {}
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                [executable, "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {}
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {}

    def _openvino_devices(self) -> dict[str, Any]:
        python = self.accelerator_python_path()
        if not python:
            return {"available": False, "devices": [], "reason": "isolated accelerator runtime not installed"}
        try:
            result = self._run_probe(["devices"], timeout=60)
            result["available"] = result.get("status") == "ok"
            result["python"] = python
            return result
        except RuntimeError as exc:
            return {"available": False, "devices": [], "reason": str(exc), "python": python}

    def _run_probe(self, arguments: list[str], *, timeout: int) -> dict[str, Any]:
        python = self.accelerator_python_path()
        if not python:
            raise RuntimeError("accelerator Python is unavailable")
        env = accelerator_environment(self.project_root)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                [python, "-m", "openagent_engine.probe", *arguments],
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"accelerator probe failed: {exc}") from exc
        output = completed.stdout.strip().splitlines()
        if not output:
            raise RuntimeError(completed.stderr.strip() or "accelerator probe returned no output")
        try:
            result = json.loads(output[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("accelerator probe returned invalid JSON") from exc
        if completed.returncode != 0 or result.get("status") == "error":
            raise RuntimeError(str(result.get("error") or completed.stderr.strip() or "accelerator probe failed"))
        return result

    def _onnx_runtime(self) -> dict[str, Any]:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            return {"available": True, "version": ort.__version__, "providers": ort.get_available_providers()}
        except ImportError:
            return {"available": False, "providers": []}

    def _llama_runtime(self) -> dict[str, Any]:
        executable = (
            os.environ.get("KERNELLOOM_LLAMA_SERVER", "").strip()
            or os.environ.get("OPENAGENT_ENGINE_LLAMA_SERVER", "").strip()
            or os.environ.get("OPENAGENT_LLAMA_SERVER", "").strip()
            or shutil.which("llama-server")
            or ""
        )
        if not executable:
            return {"available": False, "devices": [], "backends": []}
        devices: list[str] = []
        try:
            completed = subprocess.run(
                [executable, "--list-devices"], capture_output=True, text=True, timeout=15, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            text = completed.stdout + "\n" + completed.stderr
            devices = [line.strip() for line in text.splitlines() if line.strip() and any(key in line.lower() for key in ("cuda", "vulkan", "sycl", "gpu", "metal", "hip"))]
        except (OSError, subprocess.TimeoutExpired):
            pass
        lowered = " ".join(devices).lower()
        backends = [backend for backend in ("sycl", "vulkan", "cuda", "hip", "metal") if backend in lowered]
        return {"available": True, "path": str(Path(executable).resolve()), "devices": devices, "backends": backends}

    def _cpu_profile(self, info: dict[str, Any], memory: dict[str, Any], openvino: dict[str, Any], onnx: dict[str, Any], llama: dict[str, Any]) -> DeviceProfile:
        name = str(info.get("Name") or platform.processor() or "CPU")
        backends = []
        if openvino.get("available") and any(str(item.get("id", "")).startswith("CPU") for item in openvino.get("devices", [])):
            backends.append("openvino")
        if onnx.get("available"):
            backends.append("onnxruntime")
        if llama.get("available"):
            backends.append("llama-cpu")
        return DeviceProfile(
            id="cpu:0",
            kind="cpu",
            name=name,
            vendor="intel" if "intel" in name.lower() else "amd" if "amd" in name.lower() else platform.machine().lower(),
            memory_gb=float(memory.get("available_gb", 0)),
            unified_memory=True,
            compute_tops=max(0.5, (os.cpu_count() or 1) * 0.10),
            memory_bandwidth_gbps=float(
                os.environ.get("KERNELLOOM_MEMORY_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_ENGINE_MEMORY_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_MEMORY_BANDWIDTH_GBPS", 25)
            ),
            precisions=("fp32", "fp16", "bf16", "int8", "int4", "int3", "int2"),
            supported_ops=_all_ops(),
            backends=tuple(backends),
            capabilities={"cores": int(info.get("NumberOfCores", os.cpu_count() or 1) or 1), "threads": int(info.get("NumberOfLogicalProcessors", os.cpu_count() or 1) or 1)},
        )

    def _gpu_profile(self, info: dict[str, Any], index: int, memory: dict[str, Any], openvino: dict[str, Any], llama: dict[str, Any]) -> DeviceProfile:
        name = str(info.get("Name") or info.get("name") or f"GPU {index}")
        name_lower = name.lower()
        integrated = any(marker in name_lower for marker in ("iris", "uhd", "arc 1", "integrated", "igpu"))
        reported_bytes = int(info.get("AdapterRAM", 0) or 0)
        memory_match = re.search(r"\((\d+(?:\.\d+)?)\s*GB\)", name, re.IGNORECASE)
        memory_gb = float(memory_match.group(1)) if memory_match else reported_bytes / 1024**3
        if integrated:
            memory_gb = min(memory_gb or float(memory.get("total_gb", 0)) * 0.5, float(memory.get("total_gb", 0)) * 0.75)
        ov_device = next((item for item in openvino.get("devices", []) if str(item.get("id", "")).startswith("GPU")), None)
        if ov_device:
            name = str(ov_device.get("full_name") or name)
            memory_gb = max(memory_gb, float(ov_device.get("memory_bytes", 0) or 0) / 1024**3)
        backends = []
        if ov_device:
            backends.append("openvino")
        for backend in llama.get("backends", []):
            if backend == "sycl" and "intel" in name_lower:
                backends.append("llama-sycl")
            elif backend in {"vulkan", "cuda", "hip", "metal"}:
                backends.append(f"llama-{backend}")
        vendor = "intel" if "intel" in name_lower else "nvidia" if "nvidia" in name_lower else "amd" if any(value in name_lower for value in ("amd", "radeon")) else "unknown"
        return DeviceProfile(
            id=f"gpu:{index}",
            kind="gpu",
            name=name,
            vendor=vendor,
            memory_gb=round(memory_gb, 3),
            unified_memory=integrated,
            compute_tops=12.0 if "arc" in name_lower else 8.0,
            memory_bandwidth_gbps=float(
                os.environ.get("KERNELLOOM_GPU_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_ENGINE_GPU_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_GPU_BANDWIDTH_GBPS", 60 if integrated else 200)
            ),
            precisions=("fp32", "fp16", "bf16", "int8", "int4"),
            supported_ops=tuple(op for op in _all_ops() if op not in {"tokenization", "sampling"}),
            backends=tuple(dict.fromkeys(backends)),
            driver_version=str(info.get("DriverVersion", "")),
            capabilities={"integrated": integrated, "openvino": ov_device or {}},
        )

    def _npu_profile(self, info: dict[str, Any], index: int, memory: dict[str, Any], openvino: dict[str, Any]) -> DeviceProfile:
        name = str(info.get("FriendlyName") or info.get("Name") or f"NPU {index}")
        ov_device = next((item for item in openvino.get("devices", []) if str(item.get("id", "")).startswith("NPU")), None)
        if ov_device:
            name = str(ov_device.get("full_name") or name)
        npu_memory = float((ov_device or {}).get("memory_bytes", 0) or 0) / 1024**3
        return DeviceProfile(
            id=f"npu:{index}",
            kind="npu",
            name=name,
            vendor="intel" if "intel" in name.lower() else "unknown",
            memory_gb=round(npu_memory or float(memory.get("available_gb", 0)) * 0.35, 3),
            unified_memory=True,
            compute_tops=8.0,
            memory_bandwidth_gbps=float(
                os.environ.get("KERNELLOOM_NPU_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_ENGINE_NPU_BANDWIDTH_GBPS")
                or os.environ.get("OPENAGENT_NPU_BANDWIDTH_GBPS", 40)
            ),
            precisions=("fp16", "int8", "int4", "nf4"),
            supported_ops=("embedding", "rms_norm", "layer_norm", "matmul", "attention", "rope", "softmax", "activation", "conv2d", "reshape", "transpose", "residual_add"),
            backends=("openvino",) if ov_device else (),
            driver_version=str((ov_device or {}).get("driver_version", "")),
            capabilities={"status": info.get("Status", ""), "openvino": ov_device or {}, "stateful_llm": True},
        )

    def _openvino_only_profile(self, item: dict[str, Any], memory: dict[str, Any]) -> DeviceProfile:
        kind = str(item.get("id", "")).split(".", 1)[0].lower()
        return DeviceProfile(
            id=f"{kind}:openvino",
            kind=kind,
            name=str(item.get("full_name", item.get("id", kind))),
            vendor="unknown",
            memory_gb=float(memory.get("available_gb", 0)),
            unified_memory=True,
            compute_tops=4.0,
            memory_bandwidth_gbps=30.0,
            precisions=tuple(str(value).lower() for value in item.get("capabilities", [])) or ("fp16", "int8"),
            supported_ops=_all_ops(),
            backends=("openvino",),
            capabilities={"openvino": item},
        )


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def power_status() -> dict[str, Any]:
    if os.name == "nt":
        status = _SYSTEM_POWER_STATUS()
        try:
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
                return {
                    "ac_connected": status.ACLineStatus == 1,
                    "battery_percent": None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent),
                    "battery_saver": bool(status.SystemStatusFlag),
                    "thermal": "not_exposed",
                }
        except (AttributeError, OSError):
            pass
    return {"ac_connected": None, "battery_percent": None, "battery_saver": False, "thermal": "unknown"}


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and value:
        return [value]
    return []


def _first_record(value: Any) -> dict[str, Any]:
    records = _records(value)
    return records[0] if records else {}


def _all_ops() -> tuple[str, ...]:
    return (
        "embedding", "rms_norm", "layer_norm", "quantized_linear", "matmul", "attention",
        "rope", "softmax", "activation", "mixture_of_experts", "residual_add", "conv2d",
        "reshape", "transpose", "dequantize", "sampling", "tokenization", "tensor_load", "unknown",
    )
