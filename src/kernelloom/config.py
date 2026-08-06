"""Configuration objects used by the high-level model API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    """Describe one local model and how KernelLoom should run it."""

    model_path: str
    model_id: str = "default"
    backend: str = "auto"
    device: str = "CPU"
    data_dir: str = ""
    context_length: int = 4096
    batch_size: int = 512
    micro_batch_size: int = 0
    threads: int = 0
    batch_threads: int = 0
    gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    offload_kqv: bool = True
    flash_attention: bool = False
    numa: bool = False
    chat_format: str = ""
    seed: int = -1
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    system_prompt: str = ""
    device_config: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_path = str(Path(self.model_path).expanduser().resolve())
        self.model_id = self.model_id.strip() or "default"
        self.backend = self.backend.strip().lower() or "auto"
        self.device = self.device.strip().upper() or "CPU"
        if self.backend not in {"auto", "llama-cpp", "openvino"}:
            raise ValueError("backend must be auto, llama-cpp, or openvino")
        if self.context_length < 128:
            raise ValueError("context_length must be at least 128")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.micro_batch_size < 0 or self.micro_batch_size > self.batch_size:
            raise ValueError("micro_batch_size must be zero (auto) or no larger than batch_size")
        if self.threads < 0 or self.batch_threads < 0:
            raise ValueError("thread counts cannot be negative")
        if self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1 (all), 0 (CPU), or positive")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= self.temperature:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")

    @property
    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        path = Path(self.model_path)
        return "llama-cpp" if path.suffix.lower() == ".gguf" else "openvino"

    @property
    def runtime_data_dir(self) -> str:
        if self.data_dir:
            return str(Path(self.data_dir).expanduser().resolve())
        return str(Path.home() / ".kernelloom")

    def generation(self, **overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return values

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved_backend"] = self.resolved_backend
        return result
