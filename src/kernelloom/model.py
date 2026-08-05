"""A small, backend-neutral interface for local text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .config import ModelConfig


Message = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model_id: str
    backend: str
    device: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "backend": self.backend,
            "device": self.device,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class KernelLoomModel:
    """Keep a local language model resident and expose one consistent API.

    GGUF files use llama.cpp. OpenVINO GenAI directories use KernelLoom's
    isolated native worker. Dependencies are imported only when that backend
    is selected, keeping the inspection and planning tools lightweight.
    """

    def __init__(self, config: ModelConfig | str, **options: Any) -> None:
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(config, **options)
        self._backend: Any = None
        self._engine: Any = None
        self._loaded = False
        self._load_info: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> "KernelLoomModel":
        with self._lock:
            if self._loaded:
                return self
            path = Path(self.config.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            if self.config.resolved_backend == "llama-cpp":
                self._load_llama_cpp()
            else:
                self._load_openvino()
            self._loaded = True
            return self

    def invoke(
        self,
        value: str | Sequence[Message],
        **generation: Any,
    ) -> str:
        """Return generated text for a prompt or a list of chat messages."""

        return self.generate(value, **generation).text

    def chat(self, messages: Sequence[Message], **generation: Any) -> GenerationResult:
        return self.generate(messages, **generation)

    def generate(
        self,
        value: str | Sequence[Message],
        **generation: Any,
    ) -> GenerationResult:
        with self._lock:
            self.load()
            messages = self._messages(value)
            settings = self.config.generation(**generation)
            started = time.perf_counter()
            if self.config.resolved_backend == "llama-cpp":
                response = self._backend.create_chat_completion(
                    messages=messages,
                    max_tokens=int(settings["max_new_tokens"]),
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    repeat_penalty=float(settings["repetition_penalty"]),
                    stop=settings.get("stop_strings"),
                )
                choice = response["choices"][0]
                text = str(choice.get("message", {}).get("content", ""))
                metadata = {"usage": response.get("usage", {}), "finish_reason": choice.get("finish_reason")}
                device = "GPU" if self.config.gpu_layers else "CPU"
            else:
                response = self._engine.generate_direct(
                    self.config.model_id,
                    messages=messages,
                    generation={**settings, "do_sample": float(settings["temperature"]) > 0},
                )
                text = str(response.get("text", ""))
                metadata = {key: value for key, value in response.items() if key not in {"text", "model_id"}}
                device = str(response.get("device", self.config.device))
            latency_ms = (time.perf_counter() - started) * 1000
            return GenerationResult(
                text=text,
                model_id=self.config.model_id,
                backend=self.config.resolved_backend,
                device=device,
                latency_ms=round(latency_ms, 3),
                metadata=metadata,
            )

    def stream(self, value: str | Sequence[Message], **generation: Any) -> Iterator[str]:
        """Yield text fragments as the backend produces them."""

        with self._lock:
            self.load()
            messages = self._messages(value)
            settings = self.config.generation(**generation)
            if self.config.resolved_backend == "llama-cpp":
                chunks = self._backend.create_chat_completion(
                    messages=messages,
                    max_tokens=int(settings["max_new_tokens"]),
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    repeat_penalty=float(settings["repetition_penalty"]),
                    stop=settings.get("stop_strings"),
                    stream=True,
                )
                for chunk in chunks:
                    text = str(chunk["choices"][0].get("delta", {}).get("content", ""))
                    if text:
                        yield text
                return
            for event in self._engine.direct.stream_generate(
                self.config.model_id,
                messages=messages,
                generation={**settings, "do_sample": float(settings["temperature"]) > 0},
            ):
                if event.get("type") == "token" and event.get("content"):
                    yield str(event["content"])

    def info(self) -> dict[str, Any]:
        return {
            "id": self.config.model_id,
            "path": self.config.model_path,
            "backend": self.config.resolved_backend,
            "device": self.config.device,
            "loaded": self.loaded,
            "load": dict(self._load_info),
        }

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                if self._loaded:
                    try:
                        self._engine.unload_direct_model(self.config.model_id)
                    except Exception:
                        pass
                self._engine.close()
            self._backend = None
            self._engine = None
            self._loaded = False

    def __enter__(self) -> "KernelLoomModel":
        return self.load()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _load_llama_cpp(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "GGUF execution needs llama-cpp-python. Install KernelLoom with the 'llama' extra."
            ) from exc
        threads = self.config.threads or max(1, (os.cpu_count() or 2) - 1)
        started = time.perf_counter()
        self._backend = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.context_length,
            n_batch=self.config.batch_size,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=self.config.gpu_layers,
            verbose=False,
        )
        self._load_info = {"threads": threads, "load_ms": round((time.perf_counter() - started) * 1000, 3)}

    def _load_openvino(self) -> None:
        from openagent_engine import AdaptiveExecutionEngine

        self._engine = AdaptiveExecutionEngine(self.config.runtime_data_dir)
        self._load_info = self._engine.load_direct_llm(
            self.config.model_path,
            device_id=self.config.device.lower() + ":0" if ":" not in self.config.device else self.config.device.lower(),
            model_id=self.config.model_id,
            config=self.config.device_config,
            scheduler=self.config.scheduler or None,
        )

    def _messages(self, value: str | Sequence[Message]) -> list[dict[str, str]]:
        if isinstance(value, str):
            messages = [{"role": "user", "content": value}]
        else:
            messages = [
                {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
                for message in value
            ]
        if self.config.system_prompt and not any(item["role"] == "system" for item in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        if not messages or not any(item["content"].strip() for item in messages):
            raise ValueError("A prompt or at least one non-empty message is required")
        return messages
