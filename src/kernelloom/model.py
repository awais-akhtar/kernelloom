"""A small, backend-neutral interface for local text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import os
from pathlib import Path
import threading
import time
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from .config import ModelConfig


Message = Mapping[str, Any]


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

    async def aload(self) -> "KernelLoomModel":
        """Load without blocking an async application event loop."""

        return await asyncio.to_thread(self.load)

    async def ainvoke(self, value: str | Sequence[Message], **generation: Any) -> str:
        return (await self.agenerate(value, **generation)).text

    async def achat(self, messages: Sequence[Message], **generation: Any) -> GenerationResult:
        return await self.agenerate(messages, **generation)

    async def agenerate(
        self,
        value: str | Sequence[Message],
        **generation: Any,
    ) -> GenerationResult:
        return await asyncio.to_thread(self.generate, value, **generation)

    async def astream(
        self,
        value: str | Sequence[Message],
        **generation: Any,
    ) -> AsyncIterator[str]:
        """Bridge blocking native streaming into async code with cancellation."""

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        cancelled = threading.Event()

        def publish(kind: str, payload: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
            except RuntimeError:
                pass

        def produce() -> None:
            iterator: Iterator[str] | None = None
            try:
                iterator = self.stream(value, **generation)
                for fragment in iterator:
                    if cancelled.is_set():
                        break
                    publish("token", fragment)
            except BaseException as exc:
                publish("error", exc)
            finally:
                if iterator is not None:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                publish("done", None)

        worker = threading.Thread(target=produce, name=f"kernelloom-{self.config.model_id}", daemon=True)
        worker.start()
        try:
            while True:
                kind, value = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise value
                yield str(value)
        finally:
            cancelled.set()

    def chat(self, messages: Sequence[Message], **generation: Any) -> GenerationResult:
        return self.generate(messages, **generation)

    def embed(self, text: str) -> list[float]:
        """Embed one text with a local GGUF embedding model."""

        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several texts in one backend call."""

        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain at least one non-empty string")
        with self._lock:
            self.load()
            if self.config.resolved_backend != "llama-cpp":
                raise NotImplementedError("High-level embeddings currently require a GGUF embedding model")
            if not self.config.embedding:
                raise RuntimeError("Load the model with embedding=True before requesting embeddings")
            response = self._backend.create_embedding(input=list(texts))
            rows = sorted(response.get("data", []), key=lambda item: int(item.get("index", 0)))
            vectors = [item.get("embedding") for item in rows]
            if len(vectors) != len(texts) or any(not _flat_vector(vector) for vector in vectors):
                raise RuntimeError("The model did not return one sequence-level embedding per input")
            return [[float(value) for value in vector] for vector in vectors]

    async def aembed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed, text)

    async def aembed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_many, texts)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the loaded local GGUF tokenizer."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._lock:
            self.load()
            if self.config.resolved_backend != "llama-cpp":
                raise NotImplementedError("High-level token counting currently requires a GGUF model")
            return len(self._backend.tokenize(text.encode("utf-8"), add_bos=False))

    async def acount_tokens(self, text: str) -> int:
        return await asyncio.to_thread(self.count_tokens, text)

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
                backend_options = {
                    key: settings[key]
                    for key in ("tools", "tool_choice", "response_format")
                    if settings.get(key) is not None
                }
                response = self._backend.create_chat_completion(
                    messages=messages,
                    max_tokens=int(settings["max_new_tokens"]),
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    repeat_penalty=float(settings["repetition_penalty"]),
                    stop=settings.get("stop_strings"),
                    **backend_options,
                )
                choice = response["choices"][0]
                message = choice.get("message", {})
                text = str(message.get("content") or "")
                metadata = {
                    "usage": response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": message.get("tool_calls", []),
                }
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
                backend_options = {
                    key: settings[key]
                    for key in ("tools", "tool_choice", "response_format")
                    if settings.get(key) is not None
                }
                chunks = self._backend.create_chat_completion(
                    messages=messages,
                    max_tokens=int(settings["max_new_tokens"]),
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    top_k=int(settings["top_k"]),
                    repeat_penalty=float(settings["repetition_penalty"]),
                    stop=settings.get("stop_strings"),
                    stream=True,
                    **backend_options,
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
            "embedding": self.config.embedding,
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
        batch_threads = self.config.batch_threads or threads
        micro_batch_size = self.config.micro_batch_size or min(128, self.config.batch_size)
        started = time.perf_counter()
        options: dict[str, Any] = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.context_length,
            "n_batch": self.config.batch_size,
            "n_ubatch": micro_batch_size,
            "n_threads": threads,
            "n_threads_batch": batch_threads,
            "n_gpu_layers": self.config.gpu_layers,
            "use_mmap": self.config.use_mmap,
            "use_mlock": self.config.use_mlock,
            "offload_kqv": self.config.offload_kqv,
            "flash_attn": self.config.flash_attention,
            "numa": self.config.numa,
            "seed": self.config.seed,
            "embedding": self.config.embedding,
            "verbose": False,
        }
        if self.config.chat_format:
            options["chat_format"] = self.config.chat_format
        self._backend = Llama(
            **options,
        )
        self._load_info = {
            "threads": threads,
            "batch_threads": batch_threads,
            "batch_size": self.config.batch_size,
            "micro_batch_size": micro_batch_size,
            "context_length": self.config.context_length,
            "gpu_layers": self.config.gpu_layers,
            "mmap": self.config.use_mmap,
            "mlock": self.config.use_mlock,
            "flash_attention": self.config.flash_attention,
            "load_ms": round((time.perf_counter() - started) * 1000, 3),
        }

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

    def _messages(self, value: str | Sequence[Message]) -> list[dict[str, Any]]:
        if isinstance(value, str):
            messages = [{"role": "user", "content": value}]
        else:
            messages = []
            for message in value:
                item: dict[str, Any] = {
                    "role": str(message.get("role", "user")),
                    "content": message.get("content", ""),
                }
                for key in ("name", "tool_call_id", "tool_calls"):
                    if message.get(key) is not None:
                        item[key] = message[key]
                messages.append(item)
        if self.config.system_prompt and not any(item["role"] == "system" for item in messages):
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})
        if not messages or not any(str(item.get("content", "")).strip() or item.get("tool_calls") for item in messages):
            raise ValueError("A prompt or at least one non-empty message is required")
        return messages


def _flat_vector(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) for item in value)
