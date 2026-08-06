"""A small, backend-neutral interface for local text generation."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
import asyncio
import hashlib
from pathlib import Path
import threading
import time
from typing import Any, AsyncIterator, Iterator, Mapping, Sequence

from .config import ModelConfig
from .cpu import plan_cpu_execution


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
        self._embedding_cache: OrderedDict[str, array] = OrderedDict()
        self._embedding_cache_bytes = 0
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        self._cache_stats = {"embedding_hits": 0, "embedding_misses": 0, "token_hits": 0, "token_misses": 0}
        self._warmup_info: dict[str, Any] = {}
        self._generation_requests = 0

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
            if self.config.warmup:
                self.warmup()
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
            keys = [_cache_key(text) for text in texts]
            cached: dict[str, Sequence[float]] = {}
            missing: dict[str, str] = {}
            for key, text in zip(keys, texts):
                vector = _cache_get(self._embedding_cache, key)
                if vector is not None:
                    self._cache_stats["embedding_hits"] += 1
                    cached[key] = vector
                else:
                    self._cache_stats["embedding_misses"] += 1
                    missing.setdefault(key, text)
            if not missing:
                return [list(cached[key]) for key in keys]
            missing_keys = list(missing)
            vectors = self._embed_uncached([missing[key] for key in missing_keys])
            for key, vector in zip(missing_keys, vectors):
                normalized = array("f", vector)
                cached[key] = normalized
                self._cache_embedding(key, normalized)
            return [list(cached[key]) for key in keys]

    async def aembed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed, text)

    async def aembed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_many, texts)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the loaded local GGUF tokenizer."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._lock:
            key = _cache_key(text)
            cached = _cache_get(self._token_cache, key)
            if cached is not None:
                self._cache_stats["token_hits"] += 1
                return cached
            self._cache_stats["token_misses"] += 1
            self.load()
            if self.config.resolved_backend != "llama-cpp":
                raise NotImplementedError("High-level token counting currently requires a GGUF model")
            count = len(self._backend.tokenize(text.encode("utf-8"), add_bos=False))
            _cache_put(self._token_cache, key, count, self.config.token_cache_size)
            return count

    async def acount_tokens(self, text: str) -> int:
        return await asyncio.to_thread(self.count_tokens, text)

    def warmup(
        self,
        prompt: str | None = None,
        *,
        iterations: int = 1,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Fault in model pages and tokenizer state before real user traffic.

        Warmup performs actual local work: embedding models run their embedding
        path and chat models generate a bounded response.  It never downloads a
        model or opens a network connection.
        """

        if iterations < 1:
            raise ValueError("iterations must be positive")
        text = (self.config.warmup_prompt if prompt is None else prompt).strip()
        if not text:
            raise ValueError("warmup prompt cannot be empty")
        tokens = self.config.warmup_tokens if max_new_tokens is None else max_new_tokens
        if tokens < 1:
            raise ValueError("warmup max_new_tokens must be positive")
        started = time.perf_counter()
        with self._lock:
            self.load()
            if self.config.embedding:
                for _ in range(iterations):
                    self._embed_uncached([text])
                kind = "embedding"
            else:
                for _ in range(iterations):
                    self.generate(
                        text,
                        max_new_tokens=tokens,
                        temperature=0,
                    )
                kind = "generation"
            self._warmup_info = {
                "warmed": True,
                "kind": kind,
                "iterations": iterations,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "at_unix": round(time.time(), 3),
            }
            return dict(self._warmup_info)

    async def awarmup(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.warmup, *args, **kwargs)

    def cache_info(self) -> dict[str, int]:
        """Return bounded local cache metrics without exposing cached input text."""

        with self._lock:
            return {
                **self._cache_stats,
                "embedding_entries": len(self._embedding_cache),
                "embedding_bytes": self._embedding_cache_bytes,
                "token_entries": len(self._token_cache),
            }

    def clear_caches(self) -> None:
        """Release cached embeddings and token counts while leaving the model warm."""

        with self._lock:
            self._embedding_cache.clear()
            self._embedding_cache_bytes = 0
            self._token_cache.clear()
            for key in self._cache_stats:
                self._cache_stats[key] = 0

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
            self._generation_requests += 1
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
            "warmup": dict(self._warmup_info),
            "cache": self.cache_info(),
            "generation_requests": self._generation_requests,
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
            self.clear_caches()
            self._warmup_info = {}

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
        cpu_plan = plan_cpu_execution(self.config.cpu_profile, reserve_cores=self.config.reserve_cores)
        threads = self.config.threads or cpu_plan.threads
        batch_threads = self.config.batch_threads or cpu_plan.batch_threads
        batch_size = cpu_plan.recommended_batch_size if self.config.auto_batch_size else self.config.batch_size
        micro_batch_size = self.config.micro_batch_size or (
            cpu_plan.recommended_micro_batch_size if self.config.auto_batch_size else min(128, batch_size)
        )
        if micro_batch_size > batch_size:
            raise ValueError("resolved micro_batch_size cannot exceed resolved batch_size")
        started = time.perf_counter()
        options: dict[str, Any] = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.context_length,
            "n_batch": batch_size,
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
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "context_length": self.config.context_length,
            "gpu_layers": self.config.gpu_layers,
            "mmap": self.config.use_mmap,
            "mlock": self.config.use_mlock,
            "flash_attention": self.config.flash_attention,
            "cpu_plan": cpu_plan.to_dict(),
            "load_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _load_openvino(self) -> None:
        from openagent_engine import AdaptiveExecutionEngine

        self._engine = AdaptiveExecutionEngine(self.config.runtime_data_dir)
        device = self.config.device.lower()
        target = device if device in {"auto", "multi", "hetero"} or ":" in device else f"{device}:0"
        device_config = dict(self.config.device_config)
        target_root = target.split(":", 1)[0].split(".", 1)[0].upper()
        if target_root in {"CPU", "AUTO"}:
            cpu_plan = plan_cpu_execution(self.config.cpu_profile, reserve_cores=self.config.reserve_cores)
            device_config.setdefault("INFERENCE_NUM_THREADS", self.config.threads or cpu_plan.threads)
            if self.config.cpu_profile == "throughput":
                device_config.setdefault("PERFORMANCE_HINT", "THROUGHPUT")
            else:
                device_config.setdefault("PERFORMANCE_HINT", "LATENCY")
        self._load_info = self._engine.load_direct_llm(
            self.config.model_path,
            device_id=target,
            model_id=self.config.model_id,
            config=device_config,
            scheduler=self.config.scheduler or None,
        )

    def _embed_uncached(self, texts: Sequence[str]) -> list[list[float]]:
        """Execute a native embedding call; caller holds the model lock."""

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

    def _cache_embedding(self, key: str, value: array) -> None:
        """Store compact float32 vectors in a bounded LRU cache."""

        capacity = self.config.embedding_cache_size
        byte_budget = self.config.embedding_cache_max_bytes
        if capacity <= 0 or byte_budget <= 0:
            return
        previous = self._embedding_cache.pop(key, None)
        if previous is not None:
            self._embedding_cache_bytes -= len(previous) * previous.itemsize
        self._embedding_cache[key] = value
        self._embedding_cache_bytes += len(value) * value.itemsize
        while self._embedding_cache and (
            len(self._embedding_cache) > capacity or self._embedding_cache_bytes > byte_budget
        ):
            _, evicted = self._embedding_cache.popitem(last=False)
            self._embedding_cache_bytes -= len(evicted) * evicted.itemsize

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


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_get(cache: OrderedDict[str, Any], key: str) -> Any:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict[str, Any], key: str, value: Any, capacity: int) -> None:
    if capacity <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)
