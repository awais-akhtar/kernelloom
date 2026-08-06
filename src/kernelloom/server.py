"""OpenAI-style HTTP service, local control API, and browser console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

from . import __version__
from .config import ModelConfig
from .cpu import plan_cpu_execution
from .model import KernelLoomModel
from .rag import InMemoryVectorStore, RAGConfig, RAGPipeline, SQLiteVectorStore


_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(slots=True)
class RuntimeMetrics:
    requests: int = 0
    active: int = 0
    completed: int = 0
    failed: int = 0
    generation_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> float:
        with self._lock:
            self.requests += 1
            self.active += 1
        return time.perf_counter()

    def finish(self, started: float, *, failed: bool = False) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self.failed += int(failed)
            self.completed += int(not failed)
            self.generation_seconds += time.perf_counter() - started

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "requests": self.requests,
                "active": self.active,
                "completed": self.completed,
                "failed": self.failed,
                "generation_seconds": round(self.generation_seconds, 6),
            }


class ModelRegistry:
    """Thread-safe collection of resident local models."""

    def __init__(self, *, max_models: int = 4) -> None:
        self._models: dict[str, KernelLoomModel] = {}
        self._lock = threading.RLock()
        self.max_models = max_models

    def load(self, values: Mapping[str, Any]) -> dict[str, Any]:
        config = ModelConfig(**dict(values))
        with self._lock:
            existing = self._models.get(config.model_id)
            if existing is not None and existing.config.to_dict() == config.to_dict():
                return {**existing.info(), "reused": True}
            if existing is None and len(self._models) >= self.max_models:
                raise RuntimeError(f"Model limit reached ({self.max_models}); unload one before loading another")
            # Keep lifecycle operations serialized: concurrent replacements must
            # never leave two native contexts claiming the same model ID.
            model = KernelLoomModel(config).load()
            previous = self._models.get(config.model_id)
            self._models[config.model_id] = model
        if previous is not None:
            previous.close()
        return model.info()

    def get(self, model_id: str) -> KernelLoomModel:
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            raise KeyError(model_id)
        return model

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [model.info() for model in self._models.values()]

    def configurations(self) -> list[dict[str, Any]]:
        """Return reusable model configuration snapshots under one registry lock."""

        with self._lock:
            return [_model_config_values(model.config) for model in self._models.values()]

    def unload(self, model_id: str) -> bool:
        with self._lock:
            model = self._models.pop(model_id, None)
        if model is None:
            return False
        model.close()
        return True

    def warm(self, model_id: str, **options: Any) -> dict[str, Any]:
        """Warm an already-resident model without replacing its native context."""

        return self.get(model_id).warmup(**options)

    def close(self) -> None:
        with self._lock:
            models = list(self._models.values())
            self._models.clear()
        for model in models:
            model.close()


class _ResidentGenerator:
    """RAG adapter that resolves a server-owned model only when it is used."""

    def __init__(self, models: ModelRegistry, model_id: str) -> None:
        self.models = models
        self.model_id = model_id

    def invoke(self, prompt: str, **generation: Any) -> str:
        return self.models.get(self.model_id).invoke(prompt, **generation)

    async def ainvoke(self, prompt: str, **generation: Any) -> str:
        return await self.models.get(self.model_id).ainvoke(prompt, **generation)

    def warmup(self) -> dict[str, Any]:
        return self.models.warm(self.model_id)


class _ResidentEmbedder:
    """RAG adapter for a model retained by :class:`ModelRegistry`."""

    def __init__(self, models: ModelRegistry, model_id: str) -> None:
        self.models = models
        self.model_id = model_id

    def embed(self, text: str) -> list[float]:
        return self.models.get(self.model_id).embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self.models.get(self.model_id).embed_many(texts)

    def warmup(self) -> dict[str, Any]:
        return self.models.warm(self.model_id)


@dataclass(slots=True)
class _RAGCollection:
    identifier: str
    model_id: str
    embedding_model_id: str
    database: str
    pipeline: RAGPipeline
    embedding_signature: str
    created_at: float


class RAGRegistry:
    """Own RAG stores while leaving resident model ownership with the server."""

    def __init__(self, models: ModelRegistry) -> None:
        self.models = models
        self._collections: dict[str, _RAGCollection] = {}
        self._lock = threading.RLock()

    def create(self, values: Mapping[str, Any]) -> dict[str, Any]:
        identifier = _resource_id(values.get("id"), label="collection id")
        model_id = _required_id(values.get("model"), label="model")
        embedding_model_id = _required_id(values.get("embedding_model"), label="embedding_model")
        raw_config = values.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("config must be an object")
        config = RAGConfig(**dict(raw_config))
        database = str(values.get("database", "memory")).strip() or "memory"

        with self._lock:
            if identifier in self._collections:
                raise ValueError(f"RAG collection '{identifier}' already exists")
        self.models.get(model_id)
        embedding_model = self.models.get(embedding_model_id)
        if not embedding_model.config.embedding:
            raise ValueError("embedding_model must be loaded with embedding=True")

        store = self._new_store(database)
        collection = _RAGCollection(
            identifier=identifier,
            model_id=model_id,
            embedding_model_id=embedding_model_id,
            database=database,
            pipeline=RAGPipeline(
                _ResidentGenerator(self.models, model_id),
                _ResidentEmbedder(self.models, embedding_model_id),
                store=store,
                config=config,
            ),
            embedding_signature=_embedding_signature(embedding_model),
            created_at=time.time(),
        )
        with self._lock:
            if identifier in self._collections:
                collection.pipeline.close()
                raise ValueError(f"RAG collection '{identifier}' already exists")
            self._collections[identifier] = collection
        return self.describe(collection)

    def get(self, identifier: str, *, require_ready: bool = True) -> _RAGCollection:
        with self._lock:
            collection = self._collections.get(identifier)
        if collection is None:
            raise KeyError(identifier)
        if require_ready:
            self._assert_ready(collection)
        return collection

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            collections = list(self._collections.values())
        return [self.describe(item) for item in collections]

    def delete(self, identifier: str) -> bool:
        with self._lock:
            collection = self._collections.pop(identifier, None)
        if collection is None:
            return False
        collection.pipeline.close()
        return True

    def referenced_by(self, model_id: str) -> bool:
        with self._lock:
            return any(
                item.model_id == model_id or item.embedding_model_id == model_id
                for item in self._collections.values()
            )

    def close(self) -> None:
        with self._lock:
            collections = list(self._collections.values())
            self._collections.clear()
        for collection in collections:
            collection.pipeline.close()

    def describe(self, collection: _RAGCollection) -> dict[str, Any]:
        readiness: dict[str, Any] = {"ready": True}
        try:
            self._assert_ready(collection)
        except (KeyError, RuntimeError) as exc:
            readiness = {"ready": False, "reason": str(exc)}
        return {
            "id": collection.identifier,
            "model": collection.model_id,
            "embedding_model": collection.embedding_model_id,
            "database": collection.database,
            "store": type(collection.pipeline.store).__name__,
            "config": asdict(collection.pipeline.config),
            "documents": collection.pipeline.count(),
            "cache": collection.pipeline.cache_info(),
            "created_at": round(collection.created_at, 3),
            **readiness,
        }

    def _assert_ready(self, collection: _RAGCollection) -> None:
        self.models.get(collection.model_id)
        embedding_model = self.models.get(collection.embedding_model_id)
        if _embedding_signature(embedding_model) != collection.embedding_signature:
            raise RuntimeError(
                "The embedding model changed after this collection was indexed. "
                "Delete and rebuild the collection before querying it."
            )

    @staticmethod
    def _new_store(database: str) -> Any:
        normalized = database.lower()
        if normalized == "memory":
            return InMemoryVectorStore()
        if normalized in {"faiss", "faiss-flat"}:
            from .faiss_store import FaissVectorStore

            return FaissVectorStore()
        return SQLiteVectorStore(database)


def create_app(
    *,
    initial_model: ModelConfig | None = None,
    initial_models: list[ModelConfig] | None = None,
    max_models: int | None = None,
) -> Any:
    """Build the FastAPI application without importing server dependencies globally."""

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("The web service needs the 'server' extra: pip install kernelloom[server]") from exc

    limit = max_models or int(os.environ.get("KERNELLOOM_MAX_MODELS", "4"))
    registry = ModelRegistry(max_models=limit)
    rag = RAGRegistry(registry)
    metrics = RuntimeMetrics()
    hardware_profiler: Any | None = None
    hardware_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: Any):
        configured = list(initial_models or [])
        if initial_model is not None:
            configured.append(initial_model)
        for model_config in configured:
            registry.load(_model_config_values(model_config))
        yield
        rag.close()
        registry.close()

    app = FastAPI(
        title="KernelLoom",
        version=__version__,
        description="Local GGUF and OpenVINO model runtime with Python, HTTP, and RAG APIs.",
        lifespan=lifespan,
    )
    app.state.models = registry
    app.state.rag = rag
    app.state.metrics = metrics

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = os.environ.get("KERNELLOOM_API_KEY", "").strip()
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    def profiler() -> Any:
        nonlocal hardware_profiler
        with hardware_lock:
            if hardware_profiler is None:
                from openagent_engine import HardwareProfiler

                data_dir = os.environ.get("KERNELLOOM_DATA_DIR", "").strip() or str(Path.home() / ".kernelloom")
                hardware_profiler = HardwareProfiler(data_dir)
        return hardware_profiler

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console() -> str:
        return _web_asset("console.html")

    @app.get("/assets/{asset_name}", include_in_schema=False)
    def console_asset(asset_name: str) -> Any:
        media_types = {"console.css": "text/css; charset=utf-8", "console.js": "application/javascript; charset=utf-8"}
        if asset_name not in media_types:
            raise HTTPException(status_code=404, detail="Asset not found")
        return Response(_web_asset(asset_name), media_type=media_types[asset_name])

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "models": len(registry.list())}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        loaded = [item["id"] for item in registry.list()]
        return {"ready": bool(loaded), "models": loaded, "capacity": registry.max_models}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        values = metrics.snapshot()
        lines = [
            "# HELP kernelloom_models_loaded Number of resident local models.",
            "# TYPE kernelloom_models_loaded gauge",
            f"kernelloom_models_loaded {len(registry.list())}",
            "# HELP kernelloom_rag_collections Number of active RAG collections.",
            "# TYPE kernelloom_rag_collections gauge",
            f"kernelloom_rag_collections {len(rag.list())}",
        ]
        for name, value in values.items():
            metric = f"kernelloom_{name}"
            lines.extend((f"# TYPE {metric} {'gauge' if name == 'active' else 'counter'}", f"{metric} {value}"))
        return "\n".join(lines) + "\n"

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": item["id"], "object": "model", "owned_by": "local", **item}
                for item in registry.list()
            ],
        }

    @app.get("/v1/models/{model_id}", dependencies=[Depends(authorize)])
    def model_detail(model_id: str) -> dict[str, Any]:
        try:
            model = registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model is not loaded") from exc
        return {**model.info(), "config": _model_config_values(model.config)}

    @app.post("/v1/models/load", dependencies=[Depends(authorize)])
    def load_model(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            candidate = ModelConfig(**payload)
            try:
                current = registry.get(candidate.model_id)
            except KeyError:
                current = None
            if (
                current is not None
                and current.config.to_dict() != candidate.to_dict()
                and rag.referenced_by(candidate.model_id)
            ):
                raise HTTPException(
                    status_code=409,
                    detail="This model is used by a RAG collection. Delete that collection before replacing the model.",
                )
            return registry.load(payload)
        except HTTPException:
            raise
        except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/models/{model_id}", dependencies=[Depends(authorize)])
    def unload_model(model_id: str) -> dict[str, Any]:
        if rag.referenced_by(model_id):
            raise HTTPException(
                status_code=409,
                detail="This model is used by a RAG collection. Delete that collection before unloading the model.",
            )
        if not registry.unload(model_id):
            raise HTTPException(status_code=404, detail="Model is not loaded")
        return {"id": model_id, "deleted": True}

    @app.post("/v1/models/{model_id}/warm", dependencies=[Depends(authorize)])
    def warm_model(model_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        values = payload or {}
        try:
            return {
                "id": model_id,
                **registry.warm(
                    model_id,
                    prompt=values.get("prompt"),
                    iterations=int(values.get("iterations", 1)),
                    max_new_tokens=values.get("max_tokens"),
                ),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model is not loaded") from exc
        except (TypeError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/models/{model_id}/cache/clear", dependencies=[Depends(authorize)])
    def clear_model_cache(model_id: str) -> dict[str, Any]:
        try:
            model = registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Model is not loaded") from exc
        model.clear_caches()
        return {"id": model_id, "cleared": True, "cache": model.cache_info()}

    @app.get("/v1/hardware", dependencies=[Depends(authorize)])
    def hardware(refresh: bool = False) -> dict[str, Any]:
        return profiler().profile(force=refresh).to_dict()

    @app.get("/v1/cpu-plan", dependencies=[Depends(authorize)])
    def cpu_plan(profile: str = "auto", reserve_cores: int = 1) -> dict[str, Any]:
        try:
            return plan_cpu_execution(profile, reserve_cores=reserve_cores).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/runtime/config", dependencies=[Depends(authorize)])
    def runtime_config() -> dict[str, Any]:
        return {
            "server": {"host": "127.0.0.1", "port": 11435, "max_models": registry.max_models},
            "models": registry.configurations(),
        }

    @app.post("/v1/runtime/config/validate", dependencies=[Depends(authorize)])
    def validate_runtime_config(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from .settings import RuntimeConfig

            server = payload.get("server", {})
            models_payload = payload.get("models", [])
            if not isinstance(server, Mapping) or not isinstance(models_payload, list):
                raise ValueError("server must be an object and models must be a list")
            models: list[ModelConfig] = []
            for item in models_payload:
                if not isinstance(item, Mapping):
                    raise ValueError("every model entry must be an object")
                model_values = dict(item)
                model_values.pop("resolved_backend", None)
                models.append(ModelConfig(**model_values))
            config = RuntimeConfig(
                models=models,
                host=str(server.get("host", "127.0.0.1")),
                port=int(server.get("port", 11435)),
                max_models=int(server.get("max_models", registry.max_models)),
            )
            return {
                "valid": True,
                "config": {
                    "server": {"host": config.host, "port": config.port, "max_models": config.max_models},
                    "models": [_model_config_values(item) for item in config.models],
                },
            }
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/rag/collections", dependencies=[Depends(authorize)])
    def rag_collections() -> dict[str, Any]:
        return {"object": "list", "data": rag.list()}

    @app.post("/v1/rag/collections", dependencies=[Depends(authorize)])
    def create_rag_collection(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return rag.create(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Model '{exc.args[0]}' is not loaded") from exc
        except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            status = 409 if "already exists" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.get("/v1/rag/collections/{collection_id}", dependencies=[Depends(authorize)])
    def rag_collection(collection_id: str) -> dict[str, Any]:
        try:
            return rag.describe(rag.get(collection_id, require_ready=False))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection does not exist") from exc

    @app.delete("/v1/rag/collections/{collection_id}", dependencies=[Depends(authorize)])
    def delete_rag_collection(collection_id: str) -> dict[str, Any]:
        if not rag.delete(collection_id):
            raise HTTPException(status_code=404, detail="RAG collection does not exist")
        return {"id": collection_id, "deleted": True}

    @app.post("/v1/rag/collections/{collection_id}/ingest", dependencies=[Depends(authorize)])
    def ingest_rag_collection(collection_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sources = payload.get("sources", payload.get("source"))
        if isinstance(sources, list) and any(not isinstance(value, str) for value in sources):
            raise HTTPException(status_code=422, detail="sources must be a string or a list of strings")
        if not isinstance(sources, (str, list)) or not sources:
            raise HTTPException(status_code=422, detail="sources is required")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise HTTPException(status_code=422, detail="metadata must be an object")
        try:
            collection = rag.get(collection_id)
            namespace = _optional_text(payload.get("namespace"))
            indexed = collection.pipeline.ingest(
                sources,
                metadata=metadata,
                namespace=namespace,
                batch_size=int(payload.get("batch_size", 32)),
            )
            return {
                "id": collection_id,
                "indexed": indexed,
                "namespace": namespace or collection.pipeline.config.namespace,
                "documents": collection.pipeline.count(namespace=namespace),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection or model does not exist") from exc
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rag/collections/{collection_id}/retrieve", dependencies=[Depends(authorize)])
    def retrieve_rag_collection(collection_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            collection = rag.get(collection_id)
            results = collection.pipeline.retrieve(
                _required_text(payload.get("query", payload.get("question")), label="query"),
                filters=_optional_mapping(payload.get("filters"), label="filters"),
                namespace=_optional_text(payload.get("namespace")),
                top_k=_optional_int(payload.get("top_k"), label="top_k"),
            )
            return {"id": collection_id, "data": [_search_result(item) for item in results], "cache": collection.pipeline.cache_info()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection or model does not exist") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rag/collections/{collection_id}/query", dependencies=[Depends(authorize)])
    def query_rag_collection(collection_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            generation = _optional_mapping(payload.get("generation"), label="generation") or {}
            collection = rag.get(collection_id)
            answer = collection.pipeline.ask(
                _required_text(payload.get("question", payload.get("query")), label="question"),
                filters=_optional_mapping(payload.get("filters"), label="filters"),
                namespace=_optional_text(payload.get("namespace")),
                top_k=_optional_int(payload.get("top_k"), label="top_k"),
                generation=generation,
            )
            return {"id": collection_id, **answer.to_dict(), "cache": collection.pipeline.cache_info()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection or model does not exist") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/rag/collections/{collection_id}/warm", dependencies=[Depends(authorize)])
    def warm_rag_collection(collection_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        values = payload or {}
        queries = values.get("queries", [])
        if not isinstance(queries, list) or any(not isinstance(value, str) for value in queries):
            raise HTTPException(status_code=422, detail="queries must be a list of strings")
        try:
            collection = rag.get(collection_id)
            return {"id": collection_id, **collection.pipeline.warmup(queries)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection or model does not exist") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/rag/collections/{collection_id}/namespaces/{namespace}", dependencies=[Depends(authorize)])
    def clear_rag_namespace(collection_id: str, namespace: str) -> dict[str, Any]:
        try:
            collection = rag.get(collection_id)
            deleted = collection.pipeline.clear(namespace=_required_text(namespace, label="namespace"))
            return {"id": collection_id, "namespace": namespace, "deleted": deleted}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="RAG collection or model does not exist") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat_completions(payload: dict[str, Any]) -> Any:
        model = _selected_model(registry, payload, HTTPException)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=422, detail="messages must be a list")
        settings = _generation_settings(payload)
        request_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())
        if payload.get("stream"):
            return StreamingResponse(
                _chat_stream(model, messages, settings, request_id, created, metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        started = metrics.start()
        try:
            result = model.chat(messages, **settings)
        except Exception as exc:
            metrics.finish(started, failed=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        metrics.finish(started)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model.config.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}],
            "usage": result.metadata.get("usage", {}),
            "kernelloom": {"backend": result.backend, "device": result.device, "latency_ms": result.latency_ms},
        }

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    def completions(payload: dict[str, Any]) -> dict[str, Any]:
        model = _selected_model(registry, payload, HTTPException)
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str):
            raise HTTPException(status_code=422, detail="prompt must be a string")
        started = metrics.start()
        try:
            result = model.generate(prompt, **_generation_settings(payload))
        except Exception as exc:
            metrics.finish(started, failed=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        metrics.finish(started)
        return {
            "id": f"cmpl-{uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model.config.model_id,
            "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
            "usage": result.metadata.get("usage", {}),
        }

    @app.post("/v1/embeddings", dependencies=[Depends(authorize)])
    def embeddings(payload: dict[str, Any]) -> dict[str, Any]:
        model = _selected_model(registry, payload, HTTPException)
        values = payload.get("input")
        texts = [values] if isinstance(values, str) else values
        if not isinstance(texts, list) or not texts or any(not isinstance(item, str) for item in texts):
            raise HTTPException(status_code=422, detail="input must be a string or a non-empty list of strings")
        started = metrics.start()
        try:
            vectors = model.embed_many(texts)
            token_count = sum(model.count_tokens(text) for text in texts)
        except Exception as exc:
            metrics.finish(started, failed=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        metrics.finish(started)
        return {
            "object": "list",
            "model": model.config.model_id,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
        }

    return app


def _resource_id(value: Any, *, label: str) -> str:
    identifier = _required_text(value, label=label)
    if not _RESOURCE_ID.fullmatch(identifier):
        raise ValueError(f"{label} must use 1-64 letters, numbers, dots, underscores, or hyphens")
    return identifier


def _required_id(value: Any, *, label: str) -> str:
    return _required_text(value, label=label)


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    return value.strip() or None


def _optional_int(value: Any, *, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _optional_mapping(value: Any, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _embedding_signature(model: KernelLoomModel) -> str:
    values = model.config.to_dict()
    relevant = {key: values[key] for key in ("model_path", "backend", "device", "embedding")}
    return json.dumps(relevant, sort_keys=True, separators=(",", ":"))


def _model_config_values(config: ModelConfig) -> dict[str, Any]:
    return {key: value for key, value in config.to_dict().items() if key != "resolved_backend"}


def _search_result(result: Any) -> dict[str, Any]:
    return {
        "id": result.document.id,
        "text": result.document.text,
        "score": result.score,
        "metadata": dict(result.document.metadata),
    }


def _selected_model(registry: ModelRegistry, payload: dict[str, Any], error: Any) -> KernelLoomModel:
    model_id = str(payload.get("model", "default"))
    try:
        return registry.get(model_id)
    except KeyError as exc:
        raise error(status_code=404, detail=f"Model '{model_id}' is not loaded") from exc


def _generation_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "repetition_penalty": payload.get("repetition_penalty"),
        "stop_strings": payload.get("stop"),
    }


def _chat_stream(
    model: KernelLoomModel,
    messages: list[dict[str, str]],
    settings: dict[str, Any],
    request_id: str,
    created: int,
    metrics: RuntimeMetrics,
) -> Iterator[str]:
    started = metrics.start()
    try:
        for text in model.stream(messages, **settings):
            event = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model.config.model_id,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model.config.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        metrics.finish(started, failed=True)
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}})}\n\n"
        yield "data: [DONE]\n\n"
    else:
        metrics.finish(started)


@lru_cache(maxsize=3)
def _web_asset(name: str) -> str:
    """Read packaged console assets without relying on a source checkout."""

    return files("kernelloom").joinpath("web", name).read_text(encoding="utf-8")
