"""OpenAI-compatible HTTP service and local browser console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import os
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from .config import ModelConfig
from .model import KernelLoomModel
from . import __version__


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
    """Thread-safe collection of resident models."""

    def __init__(self, *, max_models: int = 4) -> None:
        self._models: dict[str, KernelLoomModel] = {}
        self._lock = threading.RLock()
        self.max_models = max_models

    def load(self, values: dict[str, Any]) -> dict[str, Any]:
        config = ModelConfig(**values)
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


def create_app(
    *,
    initial_model: ModelConfig | None = None,
    initial_models: list[ModelConfig] | None = None,
    max_models: int | None = None,
) -> Any:
    """Build the FastAPI application without importing server dependencies globally."""

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("The web service needs the 'server' extra: pip install kernelloom[server]") from exc

    limit = max_models or int(os.environ.get("KERNELLOOM_MAX_MODELS", "4"))
    registry = ModelRegistry(max_models=limit)
    metrics = RuntimeMetrics()

    @asynccontextmanager
    async def lifespan(_: Any):
        configured = list(initial_models or [])
        if initial_model is not None:
            configured.append(initial_model)
        for model_config in configured:
            registry.load({key: value for key, value in model_config.to_dict().items() if key != "resolved_backend"})
        yield
        registry.close()

    app = FastAPI(
        title="KernelLoom",
        version=__version__,
        description="Developer-friendly local model serving with hardware-aware execution.",
        lifespan=lifespan,
    )
    app.state.models = registry
    app.state.metrics = metrics

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = os.environ.get("KERNELLOOM_API_KEY", "").strip()
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console() -> str:
        return _CONSOLE_HTML

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

    @app.post("/v1/models/load", dependencies=[Depends(authorize)])
    def load_model(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return registry.load(payload)
        except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/models/{model_id}", dependencies=[Depends(authorize)])
    def unload_model(model_id: str) -> dict[str, Any]:
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


def _selected_model(registry: ModelRegistry, payload: dict[str, Any], error: Any) -> KernelLoomModel:
    model_id = str(payload.get("model", "default"))
    try:
        return registry.get(model_id)
    except KeyError as exc:
        raise error(status_code=404, detail=f"Model '{model_id}' is not loaded") from exc


def _generation_settings(payload: dict[str, Any]) -> dict[str, Any]:
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


_CONSOLE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KernelLoom</title><style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#14181d;--line:#29313a;--text:#e9eef3;--muted:#8e9aa7;--accent:#65d1a7;--danger:#ff7c86}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#17342c 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}.shell{max-width:1100px;margin:auto;padding:48px 24px}.brand{display:flex;align-items:center;gap:12px;margin-bottom:30px}.mark{width:38px;height:38px;border:2px solid var(--accent);border-radius:11px;transform:rotate(45deg);box-shadow:0 0 28px #65d1a744}.brand h1{font-size:24px;margin:0}.brand span{color:var(--muted)}.grid{display:grid;grid-template-columns:360px 1fr;gap:18px}.card{background:#14181de8;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 18px 50px #0004}h2{font-size:14px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 18px}label{display:block;color:var(--muted);font-size:12px;margin:12px 0 5px}input,select,textarea,button{width:100%;border:1px solid var(--line);background:#0d1115;color:var(--text);border-radius:8px;padding:10px 11px;font:inherit}textarea{min-height:190px;resize:vertical}button{background:var(--accent);color:#07120e;border:0;font-weight:700;cursor:pointer;margin-top:14px}button.secondary{background:#212830;color:var(--text)}#answer{white-space:pre-wrap;min-height:220px;border:1px solid var(--line);border-radius:8px;padding:14px;background:#0d1115}.status{font-size:12px;color:var(--muted);margin-top:10px}.ok{color:var(--accent)}.error{color:var(--danger)}@media(max-width:760px){.grid{grid-template-columns:1fr}.shell{padding:24px 14px}}
</style></head><body><main class="shell"><div class="brand"><div class="mark"></div><div><h1>KernelLoom</h1><span>Local model runtime</span></div></div><div class="grid"><section class="card"><h2>Load a model</h2><label>Model path</label><input id="path" placeholder="D:\models\model.gguf"><label>Model ID</label><input id="model" value="default"><label>Backend</label><select id="backend"><option value="auto">Auto</option><option value="llama-cpp">llama.cpp</option><option value="openvino">OpenVINO GenAI</option></select><label>Device</label><select id="device"><option>CPU</option><option>GPU</option><option>NPU</option></select><label>Context length</label><input id="context" type="number" value="4096"><label>API key (if configured)</label><input id="apiKey" type="password" autocomplete="off"><button onclick="loadModel()">Load model</button><div id="loadStatus" class="status">No model loaded in this session.</div></section><section class="card"><h2>Test response</h2><label>Prompt</label><textarea id="prompt" placeholder="Ask the loaded model something..."></textarea><div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><div><label>Max tokens</label><input id="tokens" type="number" value="256"></div><div><label>Temperature</label><input id="temperature" type="number" step="0.1" value="0.7"></div></div><button onclick="sendPrompt()">Generate</button><label>Response</label><div id="answer">Ready.</div><div id="responseStatus" class="status"></div></section></div></main><script>
const byId=id=>document.getElementById(id);const headers=()=>{const h={'Content-Type':'application/json'},k=byId('apiKey').value;if(k)h.Authorization=`Bearer ${k}`;return h};async function loadModel(){const status=byId('loadStatus');status.textContent='Loading…';status.className='status';try{const r=await fetch('/v1/models/load',{method:'POST',headers:headers(),body:JSON.stringify({model_path:byId('path').value,model_id:byId('model').value,backend:byId('backend').value,device:byId('device').value,context_length:Number(byId('context').value)})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not load model');status.textContent=`Loaded ${d.id} on ${d.device} with ${d.backend}`;status.className='status ok'}catch(e){status.textContent=e.message;status.className='status error'}}async function sendPrompt(){const answer=byId('answer'),status=byId('responseStatus');answer.textContent='Generating…';status.textContent='';const started=performance.now();try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:headers(),body:JSON.stringify({model:byId('model').value,messages:[{role:'user',content:byId('prompt').value}],max_tokens:Number(byId('tokens').value),temperature:Number(byId('temperature').value)})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Generation failed');answer.textContent=d.choices[0].message.content;status.textContent=`${Math.round(performance.now()-started)} ms · ${d.kernelloom.backend} · ${d.kernelloom.device}`;status.className='status ok'}catch(e){answer.textContent=e.message;status.className='status error'}}
</script></body></html>"""
