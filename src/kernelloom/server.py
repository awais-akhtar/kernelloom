"""OpenAI-compatible HTTP service and local browser console."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from uuid import uuid4

from .config import ModelConfig
from .model import KernelLoomModel


class ModelRegistry:
    """Thread-safe collection of resident models."""

    def __init__(self) -> None:
        self._models: dict[str, KernelLoomModel] = {}
        self._lock = threading.RLock()

    def load(self, values: dict[str, Any]) -> dict[str, Any]:
        config = ModelConfig(**values)
        model = KernelLoomModel(config).load()
        with self._lock:
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

    def close(self) -> None:
        with self._lock:
            models = list(self._models.values())
            self._models.clear()
        for model in models:
            model.close()


def create_app(*, initial_model: ModelConfig | None = None) -> Any:
    """Build the FastAPI application without importing server dependencies globally."""

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("The web service needs the 'server' extra: pip install kernelloom[server]") from exc

    registry = ModelRegistry()

    @asynccontextmanager
    async def lifespan(_: Any):
        if initial_model is not None:
            registry.load({key: value for key, value in initial_model.to_dict().items() if key != "resolved_backend"})
        yield
        registry.close()

    app = FastAPI(
        title="KernelLoom",
        version="0.2.0",
        description="Local model serving with hardware-aware execution.",
        lifespan=lifespan,
    )
    app.state.models = registry

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
                _chat_stream(model, messages, settings, request_id, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = model.chat(messages, **settings)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
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
        try:
            result = model.generate(prompt, **_generation_settings(payload))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "id": f"cmpl-{uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model.config.model_id,
            "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
            "usage": result.metadata.get("usage", {}),
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
) -> Iterator[str]:
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
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}})}\n\n"
        yield "data: [DONE]\n\n"


_CONSOLE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KernelLoom</title><style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#14181d;--line:#29313a;--text:#e9eef3;--muted:#8e9aa7;--accent:#65d1a7;--danger:#ff7c86}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#17342c 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}.shell{max-width:1100px;margin:auto;padding:48px 24px}.brand{display:flex;align-items:center;gap:12px;margin-bottom:30px}.mark{width:38px;height:38px;border:2px solid var(--accent);border-radius:11px;transform:rotate(45deg);box-shadow:0 0 28px #65d1a744}.brand h1{font-size:24px;margin:0}.brand span{color:var(--muted)}.grid{display:grid;grid-template-columns:360px 1fr;gap:18px}.card{background:#14181de8;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 18px 50px #0004}h2{font-size:14px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 18px}label{display:block;color:var(--muted);font-size:12px;margin:12px 0 5px}input,select,textarea,button{width:100%;border:1px solid var(--line);background:#0d1115;color:var(--text);border-radius:8px;padding:10px 11px;font:inherit}textarea{min-height:190px;resize:vertical}button{background:var(--accent);color:#07120e;border:0;font-weight:700;cursor:pointer;margin-top:14px}button.secondary{background:#212830;color:var(--text)}#answer{white-space:pre-wrap;min-height:220px;border:1px solid var(--line);border-radius:8px;padding:14px;background:#0d1115}.status{font-size:12px;color:var(--muted);margin-top:10px}.ok{color:var(--accent)}.error{color:var(--danger)}@media(max-width:760px){.grid{grid-template-columns:1fr}.shell{padding:24px 14px}}
</style></head><body><main class="shell"><div class="brand"><div class="mark"></div><div><h1>KernelLoom</h1><span>Local model runtime</span></div></div><div class="grid"><section class="card"><h2>Load a model</h2><label>Model path</label><input id="path" placeholder="D:\models\model.gguf"><label>Model ID</label><input id="model" value="default"><label>Backend</label><select id="backend"><option value="auto">Auto</option><option value="llama-cpp">llama.cpp</option><option value="openvino">OpenVINO GenAI</option></select><label>Device</label><select id="device"><option>CPU</option><option>GPU</option><option>NPU</option></select><label>Context length</label><input id="context" type="number" value="4096"><label>API key (if configured)</label><input id="apiKey" type="password" autocomplete="off"><button onclick="loadModel()">Load model</button><div id="loadStatus" class="status">No model loaded in this session.</div></section><section class="card"><h2>Test response</h2><label>Prompt</label><textarea id="prompt" placeholder="Ask the loaded model something..."></textarea><div style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><div><label>Max tokens</label><input id="tokens" type="number" value="256"></div><div><label>Temperature</label><input id="temperature" type="number" step="0.1" value="0.7"></div></div><button onclick="sendPrompt()">Generate</button><label>Response</label><div id="answer">Ready.</div><div id="responseStatus" class="status"></div></section></div></main><script>
const byId=id=>document.getElementById(id);const headers=()=>{const h={'Content-Type':'application/json'},k=byId('apiKey').value;if(k)h.Authorization=`Bearer ${k}`;return h};async function loadModel(){const status=byId('loadStatus');status.textContent='Loading…';status.className='status';try{const r=await fetch('/v1/models/load',{method:'POST',headers:headers(),body:JSON.stringify({model_path:byId('path').value,model_id:byId('model').value,backend:byId('backend').value,device:byId('device').value,context_length:Number(byId('context').value)})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not load model');status.textContent=`Loaded ${d.id} on ${d.device} with ${d.backend}`;status.className='status ok'}catch(e){status.textContent=e.message;status.className='status error'}}async function sendPrompt(){const answer=byId('answer'),status=byId('responseStatus');answer.textContent='Generating…';status.textContent='';const started=performance.now();try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:headers(),body:JSON.stringify({model:byId('model').value,messages:[{role:'user',content:byId('prompt').value}],max_tokens:Number(byId('tokens').value),temperature:Number(byId('temperature').value)})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Generation failed');answer.textContent=d.choices[0].message.content;status.textContent=`${Math.round(performance.now()-started)} ms · ${d.kernelloom.backend} · ${d.kernelloom.device}`;status.className='status ok'}catch(e){answer.textContent=e.message;status.className='status error'}}
</script></body></html>"""
