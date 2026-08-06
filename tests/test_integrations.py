from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("langchain_core")

from kernelloom import KernelLoomModel, ModelConfig
from kernelloom.langchain import KernelLoomChatModel, KernelLoomEmbeddings
from kernelloom.server import create_app
from tests.test_kernelloom import FakeLlama


def test_langchain_adapter_invokes_and_streams() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.gguf"
        path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            model = KernelLoomModel(ModelConfig(str(path)))
            llm = KernelLoomChatModel(model)
            response = llm.invoke("Hello")
            assert response.content == "local answer"
            assert response.usage_metadata == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            assert response.response_metadata["backend"] == "llama-cpp"
            assert "".join(chunk.content for chunk in llm.stream("Hello")) == "local answer"
            model.close()


def test_langchain_native_async_tools_structured_output_and_embeddings() -> None:
    from pydantic import BaseModel

    class WeatherRequest(BaseModel):
        city: str

    async def exercise(llm: KernelLoomChatModel) -> None:
        assert (await llm.ainvoke("Hello")).content == "local answer"
        chunks = [chunk.content async for chunk in llm.astream("Hello")]
        assert "".join(chunks) == "local answer"
        assert [message.content for message in await llm.abatch(["one", "two"])] == [
            "local answer", "local answer",
        ]

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.gguf"
        path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            runtime = KernelLoomModel(ModelConfig(str(path)))
            llm = KernelLoomChatModel(runtime)
            asyncio.run(exercise(llm))
            assert [message.content for message in llm.batch(["one", "two"])] == [
                "local answer", "local answer",
            ]
            tool_response = llm.bind_tools([WeatherRequest], tool_choice="WeatherRequest").invoke("Weather?")
            assert tool_response.tool_calls[0]["name"] == "WeatherRequest"
            assert tool_response.tool_calls[0]["args"] == {"city": "London"}
            structured = llm.with_structured_output(WeatherRequest).invoke("Extract the city")
            assert structured == WeatherRequest(city="London")
            runtime.close()

            embeddings = KernelLoomEmbeddings(str(path))
            assert embeddings.embed_query("hello") == [5.0, 1.0]
            assert embeddings.embed_documents(["a", "abcd"]) == [[1.0, 1.0], [4.0, 2.0]]
            assert asyncio.run(embeddings.aembed_query("async")) == [5.0, 1.0]
            embeddings.close()


def test_http_health_console_models_and_authentication() -> None:
    async def exercise() -> None:
        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).json() == {"status": "ok", "models": 0}
            assert (await client.get("/ready")).json() == {"ready": False, "models": [], "capacity": 4}
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "kernelloom_models_loaded 0" in metrics.text
            console = await client.get("/")
            assert console.status_code == 200
            assert "KernelLoom" in console.text
            assert (await client.get("/v1/models")).json()["object"] == "list"

        with patch.dict(os.environ, {"KERNELLOOM_API_KEY": "secret"}):
            protected = create_app()
            protected_transport = httpx.ASGITransport(app=protected)
            async with httpx.AsyncClient(transport=protected_transport, base_url="http://test") as client:
                assert (await client.get("/v1/models")).status_code == 401
                response = await client.get("/v1/models", headers={"Authorization": "Bearer secret"})
                assert response.status_code == 200

    asyncio.run(exercise())


def test_http_local_embeddings_endpoint() -> None:
    async def exercise(path: Path) -> None:
        app = create_app()
        app.state.models.load({"model_path": str(path), "model_id": "embed", "embedding": True})
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/embeddings", json={
                "model": "embed",
                "input": ["a", "abcd"],
            })
            assert response.status_code == 200
            body = response.json()
            assert body["model"] == "embed"
            assert [item["embedding"] for item in body["data"]] == [[1.0, 1.0], [4.0, 2.0]]
            assert body["usage"] == {"prompt_tokens": 2, "total_tokens": 2}
        app.state.models.close()

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "embed.gguf"
        path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            asyncio.run(exercise(path))


def test_server_reuses_identical_warm_model_and_exposes_warmup() -> None:
    async def exercise(path: Path) -> None:
        app = create_app()
        first = app.state.models.load({"model_path": str(path), "model_id": "embed", "embedding": True})
        second = app.state.models.load({"model_path": str(path), "model_id": "embed", "embedding": True})
        assert first["id"] == "embed"
        assert second["reused"] is True
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/models/embed/warm", json={"iterations": 2})
            assert response.status_code == 200
            assert response.json()["kind"] == "embedding"
        app.state.models.close()

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "embed.gguf"
        path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            asyncio.run(exercise(path))


def test_server_control_routes_and_rag_collections_keep_models_resident() -> None:
    async def exercise(chat_path: Path, embed_path: Path) -> None:
        app = create_app()
        app.state.models.load({"model_path": str(chat_path), "model_id": "chat"})
        app.state.models.load({"model_path": str(embed_path), "model_id": "embed", "embedding": True})
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            console = await client.get("/")
            assert "/assets/console.js" in console.text
            assert (await client.get("/assets/console.css")).status_code == 200

            detail = await client.get("/v1/models/chat")
            assert detail.status_code == 200
            assert detail.json()["config"]["model_id"] == "chat"
            assert "resolved_backend" not in detail.json()["config"]
            assert (await client.post("/v1/models/chat/cache/clear", json={})).json()["cleared"] is True
            assert (await client.get("/v1/cpu-plan?profile=latency&reserve_cores=1")).json()["profile"] == "latency"

            exported = await client.get("/v1/runtime/config")
            assert exported.status_code == 200
            validated = await client.post("/v1/runtime/config/validate", json=exported.json())
            assert validated.status_code == 200
            assert validated.json()["valid"] is True

            created = await client.post("/v1/rag/collections", json={
                "id": "notes",
                "model": "chat",
                "embedding_model": "embed",
                "database": "memory",
                "config": {"chunk_size": 100, "chunk_overlap": 10, "retrieval": "similarity"},
            })
            assert created.status_code == 200
            assert created.json()["ready"] is True
            ingested = await client.post("/v1/rag/collections/notes/ingest", json={
                "sources": "Python powers the local database service.",
                "metadata": {"team": "platform"},
            })
            assert ingested.status_code == 200
            assert ingested.json()["indexed"] == 1
            retrieved = await client.post("/v1/rag/collections/notes/retrieve", json={
                "query": "Which language powers the database?",
                "filters": {"team": "platform"},
            })
            assert retrieved.status_code == 200
            assert retrieved.json()["data"][0]["metadata"]["team"] == "platform"
            answered = await client.post("/v1/rag/collections/notes/query", json={
                "question": "Which language powers the database?",
            })
            assert answered.status_code == 200
            assert answered.json()["sources"][0]["metadata"]["team"] == "platform"

            blocked = await client.delete("/v1/models/chat")
            assert blocked.status_code == 409
            removed = await client.delete("/v1/rag/collections/notes")
            assert removed.status_code == 200
            assert app.state.models.get("chat").loaded is True
            assert (await client.delete("/v1/models/chat")).status_code == 200
        app.state.rag.close()
        app.state.models.close()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        chat_path = root / "chat.gguf"
        embed_path = root / "embed.gguf"
        chat_path.write_bytes(b"test")
        embed_path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            asyncio.run(exercise(chat_path, embed_path))
