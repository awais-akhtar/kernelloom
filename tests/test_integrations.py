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
from kernelloom.langchain import KernelLoomChatModel
from kernelloom.server import create_app
from tests.test_kernelloom import FakeLlama


def test_langchain_adapter_invokes_and_streams() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "model.gguf"
        path.write_bytes(b"test")
        with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
            model = KernelLoomModel(ModelConfig(str(path)))
            llm = KernelLoomChatModel(model)
            assert llm.invoke("Hello").content == "local answer"
            assert "".join(chunk.content for chunk in llm.stream("Hello")) == "local answer"
            model.close()


def test_http_health_console_models_and_authentication() -> None:
    async def exercise() -> None:
        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).json() == {"status": "ok", "models": 0}
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
