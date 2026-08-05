from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kernelloom import KernelLoomModel, ModelConfig, __version__


class FakeLlama:
    def __init__(self, **options):
        self.options = options

    def create_chat_completion(self, *, messages, stream=False, **settings):
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "local "}}]},
                {"choices": [{"delta": {"content": "answer"}}]},
            ])
        return {
            "choices": [{"message": {"content": "local answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }


class KernelLoomTests(unittest.TestCase):
    def test_public_version_and_backend_detection(self) -> None:
        self.assertEqual(__version__, "0.2.0")
        self.assertEqual(ModelConfig("model.gguf").resolved_backend, "llama-cpp")
        self.assertEqual(ModelConfig("openvino-model").resolved_backend, "openvino")

    def test_gguf_model_supports_invoke_chat_and_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            path.write_bytes(b"test")
            fake_module = SimpleNamespace(Llama=FakeLlama)
            with patch.dict(sys.modules, {"llama_cpp": fake_module}):
                with KernelLoomModel(ModelConfig(str(path), threads=2)) as model:
                    self.assertEqual(model.invoke("Hello"), "local answer")
                    result = model.chat([{"role": "user", "content": "Hello"}])
                    streamed = "".join(model.stream("Hello"))
                    info = model.info()
        self.assertEqual(result.backend, "llama-cpp")
        self.assertEqual(result.metadata["usage"]["completion_tokens"], 2)
        self.assertEqual(streamed, "local answer")
        self.assertEqual(info["load"]["threads"], 2)

    def test_system_prompt_is_added_once(self) -> None:
        config = ModelConfig("model.gguf", system_prompt="Be concise.")
        model = KernelLoomModel(config)
        self.assertEqual(model._messages("Hello")[0]["role"], "system")
        messages = model._messages([
            {"role": "system", "content": "Use this instead."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertEqual(len([item for item in messages if item["role"] == "system"]), 1)


if __name__ == "__main__":
    unittest.main()
