from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kernelloom import KernelLoomModel, ModelConfig, __version__
from kernelloom.cli import build_parser


class FakeLlama:
    def __init__(self, **options):
        self.options = options

    def create_chat_completion(self, *, messages, stream=False, **settings):
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "local "}}]},
                {"choices": [{"delta": {"content": "answer"}}]},
            ])
        message = {"content": "local answer"}
        if settings.get("tools"):
            function = settings["tools"][0]["function"]
            message = {
                "content": "",
                "tool_calls": [{
                    "id": "call-local-1",
                    "type": "function",
                    "function": {"name": function["name"], "arguments": '{"city":"London"}'},
                }],
            }
        return {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def create_embedding(self, *, input):
        return {
            "data": [
                {"index": index, "embedding": [float(len(text)), float(index + 1)]}
                for index, text in enumerate(input)
            ]
        }

    def tokenize(self, value, *, add_bos):
        return value.split()


class CountingLlama(FakeLlama):
    embedding_calls = 0
    token_calls = 0
    completion_calls = 0

    def create_embedding(self, *, input):
        type(self).embedding_calls += 1
        return super().create_embedding(input=input)

    def tokenize(self, value, *, add_bos):
        type(self).token_calls += 1
        return super().tokenize(value, add_bos=add_bos)

    def create_chat_completion(self, *, messages, stream=False, **settings):
        type(self).completion_calls += 1
        return super().create_chat_completion(messages=messages, stream=stream, **settings)


class KernelLoomTests(unittest.TestCase):
    def test_public_version_and_backend_detection(self) -> None:
        self.assertEqual(__version__, "0.3.0")
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
        self.assertEqual(info["load"]["micro_batch_size"], 128)

    def test_async_model_api(self) -> None:
        import asyncio

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "model.gguf"
                path.write_bytes(b"test")
                with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
                    model = KernelLoomModel(ModelConfig(str(path)))
                    self.assertEqual(await model.ainvoke("Hello"), "local answer")
                    chunks = [chunk async for chunk in model.astream("Hello")]
                    self.assertEqual("".join(chunks), "local answer")
                    model.close()

        asyncio.run(exercise())

    def test_performance_options_reach_llama_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            path.write_bytes(b"test")
            config = ModelConfig(
                str(path), batch_size=256, micro_batch_size=64, batch_threads=3,
                flash_attention=True, use_mlock=True, gpu_layers=-1,
            )
            with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
                model = KernelLoomModel(config).load()
                self.assertEqual(model._backend.options["n_ubatch"], 64)
                self.assertEqual(model._backend.options["n_threads_batch"], 3)
                self.assertTrue(model._backend.options["flash_attn"])
                self.assertTrue(model._backend.options["use_mlock"])
                model.close()

    def test_local_embedding_batch_and_async_api(self) -> None:
        import asyncio

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "embed.gguf"
            path.write_bytes(b"test")
            with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=FakeLlama)}):
                model = KernelLoomModel(ModelConfig(str(path), embedding=True))
                self.assertEqual(model.embed("hello"), [5.0, 1.0])
                self.assertEqual(model.embed_many(["a", "abcd"]), [[1.0, 1.0], [4.0, 2.0]])
                self.assertEqual(asyncio.run(model.aembed("async")), [5.0, 1.0])
                self.assertEqual(model.count_tokens("one two three"), 3)
                model.close()

    def test_system_prompt_is_added_once(self) -> None:
        config = ModelConfig("model.gguf", system_prompt="Be concise.")
        model = KernelLoomModel(config)
        self.assertEqual(model._messages("Hello")[0]["role"], "system")
        messages = model._messages([
            {"role": "system", "content": "Use this instead."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertEqual(len([item for item in messages if item["role"] == "system"]), 1)

    def test_cpu_profile_warmup_and_compact_local_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "embed.gguf"
            path.write_bytes(b"test")
            CountingLlama.embedding_calls = 0
            CountingLlama.token_calls = 0
            CountingLlama.completion_calls = 0
            with patch.dict(sys.modules, {"llama_cpp": SimpleNamespace(Llama=CountingLlama)}):
                model = KernelLoomModel(ModelConfig(
                    str(path), embedding=True, cpu_profile="throughput", auto_batch_size=True,
                    embedding_cache_size=2, embedding_cache_max_bytes=64,
                ))
                self.assertEqual(model.embed_many(["same", "same", "other"]), [[4.0, 1.0], [4.0, 1.0], [5.0, 2.0]])
                self.assertEqual(CountingLlama.embedding_calls, 1)
                model.embed("same")
                self.assertEqual(CountingLlama.embedding_calls, 1)
                model.count_tokens("one two")
                model.count_tokens("one two")
                self.assertEqual(CountingLlama.token_calls, 1)
                warm = model.warmup(iterations=2)
                self.assertEqual(warm["kind"], "embedding")
                self.assertEqual(CountingLlama.embedding_calls, 3)
                self.assertEqual(model.info()["load"]["cpu_plan"]["profile"], "throughput")
                self.assertEqual(model._backend.options["n_batch"], 1024)
                self.assertLessEqual(model.cache_info()["embedding_bytes"], 64)
                with self.assertRaises(ValueError):
                    model.warmup(prompt="", max_new_tokens=0)
                model.close()

    def test_cli_exposes_cpu_planning_and_warmup(self) -> None:
        parser = build_parser()
        plan = parser.parse_args(["cpu-plan", "--profile", "throughput", "--available-cores", "4"])
        warm = parser.parse_args(["warm", "model.gguf", "--embedding", "--cpu-profile", "latency"])
        self.assertEqual(plan.command, "cpu-plan")
        self.assertEqual(plan.available_cores, 4)
        self.assertTrue(warm.embedding)
        self.assertEqual(warm.cpu_profile, "latency")


if __name__ == "__main__":
    unittest.main()
