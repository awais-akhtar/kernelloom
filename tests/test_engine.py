from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openagent_engine import (
    AdaptiveCompiler,
    AdaptiveExecutionEngine,
    EngineStore,
    ModelFormatError,
    ModelFrontend,
)
from openagent_engine.device_runtime import DirectHardwareClient
from openagent_engine.hardware import DeviceProfile, HardwareProfile, HardwareProfiler, accelerator_environment
from openagent_engine.ir import ModelIR, OpCode, build_transformer_graph, lower_graph_to_tensor_ir
from openagent_engine.memory import VirtualKVCache
from openagent_engine.models import AuditEvent
from openagent_engine.scheduler import (
    DeadlineBatchScheduler,
    InferenceRequest,
    RequestPhase,
    RequestPriority,
)
from openagent_engine.worker import _device_config, _filter_cpu_properties, _process_alive, _session_signature


def hardware_profile() -> HardwareProfile:
    operations = tuple(item.value for item in OpCode)
    cpu = DeviceProfile(
        id="cpu:0",
        kind="cpu",
        name="Test CPU",
        vendor="generic",
        memory_gb=16,
        unified_memory=True,
        compute_tops=2,
        memory_bandwidth_gbps=30,
        precisions=("fp32", "fp16", "int8", "int4"),
        supported_ops=operations,
        backends=("openvino", "llama-cpu"),
        capabilities={"openvino": {"id": "CPU"}},
    )
    return HardwareProfile(
        id="test-hardware",
        platform="test",
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_threads=16,
        devices=(cpu,),
        runtimes={},
        power={"ac_connected": True},
        discovered_at=time.time(),
    )


def transformer_model() -> ModelIR:
    graph = build_transformer_graph(
        block_count=4,
        weight_bytes=2_000_000_000,
        parameter_count=1_000_000_000,
        embedding_length=2048,
    )
    return ModelIR(
        id="model-id",
        name="test-model",
        source_path="test.onnx",
        source_format="onnx",
        fingerprint="fingerprint",
        architecture="llama",
        parameter_count=1_000_000_000,
        weight_bytes=2_000_000_000,
        quantization_bits=16,
        context_length=4096,
        embedding_length=2048,
        block_count=4,
        graph_nodes=graph,
        tensor_nodes=lower_graph_to_tensor_ir(graph),
    )


class EngineTests(unittest.TestCase):
    def test_public_compiler_creates_separate_prefill_and_decode_plans(self) -> None:
        package = AdaptiveCompiler().compile(
            transformer_model(),
            hardware_profile(),
            prompt_tokens=128,
            context_tokens=2048,
            memory_budget_gb=3,
        )
        self.assertEqual(package.status, "planned")
        self.assertNotEqual(
            package.prefill.estimated_tokens_per_second,
            package.decode.estimated_tokens_per_second,
        )

    def test_safetensors_inspection_does_not_load_weight_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            header = {
                "weight": {"dtype": "F16", "shape": [2, 4], "data_offsets": [0, 16]},
                "__metadata__": {"format": "pt"},
            }
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(16))
            model = ModelFrontend().inspect(path)
        self.assertEqual(model.source_format, "safetensors")
        self.assertEqual(model.parameter_count, 8)

    def test_unknown_model_container_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.bin"
            path.write_bytes(b"not a model")
            with self.assertRaises(ModelFormatError):
                ModelFrontend().inspect(path)

    def test_runtime_accepts_a_directory_and_persists_a_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = AdaptiveExecutionEngine(temp)
            try:
                model_path = Path(temp) / "model.onnx"
                model_path.write_bytes(b"opaque-test-model")
                with patch.object(engine.profiler, "profile", return_value=hardware_profile()):
                    plan = engine.compile_model(str(model_path), backend_compile=False)
                self.assertEqual(plan["status"], "planned")
                self.assertEqual(engine.plans()[0]["id"], plan["id"])
            finally:
                engine.close()

    def test_store_records_structured_audit_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EngineStore(temp)
            store.add_audit_event(
                AuditEvent("test", "engine.test", "model:one", metadata={"verified": True})
            )
            event = store.list_audit_events()[0]
        self.assertEqual(event["action"], "engine.test")
        self.assertTrue(event["metadata"]["verified"])

    def test_runtime_calibration_persists_verified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            engine = AdaptiveExecutionEngine(temp)
            try:
                model_path = Path(temp) / "model.onnx"
                model_path.write_bytes(b"opaque-test-model")
                evidence = {
                    "status": "verified",
                    "winner": {"device": "CPU", "valid": True, "latency_ms": {"mean": 1.2}},
                    "devices": [
                        {"device": "CPU", "valid": True, "latency_ms": {"mean": 1.2}}
                    ],
                    "resident_model_id": "test-model",
                }
                with patch.object(engine.profiler, "profile", return_value=hardware_profile()), patch.object(
                    engine.direct, "calibrate", return_value=evidence
                ):
                    result = engine.calibrate_direct_model(
                        str(model_path), devices=["cpu:0"], model_id="test-model"
                    )
                self.assertEqual(result["winner_profile_device"], "cpu:0")
                self.assertEqual(engine.direct_calibrations()[0]["status"], "verified")
            finally:
                engine.close()

    def test_model_roles_and_orchestration_are_standalone(self) -> None:
        class Gateway:
            def specs(self):
                return [{"id": "local", "label": "Local", "available": True}]

            def discover_models(self):
                return [{"provider": "local", "model": "test-model", "available": True}]

            def complete(self, messages, *, provider_id, model, allow_external):
                system = messages[0]["content"]
                if "strict technical reviewer" in system:
                    content = "APPROVED\nAcceptance checks pass."
                elif "implementation engineer" in system:
                    content = "Implementation and tests."
                else:
                    content = "Plan and acceptance checks."
                return SimpleNamespace(
                    provider="local",
                    model=model or "test-model",
                    content=content,
                    metadata={"network": False},
                )

        with tempfile.TemporaryDirectory() as temp:
            engine = AdaptiveExecutionEngine(temp)
            try:
                engine.bind_gateway(Gateway())
                result = engine.orchestrate_models("Build a parser.", max_rounds=1)
                history = engine.orchestration_runs()
            finally:
                engine.close()
        self.assertTrue(result["approved"])
        self.assertEqual(history[0]["status"], "completed")

    def test_kv_prefix_is_shared_and_copy_on_write(self) -> None:
        cache = VirtualKVCache(1024 * 1024, block_tokens=4, bytes_per_token=64)
        cache.create_session("one", prefix_tokens=[1, 2, 3])
        cache.create_session("two", prefix_tokens=[1, 2, 3])
        self.assertEqual(cache.status()["shared_blocks"], 1)
        cache.append("two", [4])
        self.assertEqual(cache.session_status("one")["token_count"], 3)
        self.assertEqual(cache.session_status("two")["token_count"], 4)

    def test_scheduler_protects_interactive_requests(self) -> None:
        scheduler = DeadlineBatchScheduler(max_batch_size=2, interactive_reserve=1)
        scheduler.submit(
            InferenceRequest("background", "m", RequestPhase.DECODE, 10, RequestPriority.BACKGROUND)
        )
        scheduler.submit(
            InferenceRequest("interactive", "m", RequestPhase.DECODE, 10, RequestPriority.INTERACTIVE)
        )
        batch = scheduler.next_batch()
        self.assertEqual(batch[0].id, "interactive")

    def test_accelerator_environment_isolates_host_python(self) -> None:
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "host-packages", "PYTHONHOME": "host-python"},
            clear=False,
        ):
            environment = accelerator_environment(Path("engine-source"))
        self.assertEqual(environment["PYTHONPATH"], "engine-source")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertNotIn("PYTHONHOME", environment)

    def test_worker_helpers_and_status_remain_local(self) -> None:
        self.assertTrue(_process_alive(os.getpid()))
        self.assertFalse(_process_alive(-1))
        self.assertEqual(_device_config({}, device="GPU.0")["PERFORMANCE_HINT"], "LATENCY")
        cpu_config = _device_config({"inference_num_threads": 6, "untrusted": "ignored"}, device="CPU")
        self.assertEqual(cpu_config["INFERENCE_NUM_THREADS"], 6)

        class Core:
            def get_property(self, _device, _property):
                return ["INFERENCE_NUM_THREADS"]

        self.assertEqual(
            _filter_cpu_properties(Core(), "CPU", {"INFERENCE_NUM_THREADS": 6, "ENABLE_CPU_PINNING": True}),
            {"INFERENCE_NUM_THREADS": 6},
        )
        path = Path("model.xml")
        first = _session_signature(path, "CPU", {"INFERENCE_NUM_THREADS": 4})
        second = _session_signature(path, "CPU", {"INFERENCE_NUM_THREADS": 6})
        self.assertNotEqual(first, second)
        with tempfile.TemporaryDirectory() as temp:
            profiler = HardwareProfiler(temp, accelerator_python=str(Path(temp) / "missing"))
            status = DirectHardwareClient(profiler).status()
        self.assertFalse(status["running"])
        self.assertEqual(status["transport"], "inherited-jsonl-pipes")
        self.assertNotIn("port", status)

    def test_auto_direct_target_prefers_verified_accelerator(self) -> None:
        cpu = hardware_profile().devices[0]
        gpu = DeviceProfile(
            id="gpu:0", kind="gpu", name="Test GPU", vendor="generic", memory_gb=4,
            unified_memory=False, compute_tops=4, memory_bandwidth_gbps=100,
            precisions=("fp16",), supported_ops=(), backends=("openvino",),
            capabilities={"openvino": {"id": "GPU.0"}},
        )
        device, target = AdaptiveExecutionEngine._direct_target(None, (cpu, gpu), "AUTO")
        self.assertEqual(device.id, "gpu:0")
        self.assertEqual(target, "GPU.0")


if __name__ == "__main__":
    unittest.main()
