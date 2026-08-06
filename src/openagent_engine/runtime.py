"""Persistent adaptive compiler and inference runtime."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
import time
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import EngineStore
from .autotune import DeviceAutoTuner
from .backends import BackendRegistry, OpenVINOBackend
from .backends import openvino_device_id
from .compiler import AdaptiveCompiler, CompilationError
from .device_runtime import DirectHardwareClient, DirectHardwareError
from .frontends import ModelFormatError, ModelFrontend, enrich_onnx_from_probe
from .hardware import DeviceProfile, HardwareProfiler
from .memory import VirtualKVCache
from .scheduler import DeadlineBatchScheduler, InferenceRequest, RequestPhase, RequestPriority


MODEL_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "chat": {
        "label": "Chat", "provider_id": "auto", "model_id": "phi4-mini",
        "system_prompt": "You are a private local assistant. Be direct, accurate, and useful.",
    },
    "code": {
        "label": "Code builder", "provider_id": "auto", "model_id": "qwen2.5-coder:3b",
        "system_prompt": "You are the implementation engineer. Produce complete, testable code and report assumptions precisely.",
    },
    "reasoning": {
        "label": "Reasoning planner", "provider_id": "auto", "model_id": "phi4-mini",
        "system_prompt": "You are the technical planner. Return a concise decision summary, constraints, and an acceptance checklist without hidden chain-of-thought.",
    },
    "research": {
        "label": "Research", "provider_id": "auto", "model_id": "qwen2.5:7b",
        "system_prompt": "You are the research specialist. Separate evidence, inference, uncertainty, and unanswered questions.",
    },
    "supervisor": {
        "label": "Supervisor", "provider_id": "auto", "model_id": "phi4-mini",
        "system_prompt": "You are the strict technical reviewer. Reply APPROVED only when every acceptance criterion is met; otherwise reply REVISE followed by concrete defects and fixes.",
    },
    "draft": {
        "label": "Fast draft", "provider_id": "auto", "model_id": "gemma3:2b",
        "system_prompt": "You create a fast, compact first draft for a stronger model to verify.",
    },
    "vision": {
        "label": "Vision", "provider_id": "auto", "model_id": "",
        "system_prompt": "You inspect local visual evidence and describe only what the supplied media supports.",
    },
    "voice": {
        "label": "Voice", "provider_id": "local", "model_id": "faster-whisper:base.en",
        "system_prompt": "You are the local speech interface. Preserve the speaker's meaning and mark uncertain words.",
    },
}


class AdaptiveRuntimeError(RuntimeError):
    """Raised when a runtime action is invalid or cannot be proven locally."""


class AdaptiveExecutionEngine:
    """Own model inspection, planning, backend validation, and measurements."""

    def __init__(self, store: EngineStore | str | Path, *, accelerator_python: str = "") -> None:
        if isinstance(store, (str, Path)):
            store = EngineStore(store)
        self.store = store
        self.runtime_root = (store.data_dir / "adaptive-engine").resolve()
        self.cache_root = self.runtime_root / "compiled-cache"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.frontend = ModelFrontend()
        self.profiler = HardwareProfiler(store.data_dir, accelerator_python=accelerator_python)
        self.compiler = AdaptiveCompiler()
        self.backends = BackendRegistry()
        self.openvino = OpenVINOBackend(self.profiler, self.cache_root)
        self.direct = DirectHardwareClient(self.profiler)
        self.gateway: Any = None
        self.autotuner = DeviceAutoTuner(store, self.openvino)
        self.scheduler = DeadlineBatchScheduler(max_batch_size=8, interactive_reserve=2)
        self.kv_cache = VirtualKVCache(256 * 1024 * 1024, block_tokens=16, bytes_per_token=64 * 1024)
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS adaptive_models (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    model_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_adaptive_models_fingerprint
                    ON adaptive_models(project, fingerprint);

                CREATE TABLE IF NOT EXISTS adaptive_execution_plans (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_plans_project
                    ON adaptive_execution_plans(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS adaptive_device_benchmarks (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_benchmarks_project
                    ON adaptive_device_benchmarks(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS adaptive_model_calibrations (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    model_fingerprint TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    winner_device TEXT NOT NULL,
                    winner_profile_device TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_calibrations_model
                    ON adaptive_model_calibrations(project, model_fingerprint, created_at DESC);

                CREATE TABLE IF NOT EXISTS adaptive_model_roles (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    role TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project, role)
                );
                CREATE TABLE IF NOT EXISTS adaptive_orchestration_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rounds INTEGER NOT NULL,
                    approved INTEGER NOT NULL,
                    final_output TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_orchestration_runs_project
                    ON adaptive_orchestration_runs(project, created_at DESC);
                CREATE TABLE IF NOT EXISTS adaptive_orchestration_turns (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_orchestration_turns_run
                    ON adaptive_orchestration_turns(run_id, turn_index);
                """
            )

    def bind_gateway(self, gateway: Any) -> None:
        self.gateway = gateway

    def status(self, *, project: str = "default", refresh_hardware: bool = False) -> dict[str, Any]:
        profile = self.profiler.profile(force=refresh_hardware)
        with self.store.session() as connection:
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM adaptive_models WHERE project = ?) AS models,
                    (SELECT COUNT(*) FROM adaptive_execution_plans WHERE project = ?) AS plans,
                    (SELECT COUNT(*) FROM adaptive_execution_plans WHERE project = ? AND status = 'compiled') AS compiled,
                    (SELECT COUNT(*) FROM adaptive_device_benchmarks WHERE project = ?) AS benchmarks,
                    (SELECT COUNT(*) FROM adaptive_model_calibrations WHERE project = ?) AS calibrations,
                    (SELECT COUNT(*) FROM adaptive_model_roles WHERE project = ?) AS model_roles,
                    (SELECT COUNT(*) FROM adaptive_orchestration_runs WHERE project = ?) AS orchestration_runs
                """,
                (project, project, project, project, project, project, project),
            ).fetchone()
        return {
            "status": "ready",
            "project": project,
            "engine": "kernelloom",
            "runtime": "adaptive-compiler-runtime",
            "version": "0.2.0",
            "hardware_profile": profile.to_dict(),
            "backends": [item.to_dict() for item in self.backends.capabilities(profile)],
            "inventory": dict(counts) if counts else {},
            "scheduler": self.scheduler.status(),
            "kv_cache": self.kv_cache.status(),
            "direct_runtime": self.direct.status(),
            "execution_ownership": {
                "gateway_bound": self.gateway is not None,
                "entrypoint": "host application -> AdaptiveExecutionEngine -> selected local provider",
                "shared_consumers": [
                    "host applications",
                    "agents",
                    "workflows",
                    "research tools",
                    "background workers",
                    "model benchmarks",
                ],
                "external_default": False,
            },
            "state_semantics": {
                "planned": "analytical placement only",
                "compiled": "the vendor backend accepted the model and target device",
                "verified": "model output passed an explicit numerical reference check",
            },
            "privacy_boundary": "Hardware discovery, model inspection, compilation caches, and benchmarks remain on this machine. No telemetry is uploaded.",
        }

    def hardware(self, *, refresh: bool = False) -> dict[str, Any]:
        profile = self.profiler.profile(force=refresh)
        return {
            "profile": profile.to_dict(),
            "backends": [item.to_dict() for item in self.backends.capabilities(profile)],
        }

    def readiness(self, *, project: str = "default") -> dict[str, Any]:
        """Return bounded module-health evidence without probing Windows drivers."""

        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM adaptive_models WHERE project = ?) AS models,
                    (SELECT COUNT(*) FROM adaptive_execution_plans WHERE project = ?) AS plans,
                    (SELECT COUNT(*) FROM adaptive_model_calibrations WHERE project = ?) AS calibrations,
                    (SELECT COUNT(*) FROM adaptive_model_roles WHERE project = ?) AS model_roles,
                    (SELECT COUNT(*) FROM adaptive_orchestration_runs WHERE project = ?) AS orchestration_runs
                """,
                (project, project, project, project, project),
            ).fetchone()
        return {
            "status": "ready",
            "engine": "kernelloom",
            "runtime": "adaptive-compiler-runtime",
            "inventory": dict(row) if row else {},
            "accelerator_runtime": self.direct.status(),
            "gateway_bound": self.gateway is not None,
            "shared_model_execution": True,
            "hardware_discovery": "cached-or-explicit-refresh",
            "privacy_boundary": "Readiness checks local files and SQLite only; device and driver probes run on the dedicated hardware endpoint.",
        }

    def inspect_model(
        self,
        path: str,
        *,
        project: str = "default",
        parameter_count_hint: int = 0,
        quantization_bits_hint: float = 0.0,
        include_tensors: bool = False,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        try:
            model = self.frontend.inspect(
                path,
                parameter_count_hint=max(0, int(parameter_count_hint)),
                quantization_bits_hint=max(0.0, float(quantization_bits_hint)),
            )
            model = self._enrich_model(model)
        except (ModelFormatError, OSError, ValueError) as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        payload = model.to_dict(include_tensors=include_tensors, include_nodes=True)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_models (id, project, source_path, fingerprint, model_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, fingerprint) DO UPDATE SET
                    source_path = excluded.source_path,
                    model_json = excluded.model_json,
                    updated_at = excluded.updated_at
                """,
                (_storage_model_id(project, model.fingerprint), project, model.source_path, model.fingerprint, json.dumps(payload, sort_keys=True), now, now),
            )
        self._audit("adaptive.model.inspect", f"model:{model.id}", project, {"format": model.source_format, "path": model.source_path})
        return payload

    def compile_model(
        self,
        path: str,
        *,
        project: str = "default",
        prompt_tokens: int = 512,
        context_tokens: int = 4096,
        memory_budget_gb: float | None = None,
        quality_loss_limit: float = 0.08,
        power_mode: str = "balanced",
        max_device_transitions: int = 4,
        backend_compile: bool = True,
        parameter_count_hint: int = 0,
        quantization_bits_hint: float = 0.0,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        try:
            model = self.frontend.inspect(
                path,
                parameter_count_hint=max(0, int(parameter_count_hint)),
                quantization_bits_hint=max(0.0, float(quantization_bits_hint)),
            )
            model = self._enrich_model(model)
            profile = self.profiler.profile()
            preferred_device_id = self._calibrated_profile_device(project, model.fingerprint)
            package = self.compiler.compile(
                model,
                profile,
                prompt_tokens=prompt_tokens,
                context_tokens=context_tokens,
                memory_budget_gb=memory_budget_gb,
                quality_loss_limit=quality_loss_limit,
                power_mode=power_mode,
                max_device_transitions=max_device_transitions,
                preferred_device_id=preferred_device_id,
            )
        except (ModelFormatError, CompilationError, OSError, ValueError) as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        payload = package.to_dict(include_placements=True)
        compilation: dict[str, Any] = {}
        if backend_compile and model.source_format in {"onnx", "openvino"}:
            target = self._single_compute_target(package.decode.placements, profile.devices)
            if target:
                try:
                    compilation = self.openvino.compile(model.source_path, target)
                    conversion_required = bool(payload.get("constraints", {}).get("precision_summary", {}).get("conversion_required"))
                    if conversion_required:
                        compilation["scope"] = "The source model compiled, but the planned mixed-precision conversion has not been materialized."
                        compilation["planned_precision_materialized"] = False
                    else:
                        compilation["planned_precision_materialized"] = True
                        payload["status"] = "compiled"
                except RuntimeError as exc:
                    compilation = {"status": "failed", "error": str(exc), "device_id": target.id}
                    payload["warnings"].append(f"Backend compilation failed: {exc}")
            else:
                compilation = {
                    "status": "not-run",
                    "reason": "The analytical plan uses multiple compute devices; no equivalent executable partition was compiled.",
                }
        elif backend_compile:
            compilation = {
                "status": "not-run",
                "reason": "GGUF execution is validated by loading through llama.cpp; SafeTensors requires an offline conversion first.",
            }
        payload["backend_compilation"] = compilation
        payload["measured_evidence"] = {"backend_compilation": compilation} if compilation else {}
        self._save_model(model.id, project, model.source_path, model.fingerprint, model.to_dict(include_nodes=False))
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_execution_plans (id, project, model_id, status, plan_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    plan_json = excluded.plan_json,
                    updated_at = excluded.updated_at
                """,
                (package.id, project, model.id, payload["status"], json.dumps(payload, sort_keys=True), now, now),
            )
        self._audit(
            "adaptive.model.compile",
            f"plan:{package.id}",
            project,
            {"model": model.id, "status": payload["status"], "backend_compilation": compilation.get("status", "not-run")},
        )
        return payload

    def direct_status(self, *, start: bool = False) -> dict[str, Any]:
        try:
            return self.direct.status(start=start)
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc

    def load_direct_model(
        self,
        path: str,
        *,
        device_id: str,
        project: str = "default",
        model_id: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.profiler.profile()
        device, target = self._direct_target(profile.devices, device_id)
        try:
            result = self.direct.load_model(
                path,
                device=target,
                model_id=model_id,
                cache_dir=str(self.cache_root / _safe_name(target)),
                config=config,
            )
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit(
            "adaptive.direct.load", f"model:{result.get('model_id', model_id)}", project,
            {"device": target, "profile_device": device.id, "path": str(Path(path).expanduser().resolve())},
        )
        return {**result, "profile_device_id": device.id}

    def load_direct_llm(
        self,
        path: str,
        *,
        device_id: str,
        project: str = "default",
        model_id: str = "engine-direct-chat",
        config: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.profiler.profile()
        device, target = self._direct_target(profile.devices, device_id)
        configured_scheduler = scheduler or {
            "enabled": True,
            "enable_prefix_caching": True,
            "dynamic_split_fuse": True,
            "max_num_seqs": 4,
        }
        device_config = dict(config or {})
        target_root = target.split(":", 1)[0].split(".", 1)[0].upper()
        device_config.setdefault("PERFORMANCE_HINT", "LATENCY")
        if target_root == "GPU":
            device_config.setdefault("EXECUTION_MODE_HINT", "PERFORMANCE")
            device_config.setdefault("INFERENCE_PRECISION_HINT", "f16")
        elif target_root == "NPU":
            device_config.setdefault("NPU_TURBO", "YES")
            device_config.setdefault(
                "NPU_COMPILATION_MODE_PARAMS",
                "optimization-level=2 performance-hint-override=latency",
            )
        try:
            result = self.direct.load_llm(
                path,
                device=target,
                model_id=model_id,
                cache_dir=str(self.cache_root / _safe_name(target)),
                config=device_config,
                scheduler=configured_scheduler,
            )
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit(
            "adaptive.direct.llm.load", f"model:{model_id}", project,
            {"device": target, "profile_device": device.id, "path": str(Path(path).expanduser().resolve())},
        )
        return {**result, "profile_device_id": device.id}

    def infer_direct(
        self,
        model_id: str,
        *,
        inputs: dict[str, Any] | str | None = None,
        output_mode: str = "summary",
        project: str = "default",
    ) -> dict[str, Any]:
        try:
            result = self.direct.infer(model_id, inputs=inputs, output_mode=output_mode)
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit("adaptive.direct.infer", f"model:{model_id}", project, {"device": result.get("device"), "latency_ms": result.get("latency_ms")})
        return result

    def generate_direct(
        self,
        model_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        prompt: str = "",
        generation: dict[str, Any] | None = None,
        project: str = "default",
    ) -> dict[str, Any]:
        try:
            result = self.direct.generate(model_id, messages=messages, prompt=prompt, generation=generation)
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit("adaptive.direct.generate", f"model:{model_id}", project, {"device": result.get("device"), "latency_ms": result.get("latency_ms")})
        return result

    def benchmark_direct_model(
        self,
        model_id: str,
        *,
        project: str = "default",
        iterations: int = 20,
        warmup: int = 2,
        inputs: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.direct.benchmark(
                model_id, iterations=iterations, warmup=warmup, inputs=inputs,
            )
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit("adaptive.direct.benchmark", f"model:{model_id}", project, {"device": result.get("device"), "iterations": result.get("iterations")})
        return result

    def calibrate_direct_model(
        self,
        path: str,
        *,
        project: str = "default",
        devices: list[str] | None = None,
        iterations: int = 10,
        absolute_tolerance: float = 0.03,
        keep_resident: bool = True,
        model_id: str = "",
    ) -> dict[str, Any]:
        model = self.frontend.inspect(path)
        model = self._enrich_model(model)
        if model.source_format not in {"onnx", "openvino"}:
            raise AdaptiveRuntimeError("Direct calibration currently requires ONNX or OpenVINO IR.")
        profile = self.profiler.profile()
        requested = devices or [item.id for item in profile.devices if "openvino" in item.backends]
        targets: list[str] = []
        target_profiles: dict[str, str] = {}
        for requested_id in requested:
            device, target = self._direct_target(profile.devices, requested_id)
            targets.append(target)
            target_profiles[target] = device.id
        resident_id = model_id.strip() or f"adaptive-{model.fingerprint[:16]}"
        try:
            result = self.direct.calibrate(
                model.source_path,
                devices=list(dict.fromkeys(targets)),
                iterations=iterations,
                absolute_tolerance=absolute_tolerance,
                cache_dir=str(self.cache_root / "calibration" / model.fingerprint[:16]),
                keep_model_id=resident_id if keep_resident else "",
            )
        except (DirectHardwareError, OSError, ValueError) as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        winner_target = str(result.get("winner", {}).get("device", ""))
        winner_profile = target_profiles.get(winner_target) or self._profile_id_for_target(profile.devices, winner_target)
        calibration_id = uuid4().hex
        now = utc_now()
        payload = {
            **result,
            "id": calibration_id,
            "project": project,
            "model_fingerprint": model.fingerprint,
            "winner_profile_device": winner_profile,
            "created_at": now,
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_model_calibrations
                    (id, project, model_fingerprint, source_path, winner_device, winner_profile_device, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calibration_id, project, model.fingerprint, model.source_path, winner_target,
                    winner_profile, json.dumps(payload, sort_keys=True), now,
                ),
            )
        self._audit(
            "adaptive.direct.calibrate", f"model:{model.id}", project,
            {"winner": winner_target, "winner_profile": winner_profile, "verified_devices": len(result.get("devices", []))},
        )
        return payload

    def direct_calibrations(self, *, project: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM adaptive_model_calibrations WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, max(1, min(int(limit), 500))),
            ).fetchall()
        return [json.loads(row["evidence_json"]) for row in rows]

    def unload_direct_model(self, model_id: str, *, project: str = "default") -> dict[str, Any]:
        try:
            result = self.direct.unload(model_id)
        except DirectHardwareError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit("adaptive.direct.unload", f"model:{model_id}", project, {"status": result.get("status")})
        return result

    def model_role_options(self) -> dict[str, Any]:
        providers = self.gateway.specs() if self.gateway is not None else []
        discovered = self.gateway.discover_models() if self.gateway is not None else []
        return {
            "roles": [{"id": role, "label": values["label"]} for role, values in MODEL_ROLE_DEFAULTS.items()],
            "providers": providers,
            "models": discovered,
            "privacy_boundary": "Model discovery queries local providers only. External model services are never contacted by this endpoint.",
        }

    def model_roles(self, *, project: str = "default") -> list[dict[str, Any]]:
        project = project.strip() or "default"
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_model_roles WHERE project = ? ORDER BY role",
                (project,),
            ).fetchall()
        saved = {str(row["role"]): row for row in rows}
        result = []
        for role, defaults in MODEL_ROLE_DEFAULTS.items():
            row = saved.get(role)
            if row:
                result.append(self._role_record(row))
            else:
                result.append(
                    {
                        "id": _storage_model_id(project, f"role:{role}"),
                        "project": project,
                        "role": role,
                        "label": defaults["label"],
                        "provider_id": defaults["provider_id"],
                        "model_id": defaults["model_id"],
                        "model_path": "",
                        "device_id": "auto",
                        "system_prompt": defaults["system_prompt"],
                        "enabled": True,
                        "config": {},
                        "persisted": False,
                    }
                )
        return result

    def upsert_model_role(
        self,
        role: str,
        *,
        project: str = "default",
        provider_id: str = "auto",
        model_id: str = "",
        model_path: str = "",
        device_id: str = "auto",
        system_prompt: str = "",
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        role = role.strip().lower()
        if role not in MODEL_ROLE_DEFAULTS:
            raise AdaptiveRuntimeError(f"Unknown model role: {role}")
        provider_id = provider_id.strip() or "auto"
        known_providers = {"auto"}
        if self.gateway is not None:
            known_providers.update(str(item.get("id", "")) for item in self.gateway.specs())
        if provider_id not in known_providers:
            raise AdaptiveRuntimeError(f"Unknown model provider: {provider_id}")
        model_id = model_id.strip() or str(MODEL_ROLE_DEFAULTS[role]["model_id"])
        clean_path = ""
        if model_path.strip():
            path = Path(model_path).expanduser().resolve()
            if not path.exists():
                raise AdaptiveRuntimeError(f"Configured model path does not exist: {path}")
            clean_path = str(path)
        system_prompt = (system_prompt.strip() or str(MODEL_ROLE_DEFAULTS[role]["system_prompt"]))[:8000]
        device_id = device_id.strip()[:128] or "auto"
        config_payload = dict(config or {})
        serialized_config = json.dumps(config_payload, sort_keys=True)
        if len(serialized_config.encode("utf-8")) > 32_768:
            raise AdaptiveRuntimeError("Role configuration exceeds the 32 KB local limit.")
        now = utc_now()
        record_id = _storage_model_id(project, f"role:{role}")
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_model_roles
                    (id, project, role, provider_id, model_id, model_path, device_id, system_prompt, enabled, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, role) DO UPDATE SET
                    provider_id = excluded.provider_id,
                    model_id = excluded.model_id,
                    model_path = excluded.model_path,
                    device_id = excluded.device_id,
                    system_prompt = excluded.system_prompt,
                    enabled = excluded.enabled,
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id, project, role, provider_id, model_id, clean_path, device_id,
                    system_prompt, int(enabled), serialized_config, now, now,
                ),
            )
            row = connection.execute("SELECT * FROM adaptive_model_roles WHERE id = ?", (record_id,)).fetchone()
        self._audit(
            "adaptive.role.upsert", f"model-role:{role}", project,
            {"provider": provider_id, "model": model_id, "device": device_id, "enabled": enabled},
        )
        return self._role_record(row)

    def orchestration_runs(self, *, project: str = "default", limit: int = 20) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_orchestration_runs WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 200))),
            ).fetchall()
            results = []
            for row in rows:
                payload = dict(row)
                payload["approved"] = bool(payload["approved"])
                turns = connection.execute(
                    "SELECT * FROM adaptive_orchestration_turns WHERE run_id = ? ORDER BY turn_index",
                    (payload["id"],),
                ).fetchall()
                payload["turns"] = [self._turn_record(item) for item in turns]
                results.append(payload)
        return results

    def orchestrate_models(
        self,
        task: str,
        *,
        project: str = "default",
        mode: str = "supervised-code",
        roles: list[str] | None = None,
        max_rounds: int = 2,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        if self.gateway is None:
            raise AdaptiveRuntimeError("The model gateway is not bound to the adaptive engine.")
        task = task.strip()
        if not task:
            raise AdaptiveRuntimeError("A collaboration task is required.")
        if len(task) > 40_000:
            raise AdaptiveRuntimeError("The collaboration task exceeds the 40,000 character limit.")
        project = project.strip() or "default"
        mode = mode.strip().lower()
        if mode not in {"supervised-code", "sequence"}:
            raise AdaptiveRuntimeError(f"Unknown orchestration mode: {mode}")
        maximum = max(1, min(int(max_rounds), 6))
        configured = {item["role"]: item for item in self.model_roles(project=project) if item["enabled"]}
        run_id = uuid4().hex
        created_at = utc_now()
        self._insert_orchestration_run(run_id, project, mode, task, created_at)
        turns: list[dict[str, Any]] = []
        final_output = ""
        approved = False
        rounds = 0
        try:
            if mode == "supervised-code":
                required = ["reasoning", "code", "supervisor"]
                missing = [role for role in required if role not in configured]
                if missing:
                    raise AdaptiveRuntimeError(f"Enable the required model roles: {', '.join(missing)}")
                plan = self._run_role(
                    configured["reasoning"],
                    "Analyze this task and return a concise implementation plan plus an acceptance checklist:\n\n" + task,
                    project=project, run_id=run_id, turn_index=len(turns), allow_external=allow_external,
                )
                turns.append(plan)
                build = self._run_role(
                    configured["code"],
                    f"TASK\n{task}\n\nACCEPTANCE PLAN\n{_bounded(plan['content'])}\n\nProduce the complete implementation or exact patch.",
                    project=project, run_id=run_id, turn_index=len(turns), allow_external=allow_external,
                )
                turns.append(build)
                final_output = build["content"]
                for round_number in range(1, maximum + 1):
                    rounds = round_number
                    review = self._run_role(
                        configured["supervisor"],
                        f"TASK\n{task}\n\nACCEPTANCE PLAN\n{_bounded(plan['content'])}\n\nCANDIDATE\n{_bounded(final_output)}\n\nFirst line must be APPROVED or REVISE.",
                        project=project, run_id=run_id, turn_index=len(turns), allow_external=allow_external,
                    )
                    turns.append(review)
                    approved = _review_approved(review["content"])
                    if approved or round_number >= maximum:
                        break
                    revision = self._run_role(
                        configured["code"],
                        f"TASK\n{task}\n\nCURRENT CANDIDATE\n{_bounded(final_output)}\n\nSUPERVISOR FEEDBACK\n{_bounded(review['content'])}\n\nReturn a corrected complete implementation.",
                        project=project, run_id=run_id, turn_index=len(turns), allow_external=allow_external,
                    )
                    turns.append(revision)
                    final_output = revision["content"]
            else:
                ordered = [str(item).strip().lower() for item in (roles or ["reasoning", "research", "supervisor"])]
                previous = task
                for role in ordered[:8]:
                    if role not in configured:
                        raise AdaptiveRuntimeError(f"Model role is disabled or unknown: {role}")
                    turn = self._run_role(
                        configured[role],
                        f"Original task:\n{task}\n\nPrevious model output:\n{_bounded(previous)}",
                        project=project, run_id=run_id, turn_index=len(turns), allow_external=allow_external,
                    )
                    turns.append(turn)
                    previous = turn["content"]
                final_output = previous
                approved = (
                    _review_approved(turns[-1]["content"])
                    if turns and turns[-1]["role"] == "supervisor"
                    else bool(turns)
                )
                rounds = 1
            completed_at = utc_now()
            self._finish_orchestration_run(run_id, status="completed", rounds=rounds, approved=approved, final_output=final_output, error="", completed_at=completed_at)
            self._audit(
                "adaptive.orchestration.complete", f"orchestration:{run_id}", project,
                {"mode": mode, "turns": len(turns), "rounds": rounds, "approved": approved},
            )
            return {
                "id": run_id, "project": project, "mode": mode, "task": task,
                "status": "completed", "rounds": rounds, "approved": approved,
                "final_output": final_output, "turns": turns, "created_at": created_at,
                "completed_at": completed_at,
                "privacy_boundary": "Role prompts and model-to-model outputs follow each configured provider boundary. External providers require explicit allow_external approval.",
            }
        except Exception as exc:
            completed_at = utc_now()
            self._finish_orchestration_run(run_id, status="failed", rounds=rounds, approved=False, final_output=final_output, error=str(exc), completed_at=completed_at)
            self._audit(
                "adaptive.orchestration.fail", f"orchestration:{run_id}", project,
                {"mode": mode, "turns": len(turns), "error_type": type(exc).__name__},
            )
            if isinstance(exc, AdaptiveRuntimeError):
                raise
            raise AdaptiveRuntimeError(str(exc)) from exc

    def close(self) -> None:
        self.direct.close()

    def benchmark_device(
        self,
        device_id: str,
        *,
        project: str = "default",
        iterations: int = 20,
        dimension: int = 256,
    ) -> dict[str, Any]:
        profile = self.profiler.profile()
        device = self._device(profile.devices, device_id)
        try:
            evidence = self.openvino.benchmark(device, iterations=iterations, dimension=dimension)
        except RuntimeError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        benchmark_id = uuid4().hex
        now = utc_now()
        payload = {
            "id": benchmark_id,
            "project": project,
            "device_id": device.id,
            "device_name": device.name,
            "evidence": evidence,
            "scope": "synthetic matrix kernel and numerical reference; this is not a full-model throughput claim",
            "created_at": now,
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO adaptive_device_benchmarks (id, project, device_id, evidence_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (benchmark_id, project, device.id, json.dumps(payload, sort_keys=True), now),
            )
        self._audit("adaptive.device.benchmark", f"device:{device.id}", project, {"status": evidence.get("status"), "dimension": dimension})
        return payload

    def autotune_device(
        self,
        device_id: str,
        *,
        project: str = "default",
        phase: str = "decode",
        iterations: int = 12,
    ) -> dict[str, Any]:
        profile = self.profiler.profile()
        device = self._device(profile.devices, device_id)
        try:
            result = self.autotuner.tune(device, project=project, phase=phase, iterations=iterations)
        except (RuntimeError, ValueError) as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        self._audit("adaptive.device.autotune", f"device:{device.id}", project, {"phase": phase, "winner": result["winner"]["dimension"]})
        return result

    def plans(self, *, project: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT plan_json FROM adaptive_execution_plans WHERE project = ? ORDER BY updated_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 500))),
            ).fetchall()
        return [json.loads(row["plan_json"]) for row in rows]

    def benchmarks(self, *, project: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM adaptive_device_benchmarks WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 500))),
            ).fetchall()
        return [json.loads(row["evidence_json"]) for row in rows]

    def create_kv_session(self, session_id: str, *, prefix_tokens: list[int], priority: int = 50) -> dict[str, Any]:
        return self.kv_cache.create_session(session_id, prefix_tokens=prefix_tokens, priority=priority)

    def append_kv(self, session_id: str, token_ids: list[int]) -> dict[str, Any]:
        return self.kv_cache.append(session_id, token_ids)

    def release_kv(self, session_id: str) -> bool:
        return self.kv_cache.release(session_id)

    def queue_request(
        self,
        *,
        request_id: str,
        model_id: str,
        phase: str,
        remaining_tokens: int,
        priority: int = 80,
        deadline_seconds: float = 0.0,
        session_id: str = "",
    ) -> dict[str, Any]:
        try:
            request = InferenceRequest(
                id=request_id,
                model_id=model_id,
                phase=RequestPhase(phase),
                remaining_tokens=remaining_tokens,
                priority=_request_priority(priority),
                deadline_monotonic=(time.monotonic() + deadline_seconds) if deadline_seconds > 0 else 0.0,
                session_id=session_id,
            )
        except ValueError as exc:
            raise AdaptiveRuntimeError(str(exc)) from exc
        return self.scheduler.submit(request)

    def next_batch(self, *, power_mode: str = "balanced") -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.scheduler.next_batch(power_mode=power_mode)]

    def _save_model(self, model_id: str, project: str, source_path: str, fingerprint: str, payload: dict[str, Any]) -> None:
        now = utc_now()
        storage_id = _storage_model_id(project, fingerprint)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_models (id, project, source_path, fingerprint, model_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, fingerprint) DO UPDATE SET
                    source_path = excluded.source_path,
                    model_json = excluded.model_json,
                    updated_at = excluded.updated_at
                """,
                (storage_id, project, source_path, fingerprint, json.dumps(payload, sort_keys=True), now, now),
            )

    def _enrich_model(self, model: Any) -> Any:
        if model.source_format == "onnx" and model.metadata.get("opaque_graph"):
            try:
                return enrich_onnx_from_probe(model, self.profiler.inspect_onnx(model.source_path))
            except RuntimeError:
                return model
        return model

    def _single_compute_target(self, placements: tuple[Any, ...], devices: tuple[DeviceProfile, ...]) -> DeviceProfile | None:
        ids = {
            placement.device_id
            for placement in placements
            if placement.op not in {"tokenization", "sampling"} and placement.backend == "openvino"
        }
        if len(ids) != 1:
            return None
        return self._device(devices, ids.pop())

    def _calibrated_profile_device(self, project: str, fingerprint: str) -> str:
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT winner_profile_device FROM adaptive_model_calibrations
                WHERE project = ? AND model_fingerprint = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (project, fingerprint),
            ).fetchone()
        return str(row["winner_profile_device"]) if row else ""

    @staticmethod
    def _profile_id_for_target(devices: tuple[DeviceProfile, ...], target: str) -> str:
        for device in devices:
            if openvino_device_id(device).upper() == target.upper():
                return device.id
        root = target.split(".", 1)[0].split(":", 1)[0].lower()
        match = next((item for item in devices if item.kind == root and "openvino" in item.backends), None)
        return match.id if match else ""

    def _direct_target(self, devices: tuple[DeviceProfile, ...], requested: str) -> tuple[DeviceProfile, str]:
        clean = requested.strip()
        if clean.lower() in {"auto", "auto:0"}:
            # Prefer a real local accelerator when one is exposed, but always
            # keep CPU as the dependable fallback for the many CPU-only hosts.
            device = next(
                (
                    item
                    for kind in ("gpu", "npu", "cpu")
                    for item in devices
                    if item.kind == kind and "openvino" in item.backends
                ),
                None,
            )
            if device is None:
                raise AdaptiveRuntimeError("No OpenVINO CPU, GPU, or NPU target is available for AUTO selection.")
            target = openvino_device_id(device)
            if not target:
                raise AdaptiveRuntimeError(f"OpenVINO did not expose {device.name}")
            return device, target
        device = next(
            (
                item for item in devices
                if item.id.lower() == clean.lower() or openvino_device_id(item).lower() == clean.lower()
            ),
            None,
        )
        if not device or "openvino" not in device.backends:
            raise AdaptiveRuntimeError(f"Unknown or unavailable OpenVINO device: {requested}")
        target = openvino_device_id(device)
        if not target:
            raise AdaptiveRuntimeError(f"OpenVINO did not expose {device.name}")
        return device, target

    def _run_role(
        self,
        role: dict[str, Any],
        prompt: str,
        *,
        project: str,
        run_id: str,
        turn_index: int,
        allow_external: bool,
    ) -> dict[str, Any]:
        provider_id = str(role["provider_id"])
        model_id = str(role["model_id"] or "").strip() or None
        if provider_id == "openvino-direct" and role.get("model_path"):
            device_id = str(role.get("device_id", "auto"))
            if device_id == "auto":
                profile = self.profiler.profile()
                preferred = next(
                    (item for kind in ("gpu", "npu", "cpu") for item in profile.devices if item.kind == kind and "openvino" in item.backends),
                    None,
                )
                if preferred is None:
                    raise AdaptiveRuntimeError("No OpenVINO CPU, GPU, or NPU target is available for this role.")
                device_id = preferred.id
            self.load_direct_llm(
                str(role["model_path"]),
                device_id=device_id,
                project=project,
                model_id=model_id or f"engine-{role['role']}",
                config=dict(role.get("config") or {}).get("runtime", {}),
                scheduler=dict(role.get("config") or {}).get("scheduler", {}),
            )
        started = time.perf_counter()
        result = self.gateway.complete(
            [
                {"role": "system", "content": str(role["system_prompt"])},
                {"role": "user", "content": prompt},
            ],
            provider_id=provider_id,
            model=model_id,
            allow_external=allow_external,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        content = str(result.content)[:120_000]
        metadata = {
            **dict(result.metadata),
            "configured_provider": provider_id,
            "configured_model": model_id or "",
            "latency_ms": round(latency_ms, 3),
        }
        turn = {
            "id": uuid4().hex,
            "run_id": run_id,
            "turn_index": turn_index,
            "role": role["role"],
            "label": role["label"],
            "provider_id": result.provider,
            "model_id": result.model,
            "content": content,
            "metadata": metadata,
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_orchestration_turns
                    (id, run_id, turn_index, role, provider_id, model_id, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn["id"], run_id, turn_index, turn["role"], turn["provider_id"],
                    turn["model_id"], content, json.dumps(metadata, sort_keys=True, default=str), turn["created_at"],
                ),
            )
        self._audit(
            "adaptive.orchestration.turn", f"orchestration:{run_id}", project,
            {
                "turn": turn_index, "role": role["role"], "provider": result.provider,
                "model": result.model, "latency_ms": round(latency_ms, 3),
            },
        )
        return turn

    def _insert_orchestration_run(self, run_id: str, project: str, mode: str, task: str, created_at: str) -> None:
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO adaptive_orchestration_runs
                    (id, project, mode, task, status, rounds, approved, final_output, error, created_at, completed_at)
                VALUES (?, ?, ?, ?, 'running', 0, 0, '', '', ?, '')
                """,
                (run_id, project, mode, task, created_at),
            )
        self._audit("adaptive.orchestration.start", f"orchestration:{run_id}", project, {"mode": mode})

    def _finish_orchestration_run(
        self,
        run_id: str,
        *,
        status: str,
        rounds: int,
        approved: bool,
        final_output: str,
        error: str,
        completed_at: str,
    ) -> None:
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE adaptive_orchestration_runs
                SET status = ?, rounds = ?, approved = ?, final_output = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, rounds, int(approved), final_output[:120_000], error[:4000], completed_at, run_id),
            )

    @staticmethod
    def _role_record(row: Any) -> dict[str, Any]:
        payload = dict(row)
        role = str(payload["role"])
        return {
            "id": payload["id"],
            "project": payload["project"],
            "role": role,
            "label": MODEL_ROLE_DEFAULTS.get(role, {}).get("label", role.title()),
            "provider_id": payload["provider_id"],
            "model_id": payload["model_id"],
            "model_path": payload["model_path"],
            "device_id": payload["device_id"],
            "system_prompt": payload["system_prompt"],
            "enabled": bool(payload["enabled"]),
            "config": json.loads(payload["config_json"] or "{}"),
            "created_at": payload["created_at"],
            "updated_at": payload["updated_at"],
            "persisted": True,
        }

    @staticmethod
    def _turn_record(row: Any) -> dict[str, Any]:
        payload = dict(row)
        payload["metadata"] = json.loads(payload.pop("metadata_json") or "{}")
        payload["label"] = MODEL_ROLE_DEFAULTS.get(str(payload["role"]), {}).get("label", str(payload["role"]).title())
        return payload

    @staticmethod
    def _device(devices: tuple[DeviceProfile, ...], device_id: str) -> DeviceProfile:
        device = next((item for item in devices if item.id == device_id), None)
        if not device:
            raise AdaptiveRuntimeError(f"Unknown local device: {device_id}")
        return device

    def _audit(self, action: str, resource: str, project: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="adaptive-engine",
                action=action,
                resource=resource,
                project=project,
                metadata=metadata,
            )
        )


def _storage_model_id(project: str, fingerprint: str) -> str:
    return hashlib.sha256(f"{project}:{fingerprint}".encode("utf-8")).hexdigest()[:24]


def _request_priority(value: int) -> RequestPriority:
    requested = max(10, min(int(value), 100))
    return min(RequestPriority, key=lambda item: abs(int(item) - requested))


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _bounded(value: str, limit: int = 24_000) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + "\n\n[content compacted between sections]\n\n" + text[-tail:]


def _review_approved(value: str) -> bool:
    first_line = next((line.strip().upper() for line in str(value).splitlines() if line.strip()), "")
    return first_line == "APPROVED" or first_line.startswith("APPROVED:")
