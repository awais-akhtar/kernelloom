"""Measured device autotuning built on backend compiler validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any
from uuid import uuid4

from .models import utc_now
from .storage import EngineStore
from .backends import OpenVINOBackend
from .hardware import DeviceProfile


@dataclass(frozen=True)
class TuningResult:
    id: str
    project: str
    device_id: str
    phase: str
    winner: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = list(self.candidates)
        return payload


class DeviceAutoTuner:
    """Benchmark representative shapes and retain only numerically valid runs."""

    def __init__(self, store: EngineStore, backend: OpenVINOBackend) -> None:
        self.store = store
        self.backend = backend
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS adaptive_tuning_results (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adaptive_tuning_device
                    ON adaptive_tuning_results(project, device_id, created_at DESC);
                """
            )

    def tune(
        self,
        device: DeviceProfile,
        *,
        project: str = "default",
        phase: str = "decode",
        iterations: int = 12,
    ) -> dict[str, Any]:
        phase = str(phase).strip().lower()
        if phase not in {"prefill", "decode", "embedding"}:
            raise ValueError("phase must be prefill, decode, or embedding")
        dimension = {"decode": 256, "prefill": 512, "embedding": 384}[phase]
        configurations = (
            {"performance_hint": "LATENCY", "num_streams": "1"},
            {"performance_hint": "LATENCY", "num_streams": ""},
            {"performance_hint": "THROUGHPUT", "num_streams": ""},
        )
        candidates: list[dict[str, Any]] = []
        for configuration in configurations:
            started = time.perf_counter()
            try:
                evidence = self.backend.benchmark(
                    device,
                    iterations=iterations,
                    dimension=dimension,
                    performance_hint=configuration["performance_hint"],
                    num_streams=configuration["num_streams"],
                )
                valid = evidence.get("status") == "verified"
                candidate = {
                    "dimension": dimension,
                    "configuration": configuration,
                    "valid": valid,
                    "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                    "evidence": evidence,
                }
            except RuntimeError as exc:
                candidate = {
                    "dimension": dimension,
                    "configuration": configuration,
                    "valid": False,
                    "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": str(exc),
                }
            candidates.append(candidate)
        valid_candidates = [item for item in candidates if item["valid"]]
        if not valid_candidates:
            raise RuntimeError("No autotuning candidate passed numerical validation")
        winner = min(valid_candidates, key=lambda item: float(item["evidence"]["latency_ms"]["mean"]))
        result = TuningResult(
            id=uuid4().hex,
            project=project.strip() or "default",
            device_id=device.id,
            phase=phase,
            winner=winner,
            candidates=tuple(candidates),
            created_at=utc_now(),
        )
        payload = result.to_dict()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO adaptive_tuning_results (id, project, device_id, phase, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (result.id, result.project, result.device_id, result.phase, json.dumps(payload, sort_keys=True), result.created_at),
            )
        return payload

    def results(self, *, project: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT result_json FROM adaptive_tuning_results WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 500))),
            ).fetchall()
        return [json.loads(row["result_json"]) for row in rows]
