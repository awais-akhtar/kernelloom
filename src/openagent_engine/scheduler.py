"""Deadline-aware local inference scheduling with interactive protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
import heapq
import threading
import time
from typing import Any


class RequestPriority(IntEnum):
    BACKGROUND = 10
    BATCH = 30
    AGENT = 50
    INTERACTIVE = 80
    REALTIME = 100


class RequestPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    EMBEDDING = "embedding"
    RERANK = "rerank"


@dataclass
class InferenceRequest:
    id: str
    model_id: str
    phase: RequestPhase
    remaining_tokens: int
    priority: RequestPriority = RequestPriority.INTERACTIVE
    deadline_monotonic: float = 0.0
    session_id: str = ""
    submitted_at: float = field(default_factory=time.monotonic)
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["priority"] = int(self.priority)
        payload["phase"] = self.phase.value
        return payload


class DeadlineBatchScheduler:
    """Select compatible microbatches using urgency plus fairness aging."""

    def __init__(self, *, max_batch_size: int = 8, interactive_reserve: int = 2) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.interactive_reserve = max(0, min(int(interactive_reserve), self.max_batch_size))
        self._requests: dict[str, InferenceRequest] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def submit(self, request: InferenceRequest) -> dict[str, Any]:
        if not request.id.strip() or not request.model_id.strip():
            raise ValueError("request id and model id are required")
        if request.remaining_tokens <= 0:
            raise ValueError("remaining_tokens must be positive")
        with self._lock:
            if request.id in self._requests:
                raise ValueError("request already exists")
            self._sequence += 1
            request.sequence = self._sequence
            self._requests[request.id] = request
            return request.to_dict()

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            return self._requests.pop(request_id, None) is not None

    def next_batch(self, *, now: float | None = None, power_mode: str = "balanced") -> list[InferenceRequest]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            if not self._requests:
                return []
            limit = self._batch_limit(power_mode)
            ranked = sorted(self._requests.values(), key=lambda item: self._rank(item, timestamp))
            anchor = ranked[0]
            compatible = [item for item in ranked if item.model_id == anchor.model_id and item.phase == anchor.phase]
            interactive = [item for item in compatible if item.priority >= RequestPriority.INTERACTIVE]
            ordinary = [item for item in compatible if item.priority < RequestPriority.INTERACTIVE]
            chosen = interactive[: min(limit, self.interactive_reserve)]
            seen = {item.id for item in chosen}
            for item in compatible:
                if len(chosen) >= limit:
                    break
                if item.id not in seen:
                    chosen.append(item)
                    seen.add(item.id)
            if not chosen and ordinary:
                chosen = ordinary[:limit]
            return chosen

    def complete_step(self, request_id: str, *, tokens: int = 1) -> dict[str, Any]:
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                raise KeyError(request_id)
            request.remaining_tokens = max(0, request.remaining_tokens - max(1, int(tokens)))
            complete = request.remaining_tokens == 0
            payload = request.to_dict()
            payload["complete"] = complete
            if complete:
                self._requests.pop(request_id, None)
            return payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            by_priority: dict[str, int] = {}
            by_phase: dict[str, int] = {}
            for request in self._requests.values():
                by_priority[request.priority.name.lower()] = by_priority.get(request.priority.name.lower(), 0) + 1
                by_phase[request.phase.value] = by_phase.get(request.phase.value, 0) + 1
            return {
                "queued": len(self._requests),
                "max_batch_size": self.max_batch_size,
                "interactive_reserve": self.interactive_reserve,
                "by_priority": by_priority,
                "by_phase": by_phase,
            }

    def _rank(self, request: InferenceRequest, now: float) -> tuple[float, int]:
        age = max(0.0, now - request.submitted_at)
        deadline_urgency = 0.0
        if request.deadline_monotonic > 0:
            seconds_left = request.deadline_monotonic - now
            deadline_urgency = 10_000.0 if seconds_left <= 0 else 1000.0 / max(0.001, seconds_left)
        score = float(request.priority) * 100.0 + min(age, 600.0) * 2.0 + deadline_urgency
        if request.phase == RequestPhase.DECODE:
            score += 250.0
        return (-score, request.sequence)

    def _batch_limit(self, power_mode: str) -> int:
        normalized = str(power_mode).strip().lower()
        if normalized in {"battery", "background"}:
            return max(1, self.max_batch_size // 3)
        if normalized == "quiet":
            return max(1, self.max_batch_size // 2)
        return self.max_batch_size
