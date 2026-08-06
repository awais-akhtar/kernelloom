"""CPU-first execution planning that respects the cores available to this process.

The plan is deliberately deterministic rather than a guessed token-per-second
prediction.  It gives a safe starting point for llama.cpp on machines without
an accelerator, and callers can benchmark nearby values on their own hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Literal


CPUProfile = Literal["auto", "latency", "balanced", "throughput", "efficient"]


@dataclass(frozen=True, slots=True)
class CPUExecutionPlan:
    """A reproducible CPU configuration recommendation for local inference."""

    profile: str
    available_cores: int
    reserved_cores: int
    threads: int
    batch_threads: int
    recommended_batch_size: int
    recommended_micro_batch_size: int
    rationale: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def available_cpu_cores() -> int:
    """Return CPU cores this process may use, respecting Linux CPU affinity/cgroups."""

    try:
        affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        affinity = set()
    return max(1, len(affinity) if affinity else (os.cpu_count() or 1))


def plan_cpu_execution(
    profile: CPUProfile | str = "auto",
    *,
    reserve_cores: int = 1,
    available_cores: int | None = None,
) -> CPUExecutionPlan:
    """Create a CPU plan for latency, throughput, balance, or lower power use.

    ``available_cores`` makes the algorithm easy to test and lets a host apply
    its own quota.  A user-provided ``threads`` value always takes precedence
    over this recommendation when constructing :class:`ModelConfig`.
    """

    selected = str(profile).strip().lower() or "auto"
    aliases = {"auto": "balanced", "power": "efficient", "eco": "efficient"}
    selected = aliases.get(selected, selected)
    if selected not in {"latency", "balanced", "throughput", "efficient"}:
        raise ValueError("cpu profile must be auto, latency, balanced, throughput, or efficient")
    if reserve_cores < 0:
        raise ValueError("reserve_cores cannot be negative")
    total = max(1, int(available_cores or available_cpu_cores()))
    usable = max(1, total - min(int(reserve_cores), total - 1))

    if selected == "throughput":
        threads = total
        batch_threads = total
        batch_size, micro_batch_size = 1024, 256
        effective_reserved = 0
        rationale = "Uses every available logical core and larger prompt batches for batch/offline throughput."
    elif selected == "efficient":
        threads = max(1, (usable + 1) // 2)
        batch_threads = threads
        batch_size, micro_batch_size = 256, 64
        effective_reserved = max(0, total - usable)
        rationale = "Limits concurrent CPU work to preserve responsiveness, heat, and battery life."
    elif selected == "latency":
        threads = usable
        batch_threads = usable
        batch_size, micro_batch_size = 256, 128
        effective_reserved = max(0, total - usable)
        rationale = "Reserves system capacity and limits prefill batches to favor interactive response time."
    else:
        threads = usable
        batch_threads = usable
        batch_size, micro_batch_size = 512, 128
        effective_reserved = max(0, total - usable)
        rationale = "Reserves system capacity while balancing prompt throughput and interactive latency."

    return CPUExecutionPlan(
        profile=selected,
        available_cores=total,
        reserved_cores=effective_reserved,
        threads=threads,
        batch_threads=batch_threads,
        recommended_batch_size=batch_size,
        recommended_micro_batch_size=micro_batch_size,
        rationale=rationale,
    )


__all__ = ["CPUExecutionPlan", "CPUProfile", "available_cpu_cores", "plan_cpu_execution"]
