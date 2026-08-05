"""Transfer-aware compiler planning for heterogeneous local inference.

The compiler does not pretend that a schedule estimate is executable code. It
produces separate prefill and decode plans, records required model conversion,
and upgrades a plan to compiled or verified only after a backend proves it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable

from .hardware import DeviceProfile, HardwareProfile
from .ir import IRLevel, IRNode, ModelIR, OpCode
from .codegen import KernelLowerer


PLANNER_ID = "openagent-coherent-phase-planner-v1"
PRECISION_BITS = {
    "fp32": 32.0,
    "bf16": 16.0,
    "fp16": 16.0,
    "int8": 8.0,
    "q8": 8.0,
    "int4": 4.0,
    "q4": 4.0,
    "nf4": 4.0,
    "int3": 3.0,
    "q3": 3.0,
    "int2": 2.0,
    "q2": 2.0,
}


class CompilationError(RuntimeError):
    """Raised when no honest execution plan can satisfy the constraints."""


@dataclass(frozen=True)
class PrecisionAssignment:
    node_id: str
    precision: str
    storage_bytes: int
    quality_penalty: float
    conversion_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodePlacement:
    node_id: str
    node_name: str
    op: str
    device_id: str
    device_name: str
    backend: str
    precision: str
    compute_ms: float
    memory_ms: float
    transfer_ms: float
    dispatch_ms: float
    estimated_ms: float
    executable: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    prompt_tokens: int
    placements: tuple[NodePlacement, ...]
    estimated_latency_ms: float
    estimated_tokens_per_second: float
    transfer_count: int
    device_regions: tuple[dict[str, Any], ...]
    executable: bool
    estimate_source: str

    def to_dict(self, *, include_placements: bool = True) -> dict[str, Any]:
        result = {
            "phase": self.phase,
            "prompt_tokens": self.prompt_tokens,
            "estimated_latency_ms": round(self.estimated_latency_ms, 4),
            "estimated_tokens_per_second": round(self.estimated_tokens_per_second, 4),
            "transfer_count": self.transfer_count,
            "device_regions": list(self.device_regions),
            "executable": self.executable,
            "estimate_source": self.estimate_source,
        }
        if include_placements:
            result["placements"] = [placement.to_dict() for placement in self.placements]
        return result


@dataclass(frozen=True)
class ExecutionPackage:
    id: str
    planner: str
    model_id: str
    model_name: str
    model_fingerprint: str
    source_format: str
    hardware_profile_id: str
    status: str
    power_mode: str
    precision: tuple[PrecisionAssignment, ...]
    prefill: PhasePlan
    decode: PhasePlan
    memory_plan: dict[str, Any]
    kernel_plan: dict[str, Any]
    constraints: dict[str, Any]
    measured_evidence: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_placements: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "planner": self.planner,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_fingerprint": self.model_fingerprint,
            "source_format": self.source_format,
            "hardware_profile_id": self.hardware_profile_id,
            "status": self.status,
            "power_mode": self.power_mode,
            "precision": [item.to_dict() for item in self.precision],
            "prefill": self.prefill.to_dict(include_placements=include_placements),
            "decode": self.decode.to_dict(include_placements=include_placements),
            "memory_plan": self.memory_plan,
            "kernel_plan": self.kernel_plan,
            "constraints": self.constraints,
            "measured_evidence": self.measured_evidence,
            "warnings": list(self.warnings),
            "privacy_boundary": "Compilation reads local model metadata and local hardware capabilities. It does not upload model weights or telemetry.",
        }


@dataclass(frozen=True)
class _PathRecord:
    cost: float
    previous: tuple[str, int] | None
    placement: NodePlacement


class AdaptiveCompiler:
    """Compile model metadata into phase-coupled device execution plans.

    Coherent phase planning combines four signals in one dynamic program:
    operation cost, memory traffic, inter-device transfer, and migration
    hysteresis. Prefill is solved first; decode then receives a soft residency
    affinity so it can change devices when beneficial without oscillating for
    tiny local wins.
    """

    def compile(
        self,
        model: ModelIR,
        hardware: HardwareProfile,
        *,
        prompt_tokens: int = 512,
        context_tokens: int = 4096,
        memory_budget_gb: float | None = None,
        quality_loss_limit: float = 0.08,
        power_mode: str = "balanced",
        max_device_transitions: int = 4,
        preferred_device_id: str = "",
    ) -> ExecutionPackage:
        prompt_tokens = max(1, min(int(prompt_tokens), max(1, model.context_length or 1_000_000)))
        context_tokens = max(prompt_tokens, int(context_tokens))
        power_mode = _power_mode(power_mode)
        devices = tuple(device for device in hardware.devices if device.available)
        if not devices:
            raise CompilationError("No local compute device was discovered.")
        budget_gb = float(memory_budget_gb if memory_budget_gb is not None else max(0.25, hardware.available_ram_gb * 0.82))
        if budget_gb <= 0:
            raise CompilationError("Memory budget must be positive.")
        nodes = model.graph_nodes or model.tensor_nodes
        if not nodes:
            raise CompilationError("The model frontend produced no executable graph nodes.")

        precision, precision_summary = select_precision_plan(
            model,
            nodes,
            devices,
            memory_budget_bytes=int(budget_gb * 1024**3),
            quality_loss_limit=max(0.0, min(float(quality_loss_limit), 1.0)),
        )
        precision_map = {item.node_id: item for item in precision}
        prefill = self._partition(
            model,
            nodes,
            devices,
            precision_map,
            phase="prefill",
            token_count=prompt_tokens,
            power_mode=power_mode,
            max_transitions=max_device_transitions,
            affinity={},
            preferred_device_id=preferred_device_id,
        )
        prefill_affinity = {placement.node_id: placement.device_id for placement in prefill.placements}
        decode = self._partition(
            model,
            nodes,
            devices,
            precision_map,
            phase="decode",
            token_count=1,
            power_mode=power_mode,
            max_transitions=max_device_transitions,
            affinity=prefill_affinity,
            preferred_device_id=preferred_device_id,
        )
        memory_plan = build_memory_plan(
            model,
            devices,
            precision_summary["planned_weight_bytes"],
            context_tokens=context_tokens,
            budget_bytes=int(budget_gb * 1024**3),
            preferred_device=_dominant_device(decode.placements),
        )
        lowerer = KernelLowerer()
        prefill_kernels = lowerer.lower(prefill.placements, devices, phase="prefill")
        decode_kernels = lowerer.lower(decode.placements, devices, phase="decode")
        kernel_plan = {
            "lowering": "vendor-backend candidates",
            "prefill": [node.to_dict() for node in prefill_kernels],
            "decode": [node.to_dict() for node in decode_kernels],
            "correctness_gate": "No candidate becomes selected until backend compilation and numerical validation pass.",
        }
        warnings = list(model.warnings)
        if not precision_summary["memory_feasible"]:
            warnings.append("The precision plan exceeds the requested memory budget; storage-backed execution would be latency-bound.")
        if precision_summary["quality_penalty"] > quality_loss_limit:
            warnings.append("The requested memory target cannot be reached inside the quality-loss constraint.")
        if precision_summary["conversion_required"]:
            warnings.append("One or more precision assignments require an offline, calibrated model conversion before execution.")
        executable = prefill.executable and decode.executable and not precision_summary["conversion_required"]
        status = "planned"
        constraints = {
            "prompt_tokens": prompt_tokens,
            "context_tokens": context_tokens,
            "memory_budget_gb": round(budget_gb, 4),
            "quality_loss_limit": quality_loss_limit,
            "max_device_transitions": max_device_transitions,
            "precision_summary": precision_summary,
            "backend_route_executable": executable,
            "measured_preferred_device": preferred_device_id,
        }
        identity = hashlib.sha256(
            json.dumps(
                {
                    "planner": PLANNER_ID,
                    "model": model.fingerprint,
                    "hardware": hardware.id,
                    "power": power_mode,
                    "constraints": constraints,
                    "prefill": [(item.node_id, item.device_id, item.precision) for item in prefill.placements],
                    "decode": [(item.node_id, item.device_id, item.precision) for item in decode.placements],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        return ExecutionPackage(
            id=identity,
            planner=PLANNER_ID,
            model_id=model.id,
            model_name=model.name,
            model_fingerprint=model.fingerprint,
            source_format=model.source_format,
            hardware_profile_id=hardware.id,
            status=status,
            power_mode=power_mode,
            precision=tuple(precision),
            prefill=prefill,
            decode=decode,
            memory_plan=memory_plan,
            kernel_plan=kernel_plan,
            constraints=constraints,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _partition(
        self,
        model: ModelIR,
        nodes: tuple[IRNode, ...],
        devices: tuple[DeviceProfile, ...],
        precision: dict[str, PrecisionAssignment],
        *,
        phase: str,
        token_count: int,
        power_mode: str,
        max_transitions: int,
        affinity: dict[str, str],
        preferred_device_id: str = "",
    ) -> PhasePlan:
        maximum = max(0, min(int(max_transitions), 32))
        histories: list[dict[tuple[str, int], _PathRecord]] = []
        previous: dict[tuple[str, int], _PathRecord] = {}
        for index, node in enumerate(nodes):
            assignment = precision[node.id]
            candidates: list[NodePlacement] = []
            for device in devices:
                placement = estimate_placement(
                    model,
                    node,
                    device,
                    assignment,
                    phase=phase,
                    token_count=token_count,
                    power_mode=power_mode,
                )
                if placement is not None:
                    candidates.append(placement)
            if not candidates:
                raise CompilationError(f"No device supports {node.op.value} at {assignment.precision} for {phase}.")
            if preferred_device_id and node.op not in {OpCode.TOKENIZATION, OpCode.SAMPLING}:
                measured = [item for item in candidates if item.device_id == preferred_device_id]
                if measured:
                    candidates = measured
            current: dict[tuple[str, int], _PathRecord] = {}
            if index == 0:
                for placement in candidates:
                    affinity_cost = _affinity_cost(node.id, placement.device_id, affinity, placement.estimated_ms)
                    key = (placement.device_id, 0)
                    current[key] = _PathRecord(placement.estimated_ms + affinity_cost, None, placement)
            else:
                for old_key, old_record in previous.items():
                    old_device = _device_by_id(devices, old_key[0])
                    for placement in candidates:
                        transition = int(old_key[0] != placement.device_id)
                        transitions = old_key[1] + transition
                        if transitions > maximum:
                            continue
                        new_device = _device_by_id(devices, placement.device_id)
                        transfer_ms = transition_cost_ms(
                            old_device,
                            new_device,
                            node.activation_bytes,
                            phase=phase,
                        ) if transition else 0.0
                        adjusted = NodePlacement(
                            **{
                                **placement.to_dict(),
                                "transfer_ms": transfer_ms,
                                "estimated_ms": placement.estimated_ms + transfer_ms,
                            }
                        )
                        hysteresis = transfer_ms * 1.25 if transition else -min(0.02, adjusted.estimated_ms * 0.01)
                        affinity_cost = _affinity_cost(node.id, adjusted.device_id, affinity, adjusted.estimated_ms)
                        cost = old_record.cost + adjusted.estimated_ms + hysteresis + affinity_cost
                        key = (adjusted.device_id, transitions)
                        existing = current.get(key)
                        if existing is None or cost < existing.cost:
                            current[key] = _PathRecord(cost, old_key, adjusted)
            if not current:
                raise CompilationError(f"No path remains under the {maximum}-transition limit for {phase}.")
            if len(current) > 64:
                current = dict(sorted(current.items(), key=lambda item: item[1].cost)[:64])
            histories.append(current)
            previous = current
        final_key = min(previous, key=lambda key: previous[key].cost)
        reversed_placements: list[NodePlacement] = []
        key: tuple[str, int] | None = final_key
        for layer in range(len(histories) - 1, -1, -1):
            if key is None:
                raise CompilationError("Planner backtrace is incomplete.")
            record = histories[layer][key]
            reversed_placements.append(record.placement)
            key = record.previous
        placements = tuple(reversed(reversed_placements))
        latency = sum(item.estimated_ms for item in placements)
        transition_count = sum(1 for left, right in zip(placements, placements[1:]) if left.device_id != right.device_id)
        regions = _regions(placements)
        if phase == "prefill":
            token_rate = token_count / max(1e-9, latency / 1000.0)
        else:
            token_rate = 1000.0 / max(1e-9, latency)
        return PhasePlan(
            phase=phase,
            prompt_tokens=token_count if phase == "prefill" else 1,
            placements=placements,
            estimated_latency_ms=latency,
            estimated_tokens_per_second=token_rate,
            transfer_count=transition_count,
            device_regions=regions,
            executable=all(item.executable for item in placements),
            estimate_source="analytical-unbenchmarked; replace with device/model measurements before performance claims",
        )


def select_precision_plan(
    model: ModelIR,
    nodes: tuple[IRNode, ...],
    devices: tuple[DeviceProfile, ...],
    *,
    memory_budget_bytes: int,
    quality_loss_limit: float,
) -> tuple[list[PrecisionAssignment], dict[str, Any]]:
    """Allocate a precision budget while protecting sensitive model regions."""

    source_bits = max(1.0, float(model.quantization_bits or 16.0))
    total_parameter_bytes = sum(max(0, node.parameter_bytes) for node in nodes) or model.weight_bytes
    selected: dict[str, PrecisionAssignment] = {}
    alternatives: list[tuple[float, int, float, str, str]] = []
    for node in nodes:
        supported = _supported_precisions(node, devices)
        native = _native_precision(source_bits)
        if model.source_format == "gguf":
            chosen = native if native in supported else min(supported, key=lambda value: abs(PRECISION_BITS[value] - source_bits))
            conversion = False
        else:
            chosen = max(supported, key=lambda value: PRECISION_BITS[value])
            conversion = PRECISION_BITS[chosen] < source_bits - 0.5
        bytes_at_precision = _node_storage_bytes(node, chosen, source_bits)
        selected[node.id] = PrecisionAssignment(node.id, chosen, bytes_at_precision, 0.0, conversion)
        ordered = sorted((value for value in supported if PRECISION_BITS[value] < PRECISION_BITS[chosen]), key=lambda value: PRECISION_BITS[value], reverse=True)
        current = chosen
        current_bytes = bytes_at_precision
        current_penalty = 0.0
        for candidate in ordered:
            candidate_bytes = _node_storage_bytes(node, candidate, source_bits)
            savings = max(0, current_bytes - candidate_bytes)
            if not savings:
                continue
            penalty = _quality_penalty(node, current, candidate, total_parameter_bytes)
            ratio = penalty / savings
            alternatives.append((ratio, savings, penalty, node.id, candidate))
            current = candidate
            current_bytes = candidate_bytes
            current_penalty += penalty
    planned_bytes = sum(item.storage_bytes for item in selected.values())
    quality_penalty = 0.0
    for _, _, penalty, node_id, candidate in sorted(alternatives, key=lambda item: (item[0], -item[1])):
        if planned_bytes <= memory_budget_bytes:
            break
        current = selected[node_id]
        candidate_bytes = _node_storage_bytes(next(node for node in nodes if node.id == node_id), candidate, source_bits)
        savings = current.storage_bytes - candidate_bytes
        if savings <= 0 or quality_penalty + penalty > quality_loss_limit:
            continue
        selected[node_id] = PrecisionAssignment(
            node_id,
            candidate,
            candidate_bytes,
            current.quality_penalty + penalty,
            model.source_format != "gguf" and PRECISION_BITS[candidate] < source_bits - 0.5,
        )
        planned_bytes -= savings
        quality_penalty += penalty
    assignments = [selected[node.id] for node in nodes]
    summary = {
        "source_effective_bits": round(source_bits, 4),
        "source_weight_bytes": model.weight_bytes,
        "planned_weight_bytes": planned_bytes,
        "planned_weight_gb": round(planned_bytes / 1024**3, 4),
        "memory_feasible": planned_bytes <= memory_budget_bytes,
        "quality_penalty": round(sum(item.quality_penalty for item in assignments), 8),
        "conversion_required": any(item.conversion_required for item in assignments),
        "method": "sensitivity-weighted constrained precision descent",
    }
    return assignments, summary


def estimate_placement(
    model: ModelIR,
    node: IRNode,
    device: DeviceProfile,
    assignment: PrecisionAssignment,
    *,
    phase: str,
    token_count: int,
    power_mode: str,
) -> NodePlacement | None:
    if node.op.value not in device.supported_ops:
        return None
    if assignment.precision not in device.precisions and not (
        assignment.precision.startswith("q") and f"int{int(PRECISION_BITS[assignment.precision])}" in device.precisions
    ):
        return None
    backend, executable, reason = backend_for(model.source_format, device, node)
    efficiency = _efficiency(device.kind, phase, node.op)
    power_scale = _power_scale(power_mode, device.kind)
    workload_scale = max(1, token_count) if phase == "prefill" else 1
    if phase == "prefill" and node.op in {OpCode.MATMUL, OpCode.ATTENTION, OpCode.QUANTIZED_LINEAR, OpCode.MIXTURE_OF_EXPERTS}:
        workload_scale *= 0.58
    operations = max(1.0, node.operations) * workload_scale
    throughput = max(1e8, device.compute_tops * 1e12 * efficiency * power_scale)
    compute_ms = operations / throughput * 1000.0
    read_bytes = max(0, assignment.storage_bytes) + max(0, node.activation_bytes) * workload_scale
    bandwidth = max(1.0, device.memory_bandwidth_gbps) * 1e9 * max(0.2, power_scale)
    memory_ms = read_bytes / bandwidth * 1000.0
    dispatch_ms = {"cpu": 0.012, "gpu": 0.055, "npu": 0.14}.get(device.kind, 0.1)
    if phase == "prefill":
        dispatch_ms *= max(1.0, math.log2(max(2, token_count)) / 5.0)
    estimated = compute_ms + memory_ms + dispatch_ms
    return NodePlacement(
        node_id=node.id,
        node_name=node.name,
        op=node.op.value,
        device_id=device.id,
        device_name=device.name,
        backend=backend,
        precision=assignment.precision,
        compute_ms=round(compute_ms, 6),
        memory_ms=round(memory_ms, 6),
        transfer_ms=0.0,
        dispatch_ms=round(dispatch_ms, 6),
        estimated_ms=round(estimated, 6),
        executable=executable and not assignment.conversion_required,
        reason=reason if not executable else ("offline calibrated conversion required" if assignment.conversion_required else ""),
    )


def backend_for(source_format: str, device: DeviceProfile, node: IRNode) -> tuple[str, bool, str]:
    backends = set(device.backends)
    if node.op in {OpCode.TOKENIZATION, OpCode.SAMPLING}:
        return ("host-cpu", device.kind == "cpu", "tokenization and sampling stay on the CPU")
    if source_format == "gguf":
        if device.kind == "cpu" and "llama-cpu" in backends:
            return "llama-cpu", True, ""
        accelerated = next((value for value in backends if value.startswith("llama-") and value != "llama-cpu"), "")
        if accelerated:
            return accelerated, True, ""
        return "planning-only", False, "GGUF needs a compatible llama.cpp backend on this device"
    if source_format in {"onnx", "openvino"}:
        if "openvino" in backends:
            return "openvino", True, ""
        if device.kind == "cpu" and "onnxruntime" in backends:
            return "onnxruntime", True, ""
        return "planning-only", False, "ONNX/OpenVINO model is not exposed by a compatible backend"
    return "planning-only", False, "SafeTensors needs conversion to GGUF, ONNX, or OpenVINO IR"


def transition_cost_ms(left: DeviceProfile, right: DeviceProfile, size_bytes: int, *, phase: str) -> float:
    if left.id == right.id:
        return 0.0
    shared = left.unified_memory and right.unified_memory
    effective_gbps = min(left.memory_bandwidth_gbps, right.memory_bandwidth_gbps)
    effective_gbps *= 0.48 if shared else 0.18
    copy_ms = max(0, size_bytes) / max(1.0, effective_gbps * 1e9) * 1000.0
    synchronization = 0.08 if shared else 0.35
    if "npu" in {left.kind, right.kind}:
        synchronization += 0.18
    if phase == "decode":
        synchronization *= 1.35
    return round(copy_ms + synchronization, 6)


def build_memory_plan(
    model: ModelIR,
    devices: tuple[DeviceProfile, ...],
    planned_weight_bytes: int,
    *,
    context_tokens: int,
    budget_bytes: int,
    preferred_device: str,
) -> dict[str, Any]:
    hidden = max(1, model.embedding_length or 4096)
    layers = max(1, model.block_count)
    kv_bytes = context_tokens * hidden * layers * 2 * 1
    preferred = next((item for item in devices if item.id == preferred_device), devices[0])
    device_capacity = int(max(0.0, preferred.memory_gb) * 1024**3 * 0.80)
    hot_weights = min(planned_weight_bytes, device_capacity, budget_bytes)
    warm_weights = min(max(0, planned_weight_bytes - hot_weights), max(0, budget_bytes - hot_weights - kv_bytes))
    cold_weights = max(0, planned_weight_bytes - hot_weights - warm_weights)
    dense_repeated_io = cold_weights > 0 and model.expert_count == 0
    return {
        "strategy": "virtual-tiered-model-memory-v1",
        "preferred_device": preferred.id,
        "weight_tiers": {
            "accelerator_or_hot_ram_bytes": hot_weights,
            "shared_ram_bytes": warm_weights,
            "storage_backed_bytes": cold_weights,
        },
        "kv_cache": {
            "strategy": "paged-prefix-copy-on-write",
            "precision": "int8",
            "estimated_bytes": kv_bytes,
            "context_tokens": context_tokens,
            "block_tokens": 16,
        },
        "active_parameter_count": model.active_parameter_count,
        "moe_expert_tiering": model.expert_count > 0,
        "dense_storage_warning": dense_repeated_io,
        "executable_without_repeated_storage_io": not dense_repeated_io,
    }


def _supported_precisions(node: IRNode, devices: Iterable[DeviceProfile]) -> tuple[str, ...]:
    device_values = {value for device in devices for value in device.precisions}
    values = tuple(value for value in node.supported_precisions if value in PRECISION_BITS and value in device_values)
    if values:
        return values
    fallback = tuple(value for value in node.supported_precisions if value in PRECISION_BITS)
    return fallback or ("fp32",)


def _node_storage_bytes(node: IRNode, precision: str, source_bits: float) -> int:
    if node.parameter_bytes <= 0:
        return 0
    return max(1, int(node.parameter_bytes * PRECISION_BITS[precision] / max(1.0, source_bits)))


def _quality_penalty(node: IRNode, previous: str, candidate: str, total_bytes: int) -> float:
    bit_drop = max(0.0, PRECISION_BITS[previous] - PRECISION_BITS[candidate]) / 16.0
    weight = max(1, node.parameter_bytes) / max(1, total_bytes)
    low_bit_curve = 1.0 + max(0.0, 4.0 - PRECISION_BITS[candidate]) * 0.75
    boundary = 1.35 if node.sensitivity >= 0.85 else 1.0
    return bit_drop * max(0.05, node.sensitivity) * weight * low_bit_curve * boundary


def _native_precision(bits: float) -> str:
    if bits <= 2.5:
        return "int2"
    if bits <= 3.5:
        return "int3"
    if bits <= 6.0:
        return "int4"
    if bits <= 10.0:
        return "int8"
    if bits <= 18.0:
        return "fp16"
    return "fp32"


def _efficiency(kind: str, phase: str, op: OpCode) -> float:
    base = {
        ("cpu", "prefill"): 0.22,
        ("cpu", "decode"): 0.12,
        ("gpu", "prefill"): 0.58,
        ("gpu", "decode"): 0.34,
        ("npu", "prefill"): 0.62,
        ("npu", "decode"): 0.27,
    }.get((kind, phase), 0.15)
    if op in {OpCode.TOKENIZATION, OpCode.SAMPLING, OpCode.RESIDUAL_ADD}:
        return 0.45 if kind == "cpu" else base * 0.25
    if op in {OpCode.MATMUL, OpCode.QUANTIZED_LINEAR, OpCode.MIXTURE_OF_EXPERTS, OpCode.ATTENTION}:
        return base
    return base * 0.7


def _power_scale(mode: str, kind: str) -> float:
    values = {
        "maximum": {"cpu": 1.12, "gpu": 1.12, "npu": 1.05},
        "balanced": {"cpu": 1.0, "gpu": 1.0, "npu": 1.0},
        "quiet": {"cpu": 0.62, "gpu": 0.70, "npu": 0.90},
        "battery": {"cpu": 0.48, "gpu": 0.45, "npu": 0.88},
        "background": {"cpu": 0.38, "gpu": 0.42, "npu": 0.72},
    }
    return values[mode].get(kind, 0.6)


def _power_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"max": "maximum", "performance": "maximum", "battery-saver": "battery"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"maximum", "balanced", "quiet", "battery", "background"}:
        raise CompilationError("power_mode must be maximum, balanced, quiet, battery, or background.")
    return normalized


def _device_by_id(devices: tuple[DeviceProfile, ...], device_id: str) -> DeviceProfile:
    return next(device for device in devices if device.id == device_id)


def _affinity_cost(node_id: str, device_id: str, affinity: dict[str, str], local_cost: float) -> float:
    preferred = affinity.get(node_id)
    if not preferred or preferred == device_id:
        return 0.0
    return max(0.03, local_cost * 0.06)


def _regions(placements: tuple[NodePlacement, ...]) -> tuple[dict[str, Any], ...]:
    if not placements:
        return ()
    result: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(placements) + 1):
        if index == len(placements) or (
            placements[index].device_id != placements[start].device_id
            or placements[index].backend != placements[start].backend
        ):
            segment = placements[start:index]
            result.append(
                {
                    "device_id": segment[0].device_id,
                    "device_name": segment[0].device_name,
                    "backend": segment[0].backend,
                    "start_node": segment[0].node_id,
                    "end_node": segment[-1].node_id,
                    "node_count": len(segment),
                    "estimated_ms": round(sum(item.estimated_ms for item in segment), 4),
                }
            )
            start = index
    return tuple(result)


def _dominant_device(placements: tuple[NodePlacement, ...]) -> str:
    totals: dict[str, float] = {}
    for item in placements:
        totals[item.device_id] = totals.get(item.device_id, 0.0) + item.estimated_ms
    return max(totals, key=totals.get) if totals else "cpu:0"
