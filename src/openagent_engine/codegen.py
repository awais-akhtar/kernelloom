"""Kernel-level IR and vendor-backend lowering candidates.

The engine owns candidate selection and validation policy. OpenVINO, llama.cpp,
and their device compilers own final machine code generation; this avoids
shipping an unverified shader generator as a correctness-critical component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

from .hardware import DeviceProfile
from .ir import IRLevel, IRNode, OpCode


@dataclass(frozen=True)
class KernelConfiguration:
    id: str
    backend: str
    device_id: str
    phase: str
    op: str
    input_precision: str
    accumulator_precision: str
    tile_m: int
    tile_n: int
    tile_k: int
    subgroup_size: int
    vector_width: int
    pipeline_stages: int
    cooperative_matrix: bool
    fused_operations: tuple[str, ...]
    state: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fused_operations"] = list(self.fused_operations)
        return payload


class KernelLowerer:
    """Lower placed tensor/graph operations to bounded kernel candidates."""

    def lower(
        self,
        placements: Iterable[Any],
        devices: Iterable[DeviceProfile],
        *,
        phase: str,
    ) -> tuple[IRNode, ...]:
        device_map = {device.id: device for device in devices}
        emitted: dict[tuple[str, str, str], IRNode] = {}
        for placement in placements:
            device = device_map.get(str(placement.device_id))
            if not device:
                continue
            key = (str(placement.device_id), str(placement.op), str(placement.precision))
            if key in emitted:
                continue
            candidates = kernel_candidates(
                device,
                backend=str(placement.backend),
                phase=phase,
                op=str(placement.op),
                precision=str(placement.precision),
            )
            identity = hashlib.sha256(json.dumps([key, phase], sort_keys=True).encode("utf-8")).hexdigest()[:20]
            emitted[key] = IRNode(
                id=f"kernel.{identity}",
                op=_op_code(str(placement.op)),
                name=f"{phase} {placement.op} on {device.name}",
                level=IRLevel.KERNEL,
                attributes={
                    "device_id": device.id,
                    "backend": placement.backend,
                    "phase": phase,
                    "candidate_count": len(candidates),
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "selection_gate": "benchmark plus numerical reference validation",
                },
                supported_precisions=(str(placement.precision),),
            )
        return tuple(emitted.values())


def kernel_candidates(
    device: DeviceProfile,
    *,
    backend: str,
    phase: str,
    op: str,
    precision: str,
) -> tuple[KernelConfiguration, ...]:
    if op in {"tokenization", "sampling", "tensor_load", "unknown"}:
        return (_configuration(device, backend, phase, op, precision, 1, 1, 1, 1, 1, 1, False, ()),)
    if device.kind == "npu":
        return (
            _configuration(
                device,
                backend,
                phase,
                op,
                precision,
                0,
                0,
                0,
                0,
                0,
                0,
                False,
                ("vendor-graph-fusion",),
                state="vendor-compiled-candidate",
            ),
        )
    if device.kind == "gpu":
        tile_values = (16, 32) if phase == "prefill" else (8, 16)
        subgroups = (16, 32)
        vectors = (2, 4)
    else:
        tile_values = (8, 16)
        subgroups = (1,)
        vectors = (4, 8)
    configurations: list[KernelConfiguration] = []
    for tile in tile_values:
        for subgroup in subgroups:
            for vector in vectors:
                fused = ("bias", "activation") if op in {"matmul", "quantized_linear", "mixture_of_experts"} else ()
                configurations.append(
                    _configuration(
                        device,
                        backend,
                        phase,
                        op,
                        precision,
                        tile,
                        tile,
                        32 if phase == "prefill" else 16,
                        subgroup,
                        vector,
                        2 if phase == "prefill" else 1,
                        device.kind == "gpu" and "openvino" in backend,
                        fused,
                    )
                )
    return tuple(configurations[:8])


def _configuration(
    device: DeviceProfile,
    backend: str,
    phase: str,
    op: str,
    precision: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    subgroup: int,
    vector: int,
    stages: int,
    cooperative: bool,
    fused: tuple[str, ...],
    *,
    state: str = "candidate",
) -> KernelConfiguration:
    accumulator = "int32" if precision in {"int8", "int4", "int3", "int2"} else "fp32"
    values = {
        "backend": backend,
        "device": device.id,
        "phase": phase,
        "op": op,
        "precision": precision,
        "accumulator": accumulator,
        "tiles": (tile_m, tile_n, tile_k),
        "subgroup": subgroup,
        "vector": vector,
        "stages": stages,
        "cooperative": cooperative,
        "fused": fused,
    }
    identity = hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return KernelConfiguration(
        id=identity,
        backend=backend,
        device_id=device.id,
        phase=phase,
        op=op,
        input_precision=precision,
        accumulator_precision=accumulator,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        subgroup_size=subgroup,
        vector_width=vector,
        pipeline_stages=stages,
        cooperative_matrix=cooperative,
        fused_operations=fused,
        state=state,
    )


def _op_code(value: str) -> OpCode:
    try:
        return OpCode(value)
    except ValueError:
        return OpCode.UNKNOWN
