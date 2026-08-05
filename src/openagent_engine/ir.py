"""Hardware-independent intermediate representation for local AI models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class IRLevel(StrEnum):
    GRAPH = "graph"
    TENSOR = "tensor"
    KERNEL = "kernel"


class OpCode(StrEnum):
    EMBEDDING = "embedding"
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"
    QUANTIZED_LINEAR = "quantized_linear"
    MATMUL = "matmul"
    ATTENTION = "attention"
    ROPE = "rope"
    SOFTMAX = "softmax"
    ACTIVATION = "activation"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    RESIDUAL_ADD = "residual_add"
    CONV2D = "conv2d"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    DEQUANTIZE = "dequantize"
    SAMPLING = "sampling"
    TOKENIZATION = "tokenization"
    TENSOR_LOAD = "tensor_load"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IRValue:
    id: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = "row-major"
    quantization: dict[str, Any] = field(default_factory=dict)
    alignment: int = 32

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    offset: int = 0
    shard: str = ""
    quantization: dict[str, Any] = field(default_factory=dict)

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= max(0, int(dimension))
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class IRNode:
    id: str
    op: OpCode
    name: str
    level: IRLevel = IRLevel.GRAPH
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    parameter_bytes: int = 0
    operations: float = 0.0
    activation_bytes: int = 0
    sensitivity: float = 0.5
    stateful: bool = False
    supported_precisions: tuple[str, ...] = ("fp16", "int8", "int4")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["op"] = self.op.value
        payload["level"] = self.level.value
        payload["inputs"] = list(self.inputs)
        payload["outputs"] = list(self.outputs)
        payload["supported_precisions"] = list(self.supported_precisions)
        return payload


@dataclass(frozen=True)
class ModelIR:
    id: str
    name: str
    source_path: str
    source_format: str
    fingerprint: str
    architecture: str
    parameter_count: int
    weight_bytes: int
    quantization_bits: float
    context_length: int
    embedding_length: int
    block_count: int
    expert_count: int = 0
    experts_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: tuple[TensorSpec, ...] = ()
    graph_nodes: tuple[IRNode, ...] = ()
    tensor_nodes: tuple[IRNode, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def active_parameter_count(self) -> int:
        if self.expert_count > 0 and self.experts_used > 0:
            shared_fraction = 0.15
            expert_fraction = 1.0 - shared_fraction
            ratio = min(1.0, self.experts_used / self.expert_count)
            return int(self.parameter_count * (shared_fraction + expert_fraction * ratio))
        return self.parameter_count

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_nodes: bool = True,
        max_items: int = 500,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "fingerprint": self.fingerprint,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "active_parameter_count": self.active_parameter_count,
            "weight_bytes": self.weight_bytes,
            "quantization_bits": self.quantization_bits,
            "context_length": self.context_length,
            "embedding_length": self.embedding_length,
            "block_count": self.block_count,
            "expert_count": self.expert_count,
            "experts_used": self.experts_used,
            "metadata": self.metadata,
            "tensor_count": len(self.tensors),
            "graph_node_count": len(self.graph_nodes),
            "tensor_node_count": len(self.tensor_nodes),
            "warnings": list(self.warnings),
        }
        if include_nodes:
            result["graph_nodes"] = [node.to_dict() for node in self.graph_nodes[:max_items]]
            result["tensor_nodes"] = [node.to_dict() for node in self.tensor_nodes[:max_items]]
            result["nodes_truncated"] = len(self.graph_nodes) > max_items or len(self.tensor_nodes) > max_items
        if include_tensors:
            result["tensors"] = [tensor.to_dict() for tensor in self.tensors[:max_items]]
            result["tensors_truncated"] = len(self.tensors) > max_items
        return result


def build_transformer_graph(
    *,
    block_count: int,
    weight_bytes: int,
    parameter_count: int,
    embedding_length: int,
    expert_count: int = 0,
    experts_used: int = 0,
) -> tuple[IRNode, ...]:
    """Create a compact high-level graph from architecture metadata."""

    blocks = max(1, min(int(block_count or 1), 10000))
    embedding_bytes = int(weight_bytes * 0.025)
    output_bytes = int(weight_bytes * 0.025)
    block_bytes = max(0, weight_bytes - embedding_bytes - output_bytes) / blocks
    block_parameters = max(1, int(parameter_count * 0.95 / blocks))
    activation_bytes = max(4096, int(max(1, embedding_length) * 2))
    nodes: list[IRNode] = [
        IRNode(
            id="graph.tokenize",
            op=OpCode.TOKENIZATION,
            name="Tokenizer",
            parameter_bytes=0,
            operations=1_000_000,
            activation_bytes=activation_bytes,
            sensitivity=1.0,
            supported_precisions=("fp32",),
        ),
        IRNode(
            id="graph.embedding",
            op=OpCode.EMBEDDING,
            name="Token embedding",
            parameter_bytes=embedding_bytes,
            operations=max(1.0, embedding_bytes / 2),
            activation_bytes=activation_bytes,
            sensitivity=0.85,
            supported_precisions=("fp16", "int8"),
        ),
    ]
    for index in range(blocks):
        prefix = f"graph.block.{index}"
        nodes.extend(
            [
                IRNode(
                    id=f"{prefix}.norm",
                    op=OpCode.RMS_NORM,
                    name=f"Block {index} normalization",
                    operations=max(1.0, embedding_length * 8),
                    activation_bytes=activation_bytes,
                    sensitivity=0.9 if index in {0, blocks - 1} else 0.65,
                    supported_precisions=("fp16", "fp32"),
                ),
                IRNode(
                    id=f"{prefix}.attention",
                    op=OpCode.ATTENTION,
                    name=f"Block {index} attention",
                    parameter_bytes=int(block_bytes * 0.35),
                    operations=float(block_parameters * 0.7),
                    activation_bytes=activation_bytes * 4,
                    sensitivity=0.9 if index in {0, blocks - 1} else 0.72,
                    stateful=True,
                    supported_precisions=("fp16", "int8", "int4"),
                    attributes={"block": index, "uses_kv_cache": True},
                ),
                IRNode(
                    id=f"{prefix}.ffn",
                    op=OpCode.MIXTURE_OF_EXPERTS if expert_count else OpCode.QUANTIZED_LINEAR,
                    name=f"Block {index} {'experts' if expert_count else 'feed-forward'}",
                    parameter_bytes=int(block_bytes * 0.65),
                    operations=float(block_parameters * 1.3),
                    activation_bytes=activation_bytes * 3,
                    sensitivity=0.82 if index in {0, blocks - 1} else 0.55,
                    supported_precisions=("fp16", "int8", "int4", "int3"),
                    attributes={
                        "block": index,
                        "expert_count": expert_count,
                        "experts_used": experts_used,
                    },
                ),
                IRNode(
                    id=f"{prefix}.residual",
                    op=OpCode.RESIDUAL_ADD,
                    name=f"Block {index} residual",
                    operations=max(1.0, embedding_length),
                    activation_bytes=activation_bytes,
                    sensitivity=1.0,
                    supported_precisions=("fp16", "fp32"),
                ),
            ]
        )
    nodes.extend(
        [
            IRNode(
                id="graph.output_norm",
                op=OpCode.RMS_NORM,
                name="Output normalization",
                operations=max(1.0, embedding_length * 8),
                activation_bytes=activation_bytes,
                sensitivity=1.0,
                supported_precisions=("fp16", "fp32"),
            ),
            IRNode(
                id="graph.lm_head",
                op=OpCode.QUANTIZED_LINEAR,
                name="Language model head",
                parameter_bytes=output_bytes,
                operations=max(1.0, output_bytes / 2),
                activation_bytes=activation_bytes,
                sensitivity=0.95,
                supported_precisions=("fp16", "int8"),
            ),
            IRNode(
                id="graph.sampling",
                op=OpCode.SAMPLING,
                name="Token sampling",
                operations=2_000_000,
                activation_bytes=4096,
                sensitivity=1.0,
                supported_precisions=("fp32",),
            ),
        ]
    )
    return tuple(nodes)


def lower_graph_to_tensor_ir(graph_nodes: tuple[IRNode, ...]) -> tuple[IRNode, ...]:
    """Lower high-level transformer regions to explicit tensor operations."""

    result: list[IRNode] = []
    for node in graph_nodes:
        if node.op == OpCode.ATTENTION:
            shares = (0.34, 0.33, 0.33)
            for suffix, share in zip(("qkv", "rope", "softmax"), shares, strict=True):
                op = OpCode.MATMUL if suffix == "qkv" else OpCode.ROPE if suffix == "rope" else OpCode.SOFTMAX
                result.append(
                    IRNode(
                        id=f"tensor.{node.id}.{suffix}",
                        op=op,
                        name=f"{node.name} {suffix}",
                        level=IRLevel.TENSOR,
                        parameter_bytes=int(node.parameter_bytes * share) if op == OpCode.MATMUL else 0,
                        operations=node.operations * share,
                        activation_bytes=node.activation_bytes,
                        sensitivity=node.sensitivity,
                        stateful=node.stateful,
                        supported_precisions=node.supported_precisions,
                        attributes=dict(node.attributes),
                    )
                )
        elif node.op in {OpCode.QUANTIZED_LINEAR, OpCode.MIXTURE_OF_EXPERTS, OpCode.EMBEDDING}:
            result.append(
                IRNode(
                    id=f"tensor.{node.id}.matmul",
                    op=OpCode.MATMUL,
                    name=f"{node.name} matrix operation",
                    level=IRLevel.TENSOR,
                    parameter_bytes=node.parameter_bytes,
                    operations=node.operations,
                    activation_bytes=node.activation_bytes,
                    sensitivity=node.sensitivity,
                    stateful=node.stateful,
                    supported_precisions=node.supported_precisions,
                    attributes={**node.attributes, "source_op": node.op.value},
                )
            )
        else:
            result.append(
                IRNode(
                    id=f"tensor.{node.id}",
                    op=node.op,
                    name=node.name,
                    level=IRLevel.TENSOR,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    attributes=dict(node.attributes),
                    parameter_bytes=node.parameter_bytes,
                    operations=node.operations,
                    activation_bytes=node.activation_bytes,
                    sensitivity=node.sensitivity,
                    stateful=node.stateful,
                    supported_precisions=node.supported_precisions,
                )
            )
    return tuple(result)


def opcode_for_tensor_name(name: str) -> OpCode:
    lowered = name.lower()
    if "token_embd" in lowered or "embed_tokens" in lowered or "embedding" in lowered:
        return OpCode.EMBEDDING
    if "attn_norm" in lowered or "rms_norm" in lowered or "rmsnorm" in lowered:
        return OpCode.RMS_NORM
    if "layer_norm" in lowered or "layernorm" in lowered:
        return OpCode.LAYER_NORM
    if any(part in lowered for part in ("attn_q", "attn_k", "attn_v", "attn_qkv", "self_attn")):
        return OpCode.ATTENTION
    if "rope" in lowered or "rotary" in lowered:
        return OpCode.ROPE
    if "expert" in lowered or "moe" in lowered:
        return OpCode.MIXTURE_OF_EXPERTS
    if any(part in lowered for part in ("ffn", "mlp", "proj", "output.weight", "lm_head")):
        return OpCode.QUANTIZED_LINEAR
    if "conv" in lowered:
        return OpCode.CONV2D
    if "norm" in lowered:
        return OpCode.LAYER_NORM
    return OpCode.TENSOR_LOAD
