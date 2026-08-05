"""Safe local model frontends for GGUF, SafeTensors, ONNX, and OpenVINO IR."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, BinaryIO
from xml.etree import ElementTree

from .ir import (
    IRLevel,
    IRNode,
    ModelIR,
    OpCode,
    TensorSpec,
    build_transformer_graph,
    lower_graph_to_tensor_ir,
    opcode_for_tensor_name,
)


MAX_METADATA_ITEMS = 2_000_000
MAX_TENSORS = 1_000_000
MAX_STRING_BYTES = 256 * 1024 * 1024
MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
MAX_OPENVINO_XML = 128 * 1024 * 1024
SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)

GGUF_VALUE_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}

GGML_TYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
}

GGML_TYPE_BITS = {
    0: 32.0,
    1: 16.0,
    2: 4.5,
    3: 5.0,
    6: 5.5,
    7: 6.0,
    8: 8.5,
    9: 9.0,
    10: 2.625,
    11: 3.4375,
    12: 4.5,
    13: 5.5,
    14: 6.5625,
    15: 8.5,
    16: 2.0625,
    17: 2.3125,
    18: 3.0625,
    19: 1.5625,
    20: 4.5,
    21: 3.4375,
    22: 2.625,
    23: 4.25,
    24: 8.0,
    25: 16.0,
    26: 32.0,
    27: 64.0,
    28: 64.0,
    29: 1.75,
    30: 16.0,
    34: 1.6875,
    35: 2.0625,
    39: 4.0,
}


class ModelFormatError(ValueError):
    """Raised when a model container is malformed or unsupported."""


class ModelFrontend:
    """Normalize supported local model containers into one immutable IR."""

    def inspect(
        self,
        path: str | Path,
        *,
        parameter_count_hint: int = 0,
        quantization_bits_hint: float = 0.0,
    ) -> ModelIR:
        resolved = resolve_model_path(path)
        suffix = resolved.suffix.lower()
        if suffix == ".gguf":
            return inspect_gguf(resolved, parameter_count_hint=parameter_count_hint, quantization_bits_hint=quantization_bits_hint)
        if suffix == ".safetensors":
            return inspect_safetensors(resolved, parameter_count_hint=parameter_count_hint)
        if suffix == ".onnx":
            return inspect_onnx(resolved, parameter_count_hint=parameter_count_hint)
        if suffix == ".xml":
            return inspect_openvino_ir(resolved, parameter_count_hint=parameter_count_hint)
        raise ModelFormatError("Supported model inputs are GGUF, SafeTensors, ONNX, and OpenVINO XML IR.")


class _GGUFReader:
    def __init__(self, handle: BinaryIO, *, file_size: int) -> None:
        self.handle = handle
        self.file_size = file_size

    def tell(self) -> int:
        return int(self.handle.tell())

    def read_exact(self, size: int) -> bytes:
        if size < 0 or self.tell() + size > self.file_size:
            raise ModelFormatError("GGUF field extends beyond the file boundary.")
        value = self.handle.read(size)
        if len(value) != size:
            raise ModelFormatError("GGUF file ended unexpectedly.")
        return value

    def skip(self, size: int) -> None:
        if size < 0 or self.tell() + size > self.file_size:
            raise ModelFormatError("GGUF field extends beyond the file boundary.")
        self.handle.seek(size, 1)

    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read_exact(size))[0]

    def string(self, *, capture: bool = True) -> str:
        length = int(self.unpack("Q"))
        if length > MAX_STRING_BYTES:
            raise ModelFormatError("GGUF string exceeds the safe parser limit.")
        if not capture:
            self.skip(length)
            return ""
        return self.read_exact(length).decode("utf-8", errors="replace")

    def value(self, value_type: int, *, capture: bool, depth: int = 0) -> Any:
        if depth > 4:
            raise ModelFormatError("GGUF metadata arrays are nested too deeply.")
        if value_type in GGUF_VALUE_TYPES:
            fmt, _size = GGUF_VALUE_TYPES[value_type]
            return self.unpack(fmt)
        if value_type == 8:
            return self.string(capture=capture)
        if value_type == 9:
            element_type = int(self.unpack("I"))
            length = int(self.unpack("Q"))
            if length > MAX_METADATA_ITEMS:
                raise ModelFormatError("GGUF metadata array exceeds the safe parser limit.")
            values: list[Any] = []
            for index in range(length):
                keep = capture and index < 256
                item = self.value(element_type, capture=keep, depth=depth + 1)
                if keep:
                    values.append(item)
            return values if capture else None
        raise ModelFormatError(f"Unknown GGUF metadata value type {value_type}.")


def inspect_gguf(
    first_path: Path,
    *,
    parameter_count_hint: int = 0,
    quantization_bits_hint: float = 0.0,
) -> ModelIR:
    shards = discover_shards(first_path)
    all_tensors: list[TensorSpec] = []
    metadata: dict[str, Any] = {}
    versions: set[int] = set()
    for shard_index, shard in enumerate(shards):
        shard_metadata, tensors, version = _read_gguf_shard(shard, capture_metadata=shard_index == 0)
        versions.add(version)
        if shard_index == 0:
            metadata.update(shard_metadata)
        all_tensors.extend(tensors)
    if len(versions) != 1:
        raise ModelFormatError("GGUF shards use different format versions.")
    tensor_parameter_count = sum(tensor.elements for tensor in all_tensors)
    parameter_count = int(parameter_count_hint or tensor_parameter_count)
    weight_bytes = sum(path.stat().st_size for path in shards)
    tensor_bits = [GGML_TYPE_BITS.get(int(tensor.quantization.get("ggml_type", -1)), 0.0) for tensor in all_tensors]
    tensor_bits = [value for value in tensor_bits if value > 0]
    quantization_bits = float(quantization_bits_hint or (sum(tensor_bits) / len(tensor_bits) if tensor_bits else 0.0))
    if parameter_count <= 0:
        quantization_bits = quantization_bits or 4.0
        parameter_count = max(1, int(weight_bytes * 8 / quantization_bits))
    architecture = str(metadata.get("general.architecture", "unknown")).lower()
    block_count = _metadata_int(metadata, f"{architecture}.block_count", "llama.block_count", default=1)
    embedding_length = _metadata_int(metadata, f"{architecture}.embedding_length", "llama.embedding_length", default=4096)
    context_length = _metadata_int(metadata, f"{architecture}.context_length", "llama.context_length", default=4096)
    expert_count = _metadata_int(metadata, f"{architecture}.expert_count", default=0)
    experts_used = _metadata_int(metadata, f"{architecture}.expert_used_count", default=0)
    graph = build_transformer_graph(
        block_count=block_count,
        weight_bytes=weight_bytes,
        parameter_count=parameter_count,
        embedding_length=embedding_length,
        expert_count=expert_count,
        experts_used=experts_used,
    )
    name = str(metadata.get("general.name") or metadata.get("general.basename") or first_path.stem)
    fingerprint = model_fingerprint(shards)
    warnings: list[str] = []
    if parameter_count_hint and tensor_parameter_count and abs(parameter_count_hint - tensor_parameter_count) / max(1, tensor_parameter_count) > 0.1:
        warnings.append("Parameter hint differs from tensor-shape count by more than ten percent.")
    if architecture == "unknown":
        warnings.append("GGUF architecture metadata is missing; graph structure is conservative.")
    return ModelIR(
        id=fingerprint[:24],
        name=name,
        source_path=str(first_path),
        source_format="gguf",
        fingerprint=fingerprint,
        architecture=architecture,
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        quantization_bits=round(quantization_bits or 4.0, 4),
        context_length=context_length,
        embedding_length=embedding_length,
        block_count=block_count,
        expert_count=expert_count,
        experts_used=experts_used,
        metadata={
            **metadata,
            "gguf.version": next(iter(versions)),
            "gguf.shard_count": len(shards),
            "gguf.shards": [str(path) for path in shards],
        },
        tensors=tuple(all_tensors),
        graph_nodes=graph,
        tensor_nodes=lower_graph_to_tensor_ir(graph),
        warnings=tuple(warnings),
    )


def _read_gguf_shard(path: Path, *, capture_metadata: bool) -> tuple[dict[str, Any], list[TensorSpec], int]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        reader = _GGUFReader(handle, file_size=file_size)
        if reader.read_exact(4) != b"GGUF":
            raise ModelFormatError(f"Invalid GGUF magic in {path}.")
        version = int(reader.unpack("I"))
        if version not in {2, 3}:
            raise ModelFormatError(f"Unsupported GGUF version {version}; expected 2 or 3.")
        tensor_count = int(reader.unpack("Q"))
        metadata_count = int(reader.unpack("Q"))
        if tensor_count > MAX_TENSORS or metadata_count > MAX_METADATA_ITEMS:
            raise ModelFormatError("GGUF header count exceeds the safe parser limit.")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string(capture=True)
            value_type = int(reader.unpack("I"))
            capture = capture_metadata and _interesting_gguf_key(key)
            value = reader.value(value_type, capture=capture)
            if capture:
                metadata[key] = value
        raw_tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name = reader.string(capture=True)
            dimensions_count = int(reader.unpack("I"))
            if dimensions_count < 1 or dimensions_count > 8:
                raise ModelFormatError("GGUF tensor dimension count is invalid.")
            shape = tuple(int(reader.unpack("Q")) for _ in range(dimensions_count))
            tensor_type = int(reader.unpack("I"))
            offset = int(reader.unpack("Q"))
            raw_tensors.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset})
        alignment = int(metadata.get("general.alignment", 32)) if capture_metadata else 32
        alignment = alignment if alignment > 0 and alignment % 8 == 0 else 32
        data_start = _align(reader.tell(), alignment)
        sorted_offsets = sorted((item["offset"], index) for index, item in enumerate(raw_tensors))
        sizes: dict[int, int] = {}
        for position, (offset, index) in enumerate(sorted_offsets):
            next_offset = sorted_offsets[position + 1][0] if position + 1 < len(sorted_offsets) else max(offset, file_size - data_start)
            sizes[index] = max(0, next_offset - offset)
        tensors = [
            TensorSpec(
                name=item["name"],
                dtype=GGML_TYPE_NAMES.get(item["type"], f"ggml_type_{item['type']}"),
                shape=item["shape"],
                size_bytes=sizes.get(index, 0),
                offset=data_start + item["offset"],
                shard=str(path),
                quantization={
                    "ggml_type": item["type"],
                    "effective_bits": GGML_TYPE_BITS.get(item["type"]),
                },
            )
            for index, item in enumerate(raw_tensors)
        ]
        return metadata, tensors, version


def inspect_safetensors(path: Path, *, parameter_count_hint: int = 0) -> ModelIR:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_length_raw = handle.read(8)
        if len(header_length_raw) != 8:
            raise ModelFormatError("SafeTensors header is missing.")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        if header_length <= 1 or header_length > MAX_SAFETENSORS_HEADER or header_length + 8 > file_size:
            raise ModelFormatError("SafeTensors header length is invalid.")
        raw = handle.read(header_length)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFormatError("SafeTensors header is not valid UTF-8 JSON.") from exc
    if not isinstance(header, dict):
        raise ModelFormatError("SafeTensors header must be an object.")
    tensors: list[TensorSpec] = []
    dtype_bits: list[float] = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ModelFormatError(f"SafeTensors tensor '{name}' metadata is invalid.")
        shape = tuple(int(value) for value in item.get("shape", []))
        offsets = item.get("data_offsets", [])
        if not shape or not isinstance(offsets, list) or len(offsets) != 2:
            raise ModelFormatError(f"SafeTensors tensor '{name}' is missing shape or data offsets.")
        begin, end = int(offsets[0]), int(offsets[1])
        if begin < 0 or end < begin or 8 + header_length + end > file_size:
            raise ModelFormatError(f"SafeTensors tensor '{name}' points outside the file.")
        dtype = str(item.get("dtype", "UNKNOWN")).lower()
        bits = _dtype_bits(dtype)
        if bits:
            dtype_bits.append(bits)
        tensors.append(
            TensorSpec(
                name=str(name),
                dtype=dtype,
                shape=shape,
                size_bytes=end - begin,
                offset=8 + header_length + begin,
                shard=str(path),
                quantization={"effective_bits": bits or None},
            )
        )
    config = _read_json_if_small(path.with_name("config.json"))
    parameter_count = int(parameter_count_hint or sum(tensor.elements for tensor in tensors))
    weight_bytes = sum(tensor.size_bytes for tensor in tensors)
    bits = sum(dtype_bits) / len(dtype_bits) if dtype_bits else max(1.0, weight_bytes * 8 / max(1, parameter_count))
    architecture = _config_architecture(config)
    block_count = int(config.get("num_hidden_layers", config.get("n_layer", 1)) or 1)
    embedding_length = int(config.get("hidden_size", config.get("n_embd", 4096)) or 4096)
    context_length = int(config.get("max_position_embeddings", config.get("n_positions", 4096)) or 4096)
    expert_count = int(config.get("num_local_experts", config.get("n_routed_experts", 0)) or 0)
    experts_used = int(config.get("num_experts_per_tok", config.get("num_experts_per_token", 0)) or 0)
    graph = build_transformer_graph(
        block_count=block_count,
        weight_bytes=weight_bytes,
        parameter_count=parameter_count,
        embedding_length=embedding_length,
        expert_count=expert_count,
        experts_used=experts_used,
    )
    fingerprint = model_fingerprint([path, path.with_name("config.json")])
    return ModelIR(
        id=fingerprint[:24],
        name=str(config.get("_name_or_path") or path.stem),
        source_path=str(path),
        source_format="safetensors",
        fingerprint=fingerprint,
        architecture=architecture,
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        quantization_bits=round(bits, 4),
        context_length=context_length,
        embedding_length=embedding_length,
        block_count=block_count,
        expert_count=expert_count,
        experts_used=experts_used,
        metadata={"config": config, "safe_serialization": True},
        tensors=tuple(tensors),
        graph_nodes=graph,
        tensor_nodes=lower_graph_to_tensor_ir(graph),
    )


def inspect_onnx(path: Path, *, parameter_count_hint: int = 0) -> ModelIR:
    nodes: list[IRNode] = []
    metadata: dict[str, Any] = {"opaque_graph": True}
    architecture = "onnx"
    parameter_count = int(parameter_count_hint)
    try:
        import onnx  # type: ignore[import-not-found]

        model = onnx.load(str(path), load_external_data=False)
        metadata["ir_version"] = int(model.ir_version)
        metadata["opset"] = [int(item.version) for item in model.opset_import]
        metadata["opaque_graph"] = False
        for index, item in enumerate(model.graph.node):
            op = _opcode_for_framework_op(str(item.op_type))
            nodes.append(
                IRNode(
                    id=f"onnx.{index}",
                    op=op,
                    name=str(item.name or f"{item.op_type}_{index}"),
                    level=IRLevel.GRAPH,
                    inputs=tuple(item.input),
                    outputs=tuple(item.output),
                    sensitivity=_default_sensitivity(op),
                    supported_precisions=("fp32", "fp16", "int8", "int4"),
                )
            )
        parameter_count = parameter_count or sum(
            math.prod(int(value) for value in tensor.dims) for tensor in model.graph.initializer
        )
    except ImportError:
        nodes = [
            IRNode(
                id="onnx.opaque",
                op=OpCode.UNKNOWN,
                name="ONNX graph (install onnx for detailed inspection)",
                sensitivity=1.0,
                supported_precisions=("fp32", "fp16", "int8"),
            )
        ]
    weight_bytes = path.stat().st_size
    parameter_count = parameter_count or max(1, weight_bytes // 2)
    fingerprint = model_fingerprint([path])
    return ModelIR(
        id=fingerprint[:24],
        name=path.stem,
        source_path=str(path),
        source_format="onnx",
        fingerprint=fingerprint,
        architecture=architecture,
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        quantization_bits=round(weight_bytes * 8 / max(1, parameter_count), 4),
        context_length=0,
        embedding_length=0,
        block_count=max(1, len(nodes)),
        metadata=metadata,
        graph_nodes=tuple(nodes),
        tensor_nodes=tuple(nodes),
        warnings=("Detailed ONNX graph inspection requires the optional onnx package in the host environment.",)
        if metadata["opaque_graph"]
        else (),
    )


def enrich_onnx_from_probe(model: ModelIR, probe: dict[str, Any]) -> ModelIR:
    """Replace an opaque ONNX graph with bounded metadata from the isolated probe."""

    if model.source_format != "onnx" or probe.get("status") != "ok":
        return model
    initializer_map = {
        str(item.get("name", "")): item
        for item in probe.get("initializers", [])
        if isinstance(item, dict)
    }
    nodes: list[IRNode] = []
    for index, item in enumerate(probe.get("nodes", [])):
        if not isinstance(item, dict):
            continue
        op = _opcode_for_framework_op(str(item.get("op_type", "Unknown")))
        inputs = tuple(str(value) for value in item.get("inputs", []))
        parameter_bytes = sum(int(initializer_map.get(value, {}).get("size_bytes", 0)) for value in inputs)
        parameter_elements = sum(int(initializer_map.get(value, {}).get("elements", 0)) for value in inputs)
        nodes.append(
            IRNode(
                id=f"onnx.{index}",
                op=op,
                name=str(item.get("name") or f"{item.get('op_type', 'Unknown')}_{index}"),
                level=IRLevel.GRAPH,
                inputs=inputs,
                outputs=tuple(str(value) for value in item.get("outputs", [])),
                parameter_bytes=parameter_bytes,
                operations=float(max(1_000, parameter_elements * 2)),
                activation_bytes=max(4096, int(math.sqrt(max(1, parameter_elements))) * 4),
                sensitivity=_default_sensitivity(op),
                supported_precisions=("fp32", "fp16", "int8", "int4"),
            )
        )
    parameter_count = int(probe.get("parameter_count", 0) or model.parameter_count)
    weight_bytes = int(probe.get("weight_bytes", 0) or model.weight_bytes)
    return ModelIR(
        id=model.id,
        name=model.name,
        source_path=model.source_path,
        source_format=model.source_format,
        fingerprint=model.fingerprint,
        architecture=model.architecture,
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        quantization_bits=round(weight_bytes * 8 / max(1, parameter_count), 4),
        context_length=model.context_length,
        embedding_length=model.embedding_length,
        block_count=max(1, len(nodes)),
        expert_count=model.expert_count,
        experts_used=model.experts_used,
        metadata={**model.metadata, "opaque_graph": False, "isolated_probe": {"ir_version": probe.get("ir_version"), "opset": probe.get("opset", [])}},
        tensors=model.tensors,
        graph_nodes=tuple(nodes) or model.graph_nodes,
        tensor_nodes=tuple(nodes) or model.tensor_nodes,
        warnings=tuple(warning for warning in model.warnings if "Detailed ONNX" not in warning),
    )


def inspect_openvino_ir(path: Path, *, parameter_count_hint: int = 0) -> ModelIR:
    if path.stat().st_size > MAX_OPENVINO_XML:
        raise ModelFormatError("OpenVINO XML exceeds the safe parser limit.")
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ModelFormatError("OpenVINO XML is malformed.") from exc
    nodes: list[IRNode] = []
    layers_parent = root.find("layers")
    for index, layer in enumerate(list(layers_parent) if layers_parent is not None else []):
        op_name = str(layer.attrib.get("type", "Unknown"))
        op = _opcode_for_framework_op(op_name)
        nodes.append(
            IRNode(
                id=f"openvino.{layer.attrib.get('id', index)}",
                op=op,
                name=str(layer.attrib.get("name", f"{op_name}_{index}")),
                sensitivity=_default_sensitivity(op),
                supported_precisions=("fp32", "fp16", "int8", "int4") if op in {OpCode.MATMUL, OpCode.QUANTIZED_LINEAR} else ("fp32", "fp16"),
            )
        )
    bin_path = path.with_suffix(".bin")
    weight_bytes = bin_path.stat().st_size if bin_path.is_file() else path.stat().st_size
    parameter_count = int(parameter_count_hint or max(1, weight_bytes // 2))
    fingerprint = model_fingerprint([path, bin_path])
    return ModelIR(
        id=fingerprint[:24],
        name=str(root.attrib.get("name", path.stem)),
        source_path=str(path),
        source_format="openvino",
        fingerprint=fingerprint,
        architecture="openvino-ir",
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        quantization_bits=round(weight_bytes * 8 / max(1, parameter_count), 4),
        context_length=0,
        embedding_length=0,
        block_count=max(1, len(nodes)),
        metadata={"ir_version": root.attrib.get("version", ""), "bin_path": str(bin_path) if bin_path.is_file() else ""},
        graph_nodes=tuple(nodes),
        tensor_nodes=tuple(nodes),
    )


def resolve_model_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise ModelFormatError(f"Model path does not exist: {resolved}")
    candidates: list[Path] = []
    for pattern in ("*.xml", "*.gguf", "*.safetensors", "*.onnx"):
        candidates.extend(sorted(resolved.glob(pattern)))
    if not candidates:
        raise ModelFormatError("Model directory does not contain a supported model file.")
    preferred = next((item for item in candidates if item.name in {"openvino_model.xml", "model.xml", "model.safetensors"}), None)
    return preferred or candidates[0]


def discover_shards(first_path: Path) -> list[Path]:
    match = SHARD_RE.match(first_path.name)
    if not match:
        return [first_path]
    count = int(match.group("count"))
    if count < 1 or count > 100000:
        raise ModelFormatError("GGUF shard count is invalid.")
    prefix = match.group("prefix")
    paths = [first_path.with_name(f"{prefix}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ModelFormatError(f"GGUF shard set is incomplete; first missing shard: {missing[0]}")
    return paths


def model_fingerprint(paths: list[Path], *, sample_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths if item.is_file()}, key=lambda item: str(item).lower()):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        with path.open("rb") as handle:
            digest.update(handle.read(min(sample_bytes, stat.st_size)))
            if stat.st_size > sample_bytes:
                handle.seek(max(0, stat.st_size - sample_bytes))
                digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _interesting_gguf_key(key: str) -> bool:
    if key.startswith("general."):
        return True
    suffixes = (
        ".block_count",
        ".context_length",
        ".embedding_length",
        ".feed_forward_length",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".expert_count",
        ".expert_used_count",
        ".tensor_data_layout",
        ".rope.dimension_count",
    )
    return key.endswith(suffixes) or key.startswith("split.")


def _metadata_int(metadata: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and int(value) >= 0:
            return int(value)
    return default


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_bits(dtype: str) -> float:
    normalized = dtype.lower()
    mapping = {
        "bool": 1,
        "u8": 8,
        "i8": 8,
        "f8_e4m3": 8,
        "f8_e5m2": 8,
        "u16": 16,
        "i16": 16,
        "f16": 16,
        "bf16": 16,
        "u32": 32,
        "i32": 32,
        "f32": 32,
        "u64": 64,
        "i64": 64,
        "f64": 64,
    }
    return float(mapping.get(normalized, 0))


def _read_json_if_small(path: Path, *, maximum: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > maximum:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _config_architecture(config: dict[str, Any]) -> str:
    architectures = config.get("architectures", [])
    if isinstance(architectures, list) and architectures:
        return str(architectures[0]).lower()
    return str(config.get("model_type", "transformer")).lower()


def _opcode_for_framework_op(name: str) -> OpCode:
    lowered = name.lower()
    if lowered in {"matmul", "gemm", "fullyconnected", "linear"}:
        return OpCode.MATMUL
    if "attention" in lowered or lowered in {"sdpa", "scaleddotproductattention"}:
        return OpCode.ATTENTION
    if "rms" in lowered and "norm" in lowered:
        return OpCode.RMS_NORM
    if "norm" in lowered:
        return OpCode.LAYER_NORM
    if lowered in {"softmax", "logsoftmax"}:
        return OpCode.SOFTMAX
    if lowered in {"reshape", "flatten", "squeeze", "unsqueeze"}:
        return OpCode.RESHAPE
    if lowered in {"transpose", "permute"}:
        return OpCode.TRANSPOSE
    if "conv" in lowered:
        return OpCode.CONV2D
    if lowered in {"add", "sum"}:
        return OpCode.RESIDUAL_ADD
    if lowered in {"relu", "gelu", "silu", "sigmoid", "tanh", "swiglu"}:
        return OpCode.ACTIVATION
    return opcode_for_tensor_name(lowered) if "." in lowered else OpCode.UNKNOWN


def _default_sensitivity(op: OpCode) -> float:
    if op in {OpCode.LAYER_NORM, OpCode.RMS_NORM, OpCode.SOFTMAX, OpCode.SAMPLING}:
        return 1.0
    if op in {OpCode.ATTENTION, OpCode.EMBEDDING}:
        return 0.85
    if op in {OpCode.MATMUL, OpCode.QUANTIZED_LINEAR, OpCode.MIXTURE_OF_EXPERTS}:
        return 0.6
    return 0.75
