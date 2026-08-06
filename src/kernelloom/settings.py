"""Load reproducible local-runtime configurations from JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .config import ModelConfig


@dataclass(slots=True)
class RuntimeConfig:
    """A local server plus the models it should keep resident."""

    models: list[ModelConfig] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 11435
    max_models: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_models < 1:
            raise ValueError("max_models must be positive")
        identifiers = [model.model_id for model in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("model_id values must be unique")
        if len(self.models) > self.max_models:
            raise ValueError("configured models exceed max_models")


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Read a KernelLoom JSON file, resolving model paths beside that file."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    server = payload.get("server", {})
    models = payload.get("models", [])
    if not isinstance(server, dict) or not isinstance(models, list):
        raise ValueError("server must be an object and models must be a list")
    resolved: list[ModelConfig] = []
    for index, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be an object")
        values: dict[str, Any] = dict(item)
        model_path = Path(str(values.get("model_path", ""))).expanduser()
        if not model_path.is_absolute():
            model_path = source.parent / model_path
        values["model_path"] = str(model_path)
        resolved.append(ModelConfig(**values))
    return RuntimeConfig(
        models=resolved,
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 11435)),
        max_models=int(server.get("max_models", 4)),
    )
