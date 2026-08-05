"""Public API for KernelLoom."""

from importlib import import_module
from typing import Any

from .config import ModelConfig
from .model import GenerationResult, KernelLoomModel

__version__ = "0.2.0"

_ENGINE_EXPORTS = {
    "AdaptiveCompiler": "AdaptiveCompiler",
    "AdaptiveExecutionEngine": "AdaptiveExecutionEngine",
    "DirectHardwareClient": "DirectHardwareClient",
    "DirectHardwareError": "DirectHardwareError",
    "EngineStore": "EngineStore",
    "HardwareProfiler": "HardwareProfiler",
    "ModelFormatError": "ModelFormatError",
    "ModelFrontend": "ModelFrontend",
}

__all__ = [
    "GenerationResult",
    "KernelLoomModel",
    "ModelConfig",
    *_ENGINE_EXPORTS,
]


def __getattr__(name: str) -> Any:
    target = _ENGINE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module("openagent_engine"), target)
    globals()[name] = value
    return value
