"""Adaptive heterogeneous model compiler and runtime.

Imports are lazy so the lightweight device probe can run inside an isolated
accelerator environment without importing the full runtime.
"""

from importlib import import_module
from typing import Any


__version__ = "0.3.0"

_EXPORTS = {
    "AdaptiveCompiler": ("compiler", "AdaptiveCompiler"),
    "AdaptiveExecutionEngine": ("runtime", "AdaptiveExecutionEngine"),
    "DirectHardwareClient": ("device_runtime", "DirectHardwareClient"),
    "DirectHardwareError": ("device_runtime", "DirectHardwareError"),
    "HardwareProfiler": ("hardware", "HardwareProfiler"),
    "IRLevel": ("ir", "IRLevel"),
    "IRNode": ("ir", "IRNode"),
    "IRValue": ("ir", "IRValue"),
    "ModelFrontend": ("frontends", "ModelFrontend"),
    "ModelFormatError": ("frontends", "ModelFormatError"),
    "ModelIR": ("ir", "ModelIR"),
    "OpCode": ("ir", "OpCode"),
    "EngineStore": ("storage", "EngineStore"),
    "TensorSpec": ("ir", "TensorSpec"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
