"""Public API for KernelLoom."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .config import ModelConfig
from .cpu import CPUExecutionPlan, available_cpu_cores, plan_cpu_execution
from .model import GenerationResult, KernelLoomModel
from .embeddings import EmbeddingModelInfo, FastEmbedEmbedder
from .faiss_store import FaissUnavailableError, FaissVectorStore, faiss_available
from .rag import (
    Document,
    DocumentLoader,
    InMemoryVectorStore,
    KernelLoomEmbedder,
    RAGAnswer,
    RAGConfig,
    RAGPipeline,
    SQLiteVectorStore,
    SearchResult,
    TextSplitter,
)
from .settings import RuntimeConfig, load_runtime_config

_SOURCE_VERSION = "0.4.0"
try:
    _installed_version = version("kernelloom")
    __version__ = (
        _installed_version
        if _installed_version == _SOURCE_VERSION or _installed_version.startswith(f"{_SOURCE_VERSION}.post")
        else _SOURCE_VERSION
    )
except PackageNotFoundError:
    __version__ = _SOURCE_VERSION

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
    "CPUExecutionPlan",
    "available_cpu_cores",
    "plan_cpu_execution",
    "EmbeddingModelInfo",
    "FastEmbedEmbedder",
    "FaissUnavailableError",
    "FaissVectorStore",
    "faiss_available",
    "Document",
    "DocumentLoader",
    "InMemoryVectorStore",
    "KernelLoomEmbedder",
    "RAGAnswer",
    "RAGConfig",
    "RAGPipeline",
    "SQLiteVectorStore",
    "SearchResult",
    "TextSplitter",
    "RuntimeConfig",
    "load_runtime_config",
    *_ENGINE_EXPORTS,
]


def __getattr__(name: str) -> Any:
    target = _ENGINE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module("openagent_engine"), target)
    globals()[name] = value
    return value
