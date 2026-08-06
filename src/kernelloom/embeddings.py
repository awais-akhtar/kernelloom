"""Optional CPU-optimized embedding backends for the local RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    backend: str
    model_name: str
    dimensions: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {"backend": self.backend, "model_name": self.model_name, "dimensions": self.dimensions}


class FastEmbedEmbedder:
    """Run supported ONNX embedding models efficiently on CPU through FastEmbed.

    FastEmbed uses ONNX Runtime locally.  It is optional because model downloads
    and the runtime should remain an explicit choice; pre-populate its cache or
    use an offline model cache in air-gapped deployments.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        batch_size: int = 256,
        providers: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive when supplied")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbed support is optional. Install it with: pip install 'kernelloom[fastembed]'"
            ) from exc
        kwargs: dict[str, Any] = {**options}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
        if threads is not None:
            kwargs["threads"] = threads
        if providers is not None:
            kwargs["providers"] = list(providers)
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self._lock = threading.RLock()
        self._dimensions: int | None = None

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain at least one non-empty string")
        with self._lock:
            rows = [[float(value) for value in vector] for vector in self._model.embed(list(texts), batch_size=self.batch_size)]
        if len(rows) != len(texts) or not rows or any(not row for row in rows):
            raise RuntimeError("FastEmbed did not return one non-empty vector per input")
        dimensions = {len(row) for row in rows}
        if len(dimensions) != 1:
            raise RuntimeError("FastEmbed returned inconsistent embedding dimensions")
        self._dimensions = dimensions.pop()
        return rows

    def warmup(self, text: str = "KernelLoom embedding warmup.") -> dict[str, str | int | None]:
        self.embed(text)
        return self.info()

    def info(self) -> dict[str, str | int | None]:
        return EmbeddingModelInfo("fastembed", self.model_name, self._dimensions).to_dict()

    def close(self) -> None:
        # FastEmbed does not require explicit native-resource teardown today.
        return None


__all__ = ["EmbeddingModelInfo", "FastEmbedEmbedder"]
