"""Optional FAISS vector store for fast, local RAG retrieval.

The module deliberately imports neither :mod:`faiss` nor :mod:`numpy` until a
``FaissVectorStore`` is constructed.  This keeps ``import kernelloom`` usable
on a minimal CPU-only installation.  Install the optional extra when fast
native similarity search is needed::

    pip install "kernelloom[faiss]"

``FaissVectorStore`` follows the ``VectorStore`` protocol from
``kernelloom.rag`` and can therefore be passed directly to ``RAGPipeline``.
It uses a separate ``IndexFlatIP`` index for every namespace.  With the
default ``metric="cosine"``, vectors are normalized before indexing, which
makes FAISS inner-product scores cosine-similarity scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
import threading
from typing import Any, Mapping, Sequence

from .rag import Document, SearchResult


class FaissUnavailableError(ImportError):
    """Raised when the optional FAISS runtime is not installed."""


def _load_faiss() -> Any:
    try:
        return import_module("faiss")
    except ModuleNotFoundError as error:
        if error.name == "faiss":
            raise FaissUnavailableError(
                "FAISS support requires the optional dependency. "
                "Install it with: pip install \"kernelloom[faiss]\""
            ) from error
        raise


def _load_numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as error:
        if error.name == "numpy":
            raise FaissUnavailableError(
                "FAISS support requires NumPy. Install it with: "
                "pip install \"kernelloom[faiss]\""
            ) from error
        raise


def faiss_available() -> bool:
    """Return whether the optional FAISS and NumPy runtime is available."""

    try:
        _load_faiss()
        _load_numpy()
    except FaissUnavailableError:
        return False
    return True


@dataclass(slots=True)
class _NamespaceIndex:
    dimension: int
    index: Any
    by_document: dict[str, int]
    by_numeric_id: dict[int, tuple[Document, tuple[float, ...]]]
    next_numeric_id: int = 1


class FaissVectorStore:
    """Thread-safe, CPU-accelerated FAISS store for :class:`RAGPipeline`.

    Parameters
    ----------
    dimension:
        Optional embedding width.  When omitted, it is inferred from the
        first insertion.  Supplying it catches accidental use of incompatible
        embedding models sooner.
    metric:
        ``"cosine"`` (the default) normalizes stored and query vectors before
        an inner-product search.  ``"inner_product"`` preserves raw vectors.

    Notes
    -----
    Metadata stays in local Python memory so that the RAG pipeline can return
    source text and apply its simple exact-match metadata filters.  Searches
    without filters remain native FAISS searches; filtering deliberately asks
    FAISS for all candidates in a namespace to keep filter results correct.

    ``faiss_module`` and ``numpy_module`` are optional injection points for
    embedding FAISS in another runtime and for dependency-free integration
    tests.  Normal applications should leave them unset.
    """

    def __init__(
        self,
        dimension: int | None = None,
        *,
        metric: str = "cosine",
        faiss_module: Any | None = None,
        numpy_module: Any | None = None,
    ) -> None:
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be positive when supplied")
        if metric not in {"cosine", "inner_product"}:
            raise ValueError("metric must be cosine or inner_product")

        self._faiss = faiss_module if faiss_module is not None else _load_faiss()
        self._numpy = numpy_module if numpy_module is not None else _load_numpy()
        self._dimension_hint = dimension
        self.metric = metric
        self._namespaces: dict[str, _NamespaceIndex] = {}
        self._lock = threading.RLock()
        self._closed = False

    def upsert(self, records: Sequence[tuple[Document, Sequence[float]]], *, namespace: str) -> int:
        """Insert or replace documents and embeddings in ``namespace``."""

        self._ensure_open()
        prepared = _validate_records(records)
        if not prepared:
            return 0
        if not namespace:
            raise ValueError("namespace cannot be empty")

        dimensions = {len(vector) for _, vector in prepared}
        if len(dimensions) != 1:
            raise ValueError("all embeddings in one upsert must have equal dimensions")
        dimension = dimensions.pop()
        if self._dimension_hint is not None and dimension != self._dimension_hint:
            raise ValueError(
                f"embedding dimension {dimension} does not match configured dimension {self._dimension_hint}"
            )

        # A batch with duplicate document ids has the same outcome as repeated
        # single upserts: its final occurrence wins, while the return value
        # still reports how many records were accepted.
        latest: dict[str, tuple[Document, tuple[float, ...]]] = {}
        for document, vector in prepared:
            latest[document.id] = (document, vector)

        with self._lock:
            state = self._namespaces.get(namespace)
            if state is None:
                state = self._new_namespace(dimension)
                self._namespaces[namespace] = state
            elif state.dimension != dimension:
                raise ValueError(
                    f"embedding dimension {dimension} does not match namespace dimension {state.dimension}"
                )

            replaced = [state.by_document[identifier] for identifier in latest if identifier in state.by_document]
            if replaced:
                state.index.remove_ids(self._ids_array(replaced))
                for identifier in latest:
                    numeric_id = state.by_document.pop(identifier, None)
                    if numeric_id is not None:
                        state.by_numeric_id.pop(numeric_id, None)

            numeric_ids: list[int] = []
            vectors: list[tuple[float, ...]] = []
            for document, vector in latest.values():
                numeric_id = state.next_numeric_id
                state.next_numeric_id += 1
                numeric_ids.append(numeric_id)
                vectors.append(self._index_vector(vector))
                state.by_document[document.id] = numeric_id
                state.by_numeric_id[numeric_id] = (document, vector)
            state.index.add_with_ids(self._matrix(vectors), self._ids_array(numeric_ids))
        return len(prepared)

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        namespace: str,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return highest-scoring results, preserving RAG metadata filters."""

        self._ensure_open()
        if limit < 1:
            return []
        query = _vector(embedding)
        with self._lock:
            state = self._namespaces.get(namespace)
            if state is None:
                return []
            if len(query) != state.dimension:
                raise ValueError(
                    f"query dimension {len(query)} does not match namespace dimension {state.dimension}"
                )
            total = len(state.by_numeric_id)
            if not total:
                return []
            # FAISS cannot filter arbitrary metadata itself.  Asking for every
            # candidate only when a filter is present preserves correctness.
            candidates = total if filters else min(limit, total)
            scores, identifiers = state.index.search(
                self._matrix([self._index_vector(query)]), candidates
            )
            results: list[SearchResult] = []
            for score, numeric_id in zip(scores[0], identifiers[0]):
                record = state.by_numeric_id.get(int(numeric_id))
                if record is None:  # FAISS uses -1 to pad undersized searches.
                    continue
                document, vector = record
                if _matches(document.metadata, filters):
                    results.append(SearchResult(document, float(score), vector))
                    if len(results) == limit:
                        break
            return results

    def delete(self, *, namespace: str, ids: Sequence[str] | None = None) -> int:
        """Delete selected records, or an entire namespace when ``ids`` is omitted."""

        self._ensure_open()
        with self._lock:
            state = self._namespaces.get(namespace)
            if state is None:
                return 0
            if ids is None:
                removed = len(state.by_numeric_id)
                del self._namespaces[namespace]
                return removed
            selected = [identifier for identifier in dict.fromkeys(ids) if identifier in state.by_document]
            if not selected:
                return 0
            numeric_ids = [state.by_document.pop(identifier) for identifier in selected]
            state.index.remove_ids(self._ids_array(numeric_ids))
            for numeric_id in numeric_ids:
                state.by_numeric_id.pop(numeric_id, None)
            if not state.by_numeric_id:
                del self._namespaces[namespace]
            return len(numeric_ids)

    def count(self, *, namespace: str) -> int:
        """Return the number of live records in a namespace."""

        self._ensure_open()
        with self._lock:
            state = self._namespaces.get(namespace)
            return len(state.by_numeric_id) if state is not None else 0

    def warmup(self, *, namespace: str | None = None) -> int:
        """Touch FAISS indexes before serving traffic and return warmed records.

        This does not change rankings or data.  It is useful for applications
        that want native FAISS code and index pages initialized during startup
        instead of on the first user query.
        """

        self._ensure_open()
        with self._lock:
            states = (
                [self._namespaces[namespace]]
                if namespace is not None and namespace in self._namespaces
                else list(self._namespaces.values()) if namespace is None else []
            )
            warmed = 0
            for state in states:
                total = len(state.by_numeric_id)
                if total:
                    state.index.search(self._matrix([tuple(0.0 for _ in range(state.dimension))]), 1)
                    warmed += total
            return warmed

    def close(self) -> None:
        """Release Python references to FAISS indexes and metadata."""

        with self._lock:
            self._namespaces.clear()
            self._closed = True

    def __enter__(self) -> "FaissVectorStore":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _new_namespace(self, dimension: int) -> _NamespaceIndex:
        base_index = self._faiss.IndexFlatIP(dimension)
        index = self._faiss.IndexIDMap2(base_index)
        return _NamespaceIndex(dimension, index, {}, {})

    def _matrix(self, vectors: Sequence[Sequence[float]]) -> Any:
        return self._numpy.ascontiguousarray(vectors, dtype=self._numpy.float32)

    def _ids_array(self, identifiers: Sequence[int]) -> Any:
        return self._numpy.ascontiguousarray(identifiers, dtype=self._numpy.int64)

    def _index_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        if self.metric == "inner_product":
            return tuple(vector)
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector) if norm else tuple(vector)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("FAISS vector store is closed")


def _vector(value: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if not vector or any(not math.isfinite(item) for item in vector):
        raise ValueError("embedding must be a non-empty sequence of finite numbers")
    return vector


def _validate_records(records: Sequence[tuple[Document, Sequence[float]]]) -> list[tuple[Document, tuple[float, ...]]]:
    prepared = [(document, _vector(vector)) for document, vector in records]
    if any(not document.id for document, _ in prepared):
        raise ValueError("stored documents must have an id")
    return prepared


def _matches(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    return not filters or all(metadata.get(key) == value for key, value in filters.items())


__all__ = ["FaissUnavailableError", "FaissVectorStore", "faiss_available"]
