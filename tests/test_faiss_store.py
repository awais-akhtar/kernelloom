from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from kernelloom.rag import Document, RAGConfig, RAGPipeline, VectorStore
from kernelloom.faiss_store import FaissUnavailableError, FaissVectorStore, faiss_available


class FakeNumpy:
    float32 = "float32"
    int64 = "int64"

    @staticmethod
    def ascontiguousarray(value, *, dtype=None):
        if value and isinstance(value[0], (tuple, list)):
            return [list(row) for row in value]
        return list(value)


class FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension


class FakeIndexIDMap2:
    def __init__(self, base: FakeIndexFlatIP) -> None:
        self.dimension = base.dimension
        self.rows: dict[int, list[float]] = {}

    @property
    def ntotal(self) -> int:
        return len(self.rows)

    def add_with_ids(self, vectors, identifiers) -> None:
        for vector, identifier in zip(vectors, identifiers):
            if len(vector) != self.dimension:
                raise ValueError("wrong dimension")
            self.rows[int(identifier)] = list(vector)

    def remove_ids(self, identifiers) -> int:
        removed = 0
        for identifier in identifiers:
            if self.rows.pop(int(identifier), None) is not None:
                removed += 1
        return removed

    def search(self, queries, limit: int):
        query = queries[0]
        ranked = sorted(
            ((sum(left * right for left, right in zip(query, vector)), identifier)
             for identifier, vector in self.rows.items()),
            reverse=True,
        )[:limit]
        return [[score for score, _ in ranked]], [[identifier for _, identifier in ranked]]


class FakeFaiss:
    IndexFlatIP = FakeIndexFlatIP
    IndexIDMap2 = FakeIndexIDMap2


class KeywordEmbedder:
    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count("python")), float(lowered.count("database"))]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class FaissVectorStoreTests(TestCase):
    def _store(self, **options) -> FaissVectorStore:
        return FaissVectorStore(faiss_module=FakeFaiss, numpy_module=FakeNumpy, **options)

    def test_optional_dependency_error_is_clear_without_importing_faiss(self) -> None:
        def missing(_: str):
            raise ModuleNotFoundError("No module named 'faiss'", name="faiss")

        with patch("kernelloom.faiss_store.import_module", side_effect=missing):
            self.assertFalse(faiss_available())
            with self.assertRaisesRegex(FaissUnavailableError, "kernelloom\\[faiss\\]"):
                FaissVectorStore()

    def test_upsert_search_metadata_filtering_namespaces_and_delete(self) -> None:
        store = self._store()
        self.assertIsInstance(store, VectorStore)
        self.assertEqual(
            store.upsert(
                [
                    (Document("top result", "one", {"team": "platform"}), [1.0, 0.0]),
                    (Document("filtered result", "two", {"team": "docs"}), [0.9, 0.1]),
                    (Document("another filtered result", "three", {"team": "docs"}), [0.8, 0.2]),
                ],
                namespace="manuals",
            ),
            3,
        )
        store.upsert([(Document("isolated", "other"), [1.0, 0.0])], namespace="other")

        self.assertEqual(store.count(namespace="manuals"), 3)
        self.assertEqual([item.document.id for item in store.search([1.0, 0.0], limit=2, namespace="manuals")], ["one", "two"])
        self.assertEqual(
            [item.document.id for item in store.search([1.0, 0.0], limit=2, namespace="manuals", filters={"team": "docs"})],
            ["two", "three"],
        )
        self.assertEqual(store.delete(namespace="manuals", ids=["one", "missing", "one"]), 1)
        self.assertEqual(store.count(namespace="manuals"), 2)
        self.assertEqual(store.delete(namespace="manuals"), 2)
        self.assertEqual(store.count(namespace="other"), 1)

    def test_upsert_replaces_records_and_checks_dimensions(self) -> None:
        store = self._store(dimension=2)
        store.upsert([(Document("old", "same"), [1.0, 0.0])], namespace="docs")
        store.upsert([(Document("new", "same"), [0.0, 1.0])], namespace="docs")
        result = store.search([0.0, 1.0], limit=1, namespace="docs")
        self.assertEqual(store.count(namespace="docs"), 1)
        self.assertEqual(result[0].document.text, "new")
        with self.assertRaisesRegex(ValueError, "configured dimension"):
            store.upsert([(Document("wrong", "wrong"), [1.0, 0.0, 0.0])], namespace="docs")
        with self.assertRaisesRegex(ValueError, "query dimension"):
            store.search([1.0, 0.0, 0.0], limit=1, namespace="docs")

    def test_is_plug_and_play_with_the_rag_pipeline(self) -> None:
        generator = type("Generator", (), {"invoke": lambda _self, prompt, **_options: prompt})()
        pipeline = RAGPipeline(
            generator,
            KeywordEmbedder(),
            store=self._store(),
            config=RAGConfig(
                chunk_size=50,
                chunk_overlap=0,
                retrieval="similarity",
                top_k=1,
                fetch_k=1,
            ),
        )
        pipeline.ingest(Document("Python database handbook", "handbook", {"kind": "manual"}))
        results = pipeline.retrieve("python database", filters={"kind": "manual"})
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].document.text)
        self.assertEqual(pipeline.warmup()["store"], 1)

    def test_warmup_and_close(self) -> None:
        store = self._store()
        store.upsert([(Document("one", "one"), [1.0, 0.0])], namespace="a")
        store.upsert([(Document("two", "two"), [0.0, 1.0])], namespace="b")
        self.assertEqual(store.warmup(), 2)
        self.assertEqual(store.warmup(namespace="a"), 1)
        store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.count(namespace="a")
