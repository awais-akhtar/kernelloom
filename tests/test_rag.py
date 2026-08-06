from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from kernelloom import (
    Document,
    DocumentLoader,
    InMemoryVectorStore,
    RAGConfig,
    RAGPipeline,
    SQLiteVectorStore,
    TextSplitter,
)


class KeywordEmbedder:
    words = ("python", "rust", "database", "ocean")

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in self.words]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class FakeGenerator:
    def __init__(self) -> None:
        self.last_prompt = ""

    def invoke(self, prompt: str, **options) -> str:
        self.last_prompt = prompt
        return "Python is used for the service. [Source 1]"

    async def ainvoke(self, prompt: str, **options) -> str:
        return self.invoke(prompt, **options)


class RAGTests(unittest.TestCase):
    def test_full_pipeline_ingests_retrieves_filters_and_answers(self) -> None:
        generator = FakeGenerator()
        pipeline = RAGPipeline(
            generator,
            KeywordEmbedder(),
            config=RAGConfig(chunk_size=80, chunk_overlap=10, retrieval="similarity", top_k=2, fetch_k=2),
        )
        count = pipeline.ingest([
            Document("Python powers the internal service and database API.", metadata={"team": "platform"}),
            Document("Rust is used for the ocean sensor firmware.", metadata={"team": "devices"}),
        ])

        self.assertEqual(count, 2)
        self.assertEqual(pipeline.count(), 2)
        results = pipeline.retrieve("Which language powers the database?", filters={"team": "platform"})
        self.assertEqual(results[0].document.metadata["team"], "platform")
        answer = pipeline.ask("Which language powers the database?", filters={"team": "platform"})
        self.assertIn("Python", answer.answer)
        self.assertIn("Context:", generator.last_prompt)
        self.assertIn("[Source 1]", generator.last_prompt)
        self.assertEqual(answer.to_dict()["sources"][0]["metadata"]["team"], "platform")

    def test_sqlite_store_persists_namespaces_and_upserts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.db"
            first = RAGPipeline(
                FakeGenerator(), KeywordEmbedder(), store=SQLiteVectorStore(path),
                config=RAGConfig(chunk_size=100, chunk_overlap=0, namespace="manuals"),
            )
            first.ingest(Document("Python database handbook", id="handbook"))
            first.ingest(Document("Python database handbook", id="handbook"))
            self.assertEqual(first.count(), 1)
            first.close()

            store = SQLiteVectorStore(path)
            second = RAGPipeline(FakeGenerator(), KeywordEmbedder(), store=store, config=RAGConfig(namespace="manuals"))
            self.assertEqual(second.count(), 1)
            self.assertEqual(second.retrieve("python database")[0].document.text, "Python database handbook")
            self.assertEqual(second.clear(), 1)
            second.close()

    def test_loader_handles_directory_csv_jsonl_and_inline_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "notes.md").write_text("Python notes", encoding="utf-8")
            (root / "rows.csv").write_text("name,topic\nAda,database\n", encoding="utf-8")
            (root / "events.jsonl").write_text('{"topic":"ocean"}\n', encoding="utf-8")
            documents = DocumentLoader().load(root)
        self.assertEqual(len(documents), 3)
        self.assertTrue(all(item.metadata.get("source") for item in documents))
        self.assertEqual(DocumentLoader().load("inline knowledge")[0].metadata["source"], "inline")

    def test_splitter_is_deterministic_and_overlapping(self) -> None:
        splitter = TextSplitter(chunk_size=60, chunk_overlap=10)
        document = Document("Sentence one. " * 12, id="source")
        first = splitter.split(document)
        second = splitter.split(document)
        self.assertGreater(len(first), 1)
        self.assertEqual([item.id for item in first], [item.id for item in second])
        self.assertLess(first[1].metadata["start"], first[0].metadata["end"])

    def test_async_pipeline(self) -> None:
        async def exercise() -> None:
            pipeline = RAGPipeline(
                FakeGenerator(), KeywordEmbedder(), config=RAGConfig(chunk_size=100, chunk_overlap=10)
            )
            self.assertEqual(await pipeline.aingest("Python database guide"), 1)
            result = await pipeline.aask("python database")
            self.assertIn("Python", result.answer)

        asyncio.run(exercise())

    def test_configuration_validation(self) -> None:
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            RAGConfig(prompt_template="{context}")

    def test_message_returning_generator_is_supported(self) -> None:
        generator = SimpleNamespace(invoke=lambda prompt, **options: SimpleNamespace(content="message answer"))
        pipeline = RAGPipeline(
            generator, KeywordEmbedder(), config=RAGConfig(chunk_size=100, chunk_overlap=10)
        )
        pipeline.ingest("Python database guide")
        self.assertEqual(pipeline.ask("python").answer, "message answer")


if __name__ == "__main__":
    unittest.main()
